import torch
import torchvision
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import sys
from IPython import display
from tqdm import tqdm
import warnings
torch.cuda.empty_cache()
warnings.filterwarnings("ignore")  # 忽略警告


def read_voc_images(
    root="D:/Projects/DeeplabV3_ResNet_Res2Net_semantic_segmentation/data/VOCdevkit/VOC2012", is_train=True, max_num=None
):
    txt_fname = "%s/ImageSets/Segmentation/%s" % (
        root,
        "train.txt" if is_train else "val.txt",
    )
    with open(txt_fname, "r") as f:
        images = f.read().split()  # 拆分成一个个名字组成list
    if max_num is not None:
        images = images[: min(max_num, len(images))]
    features, labels = [None] * len(images), [None] * len(images)
    for i, fname in tqdm(enumerate(images)):
        # 读入数据并且转为RGB的 PIL image
        features[i] = Image.open("%s/JPEGImages/%s.jpg" % (root, fname)).convert("RGB")
        labels[i] = Image.open("%s/SegmentationClass/%s.png" % (root, fname)).convert(
            "RGB"
        )
    return features, labels  # PIL image 0-255


# 这个函数可以不需要
def set_figsize(figsize=(3.5, 2.5)):
    """在jupyter使用svg显示"""
    display.set_matplotlib_formats("svg")
    # 设置图的尺寸
    plt.rcParams["figure.figsize"] = figsize


def show_images(imgs, num_rows, num_cols, scale=2):
    # a_img = np.asarray(imgs)
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    plt.show()
    return axes


# # 根据自己存放数据集的路径修改voc_dir
# voc_dir = r"D:\Projects\Models\data\VOCdevkit\VOC2012"
# train_features, train_labels = read_voc_images(voc_dir, max_num=10)
# n = 5  # 展示几张图像
# imgs = train_features[0:n] + train_labels[0:n]  # PIL image
# show_images(imgs, 2, n)

# 标签中每个RGB颜色的值
VOC_COLORMAP = [
    [0, 0, 0],[128, 0, 0],[0, 128, 0],[128, 128, 0],[0, 0, 128],[128, 0, 128],[0, 128, 128],[128, 128, 128],
    [64, 0, 0],[192, 0, 0],[64, 128, 0],[192, 128, 0],[64, 0, 128],[192, 0, 128],[64, 128, 128],[192, 128, 128],
    [0, 64, 0],[128, 64, 0],[0, 192, 0],[128, 192, 0],[0, 64, 128],
]
# 标签其标注的类别
VOC_CLASSES = [
    "background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair",
    "cow","diningtable","dog","horse","motorbike","person","potted plant","sheep","sofa",
    "train","tv/monitor",
]


colormap2label = torch.zeros(256 ** 3, dtype=torch.uint8)  # torch.Size([16777216])
for i, colormap in enumerate(VOC_COLORMAP):
    # 每个通道的进制是256，这样可以保证每个 rgb 对应一个下标 i
    colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i

# 构造标签矩阵
def voc_label_indices(colormap, colormap2label):
    colormap = np.array(colormap.convert("RGB")).astype("int32")
    idx = (colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2]
    return colormap2label[idx]  # colormap 映射 到colormaplabel中计算的下标


# y = voc_label_indices(train_labels[0], colormap2label)
# print(y[100:110, 130:140])  # 打印结果是一个int型tensor，tensor中的每个元素i表示该像素的类别是VOC_CLASSES[i]


def voc_rand_crop(feature, label, height, width):
    """
    随机裁剪feature(PIL image) 和 label(PIL image).
    为了使裁剪的区域相同，不能直接使用RandomCrop，而要像下面这样做
    Get parameters for ``crop`` for a random crop.
    Args:
        img (PIL Image): Image to be cropped.
        output_size (tuple): Expected output size of the crop.
    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
    """
    i, j, h, w = torchvision.transforms.RandomCrop.get_params(
        feature, output_size=(height, width)
    )
    feature = torchvision.transforms.functional.crop(feature, i, j, h, w)
    label = torchvision.transforms.functional.crop(label, i, j, h, w)
    return feature, label


# 显示n张随机裁剪的图像和标签，前面的n是5
# imgs = []
# for _ in range(n):
#     imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)
# show_images(imgs[::2] + imgs[1::2], 2, n)


class VOCSegDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, crop_size, voc_dir, colormap2label, max_num=None):
        """
        crop_size: (h, w)
        """
        # 对输入图像的RGB三个通道的值分别做标准化
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.tsf = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std),
            ]
        )
        self.crop_size = crop_size  # (h, w)
        features, labels = read_voc_images(
            root=voc_dir, is_train=is_train, max_num=max_num
        )
        # 由于数据集中有些图像的尺寸可能小于随机裁剪所指定的输出尺寸，这些样本需要通过自定义的filter函数所移除
        self.features = self.filter(features)  # PIL image
        self.labels = self.filter(labels)  # PIL image
        self.colormap2label = colormap2label
        # print("read " + str(len(self.features)) + " valid examples")
        print("read " + str(len(self.features)) + " train examples" if is_train else "read " + str(len(self.features)) + " valid examples")

    def filter(self, imgs):
        return [
            img
            for img in imgs
            if (img.size[1] >= self.crop_size[0] and img.size[0] >= self.crop_size[1])
        ]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(
            self.features[idx], self.labels[idx], *self.crop_size
        )
        # float32 tensor           uint8 tensor (b,h,w)
        return (self.tsf(feature), voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)


if __name__ == "__main__":
    # 根据自己存放数据集的路径修改voc_dir
    voc_dir = r"D:\Projects\Models\data\VOCdevkit\VOC2012"
    batch_size = 10  # 实际上我的小笔记本不允许我这么做！哭了（大家根据自己电脑内存改吧）
    crop_size = (320, 480)  # 指定随机裁剪的输出图像的形状为(320,480)
    max_num = 20000  # 最多从本地读多少张图片，我指定的这个尺寸过滤完不合适的图像之后也就只有1175张~

    # 创建训练集和测试集的实例
    voc_train = VOCSegDataset(True, crop_size, voc_dir, colormap2label, max_num)
    voc_test = VOCSegDataset(False, crop_size, voc_dir, colormap2label, max_num)

    # 设批量大小为32，分别定义【训练集】和【测试集】的数据迭代器
    num_workers = 0 if sys.platform.startswith("win32") else 4
    train_iter = torch.utils.data.DataLoader(
        voc_train, batch_size, shuffle=True, drop_last=True, num_workers=num_workers
    )
    test_iter = torch.utils.data.DataLoader(
        voc_test, batch_size, drop_last=True, num_workers=num_workers
    )

    # 方便封装，把训练集和验证集保存在dict里
    dataloaders = {"train": train_iter, "val": test_iter}
    dataset_sizes = {"train": len(voc_train), "val": len(voc_test)}

    for i,(input,label) in enumerate(train_iter):
        print(label.shape)
        print(label)








