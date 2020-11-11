import time
import torch
import sys
from dataset2 import VOCSegDataset,colormap2label
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter


def train_model(model, criterion, optimizer, scheduler,dataloaders,dataset_sizes, num_epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    since = time.time()
    writer = SummaryWriter()
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # 每个epoch都有一个训练和验证阶段
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 30)
        for phase in ["train", "val"]:
            if phase == "train":
                scheduler.step()
                model.train()
            else:
                model.eval()
            runing_loss = 0.0
            runing_corrects = 0.0
            # 迭代一个epoch
            for batch_index,(inputs, labels) in enumerate(dataloaders[phase]):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()  # 零参数梯度
                # 前向，只在训练时跟踪参数
                with torch.set_grad_enabled(phase == "train"):
                    logits = model(inputs)  # [5, 21, 320, 480]
                    # print('inputs.shape', inputs.shape) # torch.Size([2, 3, 320, 480])
                    # print('logits.shape',logits.shape) # torch.Size([2, 21, 320, 480])
                    # print('labels.shape',labels.shape) # torch.Size([2, 320, 480])
                    # loss = criterion(logits, labels.long())
                    loss = criterion(logits, labels.long())
                    # 后向，只在训练阶段进行优化
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                # 统计loss和correct
                runing_loss += loss.item() * inputs.size(0)
                runing_corrects += torch.sum(
                    (torch.argmax(logits.data, 1)) == labels.data
                ) // (480 * 320)
                iteration = epoch * dataset_sizes[phase] + batch_index
                writer.add_scalar(str(phase) + "_runing_loss", runing_loss,iteration)
                writer.add_scalar(str(phase) + "_runing_corrects", runing_corrects,iteration)
                print('{}_runing_loss_{},runing_corrects_{}'.format(phase,runing_loss,runing_corrects))
            epoch_loss = runing_loss / dataset_sizes[phase]
            epoch_acc = runing_corrects.double() / dataset_sizes[phase]
            writer.add_scalar(str(phase)+"_loss", epoch_loss,epoch)
            writer.add_scalar(str(phase)+"_acc", epoch_acc,epoch)
            writer.close()
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            # 深度复制model参数
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                # best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(),'checkpoint_model_epoch_{}.pth.tar'.format(epoch))
                print("已经保存 checkpoint_model_epoch_{}.pth.tar".format(epoch))
    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))


if __name__ == "__main__":
    torch.cuda.empty_cache()
    # 根据自己存放数据集的路径修改voc_dir
    voc_dir = r"D:\Projects\Models\data\VOCdevkit\VOC2012"
    batch_size = 8  # 实际上我的小笔记本不允许我这么做！哭了（大家根据自己电脑内存改吧）
    crop_size = (320, 480)  # 指定随机裁剪的输出图像的形状为(320,480)
    max_num = 20000  # 最多从本地读多少张图片，我指定的这个尺寸过滤完不合适的图像之后也就只有1175张~

    # 创建训练集和测试集的实例
    voc_train = VOCSegDataset(True, crop_size, voc_dir, colormap2label, max_num)
    voc_test = VOCSegDataset(False, crop_size, voc_dir, colormap2label, max_num)

    # 设批量大小为32，分别定义【训练集】和【测试集】的数据迭代器
    num_workers = 0 if sys.platform.startswith("win32") else 4
    train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(voc_test, batch_size, drop_last=True, num_workers=num_workers)

    # 方便封装，把训练集和验证集保存在dict里
    dataloaders = {"train": train_iter, "val": test_iter}
    dataset_sizes = {"train": len(voc_train), "val": len(voc_test)}

    from deeplabV3plus import DeepLabv3_plus
    model = DeepLabv3_plus(
        nInputChannels=3, n_classes=21, os=16, pretrained=True, _print=True
    )

    epochs = 100  # 训练5个epoch
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4, momentum=0.9)
    # 每3个epochs衰减LR通过设置gamma=0.1
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    train_model(model.cuda(),criterion,optimizer,exp_lr_scheduler,dataloaders,dataset_sizes,epochs)