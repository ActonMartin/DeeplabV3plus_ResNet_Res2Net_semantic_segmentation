import torch
import numpy as np
import cv2
from deeplabV3plus import DeepLabv3_plus
import os
import torchvision.transforms as transforms
import time
from PIL import Image
from cityscapes_labels import label2trainid
from restore_model import restore_snapshot

# palette = [
#     128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153,
#     153, 153, 250, 170, 30, 220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130,
#     180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70, 0, 60, 100, 0, 80, 100, 0,
#     0, 230, 119, 11, 32
# ]

palette =[0, 0, 0,128, 0, 0,0, 128, 0,128, 128, 0,0, 0, 128,128, 0, 128,0, 128, 128,128, 128, 128,
64, 0, 0,192, 0, 0,64, 128, 0,192, 128, 0,64, 0, 128,192, 0, 128,64, 128, 128,192, 128, 128,
0, 64, 0,128, 64, 0,0, 192, 0,128, 192, 0,0, 64, 128]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)



def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


net = DeepLabv3_plus(
        nInputChannels=3, n_classes=21, os=16, pretrained=True, _print=True).cuda()
print('Net built.')
snapshot_path = 'D:/Projects/Models/checkpoint_model_epoch_20.pth.tar'
# net= restore_snapshot(net, snapshot=snapshot_path)
net.load_state_dict(torch.load(snapshot_path))
net.eval()
print('Net restored.')

# get data
data_dir = 'D:/Projects/Models/test_images_folder'
save_dir = 'D:/Projects/Models/test_images_result'
images = os.listdir(data_dir)
if len(images) == 0:
    print('There are no images at directory %s. Check the data path.' % (data_dir))
else:
    print('There are %d images to be processed.' % (len(images)))
images.sort()

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

start_time = time.time()
for img_id, img_name in enumerate(images):
    img_dir = os.path.join(data_dir, img_name)
    img = Image.open(img_dir).convert('RGB')
    img_tensor = img_transform(img)

    # predict
    with torch.no_grad():
        pred = net(img_tensor.unsqueeze(0).cuda())
        print('%04d/%04d: Inference done.' % (img_id + 1, len(images)))

    pred = pred.cpu().numpy().squeeze()
    pred = np.argmax(pred, axis=0)

    color_name = 'color_mask_' + img_name
    overlap_name = 'overlap_' + img_name
    pred_name = 'pred_mask_' + img_name
    # save colorized predictions
    colorized = colorize_mask(pred)
    colorized.save(os.path.join(save_dir, color_name))

    # save colorized predictions overlapped on original images
    overlap = cv2.addWeighted(np.array(img), 0.5, np.array(colorized.convert('RGB')), 0.5, 0)
    cv2.imwrite(os.path.join(save_dir, overlap_name), overlap[:, :, ::-1])

    # save label-based predictions, e.g. for submission purpose
    label_out = np.zeros_like(pred)
    for label_id, train_id in label2trainid.items():
        label_out[np.where(pred == train_id)] = label_id
        cv2.imwrite(os.path.join(save_dir, pred_name), label_out)
end_time = time.time()

print('Results saved.')
print('Inference takes %4.2f seconds, which is %4.2f seconds per image, including saving results.' % (end_time - start_time, (end_time - start_time)/len(images)))
