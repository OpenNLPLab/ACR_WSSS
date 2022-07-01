from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.backends import cudnn

cudnn.enabled = True
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import torch.nn.functional as F
import os
from datetime import datetime
from PIL import Image
from tool import pyutils, imutils, torchutils
from pamr import PAMR
import PIL.Image
import seaborn as sns
import matplotlib as mpl 


import cv2
from DPT.DPT import DPTSegmentationModel


import voc12.data

import myTool as mytool
from myTool import compute_joint_loss, compute_seg_label, compute_cam_up, compute_seg_label_two_step, decode_segmap, compute_seg_label_3

classes = ['aeroplane',
           'bicycle',
           'bird',
           'boat',
           'bottle',
           'bus',
           'car',
           'cat',
           'chair',
           'cow',
           'diningtable',
           'dog',
           'horse',
           'motorbike',
           'person',
           'pottedplant',
           'sheep',
           'sofa',
           'train',
           'tvmonitor']

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--max_epoches", default=1, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--LISTpath", default="voc12/train.txt", type=str)
    parser.add_argument("--weights", default='./netWeights/RRM_final.pth', type=str)


    parser.add_argument("--crop_size", default=256, type=int)
    parser.add_argument("--voc12_root", default='/home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/', type=str)
    parser.add_argument("--IMpath", default="/home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/JPEGImages", type=str)

    parser.add_argument('--crf_la_value', type=int, default=4)
    parser.add_argument('--crf_ha_value', type=int, default=32)

    parser.add_argument('--densecrfloss', type=float, default=1e-7,
                        metavar='M', help='densecrf loss (default: 0)')
    parser.add_argument('--rloss-scale', type=float, default=0.5,
                        help='scale factor for rloss input, choose small number for efficiency, domain: (0,1]')
    parser.add_argument('--sigma-rgb', type=float, default=15.0,
                        help='DenseCRF sigma_rgb')
    parser.add_argument('--sigma-xy', type=float, default=100,
                        help='DenseCRF sigma_xy')

    args = parser.parse_args()

    model = DPTSegmentationModel(num_classes=20)
    weights_dict = torch.load(args.weights)

    # weights_dict = torch.load('/data2/u5876230/ete_wsss_weight/weight/ori/train_from_init_10_last_vitb16_384.pth')
    model.load_state_dict(weights_dict, strict=False)

    model.cuda()    
    model.eval()

    print(vars(args))

    batch_size = args.batch_size
    img_list = mytool.read_file_2(args.LISTpath)

    # pixel adaptive refine module
    pamr = PAMR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24]).cuda()

    max_step = (len(img_list) // args.batch_size) * args.max_epoches
    print(len(img_list))
    print(max_step)

    data_list = []
    for i in range(int(max_step // 100)):
        # np.random.shuffle(img_list)
        data_list.extend(img_list)

    data_gen = mytool.chunker(data_list, batch_size)

    timer = pyutils.Timer("Session started: ")

    accuracy_list = []

    z1 = np.zeros(256)
    z2 = np.zeros(256)
    z3 = np.zeros(256)


    for iter in range(len(img_list)):
        print(iter)
        chunk = data_gen.__next__()
        img_list = chunk
        img, ori_images, label,  name_list = mytool.get_data_from_chunk_val(chunk,args)
        # if name_list[0] in ['2011_002389', '2011_002398']:
        if True:
            saliency_map_path = os.path.join('/home/users/u5876230/swin_sod/pascal/', '{}.png'.format(name_list[0]))
            saliency = PIL.Image.open(saliency_map_path)
            saliency = np.asarray(saliency)
            saliency = cv2.resize(saliency,(args.crop_size, args.crop_size))
            
            img = img.cuda()
            b, c, h, w = img.shape

            x = model.forward_cls(img)


            # generate cam and segmentation label on the go: ####################################
            cam_matrix = torch.zeros((b, 20, w, h))
            seg_label = np.zeros((b, w, h))
            saliency_pseudo = np.zeros((b, w, h))

            for cam_index in range(1):
                for batch in range(b):
                    name = name_list[batch]
                    original_img = ori_images[batch]
                    cur_label = label[batch, :]
                    output = x[batch, :]
                    for class_index in range(20):
                        if cur_label[class_index] > 1e-5:
                            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
                            one_hot[0, class_index] = 1
                            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
                            one_hot = torch.sum(one_hot.cuda() * output)
                
                            model.zero_grad()
                            one_hot.backward(retain_graph=True)

                            cam, _, cam_list = model.grad_cam(batch, start_layer=0)
                            # z1 = z1 + cam.detach().cpu().numpy()

                            # cam, _, cam_list = model.generate_cam_3(batch, start_layer=0)
                            # z2 = z2 + cam.detach().cpu().numpy()

                            # cam, _, cam_list = model.generate_cam_4(batch, start_layer=0)
                            # z3 = z3 + cam.detach().cpu().numpy()

                            # cam = cam_list[cam_index][:, 0, 1:]
                            
                            cam = cam.reshape(int(h //16), int(w //16))

                            cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), (w, h), mode='bilinear', align_corners=True)
                            cam_matrix[batch, class_index,:,:] = cam
                    
                    cam_up_single = cam_matrix[batch,:,:,:]
                    # cam_up_single = cam_up_single/(torch.amax(cam_up_single, (1, 2), keepdim=True) + 1e-5)
                    # cam_up_single = pamr((torch.from_numpy(original_img)).unsqueeze(0).float().cuda(), cam_up_single.unsqueeze(0).cuda()).squeeze(0)

                    cam_up_single = cam_up_single.cpu().data.numpy()
                    norm_cam = cam_up_single / (np.max(cam_up_single, (1, 2), keepdims=True) + 1e-5)

                    saliency_map = torch.from_numpy(saliency)
                    saliency_map[saliency_map>0] = 1
                    original_img = original_img.transpose(1,2,0).astype(np.uint8)
                    compute_seg_label_two_step(original_img, cur_label.cpu().numpy(), \
                    norm_cam, name, iter, saliency_map.data.numpy(), x[batch, :], save_heatmap=True, save_pseudo=False, cam_index=cam_index)
        else:
            break
    

    # z1 = (z1-np.min(z1))/np.max(z1)
    # z2 = (z2-np.min(z2))/np.max(z2)
    # z3 = (z3-np.min(z3))/np.max(z3)

    # mpl.rc("figure", figsize=(6,4)) 
    # sns.set_palette("hls") 
    # sns.distplot(z1,bins=None,kde_kws={"color":"blue", "lw":3 }, hist_kws={ "color": "b" }, hist=False, label='sum') 
    # sns.distplot(z2,bins=None,kde_kws={"color":"seagreen", "lw":3 }, hist_kws={ "color": "g" }, hist=False, label='dot') 
    # sns.distplot(z3,bins=None,kde_kws={"color":"red", "lw":3 }, hist_kws={ "color": "r" }, hist=False, label='multiplication') 

    # # sns.kdeplot(data=z1) 
    # # sns.kdeplot(data=z2) 
    # # sns.kdeplot(data=z3) 

    # plt.show()
    # plt.savefig('dot.png')
    
