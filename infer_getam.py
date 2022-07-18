from cv2 import transform
import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
# from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import torch.nn.functional as F
import os
# from datetime import datetime
from PIL import Image
from tool import pyutils, imutils, torchutils
import cv2
from DPT.DPT import DPTSegmentationModel
from DPT.mirrorformer import MirrorFormer
import myTool as mytool
# from DenseEnergyLoss import DenseEnergyLoss
# import shutil
# import pamr
from pamr import PAMR
# import random
import torch.multiprocessing as mp
import torch.distributed as dist
# import seaborn as sns
import matplotlib.pyplot as plt



import visdom

classes = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep',
            'sofa','train','tvmonitor']

def setup(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)



def main():
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--max_epoches", default=1, type=int)
    parser.add_argument("--lr", default=0.04, type=float)
    parser.add_argument("--step_lr", default=False, type=bool)

    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    # parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    # parser.add_argument("--val_list", default="voc12/val(id).txt", type=str)
    parser.add_argument("--LISTpath", default="voc12/train.txt", type=str)
    parser.add_argument("--backbone", default="vitb_hybrid", type=str)
    parser.add_argument("--address", default="7777", type=str)

    parser.add_argument('--densecrfloss', type=float, default=1e-7,
                        metavar='M', help='densecrf loss (default: 0)')
    parser.add_argument('--rloss-scale', type=float, default=0.5,
                        help='scale factor for rloss input, choose small number for efficiency, domain: (0,1]')
    parser.add_argument('--sigma-rgb', type=float, default=15.0,
                        help='DenseCRF sigma_rgb')
    parser.add_argument('--sigma-xy', type=float, default=100,
                        help='DenseCRF sigma_xy')
    parser.add_argument("--weights", default='./netWeights/RRM_final.pth', type=str)
    parser.add_argument("--out_cam", default='output/cam_npy', type=str)
    parser.add_argument("--heatmap", default='output/baseline', type=str)


    parser.add_argument("--session_name", default="vit_cls_seg", type=str)
    parser.add_argument("--crop_size", default=384, type=int)
    parser.add_argument("--voc12_root", default='/home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/', type=str)
    parser.add_argument("--IMpath", default="/home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/JPEGImages", type=str)

    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')

    args = parser.parse_args()

    ######################################################### set processes
    args.world_size = args.gpus * args.nodes                           #
    os.environ['MASTER_ADDR'] = 'localhost'                            #
    os.environ['MASTER_PORT'] = args.address                                #
    mp.spawn(train, nprocs=args.gpus, args=(args,), join=True)         #
    #########################################################

def train(gpu, args):
    vis = visdom.Visdom()
    rank = args.nr * args.gpus + gpu
    print(rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    setup(rank)

    # model = DPTSegmentationModel(num_classes=20, backbone_name=args.backbone)
    model = MirrorFormer(num_classes=20, backbone_name=args.backbone) 

    weights_dict = torch.load(args.weights)
    model.load_state_dict(weights_dict, strict=False)

    model.eval()
    model.cuda()

    # pixel adaptive refine module
    pamr = PAMR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24]).cuda()


    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu],output_device=[gpu], find_unused_parameters=True)

    flipper1 = transforms.RandomHorizontalFlip(p=1)
    flipper2 = transforms.RandomVerticalFlip(p=1)
    

    print(vars(args))

    # img_list = mytool.read_file('voc12/val_id.txt')
    img_list = mytool.read_file_2(args.LISTpath)

    max_step = len(img_list)

    data_gen = mytool.chunker(img_list, 1)

    timer = pyutils.Timer("Session started: ")

    for iter in range(max_step):
        print(iter)
        chunk = data_gen.__next__()
        img_list = chunk
        img, ori_images, label, name_list = mytool.get_data_from_chunk_val(chunk, args)        
        img = img.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        
        # img = flipper1(img)
        # ori_images = np.flip(ori_images, axis = 3)
        name = name_list[0]
        # rgb_img = cv2.imread('/home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/JPEGImages/{}.jpg'.format(name))
        rgb_img = cv2.imread('{}/{}.jpg'.format(args.IMpath, name))
        W,H,_ = rgb_img.shape

        # generate getam
        getam = True
        multi_scale = True
        cam_list = []
        b,c,h,w = img.shape
        if getam:
            for scale in [1]:
                for hflip in [1,2]:
                    cam_matrix = torch.zeros((b, 20, W, H))
                    model.zero_grad()

                    input = F.interpolate(img, size=(int(h * scale), int(w * scale)), mode='bilinear',align_corners=False)
                    if hflip%2 == 1:
                        input = flipper1(input)
                    
                    # cls_pred, _, attn, _ = model.forward_cls(input)
                    cls_pred, _, attn = model.forward_cam(input)


                    patch_aff = attn[:,:,1:,1:]
                    patch_aff = torch.sum(patch_aff, dim=1)
                    # print(patch_aff.shape)
                

                    original_img = ori_images[0]
                    cur_label = label[0, :]
                    output = cls_pred[0, :]
                    for class_index in range(20):
                        if cur_label[class_index] > 1e-5:
                            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
                            one_hot[0, class_index] = 1
                            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
                            one_hot = torch.sum(one_hot.cuda() * output)
                
                            model.zero_grad()
                            one_hot.backward(retain_graph=True)
                            cam, _, _ = model.getam(0, start_layer=6)
                            
                            # print(cam.shape, patch_aff.shape)

                            # patch aff refine
                            cam = torch.matmul(patch_aff, cam.unsqueeze(2))
                            # cam = torch.matmul(cam.unsqueeze(1), patch_aff)
                            # print(cam.shape)

                            cam = cam.reshape(int((h*scale) //16), int((w*scale) //16))

                            
                            # print(cam.shape)
                            
                            # cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), (args.crop_size, args.crop_size), mode='bilinear', align_corners=True)
                            # print(cam.shape)
                            # cam = cam[:,:,crop_list[0]:crop_list[0]+crop_list[1], crop_list[2]:crop_list[2]+crop_list[3]]

                            # print(cam.shape)
                            
                            cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), (W, H), mode='bilinear', align_corners=True)
                            cam_matrix[0, class_index,:,:] = cam
                    
                    # if hflip==1:
                        # cam_matrix=flipper1(cam_matrix)

                    cam_up_single = cam_matrix[0,:,:,:]
                    # print(cam_up_single.shape)
                    rgb_img = rgb_img.transpose(2,0,1)

                    # pamr ---------------------
                    # cam_up_single = pamr((torch.from_numpy(rgb_img)).unsqueeze(0).float().cuda(), cam_up_single.unsqueeze(0).cuda()).squeeze(0)
                    # cam_up_single = F.interpolate(cam_up_single.unsqueeze(0), (W, H), mode='bilinear', align_corners=True)
                    # cam_up_single = cam_up_single[0]
                    # --------------------------

                    cam_up_single = cam_up_single.cpu().data.numpy()
                    
                    if hflip%2 == 1:
                        # print(cam_up_single.shape)
                        cam_up_single = np.flip(cam_up_single, axis=2)
                    
                    cam_list.append(cam_up_single)  

            sum_cam = np.sum(cam_list, axis=0)
            norm_cam = (sum_cam - np.min(sum_cam, (1, 2), keepdims=True)) / (np.max(sum_cam, (1, 2), keepdims=True) - np.min(sum_cam, (1, 2), keepdims=True) + 1e-5 )  
            # norm_cam = cv2.resize(norm_cam, (H,W))

                        # print(norm_cam.shape)  

                        # norm_cam = cam_up_single / (np.max(cam_up_single, (1, 2), keepdims=True) + 1e-5)
                        # print(norm_cam.shape)
                        # cam_list.append(norm_cam)           

                    # original_img = original_img.transpose(1,2,0).astype(np.uint8)
            
            cam_dict = {}
            for cam_class in range(20):
                if cur_label[cam_class] > 1e-5:
                    cam_dict[cam_class] = norm_cam[cam_class]
            
            if args.out_cam is not None:
                np.save(os.path.join(args.out_cam, name + '.npy'), cam_dict)

            ori_img = ori_images[0].transpose(1, 2, 0).astype(np.uint8)
            # ori_img = rgb_img.transpose(1,2,0)
            # print(ori_img.shape, rgb_img.shape)
                    
            for cam_class in range(20):
                if cur_label[cam_class] > 1e-5:
                    mask = norm_cam[cam_class,:]
                    # print(mask.shape)

        
                    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)

                    ori_img = cv2.resize(ori_img, (heatmap.shape[1], heatmap.shape[0]))
                    # print(ori_img.shape, heatmap.shape)
                    cam_output = heatmap * 0.5 + ori_img * 0.5
                    cv2.imwrite(os.path.join(args.heatmap, name + '_{}.jpg'.format(classes[cam_class])), cam_output)

        torch.distributed.barrier()
    torch.distributed.destroy_process_group()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

    main()

