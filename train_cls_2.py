# -*- coding: UTF-8 -*- 
from curses import flash
import numpy as np
import torch
import os
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
from torch.utils.tensorboard import SummaryWriter  
import visdom
# from network.vit import *

classes = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep',
            'sofa','train','tvmonitor']

def setup(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)


def validation(model, args, optimizer, writer):
    val_list = mytool.read_file('voc12/val_id.txt')
    data_gen = mytool.chunker(val_list, 1)
    model.eval()
    val_loss_meter = pyutils.AverageMeter('loss')

    val_step = (len(val_list)//(1 * args.gpus ))

    with torch.no_grad():
        for iter in range(val_step):
            chunk = data_gen.__next__()
            img_list = chunk
            # img, ori_images, label, croppings, name_list = mytool.get_data_from_chunk_v2(chunk, args)
            img, ori_images, label, name_list = mytool.get_data_from_chunk_val(chunk, args)
            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            x,_ = model.module.forward_cls(img)
            # x, cam = model.module.forward_cam_multiscale(img)
            loss = F.multilabel_soft_margin_loss(x, label)
            val_loss_meter.add({'loss': loss.item()})
            writer.add_scalar('val loss', loss.item(), optimizer.global_step)


            save_cam = False
            # visualize cam
            if save_cam:
                label = label.cpu()
                # print(ori_images.shape)
                b, c, h, w = ori_images.shape
                ori_img = ori_images[0].transpose(1, 2, 0).astype(np.uint8)
                
                cam = F.upsample(cam, (h,w), mode='bilinear', align_corners=False)[0]
                cam = cam.detach().cpu().numpy() * label.clone().view(20, 1, 1).numpy()
                cam = cam / (np.max(cam, (1, 2), keepdims=True) + 1e-5)

                cam_dict = {}
                for i in range(20):
                    if label[0][i] > 1e-5:
                        cam_dict[i] = cam[i]
                
                keys = list(cam_dict.keys())
                for target_class in keys:
                    mask = cam_dict[target_class]
                
                    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)

                    ori_img = cv2.resize(ori_img, (heatmap.shape[1], heatmap.shape[0] ))
                    cam_output = heatmap * 0.5 + ori_img*0.5
                    cv2.imwrite(os.path.join('output/003/', name_list[0] + '_{}_cam.jpg'.format(classes[target_class])), cam_output)

    model.train()
    print('loss:', val_loss_meter.pop('loss'))

    return

def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明:y_true和y_pred的shape一致,y_true的元素非0即1,
         1表示对应的类为目标类,0表示对应的类为非目标类。
    警告:请保证y_pred的值域是全体实数,换言之一般情况下y_pred
         不用加激活函数,尤其是不能加sigmoid或者softmax!预测
         阶段则输出y_pred大于0的类。如有疑问,请仔细阅读并理解
         本文。
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat((y_pred_neg, zeros), dim=-1)
    y_pred_pos = torch.cat((y_pred_pos, zeros), dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return neg_loss + pos_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--max_epoches", default=10, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--step_lr", default=False, type=bool)

    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val(id).txt", type=str)
    parser.add_argument("--LISTpath", default="voc12/train_aug(id).txt", type=str)
    parser.add_argument("--backbone", default="vitb_hybrid", type=str)
    parser.add_argument("--address", default="8889", type=str)

    # parser.add_argument('--densecrfloss', type=float, default=1e-7,
    #                     metavar='M', help='densecrf loss (default: 0)')
    # parser.add_argument('--rloss-scale', type=float, default=0.5,
    #                     help='scale factor for rloss input, choose small number for efficiency, domain: (0,1]')
    # parser.add_argument('--sigma-rgb', type=float, default=15.0,
    #                     help='DenseCRF sigma_rgb')
    # parser.add_argument('--sigma-xy', type=float, default=100,
    #                     help='DenseCRF sigma_xy')

    parser.add_argument("--session_name", default="vit_cls_seg", type=str)
    parser.add_argument("--crop_size", default=256, type=int)
    parser.add_argument("--voc12_root", default='/home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/', type=str)
    parser.add_argument("--IMpath", default="/home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/JPEGImages", type=str)

    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')

    args = parser.parse_args()

    try:
        os.mkdir('tensorboard/{}'.format(args.session_name))
    except:
        pass

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

    if rank==0:
        writer = SummaryWriter('tensorboard/{}'.format(args.session_name))
    
    model = DPTSegmentationModel(num_classes=20, backbone_name=args.backbone) 

    torch.cuda.set_device(gpu)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(gpu)
    model.train()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu],output_device=[gpu], find_unused_parameters=True)

    flipper1 = transforms.RandomHorizontalFlip(p=1)
    flipper2 = transforms.RandomVerticalFlip(p=1)

    print(vars(args))

    batch_size = args.batch_size
    img_list = mytool.read_file(args.LISTpath)

    max_step = (len(img_list)//(args.batch_size * args.gpus )) * args.max_epoches
    # lr_step = int(max_step//6)

    data_list = []
    for i in range(int(args.max_epoches) + 1):
        np.random.shuffle(img_list)
        data_list.extend(img_list)

    data_gen = mytool.chunker(data_list, batch_size)

    params = model.parameters()
    optimizer = torchutils.PolyOptimizer(params, lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    avg_meter = pyutils.AverageMeter('loss')

    timer = pyutils.Timer("Session started: ")

    cls_loss_list = []
    for iter in range(max_step+1):
        chunk = data_gen.__next__()
        img_list = chunk
        img, ori_images, label, croppings, name_list, grad = mytool.get_data_from_chunk_v2(chunk, args)
        img = img.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        grad = grad.cuda(non_blocking=True)

        img2 = flipper1(img)
        # if np.random.uniform(0, 1)>0.75:
        #     img2 = F.interpolate(img2, \
        #     (args.crop_size*2, args.crop_size*2), mode='bilinear', align_corners=True)
        # elif np.random.uniform(0, 1)<0.25:
        #     img2 = F.interpolate(img2, \
        #     (args.crop_size//2, args.crop_size//2), mode='bilinear', align_corners=True)

        # img2 = flipper2(img2)
        # img2 = img.clone()

        with torch.cuda.amp.autocast(enabled=False):
            x1, x2, attn1, attn2 = model.module.forward_mirror(img, img2)
            # print(attn1.shape, attn2.shape)
            
            
            # attn1 = attn1[0].detach().cpu().numpy()
            # for i in range(12):
            #     attn1_layer = attn1[i,:,:]
            #     sns.heatmap(attn1_layer, yticklabels=False, xticklabels=False, cbar=False)
            #     plt.savefig("output/attn1_{}.jpg".format(i), bbox_inches='tight')
            #     plt.clf()
            
            # attn2 = attn2[0].detach().cpu().numpy()
            # for i in range(12):
            #     attn2_layer = attn2[i,:,:]
            #     sns.heatmap(attn2_layer, yticklabels=False, xticklabels=False, cbar=False)
            #     plt.savefig("output/attn2_{}.jpg".format(i), bbox_inches='tight')
            #     plt.clf()

            attn1_cls = attn1[:,:,0,1:].unsqueeze(2)
            attn2_cls = attn2[:,:,0,1:].unsqueeze(2)

            attn1_aff = attn1[:,:,1:,1:]
            attn2_aff = attn2[:,:,1:,1:]
            
            p=16 # patch size

            for i in range(p):
                attn2_cls[:,:,:,i*p:i*p+p] = attn2_cls[:,:,:,i*p:i*p+p].flip(3)
            
            for i in range(p):                              
                attn2_aff[:,:,i*p:i*p+p,:] = attn2_aff[:,:,i*p:i*p+p,:].flip(2)
            
            for i in range(p):
                attn2_aff[:,:,:,i*p:i*p+p] = attn2_aff[:,:,:,i*p:i*p+p].flip(3)

            
            cls_align_loss = F.l1_loss(attn1_cls, attn2_cls, reduction='mean')
            aff_align_loss = F.l1_loss(attn1_aff, attn2_aff, reduction='mean')
            # print(cls_align_loss.item(), aff_align_loss.item())

                
            # x, attn1 = model.module.forward_cls(img)
            # x, _ = model.module.forward_cam(img)

            # normalize 
            # croppings = torch.from_numpy(croppings).permute(2,0,1).unsqueeze(1)
            # attn1 = F.interpolate(attn1.unsqueeze(1), (args.crop_size, args.crop_size), mode='bilinear', align_corners=True)
            # attn1[croppings==0] = 0
            # attn1 = attn1 / (torch.max(attn1) + 1e-5)

            # print(attn1.shape)
            # # visualization
            # sample_attn1 = attn1[0,0,:].detach().cpu().numpy()
            # cv2.imwrite('attn1.jpg', sample_attn1*255)

            # # croppings = torch.from_numpy(croppings).permute(2,0,1).unsqueeze(1)
            # # attn2 = F.interpolate(attn2.unsqueeze(1), (args.crop_size, args.crop_size), mode='bilinear', align_corners=True)
            # # attn1[croppings==0] = 0
            # attn2 = attn2 / (torch.max(attn2) + 1e-5)
            # # visualization
            # sample_attn2 = attn2[0,0,:].detach().cpu().numpy()
            # cv2.imwrite('attn2.jpg', sample_attn2*255)


            # edge loss
            # edge_loss = F.binary_cross_entropy_with_logits(attn1, grad)

            # cls_loss_1 = F.multilabel_soft_margin_loss(x1, label) 
            # cls_loss_2 = F.multilabel_soft_margin_loss(x2, label)

            cls_loss_1 = torch.mean(multilabel_categorical_crossentropy(label, x1))
            cls_loss_2 = torch.mean(multilabel_categorical_crossentropy(label, x2))



            # print(cls_loss_1.item(), cls_loss_2.item(),cls_align_loss.item(), aff_align_loss.item())


            loss = cls_loss_1 + cls_loss_2 + cls_align_loss*1000 + aff_align_loss*1000

            writer.add_scalar('cls_align_loss', cls_align_loss.item(), optimizer.global_step)
            writer.add_scalar('aff_align_loss', aff_align_loss.item(), optimizer.global_step)
            writer.add_scalar('cls_loss_1', cls_loss_1.item(), optimizer.global_step)
            writer.add_scalar('cls_loss_2', cls_loss_2.item(), optimizer.global_step)
            writer.add_scalar('loss', loss.item(), optimizer.global_step)
            
            avg_meter.add({'loss': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # generate getam
        getam = True
        if getam:
            model.zero_grad()
            cls_pred, _ = model.module.forward_cls(img)
            b,c,h,w = img.shape

            cam_matrix = torch.zeros((b, 20, w, h))

            for batch in range(b):
                name = name_list[batch]
                original_img = ori_images[batch]
                cur_label = label[batch, :]
                output = cls_pred[batch, :]
                for class_index in range(20):
                    if cur_label[class_index] > 1e-5:
                        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
                        one_hot[0, class_index] = 1
                        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
                        one_hot = torch.sum(one_hot.cuda() * output)
            
                        model.zero_grad()
                        one_hot.backward(retain_graph=True)
                        cam, _, _, attn_map = model.module.generate_cam_2(batch, start_layer=6)
                        
                        attn_map = attn_map / (torch.max(attn_map) + 1e-5)
                        # cam = (cam.unsqueeze(1)@attn_map).squeeze(1) # add affinity 
                        # print(cam.shape)

                        cam = cam.reshape(int(h //16), int(w //16))

                        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), (w, h), mode='bilinear', align_corners=True)
                        cam_matrix[batch, class_index,:,:] = cam
                
                cam_up_single = cam_matrix[batch,:,:,:]

                cam_up_single = cam_up_single.cpu().data.numpy()
                norm_cam = cam_up_single / (np.max(cam_up_single, (1, 2), keepdims=True) + 1e-5)

                # original_img = original_img.transpose(1,2,0).astype(np.uint8)
                    
                ori_img = ori_images[batch].transpose(1, 2, 0).astype(np.uint8)
                for cam_class in range(20):
                    if cur_label[cam_class] > 1e-5:
                        mask = norm_cam[cam_class,:]
            
                        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)

                        ori_img = cv2.resize(ori_img, (heatmap.shape[1], heatmap.shape[0]))
                        cam_output = heatmap * 0.5 + ori_img * 0.5
                        cv2.imwrite(os.path.join('output/vis/', name + '_{}.jpg'.format(classes[cam_class])), cam_output)

        if (optimizer.global_step-1)%50 == 0 and gpu == 0:
            cls_loss_list.append(avg_meter.get('loss'))
            vis.line(cls_loss_list,
                    win='train_from_init_cls_part_{}_{}'.format(args.session_name, torch.distributed.get_rank()),
                    opts=dict(title='train_from_init_cls_part_{}_{}'.format(args.session_name, torch.distributed.get_rank())))

            timer.update_progress(optimizer.global_step / max_step)

            print('Rank:  -- {}'.format(torch.distributed.get_rank()),
                'Iter:%5d/%5d' % (optimizer.global_step - 1, max_step),
                'Loss:%.4f' % (avg_meter.pop('loss')),
                'imps:%.1f' % ((iter+1) * args.batch_size / timer.get_stage_elapsed()),
                'Fin:%s' % (timer.str_est_finish()),
                'lr: %.4f' % (optimizer.param_groups[0]['lr']))
        torch.distributed.barrier()

        if (optimizer.global_step-1)%3000 == 0:
            print('validating....')
            torch.distributed.barrier()
            validation(model, args, optimizer, writer)
            model.train()
        
    torch.distributed.destroy_process_group()

    if gpu==0:
        torch.save(model.module.state_dict(), os.path.join('weight', args.session_name + '_last.pth'))
        print('model saved!')

if __name__ == '__main__':

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()

