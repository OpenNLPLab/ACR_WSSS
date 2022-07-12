# -*- coding: UTF-8 -*- 
from ast import expr_context
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
from torch.utils.tensorboard import SummaryWriter  
import visdom
import shutil
# import network

from timm.models import create_model
# torch.autograd.set_detect_anomaly(True)

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
            x1, x2, _ ,_ = model.module.forward_cls(img)
            # x, cam = model.module.forward_cam_multiscale(img)
            loss = F.multilabel_soft_margin_loss(x1, label) #+ F.multilabel_soft_margin_loss(x2, label)
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
    parser.add_argument("--max_epoches", default=15, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--step_lr", default=False, type=bool)

    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val(id).txt", type=str)
    parser.add_argument("--LISTpath", default="voc12/train_aug(id).txt", type=str)
    parser.add_argument("--address", default="1111", type=str)
    parser.add_argument("--backbone", default="vitb_hybrid", type=str)

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

    try:
        shutil.rmtree('tensorboard/{}'.format(args.session_name))
    except:
        pass
    
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
    
    model = MirrorFormer(num_classes=20, backbone_name=args.backbone) 
    # model = create_model(
        # 'vitb_base_Mirrorformer',
        # pretrained=True,
        # num_classes=20,
        # drop_block_rate=None
    # )

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
        img, ori_images, label,  name_list = mytool.get_data_from_chunk_v2(chunk, args)
        img = img.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        # grad = grad.cuda(non_blocking=True)
        b,c,h,w = img.shape

        bkg_label = torch.zeros_like(label).cuda(non_blocking=True)

        img2 = flipper1(img)
        # scale = np.random.uniform(0, 1)

        # if scale>0.75:
        #     img2 = F.interpolate(img2, \
        #     (int(args.crop_size*1.5), int(args.crop_size*1.5)), mode='bilinear', align_corners=True)
        # elif scale<0.25:
        #     img2 = F.interpolate(img2, \
        #     (args.crop_size//2, args.crop_size//2), mode='bilinear', align_corners=True)

        # img2 = flipper2(img2)
        # img2 = img.clone()

        with torch.cuda.amp.autocast(enabled=False):
            cls_list, attn_list= model.module.forward_mirror(img, img2)
            
            attn1, attn2 = attn_list[0], attn_list[1]
            x1, x2 = cls_list[0], cls_list[1]
            x_p_1, x_p_2 = cls_list[2], cls_list[3]
            x_b_1, x_b_2 = cls_list[4], cls_list[5]
            # print(attn1.shape)
            # if scale>0.75:
            #     attn2 = F.interpolate(attn2, \
            #     (attn1.shape[2], attn1.shape[3]), mode='bilinear', align_corners=True)
            # elif scale<0.25:
            #     attn2 = F.interpolate(attn2, \
            #     (attn1.shape[2], attn1.shape[3]), mode='bilinear', align_corners=True)

            attn1_cls = attn1[:,:,0,1:].unsqueeze(2)
            attn2_cls = attn2[:,:,0,1:].unsqueeze(2)

            # attn1_bkg = attn1[:,:,1,2:].unsqueeze(2)
            # attn2_bkg = attn2[:,:,1,2:].unsqueeze(2)

            # visualize bkg and frg
            if False:
                cls_token_map = attn1_cls[:,0,0,:].reshape(args.batch_size, 16, 16)
                bkg_token_map = attn1_bkg[:,0,0,:].reshape(args.batch_size, 16, 16)
                cls_token_map = F.interpolate(cls_token_map.unsqueeze(1), (256, 256), mode='bilinear', align_corners=True)[:,0,:,:]
                cls_token_map = cls_token_map.detach().cpu().numpy()
                bkg_token_map = F.interpolate(bkg_token_map.unsqueeze(1), (256, 256), mode='bilinear', align_corners=True)[:,0,:,:]
                bkg_token_map = bkg_token_map.detach().cpu().numpy()

                for batch in range(args.batch_size):
                    name = name_list[batch]

                    norm_cam = cls_token_map[batch]
                    norm_cam = norm_cam/ (np.max(norm_cam, (0, 1), keepdims=True) + 1e-5)
                    heatmap = cv2.applyColorMap(np.uint8(255 * norm_cam), cv2.COLORMAP_JET)
                    ori_img = ori_images[batch].transpose(1, 2, 0).astype(np.uint8)
                    ori_img = cv2.resize(ori_img, (heatmap.shape[1], heatmap.shape[0]))
                    cls_map = heatmap * 0.5 + ori_img * 0.5
                    cv2.imwrite(os.path.join('output/attn_map/', name + '_cls.jpg'), cls_map)

                    bkg_cam = bkg_token_map[batch]
                    bkg_cam = bkg_cam/ (np.max(bkg_cam, (0, 1), keepdims=True) + 1e-5)
                    bkg_heatmap = cv2.applyColorMap(np.uint8(255 * bkg_cam), cv2.COLORMAP_JET)
                    bkg_map =bkg_heatmap * 0.5 + ori_img * 0.5
                    cv2.imwrite(os.path.join('output/attn_map/', name + '_bkg.jpg'), bkg_map)

            attn1_aff = attn1[:,:,1:,1:]
            attn2_aff = attn2[:,:,1:,1:]
            
            p = h//16 # patch size

            for i in range(p):
                attn2_cls[:,:,:,i*p:i*p+p] = attn2_cls[:,:,:,i*p:i*p+p].flip(3)
                # attn2_bkg[:,:,:,i*p:i*p+p] = attn2_bkg[:,:,:,i*p:i*p+p].flip(3)

            for i in range(p):                              
                attn2_aff[:,:,i*p:i*p+p,:] = attn2_aff[:,:,i*p:i*p+p,:].flip(2)
            
            for i in range(p):
                attn2_aff[:,:,:,i*p:i*p+p] = attn2_aff[:,:,:,i*p:i*p+p].flip(3)

            cls_align_loss = F.l1_loss(attn1_cls, attn2_cls, reduction='mean')
            aff_align_loss = F.l1_loss(attn1_aff, attn2_aff, reduction='mean')
            # bkg_align_loss = F.l1_loss(attn1_bkg, attn2_bkg, reduction='mean')

            # intra_frg_bkg_loss 
            # intra_frg_bkg_loss = 4 - (F.l1_loss(attn1_cls,  attn1_bkg, reduction='mean') + \
                                    #  F.l1_loss(attn2_cls,  attn2_bkg, reduction='mean'))
            
            # croppings = torch.from_numpy(croppings).permute(2,0,1).unsqueeze(1)
            # print(croppings.shape, cls_token_map.shape)
            # croppings =F.interpolate(croppings, (16, 16), mode='bilinear', align_corners=True)[:,0,:,:]
            

            # cls_token_map_1 = attn1_cls[:,0,0,:].reshape(args.batch_size, 16, 16)
            # bkg_token_map_1 = attn1_bkg[:,0,0,:].reshape(args.batch_size, 16, 16)
            # # print(croppings.shape, cls_token_map_1.shape)
            # # cls_token_map_1[croppings==0] = 0 
            # # bkg_token_map_1[croppings==0] = 0

            # cls_token_map_1 = (cls_token_map_1 - torch.amin(cls_token_map_1, (1, 2), keepdims=True)) / \
            #     ( torch.amax(cls_token_map_1, (1, 2), keepdims=True) - torch.amin(cls_token_map_1, (1, 2), keepdims=True) + 1e-5)
            # bkg_token_map_1 = (bkg_token_map_1 - torch.amin(bkg_token_map_1, (1, 2), keepdims=True)) / \
            #     ( torch.amax(bkg_token_map_1, (1, 2), keepdims=True) - torch.amin(bkg_token_map_1, (1, 2), keepdims=True) + 1e-5)
            # comp_map_1 = cls_token_map_1 + bkg_token_map_1
            # # print(torch.mean(cls_token_map_1, (1,2)), torch.mean(bkg_token_map_1, (1,2)))

            # # writer.add_scalar('cls_token_map_mean_1', torch.mean(cls_token_map_1, (0,1,2)).item(), optimizer.global_step)
            # # writer.add_scalar('bkg_token_map_mean_1', torch.mean(bkg_token_map_1, (0,1,2)).item(), optimizer.global_step)

            # cls_token_map_2 = attn2_cls[:,0,0,:].reshape(args.batch_size, 16, 16)
            # bkg_token_map_2 = attn2_bkg[:,0,0,:].reshape(args.batch_size, 16, 16)
            # # cls_token_map_2[croppings==0] = 0 
            # # bkg_token_map_2[croppings==0] = 0
            # cls_token_map_2 = (cls_token_map_2 - torch.amin(cls_token_map_2, (1, 2), keepdims=True)) / \
            #     ( torch.amax(cls_token_map_2, (1, 2), keepdims=True) - torch.amin(cls_token_map_2, (1, 2), keepdims=True) + 1e-5)
            # bkg_token_map_2 = (bkg_token_map_2 - torch.amin(bkg_token_map_2, (1, 2), keepdims=True)) / \
            #     ( torch.amax(bkg_token_map_2, (1, 2), keepdims=True) - torch.amin(bkg_token_map_2, (1, 2), keepdims=True) + 1e-5)
            # comp_map_2 = cls_token_map_2 + bkg_token_map_2
            # print(torch.mean(cls_token_map_2, (1,2)), torch.mean(bkg_token_map_2, (1,2)))


            # writer.add_scalar('cls_token_map_mean_2', torch.mean(cls_token_map_2, (0,1,2)).item(), optimizer.global_step)
            # writer.add_scalar('bkg_token_map_mean_2', torch.mean(bkg_token_map_2, (0,1,2)).item(), optimizer.global_step)


            # intra_frg_bkg_loss = F.l1_loss(comp_map_1, torch.ones_like(comp_map_1)) + F.l1_loss(comp_map_2, torch.ones_like(comp_map_1)) 

            
            
            cls_loss_1 = F.multilabel_soft_margin_loss(x1, label) #+ \
            # F.multilabel_soft_margin_loss(x_p_1, label) 
            cls_loss_2 = F.multilabel_soft_margin_loss(x2, label)# + \
            #  F.multilabel_soft_margin_loss(x_p_2, label) 

            # bkg_loss_1 = F.multilabel_soft_margin_loss(x_b_1, bkg_label)
            # bkg_loss_2 = F.multilabel_soft_margin_loss(x_b_2, bkg_label)

            # cls_loss_1 = torch.mean(multilabel_categorical_crossentropy(label, x1))
            # cls_loss_2 = torch.mean(multilabel_categorical_crossentropy(label, x2))

            # print(cls_loss_1.item(), cls_loss_2.item(),cls_align_loss.item(), aff_align_loss.item(), intra_frg_bkg_loss.item())

            loss = cls_loss_1 + cls_loss_2 + \
                cls_align_loss*100 + aff_align_loss*100 \
                    #  + bkg_loss_1 + bkg_loss_2  \
                    #    + intra_frg_bkg_loss * 0.1 
                    # + bkg_align_loss*1000 \
                        # + bkg_loss_1 + bkg_loss_2  \
                # + intra_frg_bkg_loss

            writer.add_scalar('cls_align_loss', cls_align_loss.item(), optimizer.global_step)
            writer.add_scalar('aff_align_loss', aff_align_loss.item(), optimizer.global_step)
            writer.add_scalar('cls_loss_1', cls_loss_1.item(), optimizer.global_step)
            writer.add_scalar('cls_loss_2', cls_loss_2.item(), optimizer.global_step)
            # writer.add_scalar('bkg_loss_1', bkg_loss_1.item(), optimizer.global_step)
            # writer.add_scalar('bkg_loss_2', bkg_loss_2.item(), optimizer.global_step)
            # writer.add_scalar('bkg_align_loss', bkg_align_loss.item(), optimizer.global_step)
            # writer.add_scalar('intra_frg_bkg_loss', intra_frg_bkg_loss.item(), optimizer.global_step)
            writer.add_scalar('loss', loss.item(), optimizer.global_step)
            
            avg_meter.add({'loss': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # generate getam
        getam = True
        if getam:
            model.zero_grad()
            cls_pred,_, _,_ = model.module.forward_cls(img)
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
                        cam, _, _ = model.module.getam(batch, start_layer=6)
                        
                        # attn_map = attn_map / (torch.max(attn_map) + 1e-5)
                        # cam = (cam.unsqueeze(1)@attn_map).squeeze(1) # add affinity 
                        # print(cam.shape)

                        cam = cam.reshape(int(h //16), int(w //16))

                        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), (w, h), mode='bilinear', align_corners=True)
                        cam_matrix[batch, class_index,:,:] = cam
                
                cam_up_single = cam_matrix[batch,:,:,:]

                cam_up_single = cam_up_single.cpu().data.numpy()
                norm_cam = (cam_up_single - np.min(cam_up_single, (1, 2), keepdims=True)) / \
                    (np.max(cam_up_single, (1, 2), keepdims=True) - np.min(cam_up_single, (1, 2), keepdims=True) + 1e-5)

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

        if (optimizer.global_step+1)%3000 == 0:
            print('validating....')
            torch.distributed.barrier()
            validation(model, args, optimizer, writer)
            model.train()
            if gpu==0:
                torch.save(model.module.state_dict(), os.path.join('weight', args.session_name + '_last.pth'))
                print('model saved!')
        
    torch.distributed.destroy_process_group()

    if gpu==0:
        torch.save(model.module.state_dict(), os.path.join('weight', args.session_name + '_last.pth'))
        print('model saved!')

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["CUDA_VISIBLE_DEVICES"]="7"
    main()

