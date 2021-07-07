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
import cv2
from DPT.DPT import DPTSegmentationModel

import myTool as mytool
from myTool import compute_joint_loss, compute_seg_label_2, compute_cam_up, decode_segmap
from DenseEnergyLoss import DenseEnergyLoss
import shutil
# import pamr
from pamr import PAMR
import random
import torch.multiprocessing as mp
import torch.distributed as dist



import visdom

def setup(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--max_epoches", default=50, type=int)
    parser.add_argument("--lr", default=0.04, type=float)
    parser.add_argument("--cls_step", default=0.5, type=float)
    parser.add_argument("--seg_lr_scale", default=0.1, type=float)

    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--LISTpath", default="voc12/train_aug(id).txt", type=str)

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

    parser.add_argument("--session_name", default="vit_cls_seg", type=str)
    parser.add_argument("--crop_size", default=256, type=int)
    parser.add_argument("--voc12_root", default='/home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/', type=str)
    parser.add_argument("--IMpath", default="/home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/JPEGImages", type=str)

    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=2, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')

    args = parser.parse_args()


    try:
        shutil.rmtree('/home/users/u5876230/ete_project/ete_output/pseudo/')
    except:
        pass
    try:
        shutil.rmtree('/home/users/u5876230/ete_project/ete_output/heatmap/')
    except:
        pass
    try:
        shutil.rmtree('/home/users/u5876230/ete_project/ete_output/seg_pred/')
    except:
        pass
    
    os.mkdir('/home/users/u5876230/ete_project/ete_output/pseudo/')
    os.mkdir('/home/users/u5876230/ete_project/ete_output/heatmap/')
    os.mkdir('/home/users/u5876230/ete_project/ete_output/seg_pred/')


    ######################################################### set processes
    args.world_size = args.gpus * args.nodes                           #
    os.environ['MASTER_ADDR'] = 'localhost'                            #
    os.environ['MASTER_PORT'] = '8888'                                 #
    mp.spawn(train, nprocs=args.gpus, args=(args,), join=True)         #
    #########################################################

def train(gpu, args):
    vis = visdom.Visdom()
    rank = args.nr * args.gpus + gpu
    print(rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    setup(0)

    model = DPTSegmentationModel(num_classes=20)

    torch.cuda.set_device(gpu)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(gpu)
    model.train()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu],output_device=[gpu], find_unused_parameters=True)

    # freeze the resnet part
    for name, param in model.named_parameters():
        # if 'backbone' in name:
            # param.requires_grad = False
        if gpu == 0:
            print(name, param.requires_grad)

    # pixel adaptive refine module
    pamr = PAMR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24]).cuda()

    #loss
    critersion = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='elementwise_mean').cuda()
    # criterion = SegmentationLosses(weight=None, cuda=True).build_loss(mode='ce')
    DenseEnergyLosslayer = DenseEnergyLoss(weight=args.densecrfloss, sigma_rgb=args.sigma_rgb,
                                     sigma_xy=args.sigma_xy, scale_factor=args.rloss_scale)

    print(vars(args))

    batch_size = args.batch_size
    img_list = mytool.read_file(args.LISTpath)

    max_step = (len(img_list)//(args.batch_size * args.gpus )) * args.max_epoches
    print(len(img_list))

    data_list = []
    for i in range(int(max_step//100)):
        np.random.shuffle(img_list)
        data_list.extend(img_list)

    data_gen = mytool.chunker(data_list, batch_size)

    params = model.parameters()
    # print(params)
    optimizer = torchutils.PolyOptimizer(params, lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    avg_meter = pyutils.AverageMeter('loss')

    timer = pyutils.Timer("Session started: ")

    TRAIN_CLS_FLAG = True

    cls_loss_list = []
    seg_loss_list = []
    for iter in range(max_step+1):
        chunk = data_gen.__next__()
        img_list = chunk
        if (optimizer.global_step-1) > args.cls_step * max_step:
            img, ori_images, label, croppings, name_list = mytool.get_data_from_chunk_v2(chunk, args)
            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            images = img.cuda(non_blocking=True)
            label  = label.cuda(non_blocking=True)

            # images = F.interpolate(images, size=(int(h * scale), int(w * scale)), mode='bilinear',align_corners=False)

            x = model.module.forward_cls(images)

            loss = F.multilabel_soft_margin_loss(x, label)

            avg_meter.add({'loss': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
                    'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)
            torch.distributed.barrier()
        
        else:
            optimizer.lr_scale = args.seg_lr_scale
            img, ori_images, label, croppings, name_list, saliency = mytool.get_data_from_chunk_v3(chunk, args)
            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
                
            b,c,h,w = img.shape

            x = model.module.forward_cls(img)

            # generate cam and segmentation label on the go: ####################################
            cam_matrix = torch.zeros((b, 20, w, h))
            seg_label = np.zeros((b, w, h))

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
                        cam, _ = model.module.generate_cam(batch, start_layer=0)

                        cam = cam.reshape(int(h //16), int(w //16))

                        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), (w, h), mode='bilinear', align_corners=True)
                        cam_matrix[batch, class_index,:,:] = cam
                
                cam_up_single = cam_matrix[batch,:,:,:]
                # cam_up_single = cam_up_single/(torch.amax(cam_up_single, (1, 2), keepdim=True) + 1e-5)
                cam_up_single = pamr((torch.from_numpy(original_img)).unsqueeze(0).float().cuda(), cam_up_single.unsqueeze(0).cuda()).squeeze(0)

                cam_up_single = cam_up_single.cpu().data.numpy()
                norm_cam = cam_up_single / (np.max(cam_up_single, (1, 2), keepdims=True) + 1e-5)

                saliency_map = saliency[batch,:]
                original_img = original_img.transpose(1,2,0).astype(np.uint8)
                seg_label[batch] = compute_seg_label_2(original_img, cur_label.cpu().numpy(), \
                norm_cam, croppings[:,:,batch], name, iter, saliency_map.data.numpy(),x[batch, :], save_heatmap=True)
            
                # rgb_pseudo_label = decode_segmap(seg_label[batch, :, :], dataset="pascal")
                # cv2.imwrite('/home/users/u5876230/ete_project/ete_output/pseudo/{}.png'.format(name),
                #             (rgb_pseudo_label * 255).astype('uint8') * 0.5 + original_img * 0.5)
                
            #########################################################
            torch.distributed.barrier()
            model.zero_grad()
            x, seg = model(img)

            # visualize sementation prediction
            seg_pred = F.interpolate(seg, (w, h), mode='bilinear', align_corners=False)
            seg_max = torch.max(F.softmax(seg_pred, dim=1), dim=1)
            seg_max_prob_index = seg_max[1]
            for batch_index in range(b):
                original_img = ori_images[batch_index]
                original_img = original_img.transpose(1,2,0).astype(np.uint8)
                seg_pred_b = seg_max_prob_index[batch_index]
                seg_pred_b = decode_segmap(seg_pred_b.cpu().numpy(), dataset="pascal")
                cv2.imwrite('/home/users/u5876230/ete_project/ete_output/seg_pred/{}_{}.png'.format(name, iter),
                            (seg_pred_b * 255).astype('uint8')*0.5 + original_img*0.5)
            
            celoss, dloss = compute_joint_loss(ori_images, seg, seg_label, croppings, critersion,DenseEnergyLosslayer)
            # celoss = criterion(seg, torch.from_numpy(seg_label).cuda())
            closs = F.multilabel_soft_margin_loss(x, label)

            # saliency loss
            # saliency_map = saliency.cuda()
            # sal_pred = ((F.softmax(seg_pred, dim=1))[:,0,:,:])
            # saliency_map = 1-(saliency_map/255).type(torch.int64)
            # sal_loss = F.binary_cross_entropy(sal_pred.float(), saliency_map.float())
                
            loss = closs + celoss + dloss#+ sal_loss

            avg_meter.add({'loss': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update learning rate
            # lr =  args.lr * (0.1 ** ((optimizer.global_step // (max_step//args.max_epoches)) // 5 ))
            # optimizer.param_groups[0]['lr'] = lr

            if (optimizer.global_step-1)%50 == 0 and gpu ==0:
                seg_loss_list.append(avg_meter.get('loss'))
                vis.line(seg_loss_list,
                        win='train_from_init_seg_part_{}'.format(args.session_name),
                        opts=dict(title='train_from_init_seg_part_{}'.format(args.session_name)))

                timer.update_progress(optimizer.global_step / max_step)

                print('Rank:  -- {}'.format(torch.distributed.get_rank()),
                    'Iter:%5d/%5d' % (optimizer.global_step - 1, max_step),
                    'Loss:%.4f' % (avg_meter.pop('loss')),
                    'imps:%.1f' % ((iter+1) * args.batch_size / timer.get_stage_elapsed()),
                    'Fin:%s' % (timer.str_est_finish()),
                    'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)
            torch.distributed.barrier()
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()

    if gpu==0:
        torch.save(model.module.state_dict(), os.path.join('weight', args.session_name + '.pth'))
        print('model saved!')

if __name__ == '__main__':
    main()