# -*- coding: UTF-8 -*- 
import numpy as np
import torch
import os
from torch.backends import cudnn
cudnn.enabled = True
from torchvision import transforms
import argparse
import torch.nn.functional as F
import os
from tool import pyutils, imutils, torchutils
import cv2
from DPT.ACR import ACR
import myTool as mytool
import torch.multiprocessing as mp
import torch.distributed as dist
import shutil


def setup(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

def validation(model, args):
    val_list = mytool.read_file('voc12/val_id.txt')
    data_gen = mytool.chunker(val_list, 1)
    model.eval()
    val_loss_meter = pyutils.AverageMeter('loss')

    val_step = (len(val_list)//(1 * args.gpus ))

    with torch.no_grad():
        for iter in range(val_step):
            chunk = data_gen.__next__()
            img_list = chunk
            img, ori_images, label, name_list = mytool.get_data_from_chunk_val(chunk, args)
            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            x1, _, _ ,_ = model.module.forward_cls(img)
            loss = F.multilabel_soft_margin_loss(x1, label) #+ F.multilabel_soft_margin_loss(x2, label)
            val_loss_meter.add({'loss': loss.item()})
            
    model.train()
    print('loss:', val_loss_meter.pop('loss'))

    return


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
    parser.add_argument("--alpha", default=100, type=int)


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
    rank = args.nr * args.gpus + gpu
    print(rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    setup(rank)
    
    model = ACR(num_classes=20, backbone_name=args.backbone) 

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
        b,c,h,w = img.shape

        img2 = flipper1(img)

        with torch.cuda.amp.autocast(enabled=False):
            cls_list, attn_list= model.module.forward_mirror(img, img2)
            
            attn1, attn2 = attn_list[0], attn_list[1]
            x1, x2 = cls_list[0], cls_list[1]
            # x_p_1, x_p_2 = cls_list[2], cls_list[3]
            # x_b_1, x_b_2 = cls_list[4], cls_list[5]
            
            # if scale>0.8:
            #     attn2 = F.interpolate(attn2, \
            #     (attn1.shape[2], attn1.shape[3]), mode='bilinear', align_corners=True)
            # elif scale<0.2:
            #     attn2 = F.interpolate(attn2, \
            #     (attn1.shape[2], attn1.shape[3]), mode='bilinear', align_corners=True)

            attn1_cls = attn1[:,:,0,1:].unsqueeze(2)
            attn2_cls = attn2[:,:,0,1:].unsqueeze(2)


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

            cls_loss_1 = F.multilabel_soft_margin_loss(x1, label)  
            cls_loss_2 = F.multilabel_soft_margin_loss(x2, label) 

        
            loss = cls_loss_1 + cls_loss_2 + \
                cls_align_loss*args.alpha + aff_align_loss*args.alpha 
            
            avg_meter.add({'loss': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (optimizer.global_step-1)%50 == 0 and gpu == 0:
            cls_loss_list.append(avg_meter.get('loss'))

            timer.update_progress(optimizer.global_step / max_step)

            print('Rank:  -- {}'.format(torch.distributed.get_rank()),
                'Iter:%5d/%5d' % (optimizer.global_step - 1, max_step),
                'Loss:%.4f' % (avg_meter.pop('loss')),
                'imps:%.1f' % ((iter+1) * args.batch_size / timer.get_stage_elapsed()),
                'Fin:%s' % (timer.str_est_finish()),
                'lr: %.4f' % (optimizer.param_groups[0]['lr']))
        torch.distributed.barrier()

        if (optimizer.global_step+1)%5000 == 0:
            print('validating....')
            torch.distributed.barrier()
            validation(model, args)
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
    main()

