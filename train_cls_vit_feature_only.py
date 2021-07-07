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
from myTool import compute_joint_loss, compute_seg_label, compute_cam_up, decode_segmap, RandomErasing
from DenseEnergyLoss import DenseEnergyLoss
import shutil
import random

import visdom
import voc12.data
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist

def cleanup():
    dist.destroy_process_group()


def setup(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_epoches", default=10, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--LISTpath", default="voc12/train_aug(id).txt", type=str)

    parser.add_argument("--session_name", default="vit_cls", type=str)
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

    ######################################################### set processes
    args.world_size = args.gpus * args.nodes                           #
    os.environ['MASTER_ADDR'] = 'localhost'                            #
    os.environ['MASTER_PORT'] = '1234'                                 #
    mp.spawn(train, nprocs=args.gpus, args=(args,), join=True)         #
    #########################################################

def train(gpu, args):
    vis = visdom.Visdom()
    rank = args.nr * args.gpus + gpu
    print(rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    setup(0)

    batch_size = args.batch_size
    img_list = mytool.read_file(args.LISTpath)

    max_step = (len(img_list)//(args.batch_size * args.gpus )) * args.max_epoches
    print(len(img_list))
    # print(max_step)

    data_list = []
    for i in range(int(max_step//100)):
        np.random.shuffle(img_list)
        data_list.extend(img_list)

    data_gen = mytool.chunker(data_list, batch_size)

    randomerasing = RandomErasing(probability = 0.5, sh = 0.3, r1 = 0.3)

    model = DPTSegmentationModel(num_classes=20, readout='add')

    torch.cuda.set_device(gpu)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(gpu)
    model.train()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    params = model.parameters()
    optimizer = torchutils.PolyOptimizer(params, lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    avg_meter = pyutils.AverageMeter('loss')

    timer = pyutils.Timer("Session started: ")

    train_loss_list = []
    seg_loss_list = []

    # scale_list = [0.5, 1, 1.5, 2]
    # scale = random.choice(scale_list)

    for iter in range(max_step+1):
        chunk = data_gen.__next__()
        img_list = chunk
        img, ori_images, label, croppings, name_list = mytool.get_data_from_chunk_v2(chunk, args)
        # img = randomerasing(img)
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
            train_loss_list.append(avg_meter.get('loss'))
            vis.line(train_loss_list,
                    win='train_loss_per_iteration_{}_{}'.format(args.session_name, torch.distributed.get_rank()),
                    opts=dict(title='train_loss_per_iteration_{}_{}'.format(args.session_name, torch.distributed.get_rank())))

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
        torch.save(model.module.state_dict(), os.path.join('weight/', args.session_name + '.pth'))
        print('model saved!')


if __name__ == '__main__':
    main()
