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
# from DPT.DPT_vit_resnet_feature import DPTSegmentationModel


import voc12.data

import myTool as mytool
from myTool import compute_joint_loss, compute_seg_label, compute_cam_up, decode_segmap
from DenseEnergyLoss import DenseEnergyLoss
import shutil
from botnet.botnet import BotNet50

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

def pred_acc(original, predicted):
    label_count = int(torch.sum(original))
    ind = np.argpartition(predicted, -label_count)[0][-label_count:]

    predicted_binary = torch.zeros_like(original)
    # print(predicted_binary)

    predicted_binary[ind] = 1

    # predicted = F.sigmoid(predicted)
    # predicted = (predicted>=0.5) * 1
    return predicted_binary.eq(original).sum().numpy()/len(original)

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--max_epoches", default=1, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--train_list", default="voc12/train.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--LISTpath", default="voc12/train.txt", type=str)

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

    # model = BotNet50(pretrained=True, img_size=args.crop_size)
    # model.load_state_dict(torch.load('botnet_cls.pth'))
    # model.cuda()

    model = DPTSegmentationModel(num_classes=20, readout='ignore')
    print(torch.cuda.is_available())
    model.load_state_dict(torch.load('weight/vit_cls_ignore.pth'), strict = 'False')
    # model.load_state_dict(torch.load('vit_cls_2.pth'), strict = 'False')

    model.cuda()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = torch.nn.DataParallel(model).cuda()
    # model.load_state_dict(torch.load('vit_hybrid_cls.pth', map_location=str(device)))

    model.eval()

    print(vars(args))

    batch_size = args.batch_size
    img_list = mytool.read_file_2(args.LISTpath)

    max_step = (len(img_list) // args.batch_size) * args.max_epoches
    print(len(img_list))
    print(max_step)

    data_list = []
    for i in range(int(max_step // 100)):
        np.random.shuffle(img_list)
        data_list.extend(img_list)

    data_gen = mytool.chunker(data_list, batch_size)

    timer = pyutils.Timer("Session started: ")

    accuracy_list = []

    for iter in range(len(img_list)):
        chunk = data_gen.__next__()
        img_list = chunk
        img, ori_images, label,  name_list = mytool.get_data_from_chunk_val(chunk,args)

        img = img.cuda()
        b, c, h, w = img.shape

        x, res_cam = model.forward_cam(img)
        # x, res_cam = model.forward_cam(img)

        res_cam = F.upsample(res_cam, (h,w), mode='bilinear', align_corners=False)[0]
        res_cam = res_cam.detach().cpu().numpy() * label.clone().view(20, 1, 1).numpy()
        # print(res_cam.shape)
        res_cam = res_cam / (np.max(res_cam, (1, 2), keepdims=True) + 1e-5)

        cam_dict = {}
        for i in range(20):
            if label[0][i] > 1e-5:
                cam_dict[i] = res_cam[i]

        img = ori_images[0].transpose(1, 2, 0).astype(np.uint8)
        
        # img = cv2.imread(args.voc12_root + '/JPEGImages/{}.jpg'.format(name_list[0]))
        # # print(img.shape)
        keys = list(cam_dict.keys())
        for target_class in keys:
            mask = cam_dict[target_class]
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)

            img = cv2.resize(img, (heatmap.shape[1], heatmap.shape[0] ))
            cam_output = heatmap * 0.5 + img*0.5
            cv2.imwrite(os.path.join('/home/users/u5876230/ete_project/ete_output/heatmap_ignore/', name_list[0] + '_{}.jpg'.format(classes[target_class])), cam_output)

        output = x.detach().cpu()
        accuracy = pred_acc(label[0], output)
        if accuracy < 1:
            accuracy = 0
        print(iter, '-----', accuracy)
        accuracy_list.append(accuracy)
        
        # generate cam and segmentation label on the go: ####################################
        for cam_strat_layer in range(1):
            cam_matrix = torch.zeros((b, 20, w, h))
            seg_label = np.zeros((b, w, h))

            for batch in range(b):
                name = name_list[batch]
                original_img = ori_images[batch].transpose(1, 2, 0).astype(np.uint8)
                cur_label = label[batch, :]
                output = x[batch, :]
                m = torch.nn.Softmax()
                output_prob = m(output)
                for class_index in range(20):
                    if cur_label[class_index] > 1e-5:
                        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
                        one_hot[0, class_index] = 1
                        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
                        one_hot = torch.sum(one_hot.cuda() * output * output_prob)

                        model.zero_grad()
                        one_hot.backward(retain_graph=True)
                        cam, attn_list = model.generate_cam(batch, start_layer=cam_strat_layer)

                        # grad_map = torch.sum(torch.stack(grad_list), dim=0)[0, 0, 1:]
                        # # print(torch.mean(grad_map))
                        # grad_map[grad_map<torch.mean(grad_map)] = 0
                        # # grad_map = grad_map.clamp(min=torch.mean(grad_map))
                        # grad_map = grad_map.reshape(args.crop_size // 16, args.crop_size // 16)
                        # grad_map = F.interpolate(grad_map.unsqueeze(0).unsqueeze(0), (w, h), mode='bilinear', align_corners=False)
                        # grad_map = grad_map.cpu().data.numpy()
                        # grad_map = grad_map / (np.max(grad_map, (2,3), keepdims=True) + 1e-5)

                        # # print(grad_map.shape)
                        # # print(grad_map)
                        # grad_map = cv2.applyColorMap(np.uint8(255 * grad_map[0,0,:,:]), cv2.COLORMAP_JET)
                        # # print(grad_map.shape)

                        # cv2.imwrite(os.path.join('/home/users/u5876230/ete_project/ete_output/heatmap_grad/', name + '_{}.jpg'.format(classes[class_index])),
                        #  (grad_map * 255) * 0.5 + original_img*0.5)

                        # max_pos = torch.argmax(cam)
                        # top_k_max = torch.topk(cam, 10)[1][0]
                        # # print(top_k_max)
                        # attn_map = np.zeros((1,1,args.crop_size, args.crop_size))

                        # for max_pos in top_k_max:
                        #     # print(max_pos)
                        #     target_map_list = []
                        #     for i in range(len(attn_list)):
                        #         target_map = attn_list[i][:, max_pos + 1, :][0, 1:]
                        #         # print(target_map.shape)
                        #         target_map_list.append(target_map)

                        #     target_attn_map =  torch.sum(torch.stack(target_map_list), dim=0)
                        #     # print(target_attn_map.shape)
                        #     target_attn_map = target_attn_map.reshape(args.crop_size // 16, args.crop_size // 16)
                        #     target_attn_map = F.interpolate(target_attn_map.unsqueeze(0).unsqueeze(0), (w, h), mode='bilinear', align_corners=False)
                        #     target_attn_map = target_attn_map.cpu().data.numpy()
                        #     target_attn_map = target_attn_map / (np.max(target_attn_map, (2,3), keepdims=True) + 1e-5) *255
                        #     attn_map = attn_map + target_attn_map
                        
                        # attn_map = attn_map / (np.max(attn_map, (2,3), keepdims=True) + 1e-5) *255
                        # cv2.imwrite(os.path.join('/home/users/u5876230/ete_project/ete_output/heatmap_3/', name + '_{}_attn.jpg'.format(classes[class_index])), attn_map[0,0,:,:])

                        cam = cam.reshape(args.crop_size // 16, args.crop_size // 16)
                        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), (w, h), mode='bilinear', align_corners=False)

                        cam_matrix[batch, class_index, :, :] = cam

                cam_up_single = cam_matrix[batch, :, :, :]
                cam_up_single = cam_up_single.cpu().data.numpy()
                norm_cam = cam_up_single / (np.max(cam_up_single, (1, 2), keepdims=True) + 1e-5)
                # seg_label[batch] = compute_seg_label(original_img, cur_label.cpu().numpy(), norm_cam, croppings[:,:, batch], name, iter,
                                                    #  save_heatmap=True)
                cam_dict = {}
                cam_np = np.zeros_like(norm_cam)
                cam_label = cur_label.cpu().numpy()
                for i in range(20):
                    if cam_label[i] > 1e-5:
                        cam_dict[i] = norm_cam[i]
                        cam_np[i] = norm_cam[i]

                img = original_img
                keys = list(cam_dict.keys())
                for target_class in keys:
                    mask = cam_dict[target_class]
                    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
                    img = cv2.resize(img, (heatmap.shape[1], heatmap.shape[0]))
                    cam_output = heatmap * 0.5 + img * 0.5

                    cv2.imwrite(os.path.join('/home/users/u5876230/ete_project/ete_output/heatmap_ignore/', name + '_{}_{}_vit.jpg'.
                    format(classes[target_class], cam_strat_layer)), cam_output)

                # rgb_pseudo_label = decode_segmap(seg_label[batch, :, :], dataset="pascal")
                # cv2.imwrite('/home/users/u5876230/ete_project/ete_output/pseudo/{}.png'.format(name),
                            # (rgb_pseudo_label * 255).astype('uint8') * 0.8 + original_img * 0.3)

    print('average accuracy:', sum(accuracy_list)/ len(accuracy_list))
