from matplotlib.pyplot import waitforbuttonpress
import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
import imageio
import importlib
from tool import imutils
import argparse
import cv2
import os.path
import torch.nn.functional as F
from DPT.DPT import DPTSegmentationModel
from myTool import *
from tool.metrics import Evaluator
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask


def getImgId(name, load_dict):
	# load_dict = json.load(open(path, 'r'))
	images = load_dict['images']

	for i in range(len(images)):
		file_name = images[i]['file_name'].split('.')[0]
		if file_name == name:
				#print(images[i])
				return images[i]['id']

cls_dict = {}
for index, item in enumerate(coco_classes):
    category_id = item['id']
    cls_dict[index] = category_id

base_dir = '/home/users/u5876230/coco/'
ann_file = os.path.join(base_dir, 'annotations/instances_{}{}.json'.format('val', 2014))
coco = COCO(ann_file)
coco_mask = mask
CAT_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49,
                50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75,
                76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

def get_coco_gt(name, h, w):
    img_id = getImgId(name, coco.dataset)

    cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
    mask = np.zeros((h, w), dtype=np.uint8)
    # print(cocotarget)
    for instance in cocotarget:
        rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
        m = coco_mask.decode(rle)
        cat = instance['category_id']

        if cat in CAT_LIST:
            c = CAT_LIST.index(cat)

        else:
            continue
        if len(m.shape) < 3:
            mask[:, :] += (mask == 0) * (m * c)
        else:
            mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
    return mask


def _crf_with_alpha(pred_prob, ori_img):
    bgcam_score = pred_prob
    crf_score = imutils.crf_inference_inf(ori_img, bgcam_score, labels=81)

    return crf_score

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default='./netWeights/RRM_final.pth', type=str)
    parser.add_argument("--out_cam_pred", default='./output/result/no_crf', type=str)
    parser.add_argument("--out_la_crf", default='./output/result/crf', type=str)
    parser.add_argument("--out_color", default='./output/result/color', type=str)
    parser.add_argument("--LISTpath", default="./voc12/val(id).txt", type=str)
    parser.add_argument("--IMpath", default="/home/users/u5876230/coco/val2014/", type=str)
    parser.add_argument("--val", default=False, type=bool)

    args = parser.parse_args()

    model = DPTSegmentationModel(num_classes=80)
    weights_dict = torch.load(args.weights)
    model.load_state_dict(weights_dict, strict=False)

    model.eval()
    model.cuda()

    evaluator = Evaluator(num_class=81) 
    im_path = args.IMpath
    img_list = os.listdir('/home/users/u5876230/coco/val2014/')

    print(len(img_list))
    pred_softmax = torch.nn.Softmax(dim=0)
    # img_list = ['2007_000464 ']
    for index, i in enumerate(img_list):
        # print(i)
        print(index)
        # i = ((i.split('/'))[2])[0:-4]
        i= i[0:-3]

        print(os.path.join(im_path, i[:-1] + '.jpg'))
        img_temp = cv2.imread(os.path.join(im_path, i[:-1] + '.jpg'))
       
        h, w, _ = img_temp.shape
        img_original = img_temp.astype(np.uint8)

        if args.val==True:

            # target_path = os.path.join('/home/users/u5876230/coco/segmentation/', '{}.png'.format(i[:-1]))
            # target = np.asarray(Image.open(target_path), dtype=np.int32)
            
            name = i[:-1]
            seg_mask = get_coco_gt(name, h, w)
            target = seg_mask


            # print(np.unique(target))


        test_size = 224
        # container = np.zeros((test_size, test_size, 3), np.float32)
        # if h>=w:
        #     img_temp = cv2.resize(img_temp, (int(test_size*w/h), test_size))
        #     # print(h,w, img_temp.shape)

        #     container[:, 0:int(test_size*w/h), :] = img_temp
        # else:
        #     img_temp = cv2.resize(img_temp, (test_size, int(test_size*h/w)))
        #     # print(h,w,img_temp.shape)
        #     container[0:int(test_size*h/w), :, :] = img_temp

        # img_temp = container

        img_temp = cv2.resize(img_temp, (test_size, test_size))

        img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB).astype(np.float)
        img_temp[:, :, 0] = (img_temp[:, :, 0] / 255. - 0.485) / 0.229
        img_temp[:, :, 1] = (img_temp[:, :, 1] / 255. - 0.456) / 0.224
        img_temp[:, :, 2] = (img_temp[:, :, 2] / 255. - 0.406) / 0.225

        input = torch.from_numpy(img_temp[np.newaxis, :].transpose(0, 3, 1, 2)).float().cuda()


        _, output= model(input)

        # output = output[]
        # print(output.shape)

        # if h>=w:
        #     output = output[:,:,:,0:int(test_size*h/w)]
        # else:
        #     output = output[:,:,0:int(test_size*h/w), :]


        output = F.interpolate(output, (h, w),mode='bilinear',align_corners=False)

        output = torch.squeeze(output)
        pred_prob = pred_softmax(output)

        output = torch.argmax(output,dim=0).cpu().numpy()

        # print(np.unique(output), np.unique(seg_mask))

        save_path = os.path.join(args.out_cam_pred,i[:-1] + '.png')
        cv2.imwrite(save_path,output.astype(np.uint8))
        
        if args.out_la_crf is not None:
            pred_prob = pred_prob.cpu().data.numpy()
            # pred_prob = _crf_with_alpha(pred_prob, img_original)

            crf_img = np.argmax(pred_prob, 0)

            if args.val:
                evaluator.add_batch(target, crf_img)
                mIoU = evaluator.Mean_Intersection_over_Union()
                print(mIoU)

            imageio.imsave(os.path.join(args.out_la_crf, i[:-1] + '.png'), crf_img.astype(np.uint8))

            rgb_pred = decode_segmap(crf_img, dataset="pascal")
            cv2.imwrite(os.path.join(args.out_color, i[:-1] + '.png'),
                        (rgb_pred * 255).astype('uint8') * 0.7 + img_original* 0.3)
            
            rgb_target = decode_segmap(target, dataset="pascal")
            cv2.imwrite(os.path.join(args.out_color, i[:-1] + '_gt.png'),
                        (rgb_target * 255).astype('uint8') * 0.7 + img_original* 0.3)

    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    print('Validation:')
    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
    # for i in range(81):
        # print(classes[i], evaluator.per_class_miou[i])
