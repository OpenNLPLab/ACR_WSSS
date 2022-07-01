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
    bgcam_score = pred_prob.cpu().data.numpy()
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

    evaluator = Evaluator(num_class=81) 
    im_path = args.IMpath
    img_list = os.listdir('/home/users/u5876230/coco/val2014/')

    print(len(img_list))
    pred_softmax = torch.nn.Softmax(dim=0)
    # img_list = ['2007_000464 ']
    for index, i in enumerate(img_list):
        print(i)
        print(index)
        # i = ((i.split('/'))[2])[0:-4]
        i= i[0:-3]

        print(os.path.join(im_path, i[:-1] + '.jpg'))
        img_temp = cv2.imread(os.path.join(im_path, i[:-1] + '.jpg'))
       
        h, w, _ = img_temp.shape
        img_original = img_temp.astype(np.uint8)

        if args.val==True:

            target_path = os.path.join('/home/users/u5876230/coco/segmentation/', '{}.png'.format(i[:-1]))
            target = np.asarray(Image.open(target_path), dtype=np.int32)
            name = i[:-1]
            print(name)
            seg_mask = get_coco_gt(name, h, w)
            print(seg_mask)
            print(np.unique(seg_mask))