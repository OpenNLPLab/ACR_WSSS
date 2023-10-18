# ACR_WSSS
[ICCV 2023 Workshop] Official implementation of the paper: [All-pairs Consistency Learning for Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/2308.04321)
**We won the best paper award of the 1st Workshop on New Ideas in Vision Transformers at ICCV 2023!**

## Step1: environment
- clone this repo 
```
git clone https://github.com/OpenNLPLab/ACR_WSSS.git
```
- optionally create a new environment python>=3.6
- install requirements.txt
```
pip install -r requirements.txt
```


## Step2: dataset preparation
[pascal voc](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) 

[MS-COCO 2014](https://cocodataset.org/#home) 


## Step3: train and inference
First, change your data path and GPU settings accordingly. Then you can perform train and inference
Run:
```
bash train_acr.sh
```
which includes train, localization generation and evaluation.



## Acknowledgement
- Thanks for codebase provided by [DPT](https://github.com/isl-org/DPT)
- Thanks for codebase provided by [RRM](https://github.com/zbf1991/RRM)




---
if you use this paper, please kindly cite:
```
@misc{sun2023allpairs,
      title={All-pairs Consistency Learning for Weakly Supervised Semantic Segmentation}, 
      author={Weixuan Sun and Yanhao Zhang and Zhen Qin and Zheyuan Liu and Lin Cheng and Fanyi Wang and Yiran Zhong and Nick Barnes},
      year={2023},
      eprint={2308.04321},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```





