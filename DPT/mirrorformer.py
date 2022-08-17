from numpy.core.numeric import roll
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from DPT.blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_vit,
)

def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )
    
class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)



class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        modules_body = []
        act = nn.ReLU(True)
        for i in range(2):
            modules_body.append(self.default_conv(channel, channel, 3, bias=True))
            if i == 0: modules_body.append(act)
        self.body = nn.Sequential(*modules_body)

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        raw_x = x
        x = self.body(x)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # print(x.shape, y.shape)
        at_x = x * y.expand_as(x)
        return at_x + raw_x


class DPT(BaseModel):
    def __init__(
        self,
        features=256,
        backbone='vitb_rn50_384',
        readout="ignore",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
        use_pretrain=True,
        use_attention=False,
        seg = False
    ):

        super(DPT, self).__init__()

        self.channels_last = channels_last
        self.attention = use_attention

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            'deitb16_384': [2,5,8,11],
            'deitb16_distil_384':[2,5,8,11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            use_pretrain,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
            seg = seg
        )

        if seg:
            self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
            self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
            self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
            self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        # classification head
        self.cls_head = nn.Linear(768, self.num_class)
        self.cls_head_2 = nn.Linear(768, self.num_class)
        # self.classifier = nn.Conv2d(768, self.num_class, kernel_size=1, bias=False)

        self.use_gap = True


    # with bkg token
    def forward_cls_2(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        _, _ = self.pretrained.model.forward_flex_2(x)
        layer_4 = self.pretrained.activations["4"]

        x_cls = layer_4[:, 0, :]
        x_bkg = layer_4[:, 1, :]
        x_patch = layer_4[:, 2:, :]

        x_patch_cls = F.avg_pool1d(x_patch.permute(0,2,1), kernel_size=x_patch.shape[1])[:,:,0]
        x_patch_cls = self.cls_head_2(x_patch_cls)

        x_cls = self.cls_head(x_cls)
        x_bkg_cls = self.cls_head(x_bkg)

        x_patch_cam = self.cls_head_2(x_patch)
        x_patch_cam = F.relu(x_patch_cam)

        attn_list = []
        for blk in self.pretrained.model.blocks:
            attn = blk.attn.get_attn()
            attn = torch.mean(attn, dim=1)
            attn_list.append(attn)
        cls_attn_sum = torch.stack(attn_list, dim=1)

        return x_cls, x_patch_cls, cls_attn_sum, x_bkg_cls, x_patch_cam
    
    def forward_cls(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        _, _ = self.pretrained.model.forward_flex(x)
        layer_4 = self.pretrained.activations["4"]

        x_cls = layer_4[:, 0, :]
        x_patch = layer_4[:, 1:, :]

        x_patch_cls = F.avg_pool1d(x_patch.permute(0,2,1), kernel_size=x_patch.shape[1])[:,:,0]
        x_patch_cls = self.cls_head_2(x_patch_cls)

        x_cls = self.cls_head(x_cls)

        attn_list = []
        for blk in self.pretrained.model.blocks:
            attn = blk.attn.get_attn()
            attn = torch.mean(attn, dim=1)
            attn_list.append(attn)
        cls_attn_sum = torch.stack(attn_list, dim=1) # b*layer*h*w
        # print(cls_attn_sum.shape)

        x_bkg_cls = None

        return x_cls, x_patch_cls, cls_attn_sum, x_bkg_cls

    def forward_cam(self, x):
    
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        _, _ = self.pretrained.model.forward_flex(x)
        layer_4 = self.pretrained.activations["4"]

        x_cls = layer_4[:, 0, :]
        x_patch = layer_4[:, 1:, :]

        x_cls = self.cls_head(x_cls)

        x_patch_cls = F.avg_pool1d(x_patch.permute(0,2,1), kernel_size=x_patch.shape[1])[:,:,0]
        x_patch_cls = self.cls_head_2(x_patch_cls)

        x_patch_cam = self.cls_head_2(x_patch)
        x_patch_cam = F.relu(x_patch_cam)

        attn_list = []
        for blk in self.pretrained.model.blocks:
            attn = blk.attn.get_attn()
            attn = torch.mean(attn, dim=1)
            attn_list.append(attn)
        cls_attn_sum = torch.stack(attn_list, dim=1)

        return x_cls, x_patch_cls, cls_attn_sum, x_patch_cam
    




class MirrorFormer(DPT):
    def __init__(self, num_classes, backbone_name, path=None, **kwargs):
        self.num_class = num_classes

        features = kwargs["features"] if "features" in kwargs else 256

        kwargs["use_bn"] = True

        backbone_dict = {"vitb_hybrid": "vitb_rn50_384",
            'vitb':"vitb16_384",
            'deit':'deitb16_384',
            'deit_distilled':'deitb16_distil_384',
            'vitl':"vitl16_384",
        }

        
        cur_backbone = backbone_dict[backbone_name]
        self.cur_backbone = cur_backbone
        print('cur_backbone:', cur_backbone)
        super().__init__(backbone=cur_backbone, **kwargs)

        if path is not None:
            self.load(path)
    
    def forward_multiscale(self, x):
        input_size_h = x.size()[2]
        input_size_w = x.size()[3]

        x2 = F.interpolate(x, size=(int(input_size_h * 0.5), int(input_size_w * 0.5)), mode='bilinear',align_corners=False)
        x3 = F.interpolate(x, size=(int(input_size_h * 1.5), int(input_size_w * 1.5)), mode='bilinear',align_corners=False)

        with torch.enable_grad():
            x_cls, x_p_cls, attn1, x_bkg_cls = super().forward_cls(x)
        with torch.no_grad():
            _, _, attn2, _ = super().forward_cls(x2)
            _, _, attn3, _ = super().forward_cls(x3)

        attn2 = F.interpolate(attn2, size=(int(attn1.shape[2]), int(attn1.shape[3])), mode='bilinear',align_corners=False)
        attn3 = F.interpolate(attn3, size=(int(attn1.shape[2]), int(attn1.shape[3])), mode='bilinear',align_corners=False)
        attn = (attn1+attn2+attn3)/3

        return x_cls, x_p_cls, attn, x_bkg_cls
    
    def forward_mirror(self, x1, x2):
        x_cls_1, x_p_cls_1, attn1, x_bkg_cls_1 = super().forward_cls(x1)
        x_cls_2, x_p_cls_2, attn2, x_bkg_cls_2 = super().forward_cls(x2)

        # x_cls_1, x_p_cls_1, attn1, x_bkg_cls_1 = self.forward_multiscale(x1)
        # x_cls_2, x_p_cls_2, attn2, x_bkg_cls_2 = self.forward_multiscale(x2)

        return [x_cls_1, x_cls_2, x_p_cls_1, x_p_cls_2, x_bkg_cls_1, x_bkg_cls_2], [attn1, attn2]

    # "GETAM" cam * gradient^2
    def getam(self, batch, start_layer=0, func = 'cam_grad_s'):
        cam_list = []
        attn_list = []
        grad_list = []
        for blk in self.pretrained.model.blocks:
            grad = blk.attn.get_attn_gradients()
            cam = blk.attn.get_attn()
            attn_list.append(torch.mean(cam, dim = 1))
            grad_list.append(torch.mean(grad, dim = 1))
            cam = cam[batch].reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad[batch].reshape(-1, grad.shape[-1], grad.shape[-1])
            
            if func == 'cam_grad_s':
                cam = grad * cam 
                cam = cam.clamp(min=0).mean(dim=0)
                positive_grad = grad.clamp(min=0).mean(dim=0)
                cam = cam * positive_grad 
            elif func == 'grad':
                cam = grad
                cam = cam.clamp(min=0).mean(dim=0)
            elif func == 'grad_s':
                cam = grad
                cam = cam.clamp(min=0).mean(dim=0)
                positive_grad = grad.clamp(min=0).mean(dim=0)
                cam = cam * positive_grad

            cam_list.append(cam.unsqueeze(0))

        cam_list = cam_list[start_layer:]
        cams = torch.stack(cam_list).sum(dim=0)
        if self.cur_backbone == 'deitb16_distil_384':
            cls_cam = torch.relu(cams[:, 0, 2:])
        else:
            cls_cam = torch.relu(cams[:, 0, 1:])

        return cls_cam, attn_list, cam_list



class MirrorFormer_SingleStage(DPT):
    def __init__(self, num_classes, backbone_name, path=None, **kwargs):
        self.num_class = num_classes

        features = kwargs["features"] if "features" in kwargs else 256

        kwargs["use_bn"] = True

        backbone_dict = {"vitb_hybrid": "vitb_rn50_384",
            'vitb':"vitb16_384",
            'deit':'deitb16_384',
            'deit_distilled':'deitb16_distil_384',
            'vitl':"vitl16_384",
        }

        cur_backbone = backbone_dict[backbone_name]
        self.cur_backbone = cur_backbone
        print('cur_backbone:', cur_backbone)
        super().__init__(backbone=cur_backbone, seg=True, **kwargs)

        self.cam_module = SELayer(channel=256)

        self.seg_head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes+1, kernel_size=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        if path is not None:
            self.load(path)
    
    def forward_mirror(self, x1, x2):
        x_cls_1, x_p_cls_1, attn1, x_bkg_cls_1 = super().forward_cls(x1)
        x_cls_2, x_p_cls_2, attn2, x_bkg_cls_2 = super().forward_cls(x2)

        # x_cls_1, x_p_cls_1, attn1, x_bkg_cls_1 = self.forward_multiscale(x1)
        # x_cls_2, x_p_cls_2, attn2, x_bkg_cls_2 = self.forward_multiscale(x2)

        return [x_cls_1, x_cls_2, x_p_cls_1, x_p_cls_2, x_bkg_cls_1, x_bkg_cls_2], [attn1, attn2]
    

    def forward_seg(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)
        
        layer_1, layer_2, layer_3, layer_4, x_cls, x_patch = forward_vit(self.pretrained, x)

        x_patch_cls = F.avg_pool1d(x_patch.permute(0,2,1), kernel_size=x_patch.shape[1])[:,:,0]
        x_patch_cls = self.cls_head_2(x_patch_cls)
        x_cls = self.cls_head(x_cls)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        path_1 = self.cam_module(path_1)
        out = self.seg_head(path_1)

        return x_cls, x_patch_cls, out

    # "GETAM" cam * gradient^2
    def getam(self, batch, start_layer=0):
        cam_list = []
        attn_list = []
        grad_list = []
        for blk in self.pretrained.model.blocks:
            grad = blk.attn.get_attn_gradients()
            cam = blk.attn.get_attn()
            attn_list.append(torch.mean(cam, dim = 1))
            grad_list.append(torch.mean(grad, dim = 1))
            cam = cam[batch].reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad[batch].reshape(-1, grad.shape[-1], grad.shape[-1])
            
            cam = grad * cam 
            cam = cam.clamp(min=0).mean(dim=0)
            
            positive_grad = grad.clamp(min=0).mean(dim=0)
            cam = cam * positive_grad

            cam_list.append(cam.unsqueeze(0))

        cam_list = cam_list[start_layer:]
        cams = torch.stack(cam_list).sum(dim=0)
        if self.cur_backbone == 'deitb16_distil_384':
            cls_cam = torch.relu(cams[:, 0, 2:])
        else:
            cls_cam = torch.relu(cams[:, 0, 1:])

        return cls_cam, attn_list, cam_list
        







