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

        self.use_gap = True
    
    def forward_cls(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        _, _ = self.pretrained.model.forward_flex(x)
        layer_4 = self.pretrained.activations["4"]

        x_cls = layer_4[:, 0, :]
        x_patch = layer_4[:, 1:, :]

        x_patch_cls = F.avg_pool1d(x_patch.permute(0,2,1), kernel_size=x_patch.shape[1])[:,:,0]
        x_patch_cls = self.cls_head(x_patch_cls)

        x_cls = self.cls_head(x_cls)

        attn_list = []
        for blk in self.pretrained.model.blocks:
            attn = blk.attn.get_attn()
            attn = torch.mean(attn, dim=1)
            attn_list.append(attn)
        cls_attn_sum = torch.stack(attn_list, dim=1) # b*layer*h*w

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
        x_patch_cls = self.cls_head(x_patch_cls)

        x_patch_cam = self.cls_head(x_patch)
        x_patch_cam = F.relu(x_patch_cam)

        attn_list = []
        for blk in self.pretrained.model.blocks:
            attn = blk.attn.get_attn()
            attn = torch.mean(attn, dim=1)
            attn_list.append(attn)
        cls_attn_sum = torch.stack(attn_list, dim=1)

        return x_cls, x_patch_cls, cls_attn_sum, x_patch_cam
    


class ACR(DPT):
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

    def forward_mirror(self, x1, x2):
        x_cls_1, x_p_cls_1, attn1, x_bkg_cls_1 = super().forward_cls(x1)
        x_cls_2, x_p_cls_2, attn2, x_bkg_cls_2 = super().forward_cls(x2)

        return [x_cls_1, x_cls_2, x_p_cls_1, x_p_cls_2, x_bkg_cls_1, x_bkg_cls_2], [attn1, attn2]

    # "GETAM" cam * gradient^2
    def getam(self, batch, start_layer=0, func = 'grad'):
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
            if func == 'cam_grad':
                cam = grad * cam 
                cam = cam.clamp(min=0).mean(dim=0)
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
        







