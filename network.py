from turtle import forward
import torch
import torch.nn as nn
from functools import partial
from vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import torch.nn.functional as F

import math



__all__ = [
    'deit_small_Mirrorformer_patch16_224',
]


class Mirrorformer(VisionTransformer):
    def __init__(self, last_opt='average', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_opt = last_opt
        self.num_class = 20
        
        self.head = nn.Linear(self.embed_dim, self.num_class)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        print(self.training)

    def forward_cls(self, x, n=12):
        x, attn_weights = self.forward_features(x, n)
        x_cls = self.head(x)

        attn_weights = torch.stack(attn_weights)

        return x_cls, attn_weights
    
    def forward_mirror(self, x1, x2):
        x_cls_1, attn1 = self.forward_cls(x1)
        x_cls_2, attn2 = self.forward_cls(x2)

        return x_cls_1, x_cls_2, attn1, attn2
    
    
    # "GETAM" cam * gradient^2
    def getam(self, batch, start_layer=0):
        cam_list = []
        attn_list = []
        grad_list = []
        for blk in self.blocks:
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

        # rollout = compute_rollout_attention(cams, start_layer=start_layer)
        cam_list = cam_list[start_layer:]
        cams = torch.stack(cam_list).sum(dim=0)
        
    
        cls_cam = torch.relu(cams[:, 0, 1:])
        # cls_cam = torch.relu(cams[:, 0, 2:]) # ditilled
        attn_map = torch.relu(cams[:,1:,1:])

        return cls_cam, attn_list, cam_list, attn_map



@register_model
def deit_small_Mirrorformer_patch16_224(pretrained=False, **kwargs):
    model = Mirrorformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        # print(checkpoint['model'].keys())
        del checkpoint['model']['head.weight']
        del checkpoint['model']['head.bias']
        
        model.load_state_dict(checkpoint["model"], strict=False)

    return model