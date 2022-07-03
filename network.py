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
    'vitb_base_Mirrorformer',
]


class Mirrorformer(VisionTransformer):
    def __init__(self, last_opt='average', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_opt = last_opt
        self.num_class = 20
        
        self.head = nn.Linear(self.embed_dim, self.num_class)
        self.head_2 = nn.Linear(self.embed_dim, self.num_class)
        self.bkg_head = nn.Linear(self.embed_dim, self.num_class)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.bkg_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        print(self.training)
    
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - self.num_classes
        N = self.pos_embed.shape[1] - self.num_classes
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0:self.num_classes]
        patch_pos_embed = self.pos_embed[:, self.num_classes:]
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)
    
    def forward_features(self, x, n):
        # B = x.shape[0]
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        bkg_tokens = self.bkg_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = torch.cat((bkg_tokens, x), dim=1)
        # x = x + self.pos_embed
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)
        attn_weights = []

        for i, blk in enumerate(self.blocks):
            x, weights = blk(x)
            if len(self.blocks) - i <= n:
                attn_weights.append(weights)

        x = self.norm(x)
        return x[:, 0], attn_weights, x

    def forward_cls(self, x, n=12):
        _, _, x = self.forward_features(x, n)
        x_bkg = x[:, 0, :]
        x_cls = x[:, 1, :]
        x_patch = x[:, 2:, :]

        x_cls = self.head(x)
        x_patch_cls = F.avg_pool1d(x_patch.permute(0,2,1), kernel_size=x_patch.shape[1])[:,:,0]
        x_patch_cls = self.head_2(x_patch_cls)

        attn_list = []
        for blk in self.blocks:
            attn = blk.attn.get_attn()
            attn = torch.mean(attn, dim=1)
            attn_list.append(attn)

        cls_attn_sum = torch.stack(attn_list, dim=1)

        # attn_weights = torch.stack(attn_weights)

        return x_cls,x_patch_cls, cls_attn_sum
    
    def forward_mirror(self, x1, x2):
        x_cls_1,x_p_cls_1, attn1 = self.forward_cls(x1)
        x_cls_2,x_p_cls_2, attn2 = self.forward_cls(x2)

        return x_cls_1, x_cls_2,x_p_cls_1, x_p_cls_2, attn1, attn2
    
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
def vitb_base_Mirrorformer(pretrained=False, **kwargs):
    model = Mirrorformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
            map_location="cpu", check_hash=True
        )
        # print(checkpoint.keys())
        del checkpoint['head.weight']
        del checkpoint['head.bias']
        
        model.load_state_dict(checkpoint, strict=False)

    return model