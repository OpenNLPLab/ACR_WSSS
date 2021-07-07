import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2



def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]

    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    # all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
    #                       for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention


from DPT.blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_vit,
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


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x

class CAM_Module(nn.Module):
    """ Channel attention module"""
    # paper: Dual Attention Network for Scene Segmentation
    def __init__(self,in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = torch.nn.Parameter(torch.zeros(1))
        self.softmax  = torch.nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature ( B X C X H X W)
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

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
        at_x = x * y.expand_as(x)
        return at_x + raw_x


class _AtrousSpatialPyramidPoolingModule(nn.Module):
    '''
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    '''

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=[6, 12, 18]):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, x_size[2:],
                                     mode='bilinear', align_corners=True)
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


class DPT(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
        use_pretrain=True,
        use_attention=False
    ):

        super(DPT, self).__init__()

        self.channels_last = channels_last
        self.attention = use_attention

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
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
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.cam_module = SELayer(channel=256)

        self.scratch.output_conv = head

        # classification head
        self.cls_head = nn.Linear(768, self.num_class)
        self.res_linear = nn.Linear(1024, 768)
        self.cls_head_2 = nn.Linear(1024, self.num_class)

    def forward(self, x):
        # x = torch.Size([1, 3, 384, 672])
        raw_x = x
        x_size = x.size()
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4, _, x_res = forward_vit(self.pretrained, x)

        x_cls = layer_4.clone()
        x_cls = F.avg_pool2d(x_cls, kernel_size=(x_cls.size(2), x_cls.size(3)), padding=0)
        x_res = F.avg_pool2d(x_res, kernel_size=(x_res.size(2), x_res.size(3)), padding=0)

        x_res = self.res_linear(x_res.squeeze(3).squeeze(2))
        x_cls = x_cls + x_res.unsqueeze(2).unsqueeze(3)
        x_cls = self.cls_head(x_cls.squeeze(3).squeeze(2))

        # x_cls = layer_4.clone()
        # x_cls = F.avg_pool2d(x_cls, kernel_size=(x_cls.size(2), x_cls.size(3)), padding=0)
        # x_cls = self.cls_head(x_cls.squeeze(3).squeeze(2))
        # x_cls = self.cls_head(x_cls)

        # x_seg = x_res.clone()

        # x_res = F.avg_pool2d(x_res, kernel_size=(x_res.size(2), x_res.size(3)), padding=0)
        # x_res = self.cls_head_2(x_res.squeeze(3).squeeze(2))

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        path_1 = self.cam_module(path_1)

        out = self.scratch.output_conv(path_1)

        # out = self.seg_head(x_seg)

        return x_cls, out #, x_res

    # def forward_cls(self, x):
    #     x_size = x.size()
    #     if self.channels_last == True:
    #         x.contiguous(memory_format=torch.channels_last)

    #     layer_1, layer_2, layer_3, layer_4, _, x_res = forward_vit(self.pretrained, x)

    #     x_cls = layer_4.clone()
    #     x_cls = F.avg_pool2d(x_cls, kernel_size=(x_cls.size(2), x_cls.size(3)), padding=0)
    #     x_cls = self.cls_head(x_cls.squeeze(3).squeeze(2))
    #     # x_cls = self.cls_head(x_cls)

    #     x_res = F.avg_pool2d(x_res, kernel_size=(x_res.size(2), x_res.size(3)), padding=0)
    #     x_res = self.cls_head_2(x_res.squeeze(3).squeeze(2))

    #     return x_cls, x_res

    def forward_cls_2(self, x):
        x_size = x.size()
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4, _, x_res = forward_vit(self.pretrained, x)

        x_cls = layer_4.clone()
        x_cls = F.avg_pool2d(x_cls, kernel_size=(x_cls.size(2), x_cls.size(3)), padding=0)
        x_res = F.avg_pool2d(x_res, kernel_size=(x_res.size(2), x_res.size(3)), padding=0)

        x_res = self.res_linear(x_res.squeeze(3).squeeze(2))
        x_cls = x_cls + x_res.unsqueeze(2).unsqueeze(3)
        x_cls = self.cls_head(x_cls.squeeze(3).squeeze(2))

        return x_cls
    
    def forward_cam_2(self, x):
        x_size = x.size()
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4, _, x_res = forward_vit(self.pretrained, x)

        x_cls = layer_4.clone()
        x_cls = F.avg_pool2d(x_cls, kernel_size=(x_cls.size(2), x_cls.size(3)), padding=0)
        x_res = F.avg_pool2d(x_res, kernel_size=(x_res.size(2), x_res.size(3)), padding=0)

        x_res = self.res_linear(x_res.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)

        res_cam = self.cls_head(x_res.permute(0,2,3,1))
        res_cam = res_cam.permute(0,3,1,2)
        res_cam = F.relu(res_cam)

        x_cls = x_cls + x_res
        x_cls = self.cls_head(x_cls.squeeze(3).squeeze(2))

        return x_cls, res_cam
    
    def forward_cam(self, x):
        x_size = x.size()
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4, _, x_res = forward_vit(self.pretrained, x)

        x_cls = layer_4.clone()
        x_cls = F.avg_pool2d(x_cls, kernel_size=(x_cls.size(2), x_cls.size(3)), padding=0)
        x_cls = self.cls_head(x_cls.squeeze(3).squeeze(2))
        # x_cls = self.cls_head(x_cls)

        res_cam = self.cls_head_2(x_res.permute(0,2,3,1))
        res_cam = res_cam.permute(0,3,1,2)
        res_cam = F.relu(res_cam)

        x_res = F.avg_pool2d(x_res, kernel_size=(x_res.size(2), x_res.size(3)), padding=0)
        x_res = self.cls_head_2(x_res.squeeze(3).squeeze(2))

        return x_cls, x_res, res_cam


class DPTSegmentationModel(DPT):
    def __init__(self, num_classes, path=None, **kwargs):
        self.num_class = num_classes

        features = kwargs["features"] if "features" in kwargs else 256

        # features = 160
        kwargs["use_bn"] = True

        head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes+1, kernel_size=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)
            
    def forward_cam_multiscale(self, x):
        input_size_h = x.size()[2]
        input_size_w = x.size()[3]

        x3 = F.interpolate(x, size=(int(input_size_h * 1.5), int(input_size_w * 1.5)), mode='bilinear',align_corners=False)
        x4 = F.interpolate(x, size=(int(input_size_h * 2), int(input_size_w * 2)), mode='bilinear',align_corners=False)

        with torch.enable_grad():
            x_cls, x_res, res_cam1 = super().forward_cam(x)
        with torch.no_grad():
            _,_,res_cam3 = super().forward_cam(x3)
            _,_,res_cam4 = super().forward_cam(x4)

        res_cam3 = F.interpolate(res_cam3, size=(int(res_cam1.shape[2]), int(res_cam1.shape[3])), mode='bilinear',align_corners=False)
        res_cam4 = F.interpolate(res_cam4, size=(int(res_cam1.shape[2]), int(res_cam1.shape[3])), mode='bilinear',align_corners=False)
        res_cam = (res_cam1+res_cam3+res_cam4)/3
        cam_list = [res_cam1, res_cam3, res_cam4]
        return x_cls, x_res, res_cam, cam_list

    def generate_cam(self, batch, start_layer=0):
        cams = []
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
            cams.append(cam.unsqueeze(0))

        rollout = compute_rollout_attention(cams, start_layer=start_layer)
        cam = rollout[:, 0, 1:]
        return cam, attn_list


class DPTSegmentationModelMultiscale(DPT):
    def __init__(self, num_classes, path=None, **kwargs):
        self.num_class = num_classes

        features = kwargs["features"] if "features" in kwargs else 256

        kwargs["use_bn"] = True

        head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes+1, kernel_size=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)
    

    def generate_cam(self, batch, start_layer=0):
        cams = []
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
            # cam = 
            cams.append(cam.unsqueeze(0))

        rollout = compute_rollout_attention(cams, start_layer=start_layer)
        cam = rollout[:, 0, 1:]
        return cam, attn_list



