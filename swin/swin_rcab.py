import torch
from .swin_encoder import SwinTransformer
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class Classifier_Module(nn.Module):
    def __init__(self,dilation_series,padding_series, NoLabels, input_channel):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series,padding_series):
            self.conv2d_list.append(nn.Conv2d(input_channel, NoLabels, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ECALayer(nn.Module):
    def __init__(self, channel, k_size):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.k_size = k_size
        self.conv = nn.Conv1d(channel, channel, kernel_size=k_size, bias=False, groups=channel)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = nn.functional.unfold(y.transpose(-1, -3), kernel_size=(1, self.k_size), padding=(0, (self.k_size - 1) // 2))
        y = self.conv(y.transpose(-1, -2)).unsqueeze(-1)
        y = self.sigmoid(y)
        x = x * y.expand_as(x)
        return x


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    # paper: Image Super-Resolution Using Very DeepResidual Channel Attention Networks
    # input: B*C*H*W
    # output: B*C*H*W
    def __init__(
        self, n_feat, kernel_size=3, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

class Edge_Module(nn.Module):

    def __init__(self, in_fea=[256, 256, 256], mid_fea=32):
        super(Edge_Module, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_fea[0], mid_fea, 1)
        self.conv4 = nn.Conv2d(in_fea[1], mid_fea, 1)
        self.conv5 = nn.Conv2d(in_fea[2], mid_fea, 1)
        self.conv5_2 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.conv5_4 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.conv5_5 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)

        self.classifer = nn.Conv2d(mid_fea * 3, 1, kernel_size=3, padding=1)
        self.rcab = RCAB(mid_fea * 3)

    def forward(self, x2, x4, x5):
        _, _, h, w = x2.size()
        edge2_fea = self.relu(self.conv2(x2))
        edge2 = self.relu(self.conv5_2(edge2_fea))
        edge4_fea = self.relu(self.conv4(x4))
        edge4 = self.relu(self.conv5_4(edge4_fea))
        edge5_fea = self.relu(self.conv5(x5))
        edge5 = self.relu(self.conv5_5(edge5_fea))

        edge4 = F.interpolate(edge4, size=(h, w), mode='bilinear', align_corners=True)
        edge5 = F.interpolate(edge5, size=(h, w), mode='bilinear', align_corners=True)

        edge = torch.cat([edge2, edge4, edge5], dim=1)
        edge = self.rcab(edge)
        edge = self.classifer(edge)
        return edge

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
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, reduction_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))

    def forward(self, x, edge):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, x_size[2:],
                                     mode='bilinear', align_corners=True)
        out = img_features

        edge_features = F.interpolate(edge, x_size[2:],
                                      mode='bilinear', align_corners=True)
        edge_features = self.edge_conv(edge_features)
        out = torch.cat((out, edge_features), 1)

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out

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

class FCDiscriminator(nn.Module):
    def __init__(self, ndf):
        super(FCDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(4, ndf, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Conv2d(ndf, 1, kernel_size=3, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.bn1 = nn.BatchNorm2d(ndf)
        self.bn2 = nn.BatchNorm2d(ndf)
        self.bn3 = nn.BatchNorm2d(ndf)
        self.bn4 = nn.BatchNorm2d(ndf)
        #self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        # #self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x


class Swin_rcab(torch.nn.Module):
    def __init__(self, img_size, pretrain,channel_size = 32):
        super(Swin_rcab, self).__init__()

        self.encoder = SwinTransformer(img_size=img_size, 
                                       embed_dim=128,
                                       depths=[2,2,18,2],
                                       num_heads=[4,8,16,32],
                                       window_size=12)
        # pretrained_dict = torch.load(pretrain)["model"]
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.encoder.state_dict()}
        # self.encoder.load_state_dict(pretrained_dict)
        self.upsample32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1 = BasicConv2d(in_planes=128, out_planes=channel_size, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(in_planes=256, out_planes=channel_size, kernel_size=3, padding=1)
        self.conv3 = BasicConv2d(in_planes=512, out_planes=channel_size, kernel_size=3, padding=1)
        self.conv4 = BasicConv2d(in_planes=1024, out_planes=channel_size, kernel_size=3, padding=1)
        self.conv5 = BasicConv2d(in_planes=1024, out_planes=channel_size, kernel_size=3, padding=1)



        self.racb5 = RCAB(channel_size * 5)
        self.racb_5 = RCAB(channel_size * 5)
        self.racb_4 = RCAB(channel_size * 4)
        self.racb_3 = RCAB(channel_size * 3)
        self.racb_2 = RCAB(channel_size * 2)
        self.racb_1 = RCAB(channel_size * 1)

        self.layer5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel_size * 5)
        self.layer6 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel_size * 4)
        self.layer7 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel_size * 3)
        self.layer8 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel_size * 2)
        self.layer9 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel_size * 1)

        self.conv_depth = BasicConv2d(6, 3, kernel_size=3, padding=1)

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x, depth=None):
        if depth != None:
            x = torch.cat((x, depth), 1)
            x = self.conv_depth(x)
        features = self.encoder(x)

        x1, x2, x3, x4, x5 = features[-5], features[-4], features[-3], features[-2], features[-1]
        # [8, 128, 96, 96], [8, 256, 48, 48], [8, 512, 24, 24], [8, 1024, 12, 12], [8, 1024, 12, 12]

        x1, x2, x3, x4, x5 = self.conv1(x1), self.conv2(x2), self.conv3(x3), self.conv4(x4), self.conv5(x5)

        feat_cat = torch.cat((x1, self.upsample2(x2), self.upsample4(x3), self.upsample8(x4), self.upsample8(x5)), 1)
        feat_cat = self.racb_5(feat_cat)
        output5 = self.upsample4(self.layer5(feat_cat))  # (b, 1, 44, 44)

        feat_cat = torch.cat((x2, self.upsample2(x3), self.upsample4(x4), self.upsample4(x5)), 1)
        feat_cat = self.racb_4(feat_cat)
        output4 = self.upsample8(self.layer6(feat_cat))  # (b, 1, 44, 44)

        feat_cat = torch.cat((x3, self.upsample2(x4), self.upsample2(x5)), 1)
        feat_cat = self.racb_3(feat_cat)
        output3 = self.upsample16(self.layer7(feat_cat))  # (b, 1, 44, 44)

        feat_cat = torch.cat((x4, x5), 1)
        feat_cat = self.racb_2(feat_cat)
        output2 = self.upsample32(self.layer8(feat_cat))  # (b, 1, 44, 44)

        output1 = self.upsample32(self.layer9(x5))  # (b, 1, 44, 44)

        # List: [8, 128, 96, 96], [8, 256, 48, 48], [8, 512, 24, 24], [8, 1024, 12, 12], [8, 1024, 12, 12]
        # outputs = self.decoder(features)

        return [output5,output4,output3,output2,output1]
