from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast

from mmcv.cnn import Linear, build_activation_layer, build_norm_layer
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.cnn.bricks.transformer import FFN
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    FEEDFORWARD_NETWORK,
)
import torch.nn.init as init

from einops import rearrange

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class UnetExtractor(nn.Module):
    def __init__(self, in_channel=3, encoder_dim=[32, 48, 96], norm_fn='group'):
        super().__init__()
        self.in_ds = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU(inplace=True)
        )

        self.res1 = nn.Sequential(
            ResidualBlock(32, encoder_dim[0], norm_fn=norm_fn),
            ResidualBlock(encoder_dim[0], encoder_dim[0], norm_fn=norm_fn)
        )
        self.res2 = nn.Sequential(
            ResidualBlock(encoder_dim[0], encoder_dim[1], stride=2, norm_fn=norm_fn),
            ResidualBlock(encoder_dim[1], encoder_dim[1], norm_fn=norm_fn)
        )
        self.res3 = nn.Sequential(
            ResidualBlock(encoder_dim[1], encoder_dim[2], stride=2, norm_fn=norm_fn),
            ResidualBlock(encoder_dim[2], encoder_dim[2], norm_fn=norm_fn),
        )

    def forward(self, x):
        x = self.in_ds(x)
        x1 = self.res1(x)
        x2 = self.res2(x1)
        x3 = self.res3(x2)

        return x1, x2, x3


class MultiBasicEncoder(nn.Module):
    def __init__(self, output_dim=[128], encoder_dim=[64, 96, 128]):
        super(MultiBasicEncoder, self).__init__()

        # output convolution for feature
        self.conv2 = nn.Sequential(
            ResidualBlock(encoder_dim[2], encoder_dim[2], stride=1),
            nn.Conv2d(encoder_dim[2], encoder_dim[2]*2, 3, padding=1))

        # output convolution for context
        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(encoder_dim[2], encoder_dim[2], stride=1),
                nn.Conv2d(encoder_dim[2], dim[2], 3, padding=1))
            output_list.append(conv_out)

        self.outputs08 = nn.ModuleList(output_list)

    def forward(self, x):
        feat1, feat2 = self.conv2(x).split(dim=0, split_size=x.shape[0]//2)

        outputs08 = [f(x) for f in self.outputs08]
        return outputs08, feat1, feat2


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers

def upsample(x):
    return F.interpolate(x, scale_factor=2, mode='bilinear')

@PLUGIN_LAYERS.register_module()
class DenseGaussianNet(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_depth_layers=1,
        depth_dim=1,
        rgb_dim=3,
        norm_fn='group'
    ):
        super().__init__()
        
        
        self.rgb_dims = [64, 256, 128]
        self.depth_dims = [32, 48, 96]
        self.decoder_dims = [48, 64, 96]
        self.head_dim = 32

        self.sh_degree = 4
        self.d_sh = (self.sh_degree + 1) ** 2

        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

        self.depth_encoder = UnetExtractor(in_channel=depth_dim, encoder_dim=self.depth_dims)

        self.decoder3 = nn.Sequential(
            ResidualBlock(self.rgb_dims[2]+self.depth_dims[2], self.decoder_dims[2], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[2], self.decoder_dims[2], norm_fn=norm_fn)
        )

        self.decoder2 = nn.Sequential(
            ResidualBlock(self.rgb_dims[1]+self.depth_dims[1]+self.decoder_dims[2], self.decoder_dims[1], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[1], self.decoder_dims[1], norm_fn=norm_fn)
        )

        self.decoder1 = nn.Sequential(
            ResidualBlock(self.rgb_dims[0]+self.depth_dims[0]+self.decoder_dims[1], self.decoder_dims[0], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[0], self.decoder_dims[0], norm_fn=norm_fn)
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.out_conv = nn.Conv2d(self.decoder_dims[0]+rgb_dim+1, self.head_dim, kernel_size=3, padding=1)
        self.out_relu = nn.ReLU(inplace=True)

        self.rot_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 4, kernel_size=1),
        )
        self.scale_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 3, kernel_size=1),
            nn.Softplus(beta=100)
        )
        self.opacity_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.sh_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 3 * self.d_sh, kernel_size=1),
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            

    def forward(self, img, feature_maps, depth, cam, focal=None, gt_depths=None):
        
        # img_feat1, img_feat2, img_feat3, _ = feature_maps
        # img_feat1, img_feat2, img_feat3 = img_feat1[:, cam, ...], img_feat2[:, cam, ...], img_feat3[:, cam, ...]
        # img_feat1, img_feat2, img_feat3 = upsample(img_feat1.float()), upsample(img_feat2.float()), upsample(img_feat3.float())
        
        img_feat1, img_feat2, img_feat3 = feature_maps
        
        depth_feat1, depth_feat2, depth_feat3 = self.depth_encoder(depth.float())

               
        feat3 = torch.concat([img_feat3, depth_feat3], dim=1)
        feat2 = torch.concat([img_feat2, depth_feat2], dim=1)
        feat1 = torch.concat([img_feat1, depth_feat1], dim=1)


        

        up3 = self.decoder3(feat3)
        up3 = self.up(up3)
        up2 = self.decoder2(torch.cat([up3, feat2], dim=1))
        up2 = self.up(up2)
        up1 = self.decoder1(torch.cat([up2, feat1], dim=1))

        up1 = self.up(up1)
        out = torch.cat([up1, img, depth], dim=1)
        out = self.out_conv(out.float())
        out = self.out_relu(out)

        # rot head
        rot_out = self.rot_head(out)
        rot_out = torch.nn.functional.normalize(rot_out, dim=1)

        # scale head
        scale_out = torch.clamp_max(self.scale_head(out), 0.01)

        # opacity head
        opacity_out = self.opacity_head(out)

        # sh head
        sh_out = self.sh_head(out)
        # sh_out: [(b * v), C, H, W]

        sh_out = rearrange(
            sh_out, "n c h w -> n (h w) c",
        )
        sh_out = rearrange(
            sh_out,
            "... (srf c) -> ... srf () c",
            srf=1,
        )

        sh_out = rearrange(sh_out, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        # [(b * v), (H * W), 1, 1 3, 25]

        # sh_out = sh_out.broadcast_to(sh_out.shape) * self.sh_mask
        sh_out = sh_out * self.sh_mask
        
        
        return rot_out, scale_out, opacity_out, sh_out


    def loss(self, depth_preds, gt_depths):
        loss = 0.0
        for pred, gt in zip(depth_preds, gt_depths):
            pred = pred.permute(0, 2, 3, 1).contiguous().reshape(-1)
            gt = gt.reshape(-1)
            fg_mask = torch.logical_and(
                gt > 0.0, torch.logical_not(torch.isnan(pred))
            )
            gt = gt[fg_mask]
            pred = pred[fg_mask]
            pred = torch.clip(pred, 0.0, self.max_depth)
            with autocast(enabled=False):
                error = torch.abs(pred - gt).sum()
                _loss = (
                    error
                    / max(1.0, len(gt) * len(depth_preds))
                    * self.loss_weight
                )
            loss = loss + _loss
        return loss

