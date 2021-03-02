# %%
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet34
from unet3d import Unet3D, AdaptiveInstanceNorm3d

#%%

def mish(x): return x * torch.tanh(F.softplus(x))
class Mish(nn.Module):
    def __init__(self, inplace=True): super().__init__()
    def forward(self, x): return mish(x)

class Noop(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x

class View(nn.Module):
    def __init__(self, *shape, exclude_batch_dim=True):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        if self.exclude_batch_dim:
            return x.view(x.size(0), *self.shape)
        else:
            return x.view(*self.shape)

class Projection(nn.Module):
    def __init__(self, im_feat, vox_feat, out_ch=4, act=nn.ReLU):
        super().__init__()
        self.out_ch = out_ch
        self.vox_feat = vox_feat
        self.im_feat = im_feat
        self.act = act(True)
        if im_feat != vox_feat*out_ch:
            self.project_w = nn.Linear(im_feat, vox_feat*out_ch)
        else:
            self.project_w = Noop()

    def to_weight(self, im_latent):
        bs = im_latent.size(0)
        return self.project_w(im_latent).view(bs*self.out_ch, self.vox_feat, 1,1,1)

    def forward(self, vol_latent, im_latent):
        bs = vol_latent.size(0)
        vol_sz = vol_latent.shape[-3:]
        return self.act(F.conv3d(
            vol_latent.view(1, bs*self.vox_feat, *vol_sz), # Mix BS with features to separate kernels for each batch item
            self.to_weight(im_latent), # Reshape (and opt. project) image features to Conv kernels, so that each group operates on 1 item of a batch
            groups=bs).reshape(bs, self.out_ch, *vol_sz)) # Reshape back to separate batch dim

class ProjectionPlus(nn.Module):
    def __init__(self, im_feat, vox_feat, out_ch=4, act=nn.ReLU):
        super().__init__()
        self.out_ch = out_ch
        self.vox_feat = vox_feat
        self.im_feat = im_feat
        if im_feat != vox_feat*out_ch*8:
            self.project_w = nn.Linear(im_feat, vox_feat*out_ch*8)
        else:
            self.project_w = Noop()
        self.final_layers = nn.Sequential(
            nn.ReLU(True),
            nn.InstanceNorm3d(out_ch*8),
            nn.Conv3d(out_ch*8, out_ch, kernel_size=1, padding=0),
            act(True)
        )

    def to_weight(self, im_latent):
        bs = im_latent.size(0)
        return self.project_w(im_latent).view(bs*self.out_ch*8, self.vox_feat, 1,1,1)

    def forward(self, vol_latent, im_latent):
        bs = vol_latent.size(0)
        vol_sz = vol_latent.shape[-3:]
        return self.final_layers(F.conv3d(
            vol_latent.view(1, bs*self.vox_feat, *vol_sz), # Mix BS with features to separate kernels for each batch item
            self.to_weight(im_latent), # Reshape (and opt. project) image features to Conv kernels, so that each group operates on 1 item of a batch
            groups=bs).reshape(bs, self.out_ch*8, *vol_sz)) # Reshape back to separate batch dim

class Standard3dConvNet(nn.Module):
    def __init__(self, layers=[16, 32, 64], first_conv_ks=1, norm=nn.InstanceNorm3d):
        super().__init__()
        vol_backbone = [nn.Sequential(
            nn.Conv3d(nin, nout, 1, 1, 0),
            nn.ReLU(True),
            norm(nout)) for nin, nout in zip(layers, layers[1:])]
        first_conv_kwargs = {'kernel_size': first_conv_ks, 'padding': first_conv_ks//2}
        self.net = nn.Sequential(
            nn.Sequential(
                nn.Conv3d(1, layers[0], stride=1, **first_conv_kwargs), # Kernel Size for first conv
                nn.ReLU(True),
                norm(layers[0])
            ),
            *vol_backbone                      # given layers
        )

    def forward(self, x): return self.net(x)

class NeuralTF(nn.Module):
    def __init__(self, backbone=resnet34(True), vol_backbone='standard', layers=[16, 32, 32], project='standard', last_act=nn.ReLU, norm=nn.InstanceNorm3d, first_conv_ks=1):
        super().__init__()
        self.im_backbone = backbone
        im_feat = self.im_backbone.fc.in_features
        self.im_backbone.fc = Noop()
        if vol_backbone == 'standard':
            self.vol_backbone = Standard3dConvNet(layers=layers, norm=norm, first_conv_ks=first_conv_ks)
            vox_feat = layers[-1]
        elif vol_backbone == 'unet':
            vox_feat = layers[-1]
            self.vol_backbone = Unet3D(output_dim=vox_feat, last_act=last_act)

        if project == 'standard':
            self.projection = Projection(im_feat, vox_feat, out_ch=4, act=last_act)
        elif project == 'plus':
            self.projection = ProjectionPlus(im_feat, vox_feat, out_ch=4, act=last_act)

    def forward(self, render, volume):
        im_latent = self.im_backbone(render)
        vol_latent = self.vol_backbone(volume)
        return self.projection(vol_latent, im_latent) # Returns RGBO Volume

class NeuralTF_Unet_Adain(nn.Module):
    def __init__(self, backbone=resnet34(True), last_act=nn.ReLU, style_dim=256):
        super().__init__()
        self.im_backbone = backbone
        im_feat = self.im_backbone.fc.in_features
        self.im_backbone.fc = Noop()
        if im_feat == style_dim:
            self.to_style = Noop()
        else:
            self.to_style = nn.Linear(im_feat, style_dim)
        self.vol_backbone = Unet3D(
            last_act=last_act,
            norm=partial(AdaptiveInstanceNorm3d, style_dim=style_dim)
        )

    def forward(self, render, volume):
        style = self.to_style(self.im_backbone(render))
        return self.vol_backbone(volume, style)
