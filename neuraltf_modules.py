# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet34

#%%
class Noop(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x

class View(nn.Module):
    def __init__(self, *shape, exclude_batch_dim=True):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        if exclude_batch_dim:
            return x.view(x.size(0), *self.shape)
        else:
            return x.view(*self.shape)

class Projection(nn.Module):
    def __init__(self, im_feat, vox_feat, out_ch=4):
        super().__init__()
        self.out_ch = out_ch
        self.vox_feat = vox_feat
        self.im_feat = im_feat
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
        return F.conv3d(
            vol_latent.view(1, bs*self.vox_feat, *vol_sz), # Mix BS with features to separate kernels for each batch item
            self.to_weight(im_latent), # Reshape (and opt. project) image features to Conv kernels, so that each group operates on 1 item of a batch
            groups=bs).reshape(bs, self.out_ch, *vol_sz) # Reshape back to separate batch dim

class NeuralTF(nn.Module):
    def __init__(self, backbone=resnet34(True), layers=[16, 32, 32], first_conv_ks=1):
        super().__init__()
        self.im_backbone = backbone
        im_feat = self.im_backbone.fc.in_features
        self.im_backbone.fc = Noop()
        vox_feat = layers[-1]
        vol_backbone = [nn.Sequential(
            nn.Conv3d(nin, nout, 1, 1, 0),
            nn.ReLU(True),
            nn.BatchNorm3d(nout)) for nin, nout in zip(layers, layers[1:])]
        first_conv_kwargs = {'kernel_size': first_conv_ks, 'padding': first_conv_ks//2}
        self.vol_backbone = nn.Sequential(
            nn.Sequential(
                nn.Conv3d(1, layers[0], stride=1, **first_conv_kwargs), # Kernel Size 3 for first conv
                nn.ReLU(True),
                nn.BatchNorm3d(layers[0])
            ),
            *vol_backbone                      # given layers
        )
        self.projection = Projection(im_feat, vox_feat, out_ch=4)

    def forward(self, render, volume):
        im_latent = self.im_backbone(render)
        vol_latent = self.vol_backbone(volume)
        return self.projection(vol_latent, im_latent) # Returns RGBO Volume


# %%  vox_feat=8, vol_sz=(5,5,5), BS=3, out_ch=4
# vol_latent = torch.stack([i*torch.ones(8, 5,5,5) for i in range(3)])               # (3, 8,  5,5,5)
# im_latent = torch.stack([i*torch.ones(8*4) for i in range(3)]).view(3,4*8,1,1,1)   # (3, 32, 1,1,1)

# vol_tfd = vol_latent.view(1, 3*8, 5,5,5)
# im_tfd = im_latent.view(3*4, 8, 1,1,1)

# res = F.conv3d(vol_tfd, im_tfd, groups=3)
# res.shape
# res2 = res.view(3, 4, 5,5,5)
# nn.Conv3d(24, 12, 1, 1, groups=3).weight.shape

# %%
