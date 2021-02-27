#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class AdaptiveInstanceNorm3d(nn.Module):
    def __init__(self, num_feature_maps, style_dim):
        super().__init__()
        self.var = nn.Linear(style_dim, num_feature_maps)
        self.mu  = nn.Linear(style_dim, num_feature_maps)
        nn.init.normal_(self.var.weight, std=1e-2)
        nn.init.constant_(self.var.bias, 1.0)
        nn.init.normal_(self.mu.weight,  std=1e-2)
        nn.init.constant_(self.mu.bias,  0.0)
        self.n_feature_maps = num_feature_maps
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, x, y):
        bs = x.size(0)

        pmu = self.mu(y).view(bs, -1, 1,1,1)   # Predicted mean
        pstd = torch.sqrt(self.var(y).view(bs, -1, 1,1,1)) # Predicted std (actually predicting variance for numerical stability)

        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()
            xmean = x.mean(dim=(2,3,4)).view(bs, self.n_feature_maps, 1,1,1) # Get mean of x
            xstd  = x.std( dim=(2,3,4)).view(bs, self.n_feature_maps, 1,1,1) # Get std of x

        std_factor = pstd.float() / (xstd + self.eps) # pre-multiply the scaling factor to adjust to the new std
        return (x - xmean) * (std_factor + self.eps) + pmu.float()  # Normalize x


class ConvBlock(nn.Module):
    def __init__(self, nin, nout, act=nn.ReLU, norm=nn.InstanceNorm3d):
        super().__init__()
        self.act = act(True)
        self.conv = nn.Conv3d(nin, nout, stride=1, padding=1, kernel_size=3)
        self.norm = norm(nout)

    def forward(self, x, y=None):
        if y is None:
            return self.act(self.norm(self.conv(x)))
        else:
            return self.act(self.norm(self.conv(x), y))

class ResBlock(nn.Module):
    def __init__(self, nin, nout, act=nn.ReLU, norm=nn.InstanceNorm3d):
        super().__init__()
        self.act = act(True)
        self.conv1 = ConvBlock(nin, nin,  act=act, norm=norm)
        self.conv2 = ConvBlock(nin, nout, act=act, norm=norm)
        if nin != nout:
            self.project = nn.Conv3d(nin, nout, kernel_size=1, stride=1, padding=0)
        else:
            self.project = lambda x: x

    def forward(self, x, y=None):
        return self.conv2(self.conv1(x, y), y) + self.project(x)

class Unet3D(nn.Module):
    def __init__(self, layers=[16, 32, 64], input_dim=1, output_dim=4,
           block=ResBlock,
             act=nn.ReLU,      norm=nn.InstanceNorm3d,
        last_act=nn.ReLU, last_norm=nn.InstanceNorm3d):
        # Init
        super().__init__()
        self.downsample = partial(F.interpolate, scale_factor=0.5, mode='nearest')
        self.upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        self.encoder, self.decoder = nn.ModuleList([]), nn.ModuleList([])
        decoder_nin_mult = [1] + [2]*(len(layers)-1)
        # FIRST CONV
        self.first_conv = nn.Conv3d(input_dim, layers[0]-1, kernel_size=3, padding=1, stride=1)
        # ENCODER
        for nin, nout in zip(layers, layers[1:]):
            self.encoder.append(block(nin, nout, act=act, norm=norm))
        # MID CONV
        self.mid_conv = block(layers[-1], layers[-1], norm=norm)
        # DECODER
        for nin, nout, m in zip(reversed(layers), reversed(layers[:-1]), decoder_nin_mult):
            self.decoder.append(block(m*nin, nout, act=act, norm=norm))
        # LAST CONV
        self.last_conv  = nn.Sequential(
            nn.Conv3d(2*layers[0], output_dim, kernel_size=1, padding=0, stride=1),
            last_norm(output_dim),
            last_act(True)
        )


    def forward(self, x, y):
        x = torch.cat([x, self.first_conv(x)], dim=1)  # to layers[0]
        skips = [x]
        for block in self.encoder:
            x = block(self.downsample(x), y)
            skips.append(x)
        x = self.mid_conv(x, y) + skips[-1]

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            x = block(x, y)
            x = torch.cat([self.upsample(x), skip], dim=1)
        x = self.last_conv(x)
        return x


# %%
