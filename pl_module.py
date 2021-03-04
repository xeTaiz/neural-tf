from argparse  import ArgumentParser
from functools import partial
from itertools import count
from pathlib   import Path
import time, math, os
import numpy as np
import wandb
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.utils  import make_grid
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.metrics.functional import accuracy, confusion_matrix

# from pytorch_msssim import ssim as ssim2d
from ranger     import Ranger
from ranger_adabelief import RangerAdaBelief
from neuraltf_modules import NeuralTF, NeuralTF_Unet_Adain
from unet3d import AdaptiveInstanceNorm3d
from adaptive_wing_loss import AdaptiveWingLoss, NormalizedReLU, NegativeScaledReLU
from ssim3d_torch import ssim3d

from torchvtk.datasets  import TorchDataset, TorchQueueDataset, dict_collate_fn
from torchvtk.utils     import make_4d, make_5d, make_nd, apply_tf_torch, tex_from_pts, TransferFunctionApplication, random_tf_from_vol, create_peaky_tf
from torchvtk.utils.tf_generate import make_trapezoid, colorize_trapeze, tf_pts_border, flatten_clip_sort_peaks, TFGenerator
from torchvtk.transforms import Composite, Lambda
from torchvision.transforms.functional import hflip
from torchvision.transforms import ColorJitter, RandomAffine

from torchvtk.rendering import show, show_tf, plot_tfs, plot_render_2tf, plot_render_tf

def normalize(tensors, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True):
    mean = torch.as_tensor(mean, dtype=tensors.dtype, device=tensors.device)[None, :, None, None]
    std  = torch.as_tensor(std,  dtype=tensors.dtype, device=tensors.device)[None, :, None, None]

    if not inplace:
        tensors = tensors.clone()

    return tensors.sub_(mean).div_(std)

class WeightedMSELoss(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, pred, targ, weight=1):
        return torch.mean(F.mse_loss(pred, targ, reduction='none') * weight)

class WeightedMAELoss(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, pred, targ, weight=1):
        return torch.mean(F.l1_loss(pred, targ, reduction='none') * weight)

class WeightedDSSIMLoss(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, pred, targ, weight=1):
        return 1.0 - torch.mean(ssim3d(pred, targ, return_average=False) * weight)

class Noop(nn.Module):
    def __init__(self, *args, **kwargs): super().__init__()
    def forward(self, x, *args, **kwargs): return x

def plot_slices(pred_slices, targ_slices):
    gs = {
        'width_ratios': [1,1,1],
        'height_ratios': [1,1,1,1]
    }
    fig, axs = plt.subplots(4, 3, gridspec_kw=gs, figsize=(15, 20))
    for i, pred, targ, tit in zip(count(), pred_slices, targ_slices, ['along Z', 'along Y', 'along X']):
        show(pred[:3], axs[0, i], title=f'Pred Color {tit}')
        show(targ[:3], axs[1, i], title=f'Targ Color {tit}')
        show(pred[[3,3,3]],  axs[2, i], title=f'Pred Opacity {tit}')
        show(targ[[3,3,3]],  axs[3, i], title=f'Targ Opacity {tit}')
    return fig

def fig_to_img(fig):
    fig.set_tight_layout(True)
    fig.set_dpi(150)
    fig.canvas.draw()
    w, h = fig.get_size_inches() * fig.get_dpi()
    w, h = int(w),int(h)
    im = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((h, w, 3))
    plt.close(fig)
    return im

def grab_bin_idxs(x, bins, value_range=None, as_tuples=False):
    ''' Bins the input to `value_range` using `bins` bins, then returns the indices of `x` which lie in one bin, for all bins

    Args:
        x (Tensor): The tensor for which to retrieve indices
        bins (int): Number of bins to discretize the `value_range`
        value_range (2-tuple/list, optional): The value range used for the bins. Defaults to None `(x.min(), x.max())`.
        as_tuples (bool, optional): Returns the indices N-tuple with `N = x.ndim` when True. See torch.nonzero's `as_tuple`. Defaults to False.

    Raises:
        Exception: fwhen `value_range` is neither `None` or a 2-tuple/list

    Returns:
        [Tensor]: List of `x.ndim`-tuples with Index Tensors of shape (N,) when `as_tuples=True`, otherwise List of Index Tensors (N, x.ndim)
    '''
    if value_range is None:
        # Add eps to x.max() to include the max value in the last upper bound
        eps = torch.finfo(x.dtype).eps if x.dtype.is_floating_point else 1
        mi, ma = x.min(), x.max() + eps
    elif isinstance(value_range, (tuple, list)) and len(value_range) == 2:
        mi, ma = value_range
    else:
        raise Exception(f'Invalid value_range given ({value_range})')
    boundaries = torch.linspace(mi, ma, bins+1, device=x.device, dtype=x.dtype)
    boundaries = torch.stack([boundaries[:-1], boundaries[1:]], dim=-1)
    return [ torch.nonzero(torch.logical_and(b[0] <= x , x < b[1]), as_tuple=as_tuples)
            for b in boundaries ]


class NeuralTransferFunction(LightningModule):
    def __init__(self, hparams=None, load_vols=True):
        super().__init__()
        self.hparams = hparams
    # Image Backbone
        if self.hparams.backbone == 'resnet18':
            im_backbone = resnet18(pretrained=hparams.pretrained)
        elif self.hparams.backbone == 'resnet34':
            im_backbone = resnet34(pretrained=hparams.pretrained)
        elif self.hparams.backbone == 'resnet50':
            im_backbone = resnet50(pretrained=hparams.pretrained)
        else:
            raise Exception(f'Invalid parameter backbone: {hparams.backbone}. Use either resnet18, resnet34, resnet50')
    # Output Activation
        if hparams.last_act == 'nrelu':
            act = NormalizedReLU
        elif hparams.last_act == 'relu':
            act = nn.ReLU
        elif hparams.last_act == 'nsrelu':
            act = NegativeScaledReLU
        elif hparams.last_act == 'sigmoid':
            act = nn.Sigmoid
        elif hparams.last_act == 'none':
            act = Noop
        else:
            raise Exception(f'Invalid last activation given ({hparams.last_act}).')
    # Volume Backbone Normalization
        if hparams.norm == 'instance':
            norm = nn.InstanceNorm3d
        elif hparams.norm == 'batch':
            norm = nn.BatchNorm3d
        elif hparams.norm == 'none':
            norm = Noop
        else:
            raise Exception(f'Invalid norm given ({hparams.norm}).')

    # Initialize Network
        if   hparams.net == 'ConvProject':
            self.network = NeuralTF(backbone=im_backbone, first_conv_ks=hparams.first_conv_ks, last_act=act, norm=norm)
        elif hparams.net == 'ConvProjectPlus':
            self.network = NeuralTF(backbone=im_backbone, first_conv_ks=hparams.first_conv_ks, last_act=act, norm=norm, project='plus')
        elif hparams.net == 'UnetProject':
            self.network = NeuralTF(backbone=im_backbone, vol_backbone='unet', first_conv_ks=hparams.first_conv_ks, last_act=act, norm=norm)
        elif hparams.net == 'UnetProjectPlus':
            self.network = NeuralTF(backbone=im_backbone, vol_backbone='unet', first_conv_ks=hparams.first_conv_ks, last_act=act, norm=norm, project='plus')
        elif hparams.net == 'UnetAdain':
            self.network = NeuralTF_Unet_Adain(backbone=im_backbone, last_act=act)
        else:
            raise Exception(f'Invalid net parameter given ({hparams.net}). Choose either ConvProject or UnetAdain')
    # AdaIN Initialization
        if hparams.net == 'UnetAdain':
            for name, l in self.network.named_modules():
                if isinstance(l, AdaptiveInstanceNorm3d):
                    torch.nn.init.normal_(l.var.weight, std=1e-2)
                    torch.nn.init.constant_(l.var.bias, 1.0)
                    torch.nn.init.normal(l.mu.weight, std=1e-2)
                    torch.nn.init.constant_(l.mu.bias, 0.0)

    # Loss Function
        if hparams.loss == 'awl':
            self.loss = AdaptiveWingLoss()
        elif hparams.loss == 'mse':
            self.loss = WeightedMSELoss()
        elif hparams.loss == 'mae':
            self.loss = WeightedMAELoss()
        elif hparams.loss == 'ssim':
            self.loss = WeightedDSSIMLoss()
        else:
            raise Exception(f'Invalid loss given ({hparams.loss}). Valid choices are mse, mae and awl')
    # Preload volumes for Training
        if load_vols:
            print(f'Loading volumes to memory (from  {hparams.cq500}).')
            self.volumes = {it['name']: it for it in TorchDataset(hparams.cq500).preload()}
        else:
            self.volumes = {}


    def forward(self, render, volume):
        return self.network(render, volume)

    def infer_1d_tex(self, images, volumes, tex_resolution=256, reduction=torch.mean):
        ''' Infer a 1D Transfer Function texture using the NeuralTF

        Args:
            images (Tensor): Either 3D image tensor or a 4D image batch. Predictions are averaged over the batch dimension.
            volumes (Tensor): Either 4D volume tensor or a 5D volume batch. Predictions are averaged over the batch dimension.
            tex_resolution (int, optional): Resolution of the resulting Transfer Function Texture. Defaults to 256.

        Returns:
            Tensor, Tensor: Transfer Function texture of shape `(C, tex_resolution)` and preclassified Volume (BS, C, D,H,W)
        '''
        images = make_4d(images)[:, :3]
        volumes = make_5d(volumes)

        rgbo = self.network(images, volumes)
        idxs = grab_bin_idxs(volumes, tex_resolution, value_range=(0,1), as_tuples=True)

        r, g, b, o = rgbo.split(1, dim=1) # Split (BS, 4, D,H,W) into 4x (BS, 1, D,H,W)
        return torch.stack([torch.tensor([
            reduction(torch.clamp(r[idx], 0, 1)),
            reduction(torch.clamp(g[idx], 0, 1)),
            reduction(torch.clamp(b[idx], 0, 1)),
            reduction(torch.clamp(o[idx], 0, 1))], dtype=r.dtype, device=r.device)
                if idx[0].nelement() > 3
                else torch.zeros(4, dtype=r.dtype, device=r.device)
                for idx in idxs], dim=-1), rgbo

    def training_step(self, batch, batch_idx):
        dtype = torch.float16 if self.hparams.precision == 16 else torch.float32
        render_gt = batch['render'].to(dtype)
        if self.hparams.pretrained:
            render_input = normalize(render_gt[:, :3], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        else:
            render_input = render_gt[:, :3]
        tf_pts    = batch['tf_pts']
        if torch.is_tensor(tf_pts): tf_pts = [t for t in tf_pts]
        if self.hparams.train_any_vol: # Load random volumes
            idx = torch.randperm(len(self.volumes))[:render_gt.size(0)]
            vol_keys = list(self.volumes.keys())
            idx = [vol_keys[i] for i in idx]
            vols = torch.stack([self.volumes[n]['vol'] for n in idx]).to(dtype).to(render_gt.device)
        else: # Load the volumes seen in the input renders
            vols = torch.stack([self.volumes[n[:n.rfind('_')]]['vol'] for n in batch['name']]).to(dtype).to(render_gt.device)

        rgbo_pred = self.forward(render_input, vols)
        rgbo_targ = apply_tf_torch(vols, tf_pts)
        if self.hparams.opacity_weight == 1:
            w = 1
        else:
            w = torch.ones_like(rgbo_targ)
            w[:, 3][rgbo_targ[:, 3] > 1e-2] *= self.hparams.opacity_weight
        loss = self.loss(rgbo_pred, rgbo_targ, weight=w)
        mae = F.l1_loss(rgbo_pred.detach(), rgbo_targ)

        self.log('metrics/train_loss', loss.detach().cpu(), prog_bar=True)
        self.log('metrics/train_mae', mae.detach().cpu(), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        dtype = torch.float16 if self.hparams.precision == 16 else torch.float32
        render_gt = batch['render'].to(dtype)
        if self.hparams.pretrained:
            render_input = normalize(render_gt[:, :3], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        else:
            render_input = render_gt[:, :3]
        tf_pts    = batch['tf_pts']
        if torch.is_tensor(tf_pts): tf_pts = [t for t in tf_pts]
        vols = torch.stack([self.volumes[n[:n.rfind('_')]]['vol'] for n in batch['name']]).to(dtype).to(render_gt.device)

        rgbo_pred = self.forward(render_input, vols)
        tf_pred_tex, _ = self.infer_1d_tex(render_input[0], vols[0])
        rgbo_targ = apply_tf_torch(vols, tf_pts)
        if self.hparams.opacity_weight == 1:
            w = 1
        else:
            w = torch.ones_like(rgbo_targ)
            w[:, 3][rgbo_targ[:, 3] > 1e-2] *= self.hparams.opacity_weight

        loss = self.loss(rgbo_pred, rgbo_targ, weight=w)
        mae = F.l1_loss(rgbo_pred, rgbo_targ)
        z,h,w = rgbo_pred.shape[-3:]

        pred_slices = torch.clamp(
                      torch.stack([rgbo_pred[:, :, z//2, :,   : ],
                                   rgbo_pred[:, :,  :, h//2,  : ],
                                   rgbo_pred[:, :,  :,   :, w//2]], dim=1),
                      0, 1)
        targ_slices = torch.stack([rgbo_targ[:, :, z//2, :,   : ],
                                   rgbo_targ[:, :,  :, h//2,  : ],
                                   rgbo_targ[:, :,  :,   :, w//2]], dim=1)

        if batch_idx < self.hparams.num_images_logged:
            image_logs = {
                'render': render_gt[[0]].cpu().float(),
                'pred_slices': pred_slices[[0]].flip(-2).cpu().float(),
                'targ_slices': targ_slices[[0]].flip(-2).cpu().float(),
                'tf_tex': tf_pred_tex[None].cpu().float(),
                'tf_targ': list(map(lambda tf: tf.cpu().float(), tf_pts[:1]))
            }
        else:
            image_logs = {}

        for name, m in self.network.named_modules():
            if isinstance(m, AdaptiveInstanceNorm3d):
                self.log(f'weight_norms/{name} Variance Weight Norm', torch.norm(m.var.weight))
                self.log(f'weight_norms/{name} Variance Bias Norm', torch.norm(m.var.bias))
                self.log(f'weight_norms/{name} Mean Weight Norm', torch.norm(m.mu.weight))
                self.log(f'weight_norms/{name} Mean Bias Norm', torch.norm(m.mu.bias))
        return {
            'loss': loss,
            'mae': mae,
            **image_logs
        }

    def validation_epoch_end(self, outputs):
        n = self.hparams.num_images_logged
        val_loss = torch.stack([o['loss'] for o in outputs]).mean()
        renders = torch.cat([o['render'] for o in outputs[:n]], dim=0)
        tf_texs = torch.cat([o['tf_tex'] for o in outputs[:n]], dim=0)
        tf_targ = [tf for o in outputs[:n] for tf in o['tf_targ']]
        pred_slices = torch.clamp(torch.cat([o['pred_slices'] for o in outputs[:n]], dim=0), 0, 1)
        targ_slices = torch.clamp(torch.cat([o['targ_slices'] for o in outputs[:n]], dim=0), 0, 1)

        self.log_dict({
            f'figs/val_linspace_tf{i}': wandb.Image(
                fig_to_img(plot_render_2tf(ren, tfp, tft)))
                for i, ren, tfp, tft in zip(count(), renders, tf_texs, tf_targ)
        })
        self.log_dict({
            f'figs/val_slices_{i}': wandb.Image(
                fig_to_img(plot_slices(ps, ts)))
                for i, ps, ts in zip(count(), pred_slices, targ_slices)
        })

        self.log('val_loss', val_loss)
        self.log('metrics/val_loss', val_loss)
        self.log('metrics/val_mae', torch.stack([o['mae'] for o in outputs]).mean())

    def configure_optimizers(self):
        params = [
            {'params': self.network.im_backbone.parameters(), 'lr': self.hparams.lr_im_backbone},
            {'params': self.network.vol_backbone.parameters(), 'lr': self.hparams.lr_vol_backbone},
        ]
        if 'Project' in self.hparams.net:
            params.append({'params': self.network.projection.parameters(), 'lr': self.hparams.lr_projection, 'weight_decay': 1e-4})

        if  self.hparams.opt.lower() == 'ranger':
            opt = Ranger(params, weight_decay=self.hparams.weight_decay)
        elif self.hparams.opt.lower() == 'rangeradabelief':
            opt = RangerAdaBelief(params, weight_decay=self.hparams.weight_decay)
        elif self.hparams.opt.lower() == 'adam':
            opt = torch.optim.Adam(params, weight_decay=self.hparams.weight_decay)
        else: raise Exception(f'Invalid optimizer given: {self.hparams.opt}')
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5)
        return {
            'optimizer': opt,
            'lr_scheduler': sched,
            'monitor': 'val_loss',
            'mode': 'min'
        }

    def denormalize(self, image):
        pass
    def transform(self, train):
        def train_augment(image):
            rot_tfm = RandomAffine(degrees=10.0)
            if (torch.rand(1) > 0.5).all():
                image = hflip(image)
            image = rot_tfm(image)
            return image

        def test_augment(image):
            return image

        if train and not self.hparams.overfit: return Composite(
            Lambda(train_augment, apply_on='render', dtype=torch.float32)
        )
        else:     return Composite(
            Lambda(test_augment, apply_on='render', dtype=torch.float32)
        )

    def train_dataloader(self):
        print('Loading Training DataLoader')
        if self.hparams.one_vol:
            split = math.floor(0.8* len(os.listdir(self.hparams.trainds)))
            if self.hparams.max_train_samples is not None:
                split = min(self.hparams.max_train_samples, split)
            ffn = lambda p: int(p.name[p.name.rfind('_')+1:-3]) <= split
        else:
            names = list(set(map(lambda n: n[:n.rfind('_')], os.listdir(Path(self.hparams.trainds)))))
            split_idx = math.floor(0.8 * len(names))
            train_names = names[:split_idx]
            ffn = lambda p: p.name[:p.name.rfind('_')] in train_names
            print('train names: ', train_names)
        ds = TorchDataset(self.hparams.trainds,
            filter_fn=ffn,
            preprocess_fn=self.transform(train=True)
        )
        print(f' Train Dataset length: {len(ds)}')
        if self.hparams.preload:
            ds = ds.preload()
        dl = torch.utils.data.DataLoader(ds,
            batch_size=self.hparams.batch_size,
            collate_fn=dict_collate_fn,
            shuffle=True,
            num_workers=0 if self.hparams.preload else 6
        )
        print(f'Train DataLoader length: {len(dl)}')
        return dl

    def val_dataloader(self):
        print('Loading Validation DataLoader')
        if self.hparams.one_vol:
            split = math.floor(0.8* len(os.listdir(self.hparams.trainds)))
            ffn = lambda p: int(p.name[p.name.rfind('_')+1:-3]) > split
        else:
            names = list(set(map(lambda n: n[:n.rfind('_')], os.listdir(Path(self.hparams.trainds)))))
            split_idx = math.floor(0.8 * len(names))
            valid_names = names[split_idx:]
            ffn = lambda p: p.name[:p.name.rfind('_')] in valid_names
            print('valid names: ', valid_names)
        ds = TorchDataset(self.hparams.trainds,
            filter_fn=ffn,
            preprocess_fn=self.transform(train=False)
        )
        if self.hparams.preload_valid:
            ds = ds.preload()
        return torch.utils.data.DataLoader(ds,
            batch_size=self.hparams.batch_size,
            collate_fn=dict_collate_fn,
            num_workers=0 if self.hparams.preload else 6
        )

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr_projection', default=1e-3, type=float, help='Learning Rate for the projection (MLP if exists)')
        parser.add_argument('--lr_im_backbone', default=1e-3, type=float, help='Learning Rate for the pretrained ResNet backbone')
        parser.add_argument('--lr_vol_backbone', default=1e-3, type=float, help='Learning Rate for the volume backbone')
        parser.add_argument('--net', default='ConvProject', type=str, help='Model for the Neural TF. ConvProject, ConvProjectPlus or UnetAdain')
        parser.add_argument('--first_conv_ks', default=1, type=int, help='Kernel Size of the first Conv layer in the NeuralNet representing the Transfer Function')
        parser.add_argument('--backbone', type=str, default='resnet34', help='What backbone to use. Either resnet18, 34 or 50')
        parser.add_argument('--pretrain', action='store_true', dest='pretrained', help='Enable to start from random init in the ResNet')
        parser.add_argument('--weight_decay',  default=1e-6, type=float, help='Weight decay for training.')
        parser.add_argument('--batch_size',    default=4,     type=int,   help='Batch Size')
        parser.add_argument('--opt', type=str, default='Ranger', help='Optimizer to use. One of Ranger, Adam')
        parser.add_argument('--loss', type=str, default='mse', help='Loss Function to use')
        parser.add_argument('--opacity-weight', type=float, default=1.0, help='Weights the loss higher for opacity (>1e-2)')
        parser.add_argument('--last-act', type=str, default='none', help='Last activation function. Otions: nrelu, relu, sigmoid, none')
        parser.add_argument('--norm', type=str, default='instance', help='Type of normalization to use in volume backbone. instance, batch or none')
        parser.add_argument('--preload', action='store_true', help='If set, preloads training data into RAM.')
        parser.add_argument('--preload-valid', action='store_true', help='If set, preloads the validation dataset into RAM')
        parser.add_argument('--one_vol', action='store_true', help='Modify dataset splitting to work on single volume datasets')
        parser.add_argument('--num_images_logged', type=int, default=10, help='Number of slices / TFs logged during validation epoch')
        parser.add_argument('--max-train-samples', type=int, default=None, help='Restrict training dataset to given amount of samples')
        parser.add_argument('--train-any-vol', action='store_true', help='Train on predicting the TF seen for a random volume, instead of the volume seen in the rendering')
        return parser
