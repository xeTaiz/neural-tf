from argparse  import ArgumentParser
from functools import partial
from itertools import count
import time, math
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
from neuraltf_modules import NeuralTF
from adaptive_wing_loss import AdaptiveWingLoss, NormalizedReLU

from torchvtk.datasets  import TorchDataset, TorchQueueDataset, dict_collate_fn
from torchvtk.utils     import make_5d, apply_tf_torch, tex_from_pts, TransferFunctionApplication, random_tf_from_vol, create_peaky_tf
from torchvtk.utils.tf_generate import make_trapezoid, colorize_trapeze, tf_pts_border, flatten_clip_sort_peaks, TFGenerator
from torchvtk.transforms import Composite, Lambda
from torchvision.transforms.functional import normalize, hflip

from torchvtk.rendering import show, show_tf, plot_tfs, plot_render_2tf, plot_render_tf

class Noop(nn.Module):
    def __init__(self, **kwargs): super().__init__()
    def forward(self, x, **kwargs): return x

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

class NeuralTransferFunction(LightningModule):
    def __init__(self, hparams=None, load_vols=True):
        super().__init__()
        self.hparams = hparams
        # Define model that predicts TF from a rendering
        if self.hparams.backbone == 'resnet18':
            im_backbone = resnet18(pretrained=hparams.pretrained)
        elif self.hparams.backbone == 'resnet34':
            im_backbone = resnet34(pretrained=hparams.pretrained)
        elif self.hparams.backbone == 'resnet50':
            im_backbone = resnet50(pretrained=hparams.pretrained)
        else:
            raise Exception(f'Invalid parameter backbone: {hparams.backbone}. Use either resnet18, resnet34, resnet50')
        if hparams.last_act == 'nrelu':
            act = NormalizedReLU()
        elif hparams.last_act == 'sigmoid':
            act = torch.sigmoid
        elif hparams.last_act == 'none':
            act = Noop()
        else:
            raise Exception(f'Invalid last activation given ({hparams.last_act}).')
        self.network = NeuralTF(first_conv_ks=hparams.first_conv_ks, act=NormalizedReLU())
        if hparams.loss == 'awl':
            self.loss = AdaptiveWingLoss()
        elif hparams.loss == 'mse':
            self.loss = F.mse_loss
        elif hparams.loss == 'mae':
            self.loss = F.l1_loss
        else:
            raise Exception(f'Invalid loss given ({hparams.loss}). Valid choices are mse, mae and awl')
        if load_vols:
            print(f'Loading volumes to memory (from  {hparams.cq500}).')
            self.volumes = {it['name']: it for it in TorchDataset(hparams.cq500).preload()}
        else:
            self.volumes = {}

    def forward(self, render, volume):
        return self.network(render, volume)

    def training_step(self, batch, batch_idx):
        dtype = torch.float16 if self.hparams.precision == 16 else torch.float32
        render_gt = batch['render'].to(dtype)
        tf_pts    = batch['tf_pts']
        if torch.is_tensor(tf_pts): tf_pts = [t for t in tf_pts]
        vols = torch.stack([self.volumes[n[:n.rfind('_')]]['vol'] for n in batch['name']]).to(dtype).to(render_gt.device)

        rgbo_pred = self.forward(render_gt[:, :3], vols)
        rgbo_targ = apply_tf_torch(vols, tf_pts)
        loss = self.loss(rgbo_pred, rgbo_targ)
        mae = F.l1_loss(rgbo_pred.detach(), rgbo_targ)

        return {
            'loss': loss,
            'log': {
                'metrics/train_loss': loss.detach().cpu().float(),
                'metrics/train_mae':   mae.detach().cpu().float()
            }
        }

    def validation_step(self, batch, batch_idx):
        dtype = torch.float16 if self.hparams.precision == 16 else torch.float32
        render_gt = batch['render'].to(dtype)
        tf_pts    = batch['tf_pts']
        if torch.is_tensor(tf_pts): tf_pts = [t for t in tf_pts]
        vols = torch.stack([self.volumes[n[:n.rfind('_')]]['vol'] for n in batch['name']]).to(dtype).to(render_gt.device)

        rgbo_pred = self.forward(render_gt[:, :3], vols).detach()
        rgbo_targ = apply_tf_torch(vols, tf_pts)
        loss = self.loss(rgbo_pred, rgbo_targ)
        mae = F.l1_loss(rgbo_pred, rgbo_targ)

        # Linspace Volumes to get 1D TF
        linsp_vols = torch.linspace(0, 1, 256, dtype=render_gt.dtype, device=render_gt.device).expand(render_gt.size(0), 1, 1,1,-1)


        tf_pred_tex = self.forward(render_gt[:, :3], linsp_vols)[:, :, 0, 0, :]
        z,h,w = rgbo_pred.shape[-3:]
        pred_slices = torch.stack([rgbo_pred[:, :, z//2, :,   : ],
                                   rgbo_pred[:, :,  :, h//2,  : ],
                                   rgbo_pred[:, :,  :,   :, w//2]], dim=1)
        targ_slices = torch.stack([rgbo_targ[:, :, z//2, :,   : ],
                                   rgbo_targ[:, :,  :, h//2,  : ],
                                   rgbo_targ[:, :,  :,   :, w//2]], dim=1)

        return {
            'loss': loss,
            'mae': mae,
            'render': render_gt[[0]].cpu(),
            'pred_slices': pred_slices[[0]].flip(-2).cpu(),
            'targ_slices': targ_slices[[0]].flip(-2).cpu(),
            'tf_tex': tf_pred_tex[[0]].cpu(),
            'tf_targ': list(map(lambda tf: tf.cpu(), tf_pts[:1]))
        }

    def validation_epoch_end(self, outputs):
        n = 10
        val_loss = torch.stack([o['loss'] for o in outputs]).mean()
        renders = torch.cat([o['render'] for o in outputs], dim=0)[:n]
        tf_texs = torch.cat([o['tf_tex'] for o in outputs], dim=0)[:n]
        tf_targ = [tf for o in outputs[:n] for tf in o['tf_targ']]
        pred_slices = torch.cat([o['pred_slices'] for o in outputs], dim=0)[:n]
        targ_slices = torch.cat([o['targ_slices'] for o in outputs], dim=0)[:n]

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

        self.log('metrics/val_loss', val_loss)
        self.log('metrics/val_mae', torch.stack([o['mae'] for o in outputs]).mean())

        return {
            'val_loss': val_loss
        }

    def configure_optimizers(self):
        params = [
            {'params': self.network.im_backbone.parameters(), 'lr': self.hparams.lr_im_backbone},
            {'params': self.network.vol_backbone.parameters(), 'lr': self.hparams.lr_vol_backbone},
            {'params': self.network.projection.parameters(), 'lr': self.hparams.lr_projection}
        ]

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

    def transform(self, train):
        def train_augment(image):
            image[:3] = normalize(image[:3], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            if (torch.rand(1) > 0.5).all():
                image = hflip(image)
            return image

        def test_augment(image):
            image[:3] = normalize(image[:3], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            return image

        if train: return Composite(
            Lambda(train_augment, apply_on='render', dtype=torch.float32)
        )
        else:     return Composite(
            Lambda(test_augment, apply_on='render', dtype=torch.float32)
        )

    def train_dataloader(self):
        print('Loading Training DataLoader')
        if self.hparams.one_vol:
            ffn = lambda p: int(p.name[p.name.rfind('_')+1:-3]) >= 5000
        else:
            ffn = lambda p: int(p.name[9:-8]) < 400
        ds = TorchDataset(self.hparams.trainds,
            filter_fn=ffn,
            # preprocess_fn=self.transform(train=True)
        )
        if self.hparams.preload:
            ds = ds.preload()
        return torch.utils.data.DataLoader(ds,
            batch_size=self.hparams.batch_size,
            collate_fn=dict_collate_fn,
            shuffle=True,
            num_workers=0 if self.hparams.preload else 6
        )

    def val_dataloader(self):
        print('Loading Validation DataLoader')
        if self.hparams.one_vol:
            ffn = lambda p: int(p.name[p.name.rfind('_')+1:-3]) < 5000
        else:
            ffn = lambda p: int(p.name[9:-8]) >= 400
        ds = TorchDataset(self.hparams.trainds,
            filter_fn=ffn,
            # preprocess_fn=self.transform(train=False)
        )
        if self.hparams.preload:
            ds = ds.preload()
        return torch.utils.data.DataLoader(ds,
            batch_size=self.hparams.batch_size,
            collate_fn=dict_collate_fn,
            num_workers=0 if self.hparams.preload else 6
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr_projection', default=1e-4, type=float, help='Learning Rate for the projection (MLP if exists)')
        parser.add_argument('--lr_im_backbone', default=1e-5, type=float, help='Learning Rate for the pretrained ResNet backbone')
        parser.add_argument('--lr_vol_backbone', default=1e-4, type=float, help='Learning Rate for the volume backbone')
        parser.add_argument('--first_conv_ks', default=1, type=int, help='Kernel Size of the first Conv layer in the NeuralNet representing the Transfer Function')
        parser.add_argument('--backbone', type=str, default='resnet34', help='What backbone to use. Either resnet18, 34 or 50')
        parser.add_argument('--no_pretrain', action='store_false', dest='pretrained', help='Enable to start from random init in the ResNet')
        parser.add_argument('--weight_decay',  default=1e-3, type=float, help='Weight decay for training.')
        parser.add_argument('--batch_size',    default=16,     type=int,   help='Batch Size')
        parser.add_argument('--opt', type=str, default='Ranger', help='Optimizer to use. One of Ranger, Adam')
        parser.add_argument('--loss', type=str, default='awl', help='Loss Function to use')
        parser.add_argument('--last-act', type=str, default='nrelu', help='Last activation function. Otions: nrelu, sigmoid, none')
        parser.add_argument('--preload', action='store_true', help='If set, preloads data into RAM.')
        parser.add_argument('--one_vol', action='store_true', help='Modify dataset splitting to work on single volume datasets')
        return parser
