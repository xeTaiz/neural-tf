from argparse  import ArgumentParser
from functools import partial
from pathlib   import Path
import random, sys, uuid, glm
import matplotlib.pyplot as plt
import numpy as np

from pytorch_lightning import Trainer, seed_everything, loggers
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pl_module import NeuralTransferFunction

from torchvtk.datasets.torch_dataset import TorchDataset
from torchvtk.datasets.queue         import TorchQueueDataset, dict_collate_fn

import torch
import torch.nn.functional as F
from torchvtk.utils import make_5d, tex_from_pts, random_tf_from_vol, TFGenerator
from torchvtk.converters.dicom.utils import hidden_errors
from torchvtk.transforms import Composite, Lambda
from torchvision.transforms import ToTensor

class QueueUsageLogging(Callback):
    def __init__(self, queue):
        super().__init__()
        self.q = queue

    def on_validation_end(self, trainer, module):
        with hidden_errors():
            plt.bar(self.q.sample_dict.keys(), self.q.sample_dict.values())
            module.logger.experiment.log({'queue/sampling_frequency': plt})

if __name__=='__main__':
    # Paramter Parsing
    parser = ArgumentParser('Trains Deep TF Prediction')
    parser.add_argument('trainds', type=str, help='Path to the main dataset')
    parser.add_argument('cq500', type=str, help='Path to CQ500 dataset (TorchDataset)')
    parser.add_argument('--seed',  default=None,     type=int, help='Random Seed')
    parser.add_argument('--precision', default=16,   type=int, help='16 to use Mixed precision (AMP O2), 32 for standard 32 bit float training')
    parser.add_argument('--dev', action='store_true', help='Activate Lightning Fast Dev Run for debugging')
    parser.add_argument('--find-lr', action='store_true', help='Use learning rate finder.')
    parser.add_argument('--overfit', action='store_true', help='If this flag is set the network is overfit to 1 batch')
    parser.add_argument('--accumulate-grads', default=1, type=int, help='Gradient Accumulation to increase batch size. (Multiplicative)')
    parser.add_argument('--min_epochs', default=50, type=int, help='Minimum number of epochs.')
    parser.add_argument('--max_epochs', default=150, type=int, help='Maximum number ob epochs to train')
    parser.add_argument('--online', action='store_true', help='Send logs to WandB directly, instead of at the end of the run')
    parser.add_argument('--n', type=str, default='', help='Run Name, additional suffix to WandB experiment name')
    parser = NeuralTransferFunction.add_model_specific_args(parser)
    args = parser.parse_args()
    # Setup Data

    if args.seed is None: # Generate random seed if none is given
        args.seed = random.randrange(4294967295) # Make sure it's logged
    seed_everything(args.seed)

    # Setup Model, Logger, Trainer
    model = NeuralTransferFunction(hparams=args) # Must be AFTER prepare_data(), because it messes with multiprocessing
    print(model)
    run_id = str(uuid.uuid4())[:6]
    logger = loggers.WandbLogger(
            project='deep-transfer-function',
            name=f'NEURALTF_{run_id}_{args.n}',
            id=run_id,
            log_model=True,
            offline=not args.online)
    ckpt_path = logger.experiment.dir + '/checkpoints'
    ckpt_cb = ModelCheckpoint(dirpath=ckpt_path, filename='{epoch}-{val_loss:.4f}-{val_mae:.2f}',save_top_k=2, verbose=True, monitor='val_loss', mode='min', save_last=True)
    callbacks = [EarlyStopping(monitor='val_loss', mode='min')]
    # if not args.overfit: callbacks.append(QueueUsageLogging(train_dl.dataset))

    trainer = Trainer.from_argparse_args(args,
        logger=logger,
        track_grad_norm=2,
        log_gpu_memory=True,
        fast_dev_run=args.dev,
        profiler=True,
        gpus=1,
        accumulate_grad_batches=args.accumulate_grads,
        overfit_batches=1 if args.overfit else 0,
        precision=args.precision,
        amp_level='O2',
        auto_lr_find=args.find_lr,
        callbacks=callbacks,
        checkpoint_callback=ckpt_cb,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
    )
    trainer.logger.log_hyperparams({
        'random_seed': args.seed,
        'gpu_name': torch.cuda.get_device_name(0),
        'gpu_capability': torch.cuda.get_device_capability(0)
        }) # Log random seed
    # Save source code !
    trainer.logger.experiment.save('pl_train.py')
    trainer.logger.experiment.save('pl_module.py')
    trainer.logger.experiment.save('neuraltf_modules.py')
    trainer.logger.experiment.save('adaptive_wing_loss.py')
    trainer.logger.experiment.save('requirements.txt')

    # Fit model
    trainer.fit(model)
    print(f'Best model with loss of {ckpt_cb.best_model_score} saved to {ckpt_cb.best_model_path}')
