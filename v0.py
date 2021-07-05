import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F

import warnings
import argparse
import albumentations as A
import pytorch_lightning as pl
warnings.filterwarnings('ignore')

from tqdm import tqdm
from pytorch_lightning.core.lightning import LightningModule

### Hyperparameters ###
parser = argparse.ArgumentParser(description='Training Pipeline')
parser.add_argument('--exp_name', type=str, required=True, help='Experiment index')
parser.add_argument('--model_name', type=str, default='unet_resnet34_cbam_v0a', help='Architecture')
parser.add_argument('--pretrained_path', type=str, default='data', help='Where have you put the pretrained weights?')

parser.add_argument('--lr', type=float, default=2e-4, help='for AdamW (default weight decay 1e-4)')
parser.add_argument('--epochs', type=int, default=64, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=16, help='As name')

parser.add_argument('--img_size', type=int, default=512, help='Resolution')
parser.add_argument('--loss_fn', type=str, default='bce', help='BCE (with logits) or lovasz_hinge')
parser.add_argument('--augs_spatial', type=int, default=0, help='Affine and cropping augmentations')
parser.add_argument('--augs_color', type=int, default=0, help='Noise, brightness, contrast, and gamma augmentations')

parser.add_argument('--n_cpus', type=int, default=0, help='Number of cores for dataloader (irrelevant)')
parser.add_argument('--gpu_no', type=int, default=0, help='Which to train on')
args = vars(parser.parse_args())

for k, v in args.items():
    print('    ', k, ':', v)
### Hyperparameters ###



### Data I/O ### 
class WMHDataset(torch.utils.data.Dataset):
    def __init__(self, df, transforms=A.Resize(args['img_size'], args['img_size'])):
        self.df = df
        self.transforms = transforms
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        img = plt.imread(self.df.img_path.values[i])
        seg = plt.imread(self.df.seg_path.values[i])
        out = self.transforms(image=img, mask=seg)
        img, seg = out['image'], torch.tensor(out['mask'])
        img = torch.tensor(img)/127. - 1
        # Turn it to 3 channels
        img = torch.stack([img]*3, 0)
        return {'img': img, 'seg': seg,\
               'patient_id': self.df.patient_id.values[i],\
               'slice_id': self.df.slice_id.values[i]}

class WMHDataModule(pl.LightningDataModule):
    def __init__(self, root='data', train_transforms=A.Resize(args['img_size'], args['img_size'])):
        super().__init__()
        self.root = root
        self.train_transforms = train_transforms
    
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_df = pd.read_csv(os.path.join(self.root, 'train.csv'))
        self.train_df = self.train_df[self.train_df.label == True]
        self.val_df = pd.read_csv(os.path.join(self.root, 'test.csv'))
        # self.val_df = self.val_df[self.val_df.label == True]
        self.train = WMHDataset(self.train_df, self.train_transforms)
        self.val = WMHDataset(self.val_df)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=args['batch_size'],\
                          shuffle=True, drop_last=True, num_workers=args['n_cpus'])

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=args['batch_size'],\
                          shuffle=False, num_workers=args['n_cpus'])

## Augmentations ##
transforms = [
        A.Flip(),
        A.RandomRotate90(),
        A.Resize(args['img_size'], args['img_size'])]

if args['augs_spatial'] == 1:
    print('Applying spatial augmentations')
    transforms += [
        A.OneOf([
            A.Affine(scale=(.85, 1.15), rotate=(-20, 20), shear=(-10, 10)),
            A.ShiftScaleRotate(shift_limit=.1, scale_limit=.15, rotate_limit=30),
            ]),
        A.OneOf([
                A.RandomCrop(int(args['img_size']*.85), int(args['img_size']*.85)),
                A.CenterCrop(int(args['img_size']*.85), int(args['img_size']*.85)),
            ], p=.75),
        A.Resize(args['img_size'], args['img_size'])]
    
if args['augs_color']:
    print('Applying pixel augmentations')
    transforms += [
        A.OneOf([
                A.RandomBrightnessContrast(),
                A.RandomGamma(),
                A.GaussNoise(var_limit=10, per_channel=False)
            ], p=.5)]

data = WMHDataModule(train_transforms=A.Compose(transforms))
### Data I/O ### 



### Criterion, metric, and miscellaneous progress bar override ###
from layers.loss_funcs.lovasz_losses import lovasz_hinge

# symmetric_lovasz = lambda y_hat, y:\
#         (lovasz_hinge(y_hat, y) + lovasz_hinge(-y_hat, 1 - y)) * .5
    
bce = lambda y_hat, y: F.binary_cross_entropy_with_logits(y_hat, y.type_as(y_hat))

# Dice score averaged across the first dimension
# Threshold of 0 for BCEwithlogits correspond to .5 threshold
def dice_stats(logits, labels, threshold=0):
    batch_size = logits.size(0)

    preds = (logits > threshold).view(batch_size, -1)
    labels = (labels > 0.5).view(batch_size, -1)
    # Return the statistics so they'll be egligible for aggregation across batches
    return {'tp': (preds*labels).sum(-1), 't': labels.sum(-1), 'p': preds.sum(-1)}

class LitProgressBar(pl.callbacks.ProgressBar):
    def init_validation_tqdm(self):
        return tqdm()
    
    def init_train_tqdm(self):
        return tqdm()
    
    def init_sanity_tqdm(self):
        return tqdm()
### Criterion, metric, and miscellaneous progress bar override ###



### Model and Pipeline ###
class SegModel(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.criterion = bce if args['loss_fn'] == 'bce' else lovasz_hinge
        self.max_metric = -1.
        self.logs = {'train_loss':[], 'train_metric':[], 'val_loss':[], 'val_metric':[], 'epoch':[]}
    
    def forward(self, x):
        return self.backbone(x).squeeze(1)
    
    def training_step(self, batch, batch_idx):
        preds = self(batch['img'])
        # Compute lovasz hinge loss
        loss = self.criterion(preds, batch['seg'])
        metric_stats = dice_stats(preds, batch['seg'])
        metric_value = (2 * metric_stats['tp'] / (metric_stats['t'] + metric_stats['p'] - 1e-8))\
                        .clamp(0, 1).mean().item()
        
        self.log('train_loss', loss.item(), sync_dist=True)
        self.log('train_metric', metric_value, sync_dist=True)
        return {'loss': loss, 'metric': metric_value}
    
    def validation_step(self, batch, batch_idx):
        preds = self(batch['img'])
        loss = self.criterion(preds, batch['seg'])
        metric_stats = dice_stats(preds, batch['seg'])
        return {'loss': loss, 'patient_id': batch['patient_id'],\
                'dice_tp': metric_stats['tp'], 'dice_t': metric_stats['t'], 'dice_p': metric_stats['p']}
    
    def training_epoch_end(self, batch_out):
        loss, metric = [], []
        for batch in batch_out:
            loss.append(batch['loss'].item())
            metric.append(batch['metric'])
        loss, metric = np.nanmean(loss), np.nanmean(metric)
        print(self.current_epoch, 'Loss:', loss, 'metric', metric)
        self.logs['train_loss'].append(loss)
        self.logs['train_metric'].append(metric)
        
    def validation_epoch_end(self, batch_out):
        loss, patient_ids, tp, t, p = [], [], [], [], []
        for batch in batch_out:
            loss.append(batch['loss'].item())
            patient_ids.append(batch['patient_id'])
            tp.append(batch['dice_tp'])
            t.append(batch['dice_t'])
            p.append(batch['dice_p'])
        f = torch.cat
        patient_ids, tp, t, p = f(patient_ids), f(tp), f(t), f(p)
        
        metric_value = []
        # Calculate dice across patients
        for patient_id in np.unique(patient_ids.cpu().numpy()):
            mask = patient_ids == patient_id
            # Skip patients who are totally negative
            if t[mask].sum() == 0:
                continue
            # Dice for this patient across slides
            dice = (2 * tp[mask].sum() / (t[mask].sum() + p[mask].sum() - 1e-8)).clamp(0, 1).item()
            metric_value.append(dice)
        
        loss, metric = np.nanmean(loss), np.nanmean(metric_value)
        if metric > self.max_metric:
            print('New best')
            self.max_metric = metric
            dev = 'cuda:'+str(args['gpu_no'])
            ckpt = torch.jit.trace(self.backbone.to(dev), torch.randn(1, 3, args['img_size'], args['img_size']).to(dev))
            torch.jit.save(ckpt, os.path.join('experiments', args['exp_name'],\
                                          args['exp_name']+'_'+str(self.current_epoch)+'.pt'))
            print('Epoch:', self.current_epoch, '(Test) Loss:', loss, 'metric', metric, '(New Best)')
        else:
            print('Epoch:', self.current_epoch, '(Test) Loss:', loss, 'metric', metric)
        # Only if there already exists train logs (there is a sanity check up front)
        if len(self.logs['train_loss']) != 0:
            self.logs['epoch'].append(self.current_epoch)
            self.logs['val_loss'].append(loss)
            self.logs['val_metric'].append(metric)
            df_logs = pd.DataFrame(self.logs)
            df_logs.to_csv(os.path.join('experiments', args['exp_name'], args['exp_name']+'.csv'))
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=args['lr'], weight_decay=1e-4)



### Training ###
from networks.imageunet import init_network
model = SegModel(init_network(args))

logger = pl.loggers.TensorBoardLogger(os.path.join('experiments', args['exp_name']))
trainer = pl.Trainer(gpus=[args['gpu_no']], checkpoint_callback=False, logger=logger, precision=16,\
                     num_sanity_val_steps=-1, amp_level='O2', benchmark=True, max_epochs=args['epochs'],\
                    callbacks=[LitProgressBar()]) 
trainer.fit(model, data)