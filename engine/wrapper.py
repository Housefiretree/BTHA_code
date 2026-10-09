from utils.model import LanGuideMedSeg
from utils.losses import HybridSegmentationLoss
from utils.ema import ModelEMA 
from torchmetrics import Accuracy, Dice
from torchmetrics.classification import BinaryJaccardIndex
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from copy import deepcopy
import pandas as pd
import sys
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
import math

class LanGuideMedSegWrapper(pl.LightningModule):

    def __init__(self, args):
        super(LanGuideMedSegWrapper, self).__init__()
        
        self.model = LanGuideMedSeg(args.bert_type, args.vision_type, args.project_dim)
        self.base_lr = args.lr 
        self.history = {}
        
        self.loss_fn = HybridSegmentationLoss(lambda_dice=1.0, lambda_focal=1.0, lambda_boundary=0.5, lambda_lovasz=0.5)
        
        # Multi-Scale Loss Weights
        self.aux_weights = {'32': 0.4, '16': 0.2, '8': 0.1}
        self.contrastive_weight = 0.1 
        
        self.ema = ModelEMA(self.model, decay=0.999)

        metrics_dict = {"acc": Accuracy(task='binary'), "dice": Dice(), "MIoU": BinaryJaccardIndex()}
        self.train_metrics = nn.ModuleDict(metrics_dict)
        self.val_metrics = deepcopy(self.train_metrics)
        self.test_metrics = deepcopy(self.train_metrics)
        
        self.save_hyperparameters()

    def configure_optimizers(self):
        # Differential Learning Rates
        backbone_params = []
        head_params = []
        backbone_ids = list(map(id, self.model.encoder.parameters())) + list(map(id, self.model.text_encoder.parameters()))
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad: continue
            if id(param) in backbone_ids:
                backbone_params.append(param)
            else:
                head_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': self.base_lr * 0.1},
            {'params': head_params, 'lr': self.base_lr}
        ], weight_decay=1e-2)

        max_epochs = self.trainer.max_epochs
        warmup_epochs = max(5, int(0.05 * max_epochs))

        def lr_lambda(current_epoch):
            if current_epoch < warmup_epochs:
                return float(current_epoch + 1) / float(warmup_epochs)
            else:
                progress = float(current_epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
    
    def on_train_start(self):
        if self.device.type == 'cuda':
            self.ema.module.to(self.device)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.ema.update(self.model)

    def on_validation_epoch_start(self):
        if self.ema.module.parameters().__next__().device != self.device:
             self.ema.module.to(self.device)
        self.original_state_dict = deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.ema.module.state_dict())
    
    def on_validation_epoch_end(self):
        self.model.load_state_dict(self.original_state_dict)
        del self.original_state_dict

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, batch_idx, stage='train'):
        x, target = batch  
        gt_mask = target['mask']
        
        if stage in ['val', 'test']:
            outputs = self(x)
            pred_prob = torch.sigmoid(outputs['pred_logits'])
            loss_total, _, _, _, _ = self.loss_fn(pred_prob, gt_mask)
            if batch_idx == 0: self.visualize_batch(x[0], gt_mask, pred_prob, batch_idx, stage)
            return {'loss': loss_total, 'preds': pred_prob.detach(), 'y': gt_mask.detach()}
            
        else:
            outputs = self(x)
            pred_prob = torch.sigmoid(outputs['pred_logits'])
            
            # 1. Main Loss
            loss_main, _, _, _, _ = self.loss_fn(pred_prob, gt_mask)
            
            # 2. Multi-Scale Aux Losses
            # Upsample all aux logits to GT size
            aux_prob_32 = torch.sigmoid(F.interpolate(outputs['aux_logits_32'], size=gt_mask.shape[-2:], mode='bilinear', align_corners=False))
            aux_prob_16 = torch.sigmoid(F.interpolate(outputs['aux_logits_16'], size=gt_mask.shape[-2:], mode='bilinear', align_corners=False))
            aux_prob_8  = torch.sigmoid(F.interpolate(outputs['aux_logits_8'],  size=gt_mask.shape[-2:], mode='bilinear', align_corners=False))
            
            # Calculate simple Dice/BCE for aux to save compute, or use full hybrid
            # Using full hybrid is better for consistency
            loss_aux32, _, _, _, _ = self.loss_fn(aux_prob_32, gt_mask)
            loss_aux16, _, _, _, _ = self.loss_fn(aux_prob_16, gt_mask)
            loss_aux8,  _, _, _, _ = self.loss_fn(aux_prob_8,  gt_mask)
            
            # 3. ITC Loss
            img_proj, text_proj, logit_scale = outputs['img_proj'], outputs['text_proj'], outputs['logit_scale']
            logit_scale = logit_scale.exp()
            logits_per_image = logit_scale * img_proj @ text_proj.t()
            logits_per_text = logits_per_image.t()
            labels = torch.arange(img_proj.shape[0], device=self.device)
            loss_itc = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
            
            # Total
            total_loss = (loss_main + 
                          self.aux_weights['32'] * loss_aux32 + 
                          self.aux_weights['16'] * loss_aux16 + 
                          self.aux_weights['8']  * loss_aux8 + 
                          self.contrastive_weight * loss_itc)
            
            return {
                'loss': total_loss,
                'loss_main': loss_main.detach(),
                'loss_itc': loss_itc.detach(),
                'preds': pred_prob.detach(),
                'y': gt_mask.detach()
            }    
    
    def visualize_batch(self, images, gt_masks, pred_probs, batch_idx, stage='val'):
        if batch_idx != 0: return
        current_epoch = self.current_epoch
        save_dir = os.path.join("vis_outputs", f"epoch_{current_epoch}")
        os.makedirs(save_dir, exist_ok=True)
        img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        gt_mask = gt_masks[0, 0].detach().cpu().numpy()
        pred_mask = pred_probs[0, 0].detach().cpu().numpy() > 0.5
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(img); ax[0].set_title("Image"); ax[0].axis('off')
        ax[1].imshow(img); ax[1].imshow(gt_mask, cmap='Greens', alpha=0.5, vmin=0, vmax=1); ax[1].set_title("GT"); ax[1].axis('off')
        ax[2].imshow(img); ax[2].imshow(pred_mask, cmap='Reds', alpha=0.5, vmin=0, vmax=1); ax[2].set_title("Pred"); ax[2].axis('off')
        plt.tight_layout(); plt.savefig(os.path.join(save_dir, f"{stage}_vis.png")); plt.close()

    def training_step(self, batch, batch_idx): return self.shared_step(batch, batch_idx, stage='train')
    def validation_step(self, batch, batch_idx): return self.shared_step(batch, batch_idx, stage='val')
    def test_step(self, batch, batch_idx): return self.shared_step(batch, batch_idx, stage='test')
    def predict_step(self, batch, batch_idx): return torch.sigmoid(self(batch[0])['pred_logits'])
        
    def shared_step_end(self, outputs, stage):
        if isinstance(outputs, list):
            loss = torch.stack([x['loss'] for x in outputs]).mean()
            preds = torch.cat([x['preds'] for x in outputs])
            y = torch.cat([x['y'] for x in outputs])
            l_main = torch.stack([x['loss_main'] for x in outputs]).mean() if 'loss_main' in outputs[0] else 0
            l_itc = torch.stack([x['loss_itc'] for x in outputs]).mean() if 'loss_itc' in outputs[0] else 0
        else:
            loss = outputs['loss']; preds = outputs['preds']; y = outputs['y']
            l_main = outputs.get('loss_main', 0)
            l_itc = outputs.get('loss_itc', 0)

        metrics = self.train_metrics if stage == "train" else (self.val_metrics if stage == "val" else self.test_metrics)
        for name in metrics:
            step_metric = metrics[name](preds, y.int()).item()
            if stage == "train": self.log(name, step_metric, prog_bar=True)
                
        if stage == 'train':
             self.log('L_main', l_main, prog_bar=True)
             self.log('L_itc', l_itc, prog_bar=True)

        return loss
    
    # ... (Epoch End Functions 保持不变) ...
    def training_step_end(self, outputs): return {'loss': self.shared_step_end(outputs, "train")}
    def validation_step_end(self, outputs): return {'val_loss': self.shared_step_end(outputs, "val")}
    def test_step_end(self, outputs): return {'test_loss': self.shared_step_end(outputs, "test")}
    def training_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs, stage="train")
        self.print(dic); self.log_dict(dic, logger=True)
    def validation_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs, stage="val")
        self.print_bar(); self.print(dic); self.log_dict(dic, logger=True)
    def test_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs, stage="test"); self.print(dic); self.log_dict(dic, logger=True)
    def shared_epoch_end(self, outputs, stage="train"):
        metrics = self.train_metrics if stage == "train" else (self.val_metrics if stage == "val" else self.test_metrics)
        epoch = self.trainer.current_epoch
        losses = []
        if isinstance(outputs, list):
            for t in outputs:
                if isinstance(t, dict):
                    if (stage + "_loss").replace('train_', '') in t: losses.append(t[(stage + "_loss").replace('train_', '')])
                    elif 'loss' in t: losses.append(t['loss'])
                    elif 'val_loss' in t: losses.append(t['val_loss'])
        stage_loss = torch.stack(losses).mean().item() if len(losses) > 0 else 0.0
        dic = {"epoch": epoch, stage + "_loss": stage_loss}
        for name in metrics:
            epoch_metric = metrics[name].compute().item(); metrics[name].reset(); dic[stage + "_" + name] = epoch_metric 
        if stage != 'test': self.history[epoch] = dict(self.history.get(epoch, {}), **dic)    
        return dic 
    def get_history(self): return pd.DataFrame(self.history.values()) 
    def print_bar(self): nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'); self.print("\n" + "=" * 80 + "%s" % nowtime)