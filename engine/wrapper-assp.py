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
        # 注意: args.lr 这里作为"基础LR" (Base LR)
        self.base_lr = args.lr 
        self.history = {}
        
        self.loss_fn = HybridSegmentationLoss(lambda_dice=1.0, lambda_focal=1.0, lambda_boundary=0.5, lambda_lovasz=0.5)
        
        self.aux_weight = 0.4
        self.contrastive_weight = 0.1 # 对比损失权重，通常设小一点
        
        self.ema = ModelEMA(self.model, decay=0.999)

        metrics_dict = {"acc": Accuracy(task='binary'), "dice": Dice(), "MIoU": BinaryJaccardIndex()}
        self.train_metrics = nn.ModuleDict(metrics_dict)
        self.val_metrics = deepcopy(self.train_metrics)
        self.test_metrics = deepcopy(self.train_metrics)
        
        self.save_hyperparameters()

    def configure_optimizers(self):
        # --- 策略一: 差分学习率 (Differential Learning Rates) ---
        # 1. 提取 Backbone 参数 (Encoder)
        backbone_params = []
        head_params = []
        
        backbone_ids = list(map(id, self.model.encoder.parameters())) + \
                       list(map(id, self.model.text_encoder.parameters()))
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in backbone_ids:
                backbone_params.append(param)
            else:
                head_params.append(param)

        # 2. 设置分组参数
        # Backbone 使用 0.1 * base_lr (例如 1e-5)
        # Head (Decoder/Fusion) 使用 base_lr (例如 1e-4)
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': self.base_lr * 0.1},
            {'params': head_params, 'lr': self.base_lr}
        ], weight_decay=1e-2)

        # --- 策略二: Warmup + Cosine Annealing ---
        # 总 Epochs
        max_epochs = self.trainer.max_epochs
        warmup_epochs = max(5, int(0.05 * max_epochs)) # 至少5个epoch预热

        # 定义 Lambda 函数
        def lr_lambda(current_epoch):
            if current_epoch < warmup_epochs:
                # Linear Warmup: 0 -> 1
                return float(current_epoch + 1) / float(warmup_epochs)
            else:
                # Cosine Annealing: 1 -> 0
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
            # 验证阶段不计算 ITC loss，也不使用 TTA (如果用户删了TTA)
            # 这里为了简洁，使用标准 forward
            outputs = self(x)
            pred_prob = torch.sigmoid(outputs['pred_logits'])
            
            # 仅计算主 Loss 方便 Log
            loss_total, loss_dice, _, _, _ = self.loss_fn(pred_prob, gt_mask)
            
            if batch_idx == 0:
                self.visualize_batch(x[0], gt_mask, pred_prob, batch_idx, stage)
                
            return {
                'loss': loss_total,
                'preds': pred_prob.detach(),
                'y': gt_mask.detach()
            }
            
        else:
            # 训练阶段
            outputs = self(x)
            pred_logits = outputs['pred_logits']
            aux_logits = outputs['aux_logits']
            img_proj = outputs['img_proj']
            text_proj = outputs['text_proj']
            logit_scale = outputs['logit_scale']
            
            pred_prob = torch.sigmoid(pred_logits)
            
            # 1. Segmentation Loss
            loss_main, _, _, _, _ = self.loss_fn(pred_prob, gt_mask)
            
            aux_logits_up = F.interpolate(aux_logits, size=gt_mask.shape[-2:], mode='bilinear', align_corners=False)
            aux_prob = torch.sigmoid(aux_logits_up)
            loss_aux, _, _, _, _ = self.loss_fn(aux_prob, gt_mask)
            
            # --- 策略三: Image-Text Contrastive (ITC) Loss ---
            # InfoNCE
            logit_scale = logit_scale.exp()
            # [B, C] @ [C, B] -> [B, B]
            logits_per_image = logit_scale * img_proj @ text_proj.t()
            logits_per_text = logits_per_image.t()
            
            # Labels: 对角线为 1 (0, 1, 2... B-1)
            batch_size = img_proj.shape[0]
            labels = torch.arange(batch_size, device=self.device)
            
            loss_i2t = F.cross_entropy(logits_per_image, labels)
            loss_t2i = F.cross_entropy(logits_per_text, labels)
            loss_itc = (loss_i2t + loss_t2i) / 2
            
            # Total Loss
            total_loss = loss_main + self.aux_weight * loss_aux + self.contrastive_weight * loss_itc
            
            return {
                'loss': total_loss,
                'loss_main': loss_main.detach(),
                'loss_aux': loss_aux.detach(),
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
             self.log('L_itc', l_itc, prog_bar=True) # 打印 Contrastive Loss

        return loss
    
    # ... (Epoch End 保持不变) ...
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