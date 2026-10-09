import json
import os
import torch
import pandas as pd
import ast
import numpy as np
from monai.transforms import (
    AddChanneld, Compose, Lambdad, NormalizeIntensityd, 
    RandCoarseShuffled, RandRotated, RandZoomd, Resized, 
    ToTensord, LoadImaged, EnsureChannelFirstd,
    RandFlipd, RandGaussianNoised, RandAdjustContrastd, 
    RandShiftIntensityd, RandAffined, Rand2DElasticd
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

class QaTa(Dataset):

    def __init__(self, csv_path=None, root_path=None, tokenizer=None, mode='train', image_size=[224,224]):

        super(QaTa, self).__init__()

        self.mode = mode
        
        with open(csv_path, 'r') as f:
            self.data = pd.read_csv(f)
            
        self.image_list = list(self.data['Image'])
        self.caption_list = list(self.data['Description'])
        
        if 'BBox' in self.data.columns:
            self.bbox_list = list(self.data['BBox'])
        else:
            self.bbox_list = None
            if mode == 'train': print("Warning: 'BBox' column not found. Using dummy boxes.")

        split_idx = int(0.8 * len(self.image_list))
        if mode == 'train':
            self.image_list = self.image_list[:split_idx]
            self.caption_list = self.caption_list[:split_idx]
            if self.bbox_list: self.bbox_list = self.bbox_list[:split_idx]
        elif mode == 'valid':
            self.image_list = self.image_list[split_idx:]
            self.caption_list = self.caption_list[split_idx:]
            if self.bbox_list: self.bbox_list = self.bbox_list[split_idx:]
        
        self.root_path = root_path
        self.image_size = image_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    def __len__(self):
        return len(self.image_list)

    def get_bbox_from_mask(self, mask_tensor):
        if isinstance(mask_tensor, torch.Tensor):
            pos = (mask_tensor[0] > 0.5).nonzero()
            H, W = mask_tensor.shape[1], mask_tensor.shape[2]
        else:
            return torch.tensor([0.0,0.0,0.0,0.0])

        if len(pos) == 0:
            return torch.tensor([0.0,0.0,0.0,0.0])
            
        y_min = pos[:, 0].min().item()
        y_max = pos[:, 0].max().item()
        x_min = pos[:, 1].min().item()
        x_max = pos[:, 1].max().item()
        
        cx = (x_min + x_max) / 2.0 / W
        cy = (y_min + y_max) / 2.0 / H
        w = (x_max - x_min) / W
        h = (y_max - y_min) / H
        return torch.tensor([cx, cy, w, h], dtype=torch.float32)

    def __getitem__(self, idx):
        trans = self.transform(self.image_size)

        image = os.path.join(self.root_path, 'img', self.image_list[idx].replace('mask_', ''))
        gt = os.path.join(self.root_path, 'labelcol', self.image_list[idx])
        caption = self.caption_list[idx]

        token_output = self.tokenizer.encode_plus(caption, padding='max_length',
                                                        max_length=32, 
                                                        truncation=True,
                                                        return_attention_mask=True,
                                                        return_tensors='pt')
        token, mask = token_output['input_ids'], token_output['attention_mask']

        data = {'image': image, 'gt': gt, 'token': token, 'mask': mask}
        data = trans(data)

        image, gt, token, mask = data['image'], data['gt'], data['token'], data['mask']
        # gt = torch.where(gt == 255, 1, 0)
        if gt.shape[0] > 1:
            gt = gt[0:1, ...] 
        
        # 2. 智能处理 GT 的数值范围 (解决 Test 效果差的问题)
        # 如果最大值大于 1 (说明是 0-255 格式)，则按照 127 阈值进行二值化
        if gt.max() > 1:
            gt = torch.where(gt >= 127, 1, 0)
        else:
            # 如果最大值已经是 1 (说明是 0-1 格式)，则保持原样或确保它是二值的
            gt = torch.where(gt > 0.5, 1, 0)
        
        text = {'input_ids': token.squeeze(dim=0), 'attention_mask': mask.squeeze(dim=0)} 

        bbox = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        if self.mode == 'train':
            bbox = self.get_bbox_from_mask(gt)
        elif self.bbox_list:
            try:
                bbox = torch.tensor(ast.literal_eval(self.bbox_list[idx]), dtype=torch.float32)
            except:
                pass

        return ([image, text], {'mask': gt, 'bbox': bbox})

    def transform(self, image_size=[224,224]):
        if self.mode == 'train':
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                
                # --- 强力增强组合 ---
                # 1. 弹性形变 (关键新增)
                Rand2DElasticd(
                    keys=["image", "gt"], 
                    prob=0.5, 
                    spacing=(20, 20), 
                    magnitude_range=(1, 2), 
                    mode=["bilinear", "nearest"], 
                    padding_mode="zeros"
                ),
                
                # 2. 旋转
                RandRotated(keys=['image', 'gt'], range_x=0.26, prob=0.5, mode=['bilinear', 'nearest']),
                
                # 3. 仿射
                RandAffined(keys=['image', 'gt'], prob=0.5, rotate_range=0.1, translate_range=10, scale_range=0.1, mode=['bilinear', 'nearest']),
                
                # 4. 缩放
                RandZoomd(['image','gt'], min_zoom=0.9, max_zoom=1.2, mode=["bicubic","nearest"], prob=0.5),
                
                # 5. 像素级增强
                RandGaussianNoised(keys=['image'], prob=0.2, mean=0.0, std=0.05),
                RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.8, 1.2)),
                RandShiftIntensityd(keys=['image'], offsets=0.1, prob=0.2),
                
                Resized(["image"], spatial_size=image_size, mode='bicubic'),
                Resized(["gt"], spatial_size=image_size, mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt","token","mask"]),
            ])
        else:
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                Resized(["image"], spatial_size=image_size, mode='bicubic'),
                Resized(["gt"], spatial_size=image_size, mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt","token","mask"]),
            ])
        return trans