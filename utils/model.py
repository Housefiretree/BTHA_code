import torch
import torch.nn as nn
from einops import rearrange, repeat
from .layers import GuideDecoder
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import math
import numpy as np

# --------------------------------------------------------
# 1. 基础模块 (位置编码 & SE Block)
# --------------------------------------------------------
class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is None: scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        x_embed = x.float()
        y_embed = x.float()
        B, C, H, W = x.shape
        not_mask = torch.ones((B, H, W), device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    轻量级的通道注意力，用于筛选融合后的特征
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# --------------------------------------------------------
# 2. 门控交叉注意力模块 (单向 + SE增强)
# --------------------------------------------------------
class GatedCrossAttentionBlock(nn.Module):
    def __init__(self, img_dim, text_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.img_dim = img_dim
        self.text_proj = nn.Linear(text_dim, img_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=img_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(img_dim, img_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(img_dim * 2, img_dim)
        )
        self.norm1 = nn.LayerNorm(img_dim)
        self.norm2 = nn.LayerNorm(img_dim)
        self.norm_text = nn.LayerNorm(img_dim)
        self.gate_attn = nn.Parameter(torch.tensor(0.0))
        self.gate_ffn = nn.Parameter(torch.tensor(0.0))
        self.pos_embed = PositionEmbeddingSine(num_pos_feats=img_dim // 2, normalize=True)
        
        # 新增: SE Block
        self.se = SEBlock(img_dim)

    def forward(self, img_feat, text_feat, text_mask=None):
        B, C, H, W = img_feat.shape
        pos = self.pos_embed(img_feat) 
        img_flat = rearrange(img_feat, 'b c h w -> b (h w) c')
        pos_flat = rearrange(pos, 'b c h w -> b (h w) c')
        
        query = self.norm1(img_flat) + pos_flat
        text_emb = self.text_proj(text_feat) 
        text_emb = self.norm_text(text_emb)
        
        padding_mask = (text_mask == 0) if text_mask is not None else None
        attn_out, _ = self.cross_attn(query=query, key=text_emb, value=text_emb, key_padding_mask=padding_mask)
        
        img_flat = img_flat + torch.tanh(self.gate_attn) * attn_out
        ffn_out = self.ffn(self.norm2(img_flat))
        img_flat = img_flat + torch.tanh(self.gate_ffn) * ffn_out
        
        # Reshape back
        out = rearrange(img_flat, 'b (h w) c -> b c h w', h=H, w=W)
        
        # Apply SE Block
        out = self.se(out)
        
        return out

# --------------------------------------------------------
# 3. Encoders (BERTModel & VisionModel)
# --------------------------------------------------------
class BERTModel(nn.Module):
    def __init__(self, bert_type, project_dim):
        super(BERTModel, self).__init__()
        self.model = AutoModel.from_pretrained(bert_type, output_hidden_states=True, trust_remote_code=True)
        for param in self.model.parameters():
            param.requires_grad = True 

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        if hasattr(output, 'last_hidden_state'):
            last_hidden_state = output.last_hidden_state
        else:
            last_hidden_state = output['last_hidden_state']

        if hasattr(output, 'pooler_output') and output.pooler_output is not None:
            pooler_output = output.pooler_output
        elif isinstance(output, dict) and 'pooler_output' in output and output['pooler_output'] is not None:
            pooler_output = output['pooler_output']
        else:
            pooler_output = last_hidden_state[:, 0, :]
            
        return last_hidden_state, pooler_output

class VisionModel(nn.Module):
    def __init__(self, vision_type, project_dim):
        super(VisionModel, self).__init__()
        self.model = AutoModel.from_pretrained(vision_type, output_hidden_states=True)   
    def forward(self, x):
        output = self.model(x, output_hidden_states=True)
        return {"feature": output['hidden_states'], "pooler_output": output['pooler_output']}

# --------------------------------------------------------
# 4. 主模型 (多尺度深层监督版)
# --------------------------------------------------------
class LanGuideMedSeg(nn.Module):
    def __init__(self, bert_type, vision_type, project_dim=512):
        super(LanGuideMedSeg, self).__init__()

        self.encoder = VisionModel(vision_type, project_dim)
        self.text_encoder = BERTModel(bert_type, project_dim)

        self.spatial_dim = [7, 14, 28, 56]
        feature_dim = [768, 384, 192, 96] 

        # Fusion Modules (单向 + SE)
        text_dim = 768
        self.fusion_os32 = GatedCrossAttentionBlock(img_dim=768, text_dim=text_dim)
        self.fusion_os16 = GatedCrossAttentionBlock(img_dim=384, text_dim=text_dim)
        self.fusion_os8 = GatedCrossAttentionBlock(img_dim=192, text_dim=text_dim)
        
        # --- Multi-Scale Deep Supervision Heads ---
        # 为每一层融合后的特征都加一个分类头
        self.aux_head_32 = nn.Conv2d(768, 1, kernel_size=1)
        self.aux_head_16 = nn.Conv2d(384, 1, kernel_size=1)
        self.aux_head_8 = nn.Conv2d(192, 1, kernel_size=1)
        # ----------------------------------------

        # ITC Projection
        self.embed_dim = 256
        self.img_proj = nn.Linear(768, self.embed_dim)
        self.text_proj = nn.Linear(768, self.embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        input_len = 32
        output_len = 32
        self.decoder16 = GuideDecoder(feature_dim[0], feature_dim[1], self.spatial_dim[0], output_len, input_text_len=input_len)
        self.decoder8 = GuideDecoder(feature_dim[1], feature_dim[2], self.spatial_dim[1], output_len, input_text_len=input_len)
        self.decoder4 = GuideDecoder(feature_dim[2], feature_dim[3], self.spatial_dim[2], output_len, input_text_len=input_len)
        self.decoder1 = SubpixelUpsample(2, feature_dim[3], 24, 4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

    def forward(self, data):
        image, text = data
        if image.shape[1] == 1:   
            image = repeat(image, 'b 1 h w -> b c h w', c=3)

        text_seq, text_global = self.text_encoder(text['input_ids'], text['attention_mask']) 
        text_mask = text['attention_mask']

        img_out = self.encoder(image)
        features = img_out['feature']
        img_global = img_out['pooler_output'] 
        
        if len(features[0].shape) == 4: features = features[1:] 
        f_os4, f_os8, f_os16, f_os32 = features[0], features[1], features[2], features[3]

        # 3. Fusion (单向)
        f_os32_fused = self.fusion_os32(f_os32, text_seq, text_mask)
        f_os16_fused = self.fusion_os16(f_os16, text_seq, text_mask)
        f_os8_fused = self.fusion_os8(f_os8, text_seq, text_mask)
        
        # 4. Multi-Scale Aux Logits
        aux_logits_32 = self.aux_head_32(f_os32_fused)
        aux_logits_16 = self.aux_head_16(f_os16_fused)
        aux_logits_8 = self.aux_head_8(f_os8_fused)

        # 5. ITC Projections
        img_feat_proj = F.normalize(self.img_proj(img_global), dim=-1)
        text_feat_proj = F.normalize(self.text_proj(text_global), dim=-1)

        # 6. Decoder
        os32_flat = rearrange(f_os32_fused, 'b c h w -> b (h w) c')
        f_os16_flat = rearrange(f_os16_fused, 'b c h w -> b (h w) c') 
        f_os8_flat = rearrange(f_os8_fused, 'b c h w -> b (h w) c')   
        f_os4_flat = rearrange(f_os4, 'b c h w -> b (h w) c')         
        
        os16 = self.decoder16(os32_flat, f_os16_flat, text_seq)
        os8 = self.decoder8(os16, f_os8_flat, text_seq)
        os4 = self.decoder4(os8, f_os4_flat, text_seq)
        os4 = rearrange(os4, 'B (H W) C -> B C H W', H=self.spatial_dim[-1], W=self.spatial_dim[-1])
        os1 = self.decoder1(os4)
        out_logits = self.out(os1)

        return {
            'pred_logits': out_logits,  
            'aux_logits_32': aux_logits_32,
            'aux_logits_16': aux_logits_16,
            'aux_logits_8': aux_logits_8,
            'img_proj': img_feat_proj,
            'text_proj': text_feat_proj,
            'logit_scale': self.logit_scale
        }