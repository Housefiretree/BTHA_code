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
# 位置编码
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

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_r1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rates[0], dilation=rates[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_r2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rates[1], dilation=rates[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_r3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rates[2], dilation=rates[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, in_channels, 1, bias=False), # 投影回原维度
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        feat1 = self.conv1x1(x)
        feat2 = self.conv_r1(x)
        feat3 = self.conv_r2(x)
        feat4 = self.conv_r3(x)
        feat5 = F.interpolate(self.global_pool(x), size=(h, w), mode='bilinear', align_corners=False)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        out = self.project(out)
        return out
    
# --------------------------------------------------------
# Gated Cross Attention
class BiGatedCrossAttentionBlock(nn.Module):
    def __init__(self, img_dim, text_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.img_dim = img_dim
        self.text_dim = text_dim
        
        # Projections
        self.text_proj = nn.Linear(text_dim, img_dim)
        self.img_proj_for_text = nn.Linear(img_dim, text_dim) # Image map to Text dim

        # 1. Text-to-Image Attention (Text queries Image) -> Update Text
        self.t2i_attn = nn.MultiheadAttention(embed_dim=text_dim, kdim=img_dim, vdim=img_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.t2i_gate = nn.Parameter(torch.tensor(0.0))
        self.t2i_norm = nn.LayerNorm(text_dim)
        self.t2i_ffn = nn.Sequential(
            nn.Linear(text_dim, text_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(text_dim * 2, text_dim)
        )
        self.t2i_norm2 = nn.LayerNorm(text_dim)

        # 2. Image-to-Text Attention (Image queries Refined Text) -> Update Image
        self.i2t_attn = nn.MultiheadAttention(embed_dim=img_dim, kdim=text_dim, vdim=text_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.i2t_gate = nn.Parameter(torch.tensor(0.0))
        self.i2t_norm = nn.LayerNorm(img_dim)
        self.i2t_ffn = nn.Sequential(
            nn.Linear(img_dim, img_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(img_dim * 2, img_dim)
        )
        self.i2t_norm2 = nn.LayerNorm(img_dim)

        self.pos_embed = PositionEmbeddingSine(num_pos_feats=img_dim // 2, normalize=True)

    def forward(self, img_feat, text_feat, text_mask=None):
        """
        img_feat: [B, C_img, H, W]
        text_feat: [B, L, C_text]
        """
        B, C, H, W = img_feat.shape
        
        # Prepare Image Features (Flatten)
        pos = self.pos_embed(img_feat) 
        img_flat = rearrange(img_feat, 'b c h w -> b (h w) c')
        pos_flat = rearrange(pos, 'b c h w -> b (h w) c')
        
        # --- Stage 1: Text Refinement (Text queries Image) ---
        # Q = Text, K = Image, V = Image
        # 我们希望文本吸收图像的上下文
        img_for_text = img_flat # [B, HW, C_img]
        
        # Mask handling for Text
        # text_mask: 1 for valid, 0 for pad.
        padding_mask = (text_mask == 0) if text_mask is not None else None
        
        # Attn: Q(Text), K(Image), V(Image)
        # Note: torch MHA key_padding_mask applies to Key. Here Key is Image, which has no padding.
        text_attn_out, _ = self.t2i_attn(
            query=self.t2i_norm(text_feat), 
            key=img_for_text, 
            value=img_for_text
        )
        text_feat = text_feat + torch.tanh(self.t2i_gate) * text_attn_out
        text_feat = text_feat + self.t2i_ffn(self.t2i_norm2(text_feat))
        
        # --- Stage 2: Image Refinement (Image queries Refined Text) ---
        # Q = Image+Pos, K = Refined Text, V = Refined Text
        img_query = self.i2t_norm(img_flat) + pos_flat
        
        img_attn_out, _ = self.i2t_attn(
            query=img_query, 
            key=text_feat, 
            value=text_feat,
            key_padding_mask=padding_mask # Mask applies to Text (Key)
        )
        
        img_flat = img_flat + torch.tanh(self.i2t_gate) * img_attn_out
        img_flat = img_flat + self.i2t_ffn(self.i2t_norm2(img_flat))
        
        out_img = rearrange(img_flat, 'b (h w) c -> b c h w', h=H, w=W)
        
        # 返回更新后的图像特征 和 更新后的文本特征 (用于下一级联)
        return out_img, text_feat
    
# --------------------------------------------------------
# Encoders
# --------------------------------------------------------
class BERTModel(nn.Module):
    def __init__(self, bert_type, project_dim):
        super(BERTModel, self).__init__()
        self.model = AutoModel.from_pretrained(bert_type, output_hidden_states=True, trust_remote_code=True)
        # 允许微调
        for param in self.model.parameters():
            param.requires_grad = True 

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        
        # 1. 获取序列特征 [B, L, 768]
        # 兼容字典访问和属性访问
        if hasattr(output, 'last_hidden_state'):
            last_hidden_state = output.last_hidden_state
        else:
            last_hidden_state = output['last_hidden_state']

        # 2. 手动获取全局特征 (CLS Token) [B, 768]
        # CXR-BERT 没有标准的 pooler_output，我们直接取第一个 token (CLS)
        if hasattr(output, 'pooler_output') and output.pooler_output is not None:
            pooler_output = output.pooler_output
        elif isinstance(output, dict) and 'pooler_output' in output and output['pooler_output'] is not None:
            pooler_output = output['pooler_output']
        else:
            # Fallback: 手动提取 CLS token
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
# 主模型
# --------------------------------------------------------
class LanGuideMedSeg(nn.Module):
    def __init__(self, bert_type, vision_type, project_dim=512):
        super(LanGuideMedSeg, self).__init__()

        self.encoder = VisionModel(vision_type, project_dim)
        self.text_encoder = BERTModel(bert_type, project_dim)

        self.spatial_dim = [7, 14, 28, 56]
        feature_dim = [768, 384, 192, 96] 
        
        # ASPP 模块 (加在最深层 768)
        self.aspp = ASPP(in_channels=768, out_channels=256) # 中间降维处理，最后投影回 768
        
        # 双向融合模块 (Bi-Directional)
        text_dim = 768
        self.fusion_os32 = BiGatedCrossAttentionBlock(img_dim=768, text_dim=text_dim)
        self.fusion_os16 = BiGatedCrossAttentionBlock(img_dim=384, text_dim=text_dim)
        self.fusion_os8 = BiGatedCrossAttentionBlock(img_dim=192, text_dim=text_dim)
        
        self.aux_head = nn.Conv2d(768, 1, kernel_size=1)

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

        # 1. Encoders
        text_seq, text_global = self.text_encoder(text['input_ids'], text['attention_mask']) 
        text_mask = text['attention_mask']

        img_out = self.encoder(image)
        features = img_out['feature']
        img_global = img_out['pooler_output'] 
        
        if len(features[0].shape) == 4: features = features[1:] 
        f_os4, f_os8, f_os16, f_os32 = features[0], features[1], features[2], features[3]

        # 2. ASPP Context Capture (Only on os32)
        # ASPP 能够扩大感受野，更好地处理多尺度病灶
        f_os32 = self.aspp(f_os32) + f_os32 # Residual connection recommended

        # 3. Bi-Directional Cascade Fusion
        # 文本特征会在每一层被更新，传递到下一层
        
        # Level 32
        f_os32_fused, text_seq_32 = self.fusion_os32(f_os32, text_seq, text_mask)
        
        # Level 16 (Use updated text from level 32)
        f_os16_fused, text_seq_16 = self.fusion_os16(f_os16, text_seq_32, text_mask)
        
        # Level 8 (Use updated text from level 16)
        f_os8_fused, text_seq_8 = self.fusion_os8(f_os8, text_seq_16, text_mask)
        
        # 4. Aux Head
        aux_logits = self.aux_head(f_os32_fused)

        # 5. ITC Projections
        img_feat_proj = F.normalize(self.img_proj(img_global), dim=-1)
        text_feat_proj = F.normalize(self.text_proj(text_global), dim=-1)

        # 6. Decoder
        # 注意: Decoder 可以使用最后更新最充分的 text_seq_8，或者原始 text_seq
        # 使用 text_seq_8 包含了最丰富的多尺度视觉信息
        final_text_seq = text_seq_8 
        
        os32_flat = rearrange(f_os32_fused, 'b c h w -> b (h w) c')
        f_os16_flat = rearrange(f_os16_fused, 'b c h w -> b (h w) c') 
        f_os8_flat = rearrange(f_os8_fused, 'b c h w -> b (h w) c')   
        f_os4_flat = rearrange(f_os4, 'b c h w -> b (h w) c')         
        
        os16 = self.decoder16(os32_flat, f_os16_flat, final_text_seq)
        os8 = self.decoder8(os16, f_os8_flat, final_text_seq)
        os4 = self.decoder4(os8, f_os4_flat, final_text_seq)
        os4 = rearrange(os4, 'B (H W) C -> B C H W', H=self.spatial_dim[-1], W=self.spatial_dim[-1])
        os1 = self.decoder1(os4)
        out_logits = self.out(os1)

        return {
            'pred_logits': out_logits,  
            'aux_logits': aux_logits,
            'img_proj': img_feat_proj,
            'text_proj': text_feat_proj,
            'logit_scale': self.logit_scale
        }