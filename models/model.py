import yaml
import math
import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.necks.sequence_neck import SequenceNeck
from models.heads.classification_head import ClassificationHead
from models.backbones.vit import ViT
from models.backbones.cnn import CNN_model


def mosaic_frames(x: torch.Tensor) -> torch.Tensor:
    # x: [B, T, C, H, W]
    B, T, C, H, W = x.shape
    # 计算最近的上界整数平方尺寸 k
    k = math.ceil(math.sqrt(T))
    total_patches = k * k
    # 如有需要，补充零帧至 T 达到 k^2
    if T < total_patches:
        pad_frames = total_patches - T
        pad_shape = (B, pad_frames, C, H, W)
        pad_tensor = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        x = torch.cat([x, pad_tensor], dim=1)  # 在时间维度拼接零帧
    # 重塑并permute维度，将 k×k 个帧拼成大图
    x = x.view(B, k, k, C, H, W)                       # [B, k, k, C, H, W]
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()       # [B, C, k, H, k, W]
    x = x.view(B, C, k * H, k * W)                     # [B, C, H*k, W*k]
    return x


class VitModel(nn.Module):
    """
    ViTPose+LSTM 序列分类模型
    输入 x: Tensor [B, T, C, H, W]
    输出 logits: Tensor [B, output_dim]
    """
    def __init__(self, cfg: dict):
        super().__init__()
        backbone_cfg = cfg['model']['backbone']
        neck_cfg = cfg['model']['neck']
        head_cfg = cfg['model']['head']

        # 1) Backbone：ViTPose
        self.backbone = ViT(**backbone_cfg)
        # 2) Neck：序列特征融合（LSTM 或平均池化）
        self.neck = SequenceNeck(**neck_cfg)
        # 3) Head：分类层
        self.head = ClassificationHead(**head_cfg)

        # 添加一个全局平均池化层用于处理特征图
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        B, T, C, H, W = x.shape

        # 将 T 帧拼接成大图 [B, C, H*k, W*k]
        big_image = mosaic_frames(x) 
        # print("大图尺寸 for yaml:", big_image.shape)

        # 1) Backbone：ViTPose - 逐帧处理
        feat_map = self.backbone(big_image)  # [B, embed_dim, Hp, Wp]
        # 此时 Hp = k, Wp = k（每个patch等于原始帧大小）
        patch_tokens = feat_map.flatten(2).transpose(1, 2)  # [B, Hp*Wp, embed_dim]
        # 如需要保持与原有管线一致的每帧特征序列：
        if T < patch_tokens.shape[1]:
            patch_tokens = patch_tokens[:, :T, :]  # 去除末尾填充的空白patch
        # patch_tokens 形状现为 [B, T, embed_dim]，每帧对应一个特征向量
        # 后续可接入序列Neck（如LSTM或均值池化）或直接用于分类头
        
        # 2) Neck：序列特征融合 (LSTM或平均池化)
        seq_features = self.neck(patch_tokens)  # [B, neck_output_dim]
        
        # 3) Head：最终分类
        logits = self.head(seq_features)  # [B, output_dim]
        
        return logits

# python -m models.model
if __name__ == '__main__':
    config_path='/home/qiangubuntu/research/har_rgbe/configs/har_rgbe.yaml'
    # config_path='/home/qiangubuntu/research/har_rgbe/configs/har_rgbd.yaml'
    # config_path='/home/qiangubuntu/research/har_rgbe/configs/har_rgb.yaml'
    # config_path='/home/qiangubuntu/research/har_rgbe/configs/har_event.yaml'
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # peusdo data
    x = torch.randn(32, 9, 4, 192, 256).to('cuda')  # [B, T, C, H, W] 

    # Vit model 测试
    # model = VitModel(cfg).to('cuda')

    # CNN model 测试
    # input_dim和Channel一致, output_dim是输出类别
    model = CNN_model(cfg).to('cuda')

    # forward
    results = model(x)
    print("[results.shape]: ", results.shape) # [B, output_dim]