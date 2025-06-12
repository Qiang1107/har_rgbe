import os
import sys
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.extensions import resize_and_normalize


# 事件数据以稀疏矩阵的形式保存为npy,然后通过ESequenceDataset加载
class ESequenceDataset(Dataset):
    