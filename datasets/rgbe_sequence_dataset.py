import os
import numpy as np
import torch
from torch.utils.data import Dataset

LABEL_MAP = {
    'Align_screwdriver':     0,
    'Approach':              1,
    'Idle':                  2,
    'No_human':              3,
    'Pick_bolt':             4,
    'Pick_cover':            5,
    'Pick_screwdriver':      6,
    'Place_bolt':            7,
    'Place_cover':           8,
    'Put_down_screwdriver':  9,
    'Screw':                10,
    'Transition':           11
}


def resize_and_normalize(img):
    import torchvision.transforms as T
    target_size = (192, 256) # (H, W)
    resize_transform = T.Compose([
        T.ToPILImage(),
        T.Resize(target_size),
        T.ToTensor()
    ])

    img = resize_transform(img)
    # print("Function [transform]: img.shape", img.shape)
    return img.squeeze(0)
    

class RGBESequenceDataset(Dataset):
    def __init__(self, data_root, window_size, stride, enable_transform=True):
        """
        data_root: 根目录，下面按类别子文件夹存 npy
        window_size: 每个样本固定帧数
        stride: 滑动窗口步长
        enable_transform: 可选的数据增强/预处理方法，对每个帧进行处理
        """
        self.window_size = window_size
        self.stride = stride
        self.enable_transform = enable_transform

        # --- 预扫描 data_root，生成 (npy_path, start_idx, label) 列表 ---
        self.samples = []
        for cls_name, cls_idx in LABEL_MAP.items():
            cls_dir = os.path.join(data_root, cls_name)
            for fname in os.listdir(cls_dir):
                if not fname.endswith('.npy'):
                    continue
                full = os.path.join(cls_dir, fname)
                arr = np.load(full, mmap_mode='r')
                N = arr.shape[0] # 每个 npy 文件的帧数
                # 对每个np文件，用滑窗切出定长片段
                for start in range(0, N - window_size + 1, stride):
                    self.samples.append((full, start, cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 1. 拆出这个样本对应的 npy 文件、起始帧、标签
        npy_path, start, label = self.samples[idx]
        arr = np.load(npy_path).astype(np.float32)  # (B, H, W, C)
        # print("arr.shape", arr.shape, "start", start, "label", label)

        # 2. 截取 [start : start+window_size] 帧
        clip = arr[start : start + self.window_size]  # (window_size, H, W, C)
        # print("clip.shape", clip.shape)

        # 3. 归一化
        ##### 3.1 RGB 归一化
        # clip[..., :3] /= 255.0

        ##### 3.2 RGBE 归一化
        clip[..., :3] /= 255.0
        e = clip[..., 3]
        clip[..., 3] = np.where(e == 0, 0.0, 1.0)

        ##### 3.3 RGBD 归一化
        # clip[..., :3] /= 255.0
        # d = clip[..., 3]
        # # Normalize depth channel
        # d_min, d_max = d.min(), d.max()
        # if d_max > d_min:  # Avoid division by zero
        #     clip[..., 3] = (d - d_min) / (d_max - d_min)
        # else:
        #     clip[..., 3] = 0.0  # Set to zero if there's no depth variation

        ##### 3.4 Event 归一化
        # e = clip[..., 0]
        # clip[..., 0] = np.where(e == 0, 0.0, 1.0)
        
        # 4. to Tensor & permute -> (T,C,H,W)
        clip = torch.from_numpy(clip).permute(0,3,1,2)
        # print("clip.shape after permute", clip.shape)

        # 5. 对每一帧做缩放处理
        if self.enable_transform:
            frames = []
            for t in range(clip.size(0)):
                # print("clip[t].shape", clip[t].shape)
                frame = resize_and_normalize(clip[t])
                # Make sure we have the channel dimension
                if frame.dim() == 2:
                    frame = frame.unsqueeze(0)
                frames.append(frame)
            clip = torch.stack(frames, dim=0)
            # print("clip.shape after transform", clip.shape)
        return clip, label
    

# python -m datasets.rgbe_sequence_dataset
if __name__ == '__main__':
    # 测试数据集
    ds = RGBESequenceDataset(
        # data_root   = '/home/qiangubuntu/research/har_rgbe/data/train',
        data_root   = 'data/train',
        window_size = 9,
        stride      = 3,
        enable_transform = True
    )
    print("len(ds)", len(ds))
    clip, label = ds[0]
    print("clip.shape", clip.shape)
    print("label", label)
