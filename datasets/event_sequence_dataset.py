import os
import sys
import yaml
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import random
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class ESequenceDataset(Dataset):
    """事件序列数据集,将事件数据转换为适合PointNet++的点云格式"""
    
    def __init__(self, data_root, window_size_us, max_points, 
                 time_dimension, enable_augment, stride_us=None):
        """
        参数:
            data_root: 数据根目录，包含各个动作类别的子文件夹
            window_size_us: 时间窗口大小，单位为微秒（用于分割长序列）
            stride_us: 滑动窗口步长,单位为微秒,若为None则默认为window_size_us的一半
            max_points: 每个样本的最大点数，超出则随机采样
            time_dimension: 是否将时间戳作为点云的第三维
            enable_augment: 是否进行数据增强
        """
        print(f"[DEBUG] Initializing ESequenceDataset with params: window_size_us={window_size_us}, max_points={max_points}, time_dimension={time_dimension}, enable_augment={enable_augment}")
        self.data_root = data_root
        self.window_size_us = window_size_us
        self.stride_us = stride_us if stride_us is not None else window_size_us // 2
        self.max_points = max_points
        self.time_dimension = time_dimension
        self.enable_augment = enable_augment
        
        # 收集所有文件路径和初始标签
        initial_samples = []
        self.classes = sorted([d for d in os.listdir(data_root) 
                              if os.path.isdir(os.path.join(data_root, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        print(f"[DEBUG] Found classes: {self.classes}")
        print(f"[DEBUG] Class to idx mapping: {self.class_to_idx}")
        
        # 遍历文件夹，收集所有.npy文件的初始信息
        for class_name in self.classes:
            class_dir = os.path.join(data_root, class_name)
            files = glob.glob(os.path.join(class_dir, "*.npy"))
            for file_path in files:
                initial_samples.append((file_path, self.class_to_idx[class_name]))
            print(f"[DEBUG] Found {len(files)} files in class {class_name}")
        
        print(f"找到了来自{len(self.classes)}个类别的{len(initial_samples)}个原始文件")
        
        # 应用滑动窗口处理，生成最终样本列表
        self.samples = []
        total_windows = 0
        
        for file_path, label in initial_samples:
            # 加载事件数据
            print(f"[DEBUG] Processing file: {file_path}, label: {label}")
            events = np.load(file_path)
            print(f"[DEBUG] Loaded events shape: {events.shape}, time range: [{events[:, 0].min()}, {events[:, 0].max()}] us")
            
            # 应用滑动窗口生成多个子序列
            windows = self.sliding_window_events(events)
            print(f"[DEBUG] Generated {len(windows)} windows from file {os.path.basename(file_path)}")
            
            # 将每个窗口添加为单独的样本
            for window_idx, window_events in enumerate(windows):
                # 为每个窗口样本添加元数据，以便在__getitem__中直接使用
                self.samples.append((file_path, label, window_idx, 
                                    window_events[:, 0].min(), window_events[:, 0].min() + self.window_size_us))
            
            total_windows += len(windows)
        
        print(f"使用滑动窗口处理后，共生成{total_windows}个训练样本")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取单个样本并转换为点云格式"""
        file_path, label, window_idx, start_time, end_time = self.samples[idx]
        print(f"[DEBUG] Getting item {idx}: file={os.path.basename(file_path)}, label={label}, window={window_idx}, time=[{start_time}, {end_time}]")
        
        # 加载事件数据并提取指定时间窗口内的事件
        events = np.load(file_path)
        mask = (events[:, 0] >= start_time) & (events[:, 0] < end_time)
        window_events = events[mask]
        print(f"[DEBUG] Window events shape: {window_events.shape}")
        
        # 处理极端情况：窗口内事件过少
        if len(window_events) < 100:
            print(f"[DEBUG] Too few events ({len(window_events)}), adding random events")
            # 创建一些随机事件以达到最小数量
            random_events = np.zeros((100, 4), dtype=events.dtype)
            random_events[:, 0] = np.random.uniform(start_time, end_time, 100)
            random_events[:, 1] = np.random.randint(0, 640, 100)  # 假设x范围为0-640
            random_events[:, 2] = np.random.randint(0, 480, 100)  # 假设y范围为0-480
            random_events[:, 3] = np.random.randint(0, 2, 100)    # 随机极性
            window_events = np.vstack([window_events, random_events])
            print(f"[DEBUG] After adding random events, shape: {window_events.shape}")
        
        # 将事件数据转换为PointNet++兼容的点云格式
        point_cloud = self.events_to_pointcloud(window_events)
        print(f"[DEBUG] Converted point cloud shape: {point_cloud.shape}")
        
        # 转换为PyTorch张量
        point_cloud_tensor = torch.from_numpy(point_cloud).float()
        
        return point_cloud_tensor, label
    
    def events_to_pointcloud(self, events):
        """将事件数据转换为点云格式,适合PointNet++处理"""
        print(f"[DEBUG] Converting events to pointcloud, events shape: {events.shape}")
        # 如果事件数太多，随机采样
        if len(events) > self.max_points:
            print(f"[DEBUG] Sampling {self.max_points} points from {len(events)} events")
            indices = np.random.choice(len(events), self.max_points, replace=False)
            events = events[indices]
        
        # 1. 时间戳归一化 (第一列)
        t_min, t_max = events[:, 0].min(), events[:, 0].max()
        print(f"[DEBUG] Time range: [{t_min}, {t_max}]")
        if t_max > t_min:
            t_normalized = (events[:, 0] - t_min) / (t_max - t_min)
        else:
            t_normalized = np.zeros_like(events[:, 0])
        
        # 2. 空间坐标归一化
        x_max = np.max(events[:, 1]) if np.max(events[:, 1]) > 0 else 640  # 假设最大宽度为640
        y_max = np.max(events[:, 2]) if np.max(events[:, 2]) > 0 else 480  # 假设最大高度为480
        print(f"[DEBUG] Spatial max: x_max={x_max}, y_max={y_max}")
        
        x_normalized = events[:, 1] / x_max
        y_normalized = events[:, 2] / y_max
        
        # 3. 构建点云
        if self.time_dimension:
            # 使用时间作为z坐标：[x, y, t, polarity]
            print(f"[DEBUG] Using time dimension, point cloud will have 4 features")
            point_cloud = np.zeros((len(events), 4), dtype=np.float32)
            point_cloud[:, 0] = x_normalized
            point_cloud[:, 1] = y_normalized
            point_cloud[:, 2] = t_normalized
            point_cloud[:, 3] = events[:, 3]  # 极性作为特征
        else:
            # 仅使用空间坐标和极性：[x, y, polarity]
            print(f"[DEBUG] Not using time dimension, point cloud will have 3 features")
            point_cloud = np.zeros((len(events), 3), dtype=np.float32)
            point_cloud[:, 0] = x_normalized
            point_cloud[:, 1] = y_normalized
            point_cloud[:, 2] = events[:, 3]  # 极性作为特征
        
        # 4. 填充到固定大小 (如果点数不足max_points)
        if len(point_cloud) < self.max_points:
            pad_count = self.max_points - len(point_cloud)
            print(f"[DEBUG] Padding with {pad_count} additional points")
            # 创建填充点 (在现有点的基础上添加小随机偏移)
            if len(point_cloud) > 0:
                random_indices = np.random.choice(len(point_cloud), pad_count)
                pad_points = point_cloud[random_indices].copy()
                # 添加小随机偏移以增加多样性
                pad_points[:, :2] += np.random.normal(0, 0.01, pad_points[:, :2].shape)
                point_cloud = np.vstack([point_cloud, pad_points])
            else:
                # 如果没有点，创建随机点
                print(f"[DEBUG] No points available, creating fully random point cloud")
                point_cloud = np.random.random((self.max_points, point_cloud.shape[1]))
        
        # 5. 数据增强 (可选)
        if self.enable_augment:
            print(f"[DEBUG] Applying point cloud augmentation")
            point_cloud = self.augment_point_cloud(point_cloud)
        
        return point_cloud
    
    def augment_point_cloud(self, point_cloud):
        """对点云进行简单的数据增强"""
        print(f"[DEBUG] Augmenting point cloud of shape {point_cloud.shape}")
        # 1. 随机旋转 (围绕z轴，因为事件数据是2D的)
        if np.random.random() > 0.5:
            theta = np.random.uniform(0, 2*np.pi)
            print(f"[DEBUG] Rotating by {theta:.2f} radians")
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            # 只旋转坐标部分，保持特征不变
            points_xyz = point_cloud[:, :3]
            points_xyz = np.dot(points_xyz, rotation_matrix)
            point_cloud[:, :3] = points_xyz
        
        # 2. 随机抖动
        if np.random.random() > 0.5:
            print(f"[DEBUG] Adding random jitter")
            jitter = np.random.normal(0, 0.01, point_cloud.shape)
            point_cloud += jitter
        
        # 3. 随机缩放
        if np.random.random() > 0.5:
            scale = np.random.uniform(0.8, 1.2)
            print(f"[DEBUG] Scaling by factor {scale:.2f}")
            point_cloud[:, :3] *= scale
        
        return point_cloud

    def sliding_window_events(self, events):
        """使用滑动窗口将长事件序列分割成多个短序列"""
        print(f"[DEBUG] Applying sliding window, events shape: {events.shape}")
        windows = []
        if len(events) == 0:
            print(f"[DEBUG] No events to process")
            return windows
        
        t_min, t_max = events[:, 0].min(), events[:, 0].max()
        t_range = t_max - t_min
        print(f"[DEBUG] Time range: {t_range} us (from {t_min} to {t_max})")
        
        # 如果总时间范围小于窗口大小，直接返回整个序列
        if t_range <= self.window_size_us:
            print(f"[DEBUG] Time range smaller than window size, returning entire sequence")
            return [events]
        
        # 滑动窗口切分
        start_time = t_min
        window_count = 0
        while start_time + self.window_size_us <= t_max:
            mask = (events[:, 0] >= start_time) & (events[:, 0] < start_time + self.window_size_us)
            window_events = events[mask]
            
            # 只有当窗口内有足够事件时才保留
            if len(window_events) >= 100:  # 最小事件数阈值
                windows.append(window_events)
                window_count += 1
                print(f"[DEBUG] Window {window_count}: start={start_time}, events={len(window_events)}")
            else:
                print(f"[DEBUG] Skipped window at start={start_time} (only {len(window_events)} events)")
            
            # 使用配置的步长移动窗口
            start_time += self.stride_us
        
        # 确保至少有一个窗口
        if not windows and len(events) > 0:
            print(f"[DEBUG] No valid windows found, using entire sequence")
            windows.append(events)
        
        return windows


if __name__ == '__main__':
    config_path='configs/har_train_config.yaml'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    print(f"[DEBUG] Loaded config: {cfg}")
    # 测试数据集
    ds_cfg = cfg['dataset']
    ds = ESequenceDataset(
        data_root          = ds_cfg['train_dir'],
        window_size_us     = ds_cfg['window_size_us'],
        stride_us          = ds_cfg['stride_us'],
        max_points         = ds_cfg['max_points'],
        time_dimension     = ds_cfg['time_dimension'],
        enable_augment     = ds_cfg['enable_augment']
    )
    print("len(ds)", len(ds))
    clip, label = ds[0]
    print("ds[0]: clip.shape", clip.shape, "label", label)