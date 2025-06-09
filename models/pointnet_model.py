import torch
import torch.nn as nn
from models.backbones.pointnet2_cls_ssg import PointNet2Classifier

class PointNet2Model(nn.Module):
    """Wrapper to apply PointNet++ on RGBE frame sequences."""
    def __init__(self, cfg: dict):
        super().__init__()
        num_classes = cfg['pointnet2']['num_classes']
        self.model = PointNet2Classifier(num_class=num_classes, normal_channel=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert batch of frames to point cloud and run PointNet++.

        Args:
            x (Tensor): [B, T, C, H, W]
        Returns:
            Tensor: [B, num_classes] logits
        """
        B, T, C, H, W = x.shape
        # Normalize coordinates to [0,1]
        t_coords = torch.linspace(0, 1, steps=T, device=x.device)
        y_coords = torch.linspace(0, 1, steps=H, device=x.device)
        x_coords = torch.linspace(0, 1, steps=W, device=x.device)
        grid_t, grid_y, grid_x = torch.meshgrid(t_coords, y_coords, x_coords, indexing='ij')
        coords = torch.stack((grid_x, grid_y, grid_t), dim=0)  # [3, T, H, W]
        coords = coords.unsqueeze(0).repeat(B, 1, 1, 1, 1)  # [B,3,T,H,W]
        points = torch.cat([coords, x], dim=1)  # [B,3+C,T,H,W]
        points = points.reshape(B, 3 + C, -1)  # [B, 3+C, T*H*W]
        logits = self.model(points)
        return logits
