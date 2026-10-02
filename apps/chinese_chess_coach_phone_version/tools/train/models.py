"""模型定义（方案 v2 · 阶段 1）。

- ``CornerModel``：轻量 CNN 骨干 + 热力图头 + DSNT 软 argmax，输出棋盘四角归一化坐标 + 置信度。
- ``PieceClassifier``：整盘分类器，输入 canonical 棋盘，输出 10×9×15 logits。

只使用 Conv2d / BatchNorm / ReLU / 池化等基础算子，保证能被 litert-torch 导出为 TFLite。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(in_c: int, out_c: int, k: int, s: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, stride=s, padding=k // 2, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class DSNT(nn.Module):
    """软 argmax（soft-argmax）坐标解码：对热力图 logits 做空间 softmax，再与归一化网格点乘。

    输入 logits ``(B, C, H, W)``，输出归一化 ``[0,1]`` 坐标 ``(B, C, 2)``。
    """

    def __init__(self, H: int, W: int):
        super().__init__()
        gx = torch.linspace(0.0, 1.0, W)
        gy = torch.linspace(0.0, 1.0, H)
        # 用 expand + contiguous 生成显式内存布局，避免 meshgrid 产生的非连续视图
        # 在 load_state_dict 时触发「multiple elements refer to one location」报错。
        self.register_buffer("grid_x", gx.view(1, 1, 1, W).expand(1, 1, H, W).contiguous())
        self.register_buffer("grid_y", gy.view(1, 1, H, 1).expand(1, 1, H, W).contiguous())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logits.shape
        hm = logits.reshape(B, C, -1).softmax(dim=-1).reshape(B, C, H, W)
        x = (hm * self.grid_x).sum(dim=(2, 3))
        y = (hm * self.grid_y).sum(dim=(2, 3))
        return torch.stack([x, y], dim=-1)  # (B, C, 2)


class CornerModel(nn.Module):
    """棋盘角点模型：输入 RGB ``[0,1]``，返回 ``(coords, conf, heatmap)``。

    - ``coords``:  (B, 4, 2) 归一化坐标，顺序 TL, TR, BR, BL。
    - ``conf``:    (B, 4)    各角点置信度（sigmoid 峰值，用于「无棋盘」门控）。
    - ``heatmap``: (B, 4, H, W) sigmoid 后的热力图（用于训练损失）。
    """

    def __init__(self, input_size: int = 320, num_corners: int = 4):
        super().__init__()
        self.input_size = input_size
        self.backbone = nn.Sequential(
            conv_bn(3, 32, 3, s=2),   # /2
            conv_bn(32, 64, 3, s=2),  # /4
            conv_bn(64, 96, 3, s=1),
            conv_bn(96, 96, 3, s=1),
        )
        self.head = nn.Conv2d(96, num_corners, 1)
        self.dsnt = DSNT(input_size // 4, input_size // 4)

    def forward(self, x: torch.Tensor):
        logits = self.head(self.backbone(x))
        heatmap = torch.sigmoid(logits)
        coords = self.dsnt(logits)
        conf = heatmap.amax(dim=(2, 3))
        return coords, conf, heatmap


class CornerModelExport(nn.Module):
    """导出包装：输入 NHWC ``(B,H,W,3)``，输出单个 ``(B,12)`` 张量。

    输出顺序：``x0,y0,x1,y1,x2,y2,x3,y3,c0,c1,c2,c3``（四角 TL,TR,BR,BL + 各自置信度）。
    """

    def __init__(self, model: CornerModel):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).contiguous()  # NHWC → NCHW
        coords, conf, _ = self.model(x)
        return torch.cat([coords.reshape(coords.shape[0], -1), conf], dim=1)


class PieceClassifier(nn.Module):
    """整盘棋子分类器：输入 canonical 棋盘（resize 到 360×400），输出 10×9×15 logits。

    交叉点像素（canonical 0.8×）落在 stride-4 特征图后每 10px 的网格中心，
    因此 ``avg_pool2d(kernel=10, stride=10)`` 与 90 个交叉点严格对齐，无需插值近似。
    """

    def __init__(self, num_classes: int = 15):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn(3, 32, 3, s=2),   # 180×200
            conv_bn(32, 32, 3, s=1),
            conv_bn(32, 64, 3, s=2),  # 90×100
            conv_bn(64, 64, 3, s=1),
            conv_bn(64, 96, 3, s=1),
            conv_bn(96, 96, 3, s=1),
        )
        self.head = nn.Conv2d(96, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.features(x))                 # (B, 15, 100, 90)
        return F.avg_pool2d(logits, kernel_size=10, stride=10)  # (B, 15, 10, 9)


class PieceClassifierExport(nn.Module):
    """导出包装：输入 NHWC ``(B,H,W,3)``，输出 ``(B,15,10,9)`` logits。"""

    def __init__(self, model: PieceClassifier):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).contiguous()  # NHWC → NCHW
        return self.model(x)


def gaussian_heatmap(coords: torch.Tensor, H: int, W: int, sigma: float = 2.0) -> torch.Tensor:
    """由归一化坐标生成 ``(B, C, H, W)`` 高斯热力图目标（CenterNet 风格，峰值=1）。"""
    B, C, _ = coords.shape
    dev = coords.device
    yy = torch.arange(H, device=dev, dtype=torch.float32)
    xx = torch.arange(W, device=dev, dtype=torch.float32)
    cy = coords[..., 1] * (H - 1)  # (B, C)
    cx = coords[..., 0] * (W - 1)
    dy = yy.view(1, 1, H, 1) - cy.view(B, C, 1, 1)
    dx = xx.view(1, 1, 1, W) - cx.view(B, C, 1, 1)
    return torch.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
