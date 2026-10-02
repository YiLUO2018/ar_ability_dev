"""数据集加载（阶段 1）。

读取 ``tools/data_gen`` 生成的 ``out/train|val/{corner,classify}`` 目录，
产出 PyTorch 张量。标签格式见 ``data_gen/README.md``。
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _list_names(img_dir: str):
    return sorted(
        os.path.splitext(n)[0]
        for n in os.listdir(img_dir)
        if n.lower().endswith(".jpg")
    )


def _load_rgb(img_path: str, size, is_square: bool = True) -> torch.Tensor:
    """读图 → BGR→RGB → resize → float32 [0,1] → (3,H,W)。"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if is_square:
        img = cv2.resize(img, (size, size))
    else:
        img = cv2.resize(img, size)  # size=(W, H)
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)


class CornerDataset(Dataset):
    """角点数据集：返回 ``(img, corners, valid)``。

    - ``img``: (3, size, size) RGB [0,1]。
    - ``corners``: (4, 2) 归一化 [0,1]，顺序 TL,TR,BR,BL；负样本为全零。
    - ``valid``: 标量 1.0（有棋盘）/ 0.0（负样本，标签 ``no_board``）。
    """

    def __init__(self, root: str, size: int = 320):
        self.img_dir = os.path.join(root, "images")
        self.lbl_dir = os.path.join(root, "labels")
        self.size = size
        self.names = _list_names(self.img_dir)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        img = _load_rgb(os.path.join(self.img_dir, name + ".jpg"), self.size, is_square=True)

        with open(os.path.join(self.lbl_dir, name + ".txt"), "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        if len(lines) == 1 and lines[0] == "no_board":
            corners = torch.zeros(4, 2, dtype=torch.float32)
            valid = torch.tensor(0.0)
        else:
            corners = torch.tensor(
                [[float(v) for v in ln.split()] for ln in lines[:4]],
                dtype=torch.float32,
            )
            valid = torch.tensor(1.0)
        return img, corners, valid


class ClassifyDataset(Dataset):
    """整盘分类数据集：返回 ``(img, grid)``。

    - ``img``: (3, 400, 360) RGB [0,1]（canonical 棋盘 resize 到 360×400）。
    - ``grid``: (10, 9) long，rank-major，取值 0..14（0=空）。
    """

    def __init__(self, root: str, size=(360, 400)):
        self.img_dir = os.path.join(root, "images")
        self.lbl_dir = os.path.join(root, "labels")
        self.size = size  # (W, H)
        self.names = _list_names(self.img_dir)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        img = _load_rgb(os.path.join(self.img_dir, name + ".jpg"), self.size, is_square=False)

        with open(os.path.join(self.lbl_dir, name + ".txt"), "r", encoding="utf-8") as f:
            grid = [[int(v) for v in ln.split()] for ln in f if ln.strip()]
        grid = torch.tensor(grid, dtype=torch.long)  # (10, 9)
        return img, grid
