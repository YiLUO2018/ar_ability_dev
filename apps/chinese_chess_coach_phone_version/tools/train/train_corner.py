"""训练棋盘角点模型。

损失 = 高斯热力图 MSE（正负样本都参与，负样本目标为全零）+ 正样本坐标 L1。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import CornerDataset
from models import CornerModel, gaussian_heatmap


def evaluate(model, dl, device, H, W):
    model.eval()
    hm_sum = 0.0
    err_sum = 0.0
    n_pos = 0
    n = 0
    with torch.no_grad():
        for img, corners, valid in dl:
            img = img.to(device)
            coords, conf, heatmap = model(img)
            corners = corners.to(device)
            valid = valid.to(device)
            gauss = gaussian_heatmap(corners, H, W) * valid.view(-1, 1, 1, 1)
            hm_sum += F.mse_loss(heatmap, gauss).item() * img.size(0)
            pos = valid > 0.5
            if pos.any():
                err_sum += (coords[pos] - corners[pos]).abs().sum().item()
                n_pos += int(pos.sum().item())
            n += img.size(0)
    # val_err：每个角点的平均绝对坐标误差（归一化 [0,1]）
    return hm_sum / n, err_sum / max(n_pos * 4, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out", help="data_gen 输出目录（含 train/val）")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="checkpoints/corner")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    model = CornerModel(input_size=args.size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_ds = CornerDataset(os.path.join(args.data, "train", "corner"), size=args.size)
    val_ds = CornerDataset(os.path.join(args.data, "val", "corner"), size=args.size)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)

    print(f"train={len(train_ds)} val={len(val_ds)}")
    os.makedirs(args.out, exist_ok=True)
    H = W = args.size // 4
    best = float("inf")

    for ep in range(args.epochs):
        model.train()
        run = 0.0
        for img, corners, valid in train_dl:
            img = img.to(device)
            corners = corners.to(device)
            valid = valid.to(device)
            coords, conf, heatmap = model(img)
            gauss = gaussian_heatmap(corners, H, W) * valid.view(-1, 1, 1, 1)
            hm_loss = F.mse_loss(heatmap, gauss)
            coord_loss = (F.l1_loss(coords, corners, reduction="none").sum(dim=-1)
                          * valid.view(-1, 1)).mean()
            loss = hm_loss + coord_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item() * img.size(0)

        vloss, verr = evaluate(model, val_dl, device, H, W)
        print(f"[{ep + 1}/{args.epochs}] loss={run / len(train_ds):.4f} "
              f"val_hm={vloss:.4f} val_err={verr:.5f}")
        if vloss < best:
            best = vloss
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))

    torch.save(model.state_dict(), os.path.join(args.out, "last.pt"))
    print("saved →", args.out)


if __name__ == "__main__":
    main()
