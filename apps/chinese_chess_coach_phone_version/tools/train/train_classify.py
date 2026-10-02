"""训练整盘棋子分类模型（10×9×15，交叉熵）。"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import ClassifyDataset
from models import PieceClassifier


def evaluate(model, dl, device, criterion):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for img, grid in dl:
            img = img.to(device)
            grid = grid.to(device)
            logits = model(img)
            loss_sum += criterion(logits, grid).item() * img.size(0)
            correct += (logits.argmax(dim=1) == grid).sum().item()
            total += grid.numel()
    return loss_sum / len(dl.dataset), correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out", help="data_gen 输出目录（含 train/val）")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="checkpoints/classify")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    model = PieceClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_ds = ClassifyDataset(os.path.join(args.data, "train", "classify"))
    val_ds = ClassifyDataset(os.path.join(args.data, "val", "classify"))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)

    print(f"train={len(train_ds)} val={len(val_ds)}")
    os.makedirs(args.out, exist_ok=True)
    best = float("inf")

    for ep in range(args.epochs):
        model.train()
        run = 0.0
        for img, grid in train_dl:
            img = img.to(device)
            grid = grid.to(device)
            logits = model(img)
            loss = criterion(logits, grid)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item() * img.size(0)

        vloss, vacc = evaluate(model, val_dl, device, criterion)
        print(f"[{ep + 1}/{args.epochs}] loss={run / len(train_ds):.4f} "
              f"val_loss={vloss:.4f} val_acc={vacc:.4f}")
        if vloss < best:
            best = vloss
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))

    torch.save(model.state_dict(), os.path.join(args.out, "last.pt"))
    print("saved →", args.out)


if __name__ == "__main__":
    main()
