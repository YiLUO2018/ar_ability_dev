"""训练 YOLO-nano 检测象棋棋子（14 类），并评估。

在 WSL（le_wm env, torch cu130, RTX 4060）运行：
  python train_yolo.py [--model yolov8n.pt] [--epochs 100] [--imgsz 640] [--batch 16]

数据：m5_yolo/（39 张裁剪棋盘图 + 归一化框）。
划分：固定种子 80/20 → 31 train / 8 val（写 train.txt / val.txt）。
输出：m5_yolo/runs/train/ 下 best.pt + 评估图。
"""
import os
import sys
import random
from glob import glob

from ultralytics import YOLO

DBG = "/mnt/l/projects/ar_ability_dev/apps/chinese_chess_coach_phone_version/debug_frames"
# 数据集目录名（debug_frames 下）；飞轮标注用 `--yolo-dir flywheel_yolo`，默认沿用 SAM 合成集。
YOLO_BASE = sys.argv[sys.argv.index("--yolo-dir") + 1] if "--yolo-dir" in sys.argv else "m5_yolo_v3"
YOLO_DIR = os.path.join(DBG, YOLO_BASE)
IMG_DIR = os.path.join(YOLO_DIR, "images")
RUN_DIR = os.path.join(YOLO_DIR, "runs")

CLASS_NAMES = ['帅', '仕', '相', '车', '马', '炮', '兵',
               '将', '士', '象', '车', '马', '炮', '卒']


def main():
    model_name = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "yolov8n.pt"
    epochs = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 100
    imgsz = int(sys.argv[sys.argv.index("--imgsz") + 1]) if "--imgsz" in sys.argv else 640
    batch = int(sys.argv[sys.argv.index("--batch") + 1]) if "--batch" in sys.argv else 16

    # 1) 划分 train/val：优先用数据集自带 train.txt/val.txt（时间序切分，防相邻帧泄漏）；
    #    没有才回退固定种子随机切分。
    if (os.path.exists(os.path.join(YOLO_DIR, "train.txt"))
            and os.path.exists(os.path.join(YOLO_DIR, "val.txt"))):
        train_imgs = [l.strip() for l in open(os.path.join(YOLO_DIR, "train.txt")) if l.strip()]
        val_imgs = [l.strip() for l in open(os.path.join(YOLO_DIR, "val.txt")) if l.strip()]
        print(f"train={len(train_imgs)} val={len(val_imgs)}（用自带时间序切分）", flush=True)
    else:
        imgs = sorted(glob(os.path.join(IMG_DIR, "*.jpg")))
        random.seed(42)
        random.shuffle(imgs)
        n_val = max(1, len(imgs) // 5)          # ~20%
        val_imgs = imgs[:n_val]
        train_imgs = imgs[n_val:]
        print(f"train={len(train_imgs)} val={len(val_imgs)}（随机切分）", flush=True)

    def write_list(path, lst):
        with open(path, "w") as fh:
            for p in lst:
                fh.write(p + "\n")

    train_txt = os.path.join(YOLO_DIR, "train.txt")
    val_txt = os.path.join(YOLO_DIR, "val.txt")
    write_list(train_txt, train_imgs)
    write_list(val_txt, val_imgs)

    # 2) data.yaml（WSL 路径）
    data_yaml = os.path.join(YOLO_DIR, "data_wsl.yaml")
    names_str = "[" + ", ".join(f"'{n}'" for n in CLASS_NAMES) + "]"
    with open(data_yaml, "w", encoding="utf-8") as fh:
        fh.write(f"path: {YOLO_DIR}\n")
        fh.write("train: train.txt\n")
        fh.write("val: val.txt\n")
        fh.write("nc: 14\n")
        fh.write(f"names: {names_str}\n")
    print(f"data_yaml -> {data_yaml}", flush=True)

    # 3) 训练
    model = YOLO(model_name)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0,
        project=RUN_DIR,
        name="train",
        patience=30,
        seed=42,
        workers=2,
        exist_ok=True,
        verbose=True,
    )

    # 4) 评估（val 集 mAP）
    best = os.path.join(RUN_DIR, "train", "weights", "best.pt")
    print(f"\n===== 评估 best.pt on val =====\n", flush=True)
    m = YOLO(best)
    metrics = m.val(data=data_yaml, imgsz=imgsz, device=0, project=RUN_DIR, name="val")
    print(f"mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}", flush=True)


if __name__ == "__main__":
    main()
