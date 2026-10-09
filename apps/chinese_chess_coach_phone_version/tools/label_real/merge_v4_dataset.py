"""合并训练集（时间序 80/20 切分，写 WSL 路径 train.txt/val.txt 防相邻帧泄漏）。

v4 = m5_yolo_v3（debug_video_01 中局对局）+ std02_yolo（debug_video_02 标准开局多角度）
v5 = v4 两源 + flywheel_std_yolo（真机飞轮标准开局帧，设备域 480×640，空交点无框=压假阳负样本）

用法：python merge_v4_dataset.py [v4|v5]
"""
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DBG = os.path.normpath(os.path.join(HERE, "..", "..", "debug_frames"))

SOURCES_V7 = {
    "m5_yolo_v3_clean": os.path.join(DBG, "m5_yolo_v3_clean"),   # m3 占位复核，删 21% 空交点幽灵框
    "std02_all_yolo": os.path.join(DBG, "std02_all_yolo"),
    "flywheel_std_yolo": os.path.join(DBG, "flywheel_std_yolo"),
}
SOURCES_V8 = {
    "m5_yolo_v3_clean2": os.path.join(DBG, "m5_yolo_v3_clean2"), # 清洗 v2：框中心 patch 直读（v1 误删真实红兵已恢复）
    "std02_all_yolo": os.path.join(DBG, "std02_all_yolo"),
    "flywheel_std_yolo": os.path.join(DBG, "flywheel_std_yolo"),
}
SOURCES_V6 = {
    "m5_yolo_v3": os.path.join(DBG, "m5_yolo_v3"),
    "std02_all_yolo": os.path.join(DBG, "std02_all_yolo"),   # std02_yolo 30 + YOLO助力回收 34，已时间序切分
    "flywheel_std_yolo": os.path.join(DBG, "flywheel_std_yolo"),
}
SOURCES_V5 = {
    "m5_yolo_v3": os.path.join(DBG, "m5_yolo_v3"),
    "std02_yolo": os.path.join(DBG, "std02_yolo"),
    "flywheel_std_yolo": os.path.join(DBG, "flywheel_std_yolo"),
}
SOURCES_V4 = {k: v for k, v in SOURCES_V5.items() if k != "flywheel_std_yolo"}

VER = sys.argv[1] if len(sys.argv) > 1 else "v7"
SOURCES = {"v4": SOURCES_V4, "v5": SOURCES_V5, "v6": SOURCES_V6,
           "v7": SOURCES_V7, "v8": SOURCES_V8}[VER]
OUT = os.path.join(DBG, f"train_{VER}_yolo")


def frame_no(name):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 0


def main():
    img_dir = os.path.join(OUT, "images")
    lbl_dir = os.path.join(OUT, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    train, val = [], []
    for src_name, src in SOURCES.items():
        s_img = os.path.join(src, "images")
        names = sorted(
            (f for f in os.listdir(s_img) if f.endswith(".jpg")),
            key=frame_no)                                    # 数字序 = 时间序
        copied = 0
        for f in names:
            lbl = f[:-4] + ".txt"
            if not os.path.exists(os.path.join(src, "labels", lbl)):
                continue
            shutil.copy2(os.path.join(s_img, f), os.path.join(img_dir, f))
            shutil.copy2(os.path.join(src, "labels", lbl), os.path.join(lbl_dir, lbl))
            copied += 1
        n_val = max(1, copied // 5)
        tr, va = names[:-n_val], names[-n_val:]              # 后 20% 时间段做 val
        train += [os.path.join(img_dir, f) for f in tr]
        val += [os.path.join(img_dir, f) for f in va]
        print(f"{src_name}: {copied} 图 → train {len(tr)} / val {len(va)}（时间序）")

    def wsl(p):
        p = p.replace("\\", "/")
        if p[:3].lower() == "l:/":
            p = "/mnt/l/" + p[3:]
        return p

    with open(os.path.join(OUT, "train.txt"), "w") as fh:
        fh.write("\n".join(wsl(p) for p in train) + "\n")
    with open(os.path.join(OUT, "val.txt"), "w") as fh:
        fh.write("\n".join(wsl(p) for p in val) + "\n")
    print(f"\n[done] train={len(train)} val={len(val)} -> {OUT}")


if __name__ == "__main__":
    main()
