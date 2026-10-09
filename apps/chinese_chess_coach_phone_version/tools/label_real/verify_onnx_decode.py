"""校验导出 ONNX 的解码逻辑（为端侧 YoloPieceDetector 写解码做依据）。

对一张图同时跑：
  1. ultralytics predict（ground truth 解码）
  2. onnxruntime 直跑 + 手写解码（[1,18,8400] → box + class → conf 阈值 → NMS）
对比两者数量/框，确认：
  - 输出布局 [1, 18, 8400]：0:4=box(xywh 像素)、4:18=14 类分数
  - 类分数是否已 sigmoid（概率）还是 logits
  - box 是否需要 letterbox 反缩放

在 WSL（le_wm env）运行：python verify_onnx_decode.py
"""
import os
import sys
from glob import glob

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
ONNX = os.path.join(ROOT, "app", "src", "main", "assets", "models", "yolov8n_chess.onnx")
BEST = os.path.join(ROOT, "debug_frames", "m5_yolo_v3", "runs", "train", "weights", "best.pt")
IMG_DIR = os.path.join(ROOT, "debug_frames", "m5_yolo_v3", "images")

CLASS_NAMES = ['帅', '仕', '相', '车', '马', '炮', '兵',
               '将', '士', '象', '车', '马', '炮', '卒']
CONF = 0.25
IOU = 0.45


def letterbox(img, size=640):
    """等比例缩放 + 右下补灰（与 ultralytics 默认一致：长边缩到 size，短边补 114）。"""
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    return canvas, r, (nh, nw)


def nms(boxes, scores, iou_thr):
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area + (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]) - inter)
        order = order[1:][iou <= iou_thr]
    return keep


def main():
    img_path = sorted(glob(os.path.join(IMG_DIR, "*.jpg")))[0]
    img_bgr = cv2.imread(img_path)
    H, W = img_bgr.shape[:2]
    print(f"image: {os.path.basename(img_path)}  {W}x{H}")

    # ---- 1) ultralytics ground truth ----
    from ultralytics import YOLO
    m = YOLO(BEST)
    res = m.predict(img_bgr, imgsz=640, conf=CONF, iou=IOU, verbose=False)[0]
    gt_boxes = res.boxes.xyxy.cpu().numpy()
    gt_cls = res.boxes.cls.cpu().numpy().astype(int)
    gt_conf = res.boxes.conf.cpu().numpy()
    print(f"\n[ultralytics] {len(gt_boxes)} boxes:")
    for b, c, cf in zip(gt_boxes, gt_cls, gt_conf):
        print(f"  {CLASS_NAMES[c]:<2} conf={cf:.3f}  box=({b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f})")

    # ---- 2) onnxruntime + 手写解码 ----
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])

    canvas, r, (nh, nw) = letterbox(img_bgr, 640)
    # RGB + normalize 0-1 + NCHW
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = np.transpose(rgb, (2, 0, 1))[None]  # [1,3,640,640]

    out = sess.run(None, {"images": x})[0]  # [1,18,8400]
    print(f"\n[onnx] output shape {out.shape}  dtype {out.dtype}")
    pred = out[0]            # [18, 8400]
    boxes = pred[:4].T       # [8400, 4]  xywh（像素，letterbox 坐标）
    scores = pred[4:].T      # [8400, 14]

    # 类分数是否已 sigmoid？打印最大值的量级判断
    print(f"  class score max={scores.max():.4f} min={scores.min():.6f}  (若已 sigmoid，应∈[0,1])")

    # 每个 anchor 取 argmax 类 + 分数
    cls_ids = scores.argmax(1)
    cls_scores = scores.max(1)
    mask = cls_scores > CONF
    keep_boxes = boxes[mask]
    keep_cls = cls_ids[mask]
    keep_scores = cls_scores[mask]

    # xywh -> xyxy
    xyxy = np.zeros_like(keep_boxes)
    xyxy[:, 0] = keep_boxes[:, 0] - keep_boxes[:, 2] / 2
    xyxy[:, 1] = keep_boxes[:, 1] - keep_boxes[:, 3] / 2
    xyxy[:, 2] = keep_boxes[:, 0] + keep_boxes[:, 2] / 2
    xyxy[:, 3] = keep_boxes[:, 1] + keep_boxes[:, 3] / 2

    # letterbox 反缩放（除以 r，再钳到原图）
    xyxy /= r
    xyxy[:, 0] = np.clip(xyxy[:, 0], 0, W)
    xyxy[:, 1] = np.clip(xyxy[:, 1], 0, H)
    xyxy[:, 2] = np.clip(xyxy[:, 2], 0, W)
    xyxy[:, 3] = np.clip(xyxy[:, 3], 0, H)

    # 分类别 NMS（与 ultralytics 一致：每类单独 NMS）
    final_boxes, final_cls, final_conf = [], [], []
    for c in np.unique(keep_cls):
        idx = np.where(keep_cls == c)[0]
        sel = nms(xyxy[idx], keep_scores[idx], IOU)
        for i in sel:
            final_boxes.append(xyxy[idx][i])
            final_cls.append(c)
            final_conf.append(keep_scores[idx][i])

    print(f"\n[手写解码] {len(final_boxes)} boxes:")
    order = np.argsort(final_conf)[::-1]
    for i in order:
        b = final_boxes[i]
        print(f"  {CLASS_NAMES[final_cls[i]]:<2} conf={final_conf[i]:.3f}  box=({b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f})")

    print(f"\n结论: ultralytics={len(gt_boxes)} vs 手写={len(final_boxes)}")


if __name__ == "__main__":
    main()
