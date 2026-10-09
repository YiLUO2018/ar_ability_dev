"""把 m5_yolo_v3 训练出的 best.pt 导出为 ONNX（供端侧 ONNX Runtime 推理）。

在 WSL（le_wm env）运行：
  python export_onnx.py

导出：
  - 输入  images [1,3,640,640] float32 NCHW（letterbox 到 640×640，RGB）
  - 输出  output0 [1, 4+nc, 8400] float32（YOLOv8 头：4 box + 14 类分数，无 objectness）
    8400 = 80×80 + 40×40 + 20×20 三个尺度的 anchor 数。
  - box 为 xywh（中心点 + 宽高），像素坐标（0–640），需再按 letterbox 反算回原图。

产物：app/src/main/assets/models/yolov8n_chess.onnx（并打印输入/输出形状做校验）。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
# 数据集目录名（debug_frames 下）；飞轮训练用 `--yolo-dir flywheel_yolo`，默认沿用 SAM 合成集。
YOLO_BASE = sys.argv[sys.argv.index("--yolo-dir") + 1] if "--yolo-dir" in sys.argv else "m5_yolo_v3"
BEST = os.path.join(ROOT, "debug_frames", YOLO_BASE, "runs", "train", "weights", "best.pt")
OUT_ONNX = os.path.join(ROOT, "app", "src", "main", "assets", "models", "yolov8n_chess.onnx")


def main():
    from ultralytics import YOLO

    print(f"best.pt -> {BEST}", flush=True)
    print(f"exists: {os.path.exists(BEST)}", flush=True)

    model = YOLO(BEST)
    # 导出 ONNX：640 输入，opset 12，simplify，静态 shape（端上固定 640）
    path = model.export(
        format="onnx",
        imgsz=640,
        opset=12,
        simplify=True,
        dynamic=False,
        half=False,          # 先 float32，端上 CPU 稳；fp16 后续再议
    )
    print(f"exported -> {path}", flush=True)

    # 校验输入/输出形状
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    print("\n== inputs ==")
    for i in sess.get_inputs():
        print(f"  {i.name}: {i.shape} ({i.type})")
    print("== outputs ==")
    for o in sess.get_outputs():
        print(f"  {o.name}: {o.shape} ({o.type})")

    # 拷贝到 assets（若导出路径与目标不同）
    if os.path.abspath(path) != os.path.abspath(OUT_ONNX):
        import shutil
        os.makedirs(os.path.dirname(OUT_ONNX), exist_ok=True)
        shutil.copyfile(path, OUT_ONNX)
        print(f"\ncopied -> {OUT_ONNX}", flush=True)

    print("[done]", flush=True)


if __name__ == "__main__":
    main()
