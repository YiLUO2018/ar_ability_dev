# models/ — 基准模型落盘区

大文件不入库（.gitignore 已忽略本目录除 README 外内容）。放入/更换模型时更新下表。

| 文件 | 版本/来源 | 输入形状 | 状态 |
|---|---|---|---|
| edge_sam_3x_encoder.onnx | 象棋 `assets/models/`（hf-mirror 下载，22MB） | [1,3,1024,1024] fp32 | ⬜ 待拷入 |
| edge_sam_3x_decoder.onnx | 同上（16MB） | embeddings + 点提示（动态 N） | ⬜ 待拷入 |
| yolov8n_chess.onnx | 象棋 v9（`tools/label_real/export_onnx.py` 产物） | [1,3,480,640] | ⬜ 待拷入 |

转换产物（如 `.tflite`）也放此处并在表内登记。

**推到真机**（benchmark App 从应用外部专有目录读取，免权限）：
```
adb push <model>.onnx /storage/emulated/0/Android/data/com.arability.gpu_on_phone/files/models/
```
PC 侧此目录即本仓库 `gpu_on_phone/models/`。
