# C9 BoardTracker 帧间几何跟踪方案（T1 已验证 / T2 设计）

> 日期：2026-10-14 ｜ 状态：**T1 PC 已验证（video_03），T2 IMU 融合设计完成待真机**
> 前验：实际对局中棋盘世界位姿恒定、动的只有相机（用户先验）→ 帧间要解的只是**相机位姿**（图像间单应），跟踪替代逐帧检测。
> 实现：`tools/label_real/replay_pipeline.py` 的 `BoardTracker` 类 + `run(--track on)`。

## 0. 棋锚纠偏与准入闸（2026-10-14 晚定稿，track4→track10 迭代）

**问题链（video_03 斜视角段 f400-f560）**：LK 累积漂移无人纠正（f491 漂 127px→f527 749px）；SAM 候选同一时段也是坏的（内收/膨胀 300-1600px）；两者互当真值 = 循环信任。

**解法 = 棋子是独立于 SAM/LK 的第三方真值源**（32 颗静态棋子每帧都在 YOLO 视野里）：

1. **棋锚纠偏 `piece_correct`（每 CORRECT_EVERY=10 帧）**：`ref_board`（棋盘系质心，静止真值）↔ 当前 YOLO 质心，两级匹配 → RANSAC 单应直接钉住 H（内点 0.35 格、≥10、det∈[0.25,4]）。实测漂移被钉在 **~3.5px/10 帧**。
   - ① 就近唯一匹配（≤0.75 格）——小漂移快速路径；**捕获范围只有 0.75 格**，拉近段 300px 漂移时恰好失灵（track5 教训）；
   - ② PCA 粗对齐（免对应，`video_label_yoloassist` 同款）：两团点主轴 8 组合相似变换对齐后吸附——大漂移恢复；镜像/翻转组合由 det>0 闸拒绝（反射 det<0）。
   - 约定（ref_board 坐标）只能被**保持**不能被纠正——种子四角误差会被忠实继承（见下「种子约定误差」）。
2. **重锚准入闸 v2+v3**：SAM 四角须同时过 ① 对投影参考包含率 ≥0.9（拦内收）② 最小边距 ≤1.0 格（拦外扩；min 统计较弱，v3 补强）③ **棋锚背书时与传播四角逐角一致 ≤AGREE_CELL=1.5 格**（拦形变/膨胀——track5/6 教训：551-1644px 的坏 SAM 靠 0.90-0.96 包含率混入，投毒 ref_board 后纠偏把错误约定自洽钉死、detect 再不成功）。近期纠偏成功（≤CORRECT_BACKING=30 帧）= 有背书，永不走陈旧逃生；无背书且 age≥STALE_AGE=75 才信 SAM 重建（防快甩后走歪卡死）。
3. **跟踪模式旁路 detect 内部包含门**（`edge.track_mode`）：真值参考下斜视角 SAM 系统性内收（0.83-0.9 < 0.95）候选被内部闸全拒 → detect 全程 None、重锚断供（track5-9 教训；track4 反而因参考漂移意外放行）。候选校验职责由准入闸接管；box 提示仍用参考。
4. **标准开局约定重置**：`matches_std_opening`（on_std≥20 且 extra≤4，几何性判据——越位子只是 RANSAC 外点、身份与位置拟合无关）时 ref_board := 32 标准交点真值 → 纠偏把四角钉到真实物理位置。任意一子走动自动失效。video_03 摆位非标准（on_std=12/extra=11）未触发，真机标准开局场景生效。
5. **证伪记录**：`grid_refine`（C7② 网格线精修）在 track 模式试过又撤——video_03 斜视角段 refine_corners 锁错线组（候选包含率 0.62-0.76 大量出现），P3 的「自洽锁错线组」陷阱在移动场景同样成立。

**残留：种子约定误差——已裁决（2026-10-14 用户看图定夺）**。SAM 种子自带 ±20-50px 逐角随机误差（f41 的 BR 偏 30px→单种子策略下 f426 继承；f218 的 TL 偏 30px→种子刷新策略下 f476 继承）——换种子 = 掷骰子。A/B 对比（`debug_frames/compare_seed_ab/`，A=track7 单种子 / B=track10 含 f218 刷新）：**用户判定 f426/f476/f560 全部 A 胜 → 定稿「单种子+棋锚纠偏」**。据此：
- **detect 内部包含门保持开启**（曾试跟踪模式旁路以解「斜视角 SAM 内收 0.83-0.9<0.95 被全拒→重锚断供」，但旁路放进 47px 偏差的 f218 种子反而更差——重锚断供不再是问题，棋锚纠偏承担传播精度，SAM 只在好几何时供种）；
- tracker 死亡时 ref_img=None → 中心提示全新重锁兜底；
- 定稿版 v03_final 与用户认可的 track7 **逐像素一致**（f138/f426/f476/f560 diff=0）。
候选根治方向（后续可选）：多种子约定聚合 / 更准的独立几何源（棋盘线检测的可靠形态）。

## 1. T1 架构（纯视觉，已验证）

```
LK 光流（goodFeaturesToTrack 棋盘 ROI 掩码 250 点 → calcOpticalFlowPyrLK 前向+后向校验
  → RANSAC findHomography）→ H_g2i ← H_lk @ H_g2i（棋盘系→当前图像系累积单应）
每帧四角 = H_g2i ∙ PHYSICAL_CORNERS（~10ms → 30fps 显示）
EdgeSAM 降频为事件触发重锚：无 H / 传播失败（内点<35、FB>2.5px、单应退化）/ 定周期 25 帧
```

三个关键设计：
- **参考点棋盘系化**：YOLO 棋子质心经 imageToGrid 存棋盘坐标（静止真值），逐帧经当前 H 投回图像 → 移动场景下包含门/box 提示/P0-b 精修全部持续有效（图像系参考相机一动即作废）。
- **重锚准入闸**：SAM 重锚四角须包住 ≥0.9 投影参考才播种/更新显示——斜视角段 SAM 系统性内收 1 格的候选（包含率恒 0.83 = 远端一行棋子被切）即使经救援二级（MOVING 绕门）混进来也会在此被拦；传播陈旧（≥75 帧未过闸）时无条件信 SAM 并重建参考（防快甩后走歪卡死）。
- **遮挡鲁棒性**：跟踪不需要看全棋盘——可见部分特征足以拟合全局单应，手部/运动棋子点被 RANSAC 剔为外点。这是逐帧分割做不到的。

## 2. video_03 实测（787 帧手持移动：俯拍全盘 → 拉近 → 45° 斜视角、棋盘部分出画）

| 指标 | 基线（SAM every-5 + EMA） | T1 跟踪版 |
|---|---|---|
| 显示覆盖率（有四角帧） | ~15%（118 OK + 保持段） | **95%（746/787，0 传播失败）** |
| EdgeSAM 调用成功次数 | 118 | **16（降 86%）** |
| 救援 | 46 | 11 |
| 相邻帧四角跳变(>30px) | —（1Hz 跳变本身大） | **24/787（3%，含 2 次快甩大纠偏）** |
| 准入闸拦截坏重锚 | — | 6 次（含 331/655px 跳变与 0.83 内收段） |
| f500（斜视角）四角 | 旧版 1 格内收 | 修复后贴合真实外角（复核确认） |

## 3. T2 IMU 融合（设计，待真机）

分工：**LK 为主、IMU 旋转预测补位**——两者失效模式互补（LK 怕运动模糊/快甩/弱纹理，IMU 怕漂移但旋转短期精确）：

```
每帧：
  LK 成功 → H_lk（主），同时用 ΔR_imu 与 H_lk 旋转分量差在线估计陀螺零偏
  LK 失败（模糊/快甩）→ H_rot = K · ΔR · K⁻¹（旋转单应；平移小增量忽略或短窗积分）
    → 传播继续，直至 LK 恢复或 EdgeSAM 重锚
显示模式选择（已有 ImuMotion 三态）：
  STATIC → 静止锚显示模式（恒定，IMU静止分支）
  MOVING → T1/T2 跟踪传播（本文档）
  UNKNOWN → S7 检测行为（现状兜底）
```

必需工程件：
1. **相机内参 K**：fx=fy=W/(2·tan(fovH/2))，CameraX Camera2Interop 读 FOV 或标定一次；主点取画面中心（畸变忽略——手机主摄边缘畸变 <1-2px @ 棋盘区域）。
2. **帧-IMU 时间戳对齐**：CameraX imageProxy.imageInfo.timestamp（相机时基）vs SensorEvent.timestamp（system elapsed realtime）——用「静止时 ΔR 应为零」在线校准常数偏移（T2 的命门，方案 §9 已预警）。
3. 陀螺积分：四元数或旋转矩阵增量积分，帧间区间 [t_prev, t_curr] 取 ΔR；零偏由 LK 成功帧在线估计。

## 4. 端侧移植选型（唯一待拍板项）

| 方案 | 代价 | 收益 | 风险 |
|---|---|---|---|
| **A. OpenCV Android（推荐）** | APK +~35MB（so 库）、引入首个原生依赖 | calcOpticalFlowPyrLK/goodFeaturesToTrack/findHomography 全现成、成熟稳定、与 PC 复刻一一对应 | 包体；ABI 兼容（官方 so 全 ABI 覆盖） |
| B. 纯 Kotlin LK | 零依赖 | 无 | 金字塔 LK 自写 ~500 行+调参；findHomography RANSAC 也要自写；工期数天且难与 PC 版对齐 |

推荐 A：本项目已接受 onnxruntime-android（20MB+ 原生库），再加 OpenCV 的边际成本可控；跟踪正确性依赖成熟实现。

## 5. 移植清单（2026-10-15 已落地，真机矩阵待跑）

| 组件 | 状态 |
|---|---|
| `chess/HomographyMath.kt` | ✅ 纯 Kotlin 数学层（3×3 乘/逆/apply、RANSAC 单应、唯一就近匹配、PCA 粗对齐 8 组合、旋转一致闸） |
| `chess/BoardTracker.kt` | ✅ 纯逻辑（seed/accumulate 质量闸/pieceCorrect 两级纠偏/admitReanchor 准入闸 v2+v3/propagateExternal） |
| `chess/LkFlowEngine.kt` | ✅ OpenCV 薄层（goodFeaturesToTrack + calcOpticalFlowPyrLK 前向/后向 + FB 闸；仅真机加载） |
| `chess/StdOpening.kt` | ✅ 标准开局约定重置（32 交点真值；on_std≥20 且 extra≤4） |
| `sensors/RotationPredictor.kt` | ✅ T2（Rodrigues 积分 + K·ΔR·K⁻¹ + 零偏 EMA 估计 + 尾段零阶保持）；旋转补位默认关（ENABLE_T2_ROTATION_FALLBACK=false，真机校准时基后开） |
| Gradle | ✅ org.opencv:opencv 4.10.0（Maven Central 官方 AAR）+ abiFilters arm64-v8a（全 ABI 254MB → 91MB） |
| `MainActivity` | ✅ 三态显示（STATIC=锚 / MOVING=传播 30fps / UNKNOWN=S7）；检测路径原样保留、tracker 搭便车（检测成功候选过准入闸播种）；每 CORRECT_EVERY=10 帧棋锚纠偏（YOLO 质心，外扩 1.5 倍裁剪兜底）；跟踪活着不清屏；unlock 同步 tracker.reset；陀螺经 StaticMotionMonitor.onGyroSample 喂 RotationPredictor |
| 单测 | ✅ 77 项全绿（HomographyMath 7 / BoardTracker 8 / RotationPredictor 6 / AdmissionGate+StdOpening 4 / 原有全套） |
| 真机矩阵 | ⏳ 手持移动/静止/切换/快甩 + IMU 静止分支四场景（用户暂无法连真机） |

**移植期修复的两个移植性 bug（PC 与 Kotlin 双修）**：
1. **PCA 180° 翻转陷阱**：标准开局点云 180° 旋转对称 → PCA 翻转组合 count 满分且 det>0 闸拦不住（对称云上镜像位置即真实棋子位置）→ 两端加**旋转一致闸**（纠偏周期内棋盘不可能转 40°+，PC piece_correct 与 Kotlin pieceCorrect 同款）。
2. **标准开局表列数错误**：底线是 9 列全占（车马象士将士象马车），首版误写成 0,2,4,6,8 五列（24 子）——真机标准开局会因 files 1,3,5,7 的 8 个「越位」被判不匹配。两端修为 32 子。

## 5.1 真机首轮验证（2026-10-15 晚，V2509A）

全链路打通（LK 传播/播种节拍/棋锚纠偏 0.7-26px/标准开局重置/陈旧逃生/IMU 三态/D1/T2 零偏收敛），另修复 2 个真机 bug：
1. **det2 比值闸**：绝对闸 [0.25,4] 是 PC 大画幅标定，480p 手机棋盘 ~200px 时合法 det≈0.1 被误杀（纠偏恒失败）→ 改 det/det0∈[0.5,2]。
2. **LK 死亡死锁**（移植独有）：age 冻结 → 背书永不过期 → v3 拦死 SAM 无法重播种 → admitReanchor 加 propagationAlive（死了直接 ACCEPT，PC 语义对齐）。

**残留已定位未修**：① refBoard 降级替换（19 质心帧顶掉 30+ 好参考 → 纠偏连环失败；修复=不用更差参考替换）② LK 480p 快速移动偏脆（40s 死 32 次，恢复 ~3s 停顿；候选：LK 参数/提分辨率/T2 补位）③ T2 零偏应只在 STATIC 帧更新（移动段被平移项污染）。详见开发日志 2026-10-15 晚。

1. `chess/BoardTracker.kt`：BoardTracker 移植（seed/track/corners/ref_to_image/准入闸常量）；LK 调 OpenCV。
2. `sensors/RotationPredictor.kt`：陀螺积分 + K·ΔR·K⁻¹ + 零偏在线估计 + 时基对齐（纯逻辑可 JVM 单测）。
3. `MainActivity`：ImageAnalysis 加灰度抽取（Y 平面零拷贝）→ tracker.track 每帧；显示模式三态切换（STATIC 锚 / MOVING 传播 / UNKNOWN 检测）；事件触发 EdgeSAM 重锚。
4. 单测：旋转积分、准入闸、棋盘系参考投影；真机矩阵：手持移动 / 静止 / 静止↔移动切换 / 快甩。
