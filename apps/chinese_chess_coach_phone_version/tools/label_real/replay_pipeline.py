"""端侧管线 PC 复刻：对视频逐帧跑与 App 完全一致的识别链路并可视化。

忠实复刻（逻辑与参数一一对应，出处标注）：
  - SegmentationBoardDetector.kt：EdgeSAM encoder/decoder、掩码面积带选择、四角提取
    （连通域→边界→凸包→贪心减点）、isValidQuad、贴边裁切拒、P0-a 时间一致性闸、
    提示点回灌、自适应 EMA、P0-b refineQuad。
  - QuadRefiner.kt：质心反馈校正（吸附 0.8 格 / 内点 0.35 格 / 校正钳 1.5 格 /
    唯一性内点 ≥16 + 行列覆盖 ≥6/6）。
  - YoloPieceDetector.kt：棋盘裁剪外扩 6%、居中 letterbox 640、逐 anchor argmax≥0.25、
    按类 NMS 0.45（枚举序先到先得）、置信度降序就近取整吸附先到先得。
  - MainActivity.processFrame：DETECT_EVERY_N=5、LOCK_MAX_MISSES=3、voteGrid 多数投票
    （平票取较新）、buildBoard→FEN（红大写 rank0=黑方顶）、XQWLight searchMain(64,1000)
    红方执走、ChineseNotation 记谱。
中间帧（非检测帧）绘制保持态（与端上 overlay 每帧重绘最新状态一致）。

用法：python replay_pipeline.py <video> <out_dir> [--max N] [--every 5]
"""
import os
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from edge_line_quad import line_intersect_quad   # C5-P1 边线求交（--quad-mode line 时启用）
from refine import refine_corners                 # C7② 网格线精修（video_01 TR 内凹根治）
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
DBG = os.path.join(ROOT, "debug_frames")
ASSETS = os.path.join(ROOT, "app", "src", "main", "assets", "models")
EDGE_ENC = os.path.join(ASSETS, "edge_sam_3x_encoder.onnx")
EDGE_DEC = os.path.join(ASSETS, "edge_sam_3x_decoder.onnx")
YOLO_ONNX = os.path.join(ASSETS, "yolov8n_chess.onnx")
JAVA = r"L:\Program Files\AndroidStudio\jbr\bin\java.exe"
XQW_CP = os.path.join(HERE, "xqw_driver", "build")
FONT = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 26)
FONT_S = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 20)

CLASS_NAMES = ['帅', '仕', '相', '车', '马', '炮', '兵', '将', '士', '象', '车', '马', '炮', '卒']
TYPE_ORDINALS = [0, 1, 2, 4, 3, 5, 6]        # YOLO c%7 → PieceType ordinal（车在马前）
PT = ['帅', '仕', '相', '马', '车', '炮', '兵']  # PieceType ordinal → 名
FEN_CHAR = {                                   # (ordinal, red?) → FEN 字符
    (0, True): 'K', (0, False): 'k', (1, True): 'A', (1, False): 'a',
    (2, True): 'B', (2, False): 'b', (3, True): 'N', (3, False): 'n',
    (4, True): 'R', (4, False): 'r', (5, True): 'C', (5, False): 'c',
    (6, True): 'P', (6, False): 'p',
}
# PieceType 约束（FlywheelCollector.isIllegalPosition 同语义，用于建议门）
MAX_COUNTS = [1, 2, 2, 2, 2, 2, 5]


def opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


# ================================================================ 几何（Homography.kt 移植）

def h_compute(src, dst):
    """4 点对应 → 3×3 单应（与 Homography.compute 同：高斯消元，h33=1）。"""
    a = np.zeros((8, 9), np.float64)
    for i in range(4):
        x, y = src[i]
        u, v = dst[i]
        a[i * 2] = [x, y, 1, 0, 0, 0, -x * u, -y * u, u]
        a[i * 2 + 1] = [0, 0, 0, x, y, 1, -x * v, -y * v, v]
    # 高斯消元
    for col in range(8):
        piv = col + int(np.argmax(np.abs(a[col:, col])))
        a[[col, piv]] = a[[piv, col]]
        if abs(a[col, col]) < 1e-9:
            continue
        a[col] /= a[col, col]
        for r in range(8):
            if r != col:
                a[r] -= a[r, col] * a[col]
    h = np.append(a[:, 8], 1.0)
    return h.reshape(3, 3)


def h_transform(h, pts):
    """点集 (N,2) 经 3×3 单应映射（与 Homography.transform 同）。"""
    pts = np.asarray(pts, np.float64)
    one = np.hstack([pts, np.ones((len(pts), 1))])
    out = one @ h.T
    return out[:, :2] / out[:, 2:3]


PHYSICAL_CORNERS = np.array([[0, 0], [450, 0], [450, 500], [0, 500]], np.float32)
MARGIN, CELL = 25, 50


# ================================================================ EdgeSAM 定位器（SegmentationBoardDetector.kt 移植）

class EdgeSamDetector:
    IMG_SIZE = 1024
    MIN_AREA_RATIO, MAX_AREA_RATIO = 0.02, 0.35
    EDGE_MARGIN = 8.0
    QUAD_MIN_DIST_RATIO, QUAD_MIN_AREA_RATIO = 0.02, 0.01
    EMA_ALPHA_MIN, EMA_ALPHA_MAX, EMA_FOLLOW_RATIO = 0.15, 0.9, 0.08
    JUMP_AREA_LO, JUMP_AREA_HI, JUMP_CORNER_RATIO, JUMP_ACCEPT_STREAK = 0.75, 1.35, 0.22, 3
    PIXEL_MEAN = np.array([123.675, 116.28, 103.53], np.float32)
    PIXEL_STD = np.array([58.395, 57.12, 57.375], np.float32)

    def __init__(self):
        import onnxruntime as ort
        so = ort.SessionOptions()
        self.enc = ort.InferenceSession(EDGE_ENC, so, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(EDGE_DEC, so, providers=["CPUExecutionProvider"])
        self.prompt_x, self.prompt_y = 0.5, 0.5
        self.prev_corners = None      # EMA 状态
        self.jump_candidate = None
        self.jump_streak = 0
        self.fail_streak = 0          # 连续失败计数（移动重锁修复 A/B/C，与端上同）
        # A/B 分析开关（corner_trace.py 用）：False = 修复前行为。
        self.relock_fixes = True      # 历史失效 + 中心提示救援
        self.shape_gate = True        # C4 形状闸
        self.grid_refine = False      # C7② 网格线精修（默认关：P3「锁错线组」陷阱在
        #                               video_01 复现——初始角偏 ~1 格时线匹配按最近原
        #                               则锁错索引，rms 小但整体平移 1 格，越修越错）
        self.GRID_RMS_MAX = 3.0       # 精修接受闸：canonical px
        # 轨迹自省（corner_trace.py 用）
        self._last_rescue = False
        self._last_recon = False
        self._last_refined = False
        self._last_area = 0
        # D1 静止锚（IMU 静止分支，IMU静止分支设计方案.md）。PC 模拟由 --imu 开关驱动；
        # 真机由 StaticMotionMonitor 三态驱动（UNKNOWN/MOVING 一律不启用锚特权）。
        self.imu_static = False
        self.imu_moving = False       # MOVING：救援启用二级（无参考重选）；UNKNOWN=退回 S7
        self.anchor = None            # 锚四角 (4,2)：连续一致成功检测的中位数
        self._anchor_buf = []         # 建锚积累（贪心连续一致计数）
        self._anchor_win = []         # 滚动锚窗口（有锚后被接受检测进窗取中位数）
        self._anchor_rej = []         # 连续锚拒绝（双确认超时候选用）
        self.n_anchor_rej = 0         # 累计锚拒绝次数（指标）
        self.n_reanchor = 0           # 仲裁重锚次数（指标）

    # ---- 推理 ----
    def _encode(self, img):
        h, w = img.shape[:2]
        scale = 1024.0 / max(h, w)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else img
        canvas = np.zeros((1024, 1024, 3), np.float32)
        canvas[:nh, :nw] = resized.astype(np.float32)
        x = (canvas - self.PIXEL_MEAN) / self.PIXEL_STD
        x = np.transpose(x, (2, 0, 1))[None]
        emb = self.enc.run(None, {"image": x})[0]
        return emb, nw, nh, scale

    def _decode(self, emb, px, py):
        coords = np.array([[[px, py]]], np.float32)
        labels = np.array([[1]], np.float32)
        out = self.dec.run(None, {
            "image_embeddings": emb, "point_coords": coords, "point_labels": labels})
        scores, masks = out[0], out[1]        # [1,C], [1,C,256,256]
        return scores[0], masks[0]

    BOX_EXPAND = 0.05       # box 提示外扩比例（相对参考 bbox 对角线）

    def _box_hint(self, w, h, ref_pts=None):
        """box 提示来源（优先级）：①静止锚 bbox——锚是经建锚验证的**固定**物理位置，
        掩码被钉在锚附近 → 检测方差坍缩（vis5 教训：质心 bbox 会随棋子移动/遮挡期
        崩塌 <10 个而消失/漂移，中心点提示掩码游走 195px，0.2 格锚闸被逼成永久拒绝）。
        冻结锚非 S4 自强化：S4 的坑是 box 逐帧追掩码输出（EMA/prev），锚不追。
        ②参考质心 bbox（移动态）。③无（纯点提示）。均外扩 5% 对角线≈物理边。"""
        if self.imu_static and self.anchor is not None:
            src = self.anchor
        elif ref_pts is not None and len(ref_pts) >= self.MIN_REF_PTS:
            src = np.asarray(ref_pts, np.float64)
        else:
            return None
        x0, y0 = src[:, 0].min(), src[:, 1].min()
        x1, y1 = src[:, 0].max(), src[:, 1].max()
        expand = self.BOX_EXPAND * np.hypot(x1 - x0, y1 - y0)
        return (x0 - expand, y0 - expand, x1 + expand, y1 + expand)

    def _decode_box(self, emb, box, px, py, w, h):
        """box+点混合提示（labels [2,3,1]）：box 约束物体范围，点定位板心。
        实测（video_01）：暗对暗物理边使点提示掩码「改跟」内部高对比线（TR 内凹
        40-150px），box 提示把问题帧 TR 压到 7-26px。坐标转 1024 系（×长边比例）。"""
        s = 1024.0 / max(w, h)
        coords = np.array([[[box[0] * s, box[1] * s],
                            [box[2] * s, box[3] * s],
                            [px, py]]], np.float32)
        labels = np.array([[2, 3, 1]], np.float32)
        out = self.dec.run(None, {
            "image_embeddings": emb, "point_coords": coords, "point_labels": labels})
        return out[0][0], out[1][0]

    # ---- 掩码后处理（连通域/凸包/贪心减点，Kotlin 同构）----
    @staticmethod
    def _largest_component_area(binary, w, h):
        n, _ = cv2.connectedComponentsWithStats(
            (binary > 0).astype(np.uint8), connectivity=4)[:2]
        if n <= 1:
            return 0
        return int(cv2.connectedComponentsWithStats(
            (binary > 0).astype(np.uint8), connectivity=4)[2][1:, 4].max())

    @staticmethod
    def _mask_hull(binary):
        """最大连通域 → 边界 → 凸包顶点环。返回 (hull, 域掩码) 或 (None, None)。"""
        mask = (binary > 0).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        if n <= 1:
            return None, None
        best = 1 + int(np.argmax(stats[1:, 4]))
        if stats[best, 4] < mask.size * 0.02:
            return None, None
        m = (lab == best)
        h, w = m.shape
        # 边界像素（与 Kotlin 同判据：任一 4 邻域不同标签/出界）
        pad = np.pad(m, 1, constant_values=False)
        up, dn = pad[:-2, 1:-1], pad[2:, 1:-1]
        lf, rt = pad[1:-1, :-2], pad[1:-1, 2:]
        boundary = m & ~(up & dn & lf & rt)
        ys, xs = np.nonzero(boundary)
        if len(xs) < 4:
            return None, None
        pts = np.unique(np.stack([xs, ys], 1).astype(np.int64), axis=0).astype(np.float64)
        pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in pts[::-1]:
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        hull = lower[:-1] + upper[:-1]
        return hull, m

    @staticmethod
    def _greedy_quad(hull):
        """凸包环 → 贪心减点到 4 顶点（全可见路径）。"""
        if len(hull) < 4:
            return None
        poly = list(hull)

        def seg_dist(p, a, b):
            dx, dy = b[0] - a[0], b[1] - a[1]
            l2 = dx * dx + dy * dy
            if l2 == 0:
                return np.hypot(p[0] - a[0], p[1] - a[1])
            t = np.clip(((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2, 0, 1)
            return np.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)

        while len(poly) > 4:
            n2 = len(poly)
            costs = [seg_dist(poly[i], poly[(i - 1) % n2], poly[(i + 1) % n2]) for i in range(n2)]
            poly.pop(int(np.argmin(costs)))
        q = np.array(poly)
        s, d = q.sum(1), q[:, 1] - q[:, 0]
        return np.array([q[np.argmin(s)], q[np.argmin(d)], q[np.argmax(s)], q[np.argmax(d)]])

    BORDER_TOL = 4.0          # 掩码坐标系贴边判据（px）：解码器上采样掩码距内容边界有 1~3px 内缩，须宽容差
    QUAD_MODE = "hull"        # C5-P1：全可见路径四角提取模式（hull=贪心减点线上行为；line=边线求交，失败回退 hull）

    @staticmethod
    def _line_intersect(la, lb):
        px, py, dx, dy = la
        qx, qy, ex, ey = lb
        d = dx * ey - dy * ex
        if abs(d) < 1e-9:
            return None
        t = ((qx - px) * ey - (qy - py) * ex) / d
        return (px + t * dx, py + t * dy)

    @staticmethod
    def _simplify_ring(ring, eps=2.5):
        """保轮廓简化：贪心删除到邻点弦垂距 < eps 的顶点（栅格阶梯毛刺），
        真角点垂距大得以保留。凸环上与 RDP 等效。"""
        poly = list(ring)

        def seg_dist(p, a, b):
            dx, dy = b[0] - a[0], b[1] - a[1]
            l2 = dx * dx + dy * dy
            if l2 == 0:
                return np.hypot(p[0] - a[0], p[1] - a[1])
            t = np.clip(((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2, 0, 1)
            return np.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)

        while len(poly) > 4:
            n2 = len(poly)
            costs = [seg_dist(poly[i], poly[(i - 1) % n2], poly[(i + 1) % n2]) for i in range(n2)]
            j = int(np.argmin(costs))
            if costs[j] >= eps:
                break
            poly.pop(j)
        return poly

    @staticmethod
    def _reconstruct_clipped(ring, cw, ch):
        """角点出画面：剔除贴边框段，剩余可见边线（含被截边的可见段）延长求交，
        复原画面外角点——覆盖 1 角出画 / 2 对角出画；整条边出画则无解返回 False。

        ring=有序顶点环（凸包）；cw,ch=掩码坐标系里的图像内容边界（非方形输入时
        内容只占掩码画布左上区，右侧/底部是 letterbox 零填充——贴边判据必须用
        内容边界而非画布边界）。「两端点都贴内容边界」的段=掩码被画面裁切的痕迹；
        剩余段按环序为棋盘可见边，近共线链合并后须恰 4 条边线，
        相邻边线交点=四角（允许出画面）。返回 (corners, ok)。
        """
        tol = EdgeSamDetector.BORDER_TOL
        ring = EdgeSamDetector._simplify_ring(ring)

        def on_border(p):
            return (p[0] <= tol or p[0] >= cw - 1 - tol or
                    p[1] <= tol or p[1] >= ch - 1 - tol)

        def collinear(pa, pb, c, d):
            """链(pa→pb) 延伸到 (c→d) 是否近共线（夹角<~20°，按弦方向判）。"""
            v1x, v1y = pb[0] - pa[0], pb[1] - pa[1]
            v2x, v2y = d[0] - c[0], d[1] - c[1]
            l1 = np.hypot(v1x, v1y)
            l2 = np.hypot(v2x, v2y)
            if l1 < 1e-6 or l2 < 1e-6:
                return False
            return abs((v1x * v2x + v1y * v2y) / (l1 * l2)) > 0.94

        n = len(ring)
        border_v = [on_border(p) for p in ring]
        if not any(border_v):
            return None, False            # 无贴边痕迹（全可见）→ 走贪心减点路径
        # 贴边段（两端点都贴边，如沿边框走线/画面角处的边框折线）整段丢弃；
        # 只有一端贴边的段保留——贴边端点=棋盘边与边框的交点，在真实边线上。
        board = [(ring[i], ring[(i + 1) % n]) for i in range(n)
                 if not (border_v[i] and border_v[(i + 1) % n])]
        if not board:
            return None, False            # 全部贴边（异常掩码）
        chains = []
        for a, b in board:
            if chains and collinear(*chains[-1], a, b):
                chains[-1] = (chains[-1][0], b)
            else:
                chains.append((a, b))
        # 环形接缝：首尾链若共线则合并（环序跨列表边界）
        if len(chains) >= 2 and collinear(*chains[-1], *chains[0]):
            chains[0] = (chains[-1][0], chains[0][1])
            chains.pop()
        if len(chains) != 4:
            return None, False
        lines = []
        for a, b in chains:
            dx, dy = b[0] - a[0], b[1] - a[1]
            if abs(dx) + abs(dy) < 1e-6:
                return None, False
            lines.append((a[0], a[1], dx, dy))
        qs = []
        for i in range(4):
            p = EdgeSamDetector._line_intersect(lines[i], lines[(i + 1) % 4])
            if p is None:
                return None, False
            qs.append(p)
        q = np.array(qs, np.float64)
        s, d = q.sum(1), q[:, 1] - q[:, 0]
        quad = np.array([q[np.argmin(s)], q[np.argmin(d)],
                         q[np.argmax(s)], q[np.argmax(d)]])
        return quad, True

    @staticmethod
    def _extract_corners_ex(binary, cw=None, ch=None):
        """→ (四角[掩码坐标], 是否外推复原, 最大连通域掩码)；失败 (None, False, None)。

        cw,ch=内容边界（掩码坐标，默认=画布尺寸）。"""
        hull, comp = EdgeSamDetector._mask_hull(binary)
        if hull is None:
            return None, False, None
        mh, mw = binary.shape[:2]
        cw = mw if cw is None else cw
        ch = mh if ch is None else ch
        rec, ok = EdgeSamDetector._reconstruct_clipped(hull, cw, ch)
        if ok:
            return rec, True, comp
        if EdgeSamDetector.QUAD_MODE == "line":
            # C5-P1：边线求交（角点检测在简化环、边线拟合用原始边界像素云），失败回退贪心。
            pad = np.pad(comp, 1, constant_values=False)
            bmask = comp & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
            ys, xs = np.nonzero(bmask)
            bpts = np.stack([xs, ys], 1).astype(np.float64) if len(xs) else None
            li = line_intersect_quad(np.asarray(hull, np.float64), bpts)
            if li is not None:
                return li, False, comp
        return EdgeSamDetector._greedy_quad(hull), False, comp

    @staticmethod
    def _extract_corners(binary):
        """兼容旧签名：仅全可见路径（凸包 → 贪心减点）。"""
        hull, _ = EdgeSamDetector._mask_hull(binary)
        if hull is None:
            return None
        return EdgeSamDetector._greedy_quad(hull)

    # ---- 校验 ----
    @staticmethod
    def _quad_area(c):
        s = 0.0
        for i in range(4):
            j = (i + 1) % 4
            s += c[i, 0] * c[j, 1] - c[j, 0] * c[i, 1]
        return abs(s) / 2

    def _is_valid_quad(self, c, w, h):
        min_dist = np.hypot(w, h) * self.QUAD_MIN_DIST_RATIO
        for i in range(4):
            for j in range(i + 1, 4):
                if np.hypot(*(c[i] - c[j])) < min_dist:
                    return False
        return self._quad_area(c) >= w * h * self.QUAD_MIN_AREA_RATIO

    @staticmethod
    def _is_cut_off(c, w, h):
        return bool((c[:, 0].min() < 8 or c[:, 0].max() > w - 8 or
                     c[:, 1].min() < 8 or c[:, 1].max() > h - 8))

    def _is_jump(self, q, ref):
        ratio = self._quad_area(q) / max(self._quad_area(ref), 1.0)
        if not (self.JUMP_AREA_LO <= ratio <= self.JUMP_AREA_HI):
            return True
        diag = max(np.hypot(*(ref[1] - ref[0])), np.hypot(*(ref[2] - ref[3])))
        return bool(np.max(np.hypot(*(q - ref).T)) > diag * self.JUMP_CORNER_RATIO)

    def _smooth(self, raw, w, h):
        prev = self.prev_corners
        if prev is None:
            self.prev_corners = raw.copy()
            return raw.copy()
        dx, dy = (raw - prev).mean(0)
        dist = np.hypot(dx, dy)
        diag = np.hypot(w, h)
        alpha = np.clip(dist / (diag * self.EMA_FOLLOW_RATIO),
                        self.EMA_ALPHA_MIN, self.EMA_ALPHA_MAX)
        out = alpha * raw + (1 - alpha) * prev
        self.prev_corners = out.copy()
        return out

    SHAPE_MIN_ANGLE, SHAPE_MAX_ANGLE = 50.0, 130.0    # C4 形状闸（与端上同构）
    SHAPE_EDGE_LO, SHAPE_EDGE_HI = 0.35, 2.9

    @staticmethod
    def _has_board_shape(c):
        """C4：内角∈[50°,130°] 且对边比∈[0.35,2.9]，杀溢出细条/尖菱形。"""
        for i in range(4):
            v1, v2 = c[(i - 1) % 4] - c[i], c[(i + 1) % 4] - c[i]
            l1, l2 = np.hypot(*v1), np.hypot(*v2)
            if l1 < 1e-6 or l2 < 1e-6:
                return False
            ang = np.degrees(np.arccos(np.clip(float(v1 @ v2) / (l1 * l2), -1, 1)))
            if not (EdgeSamDetector.SHAPE_MIN_ANGLE <= ang <= EdgeSamDetector.SHAPE_MAX_ANGLE):
                return False
        top, bottom = np.hypot(*(c[1] - c[0])), np.hypot(*(c[3] - c[2]))
        left, right = np.hypot(*(c[3] - c[0])), np.hypot(*(c[2] - c[1]))
        if bottom <= 0 or right <= 0:
            return False
        return (EdgeSamDetector.SHAPE_EDGE_LO <= top / bottom <= EdgeSamDetector.SHAPE_EDGE_HI and
                EdgeSamDetector.SHAPE_EDGE_LO <= left / right <= EdgeSamDetector.SHAPE_EDGE_HI)

    MIN_REF_PTS, CONTAIN_MIN = 10, 0.95   # C7①：参考点最少数量 / 包含率阈值（0.95=好候选实测下界；
    #                                    0.85 会放过手遮挡的 TR 60-126px 中带候选）

    # D1 静止锚参数（IMU静止分支设计方案.md §6，2026-10-13 收尾版）
    ANCHOR_SETTLE = 5            # 连续成功且互偏 <JITTER 才建锚
    ANCHOR_JITTER = 5.0          # px（S7 正常帧角点抖动中位 4px）
    ANCHOR_ROLL = 8              # 滚动锚窗口：被接受检测进窗取中位数——跟随 SAM 慢漂
    #                            # （box 随棋子走子呼吸 ±20-40px 分钟级，vis5/6 实测固定锚
    #                            # 与稳态检测互斥卡死），中位数抗 ≤3 帧突变；**显示=锚**（恒定），
    #                            # 检测的慢漂不进显示（用户标准：静止场景四角应完全一致）。
    ANCHOR_CORNER = 0.2          # 拒绝阈：任一角 vs 锚 > 0.2 格（收紧自 0.4——好帧抖动 p95≤11px
    #                            # ＜0.2格，门内残余清除：静止视频单角 max 125→21px、均值 err>15px 9.7%→0%）。
    #                            # 静止态永不自动重锚（v1 六连击一致性 / v2v3 超时 30/60 均证伪——全管线长
    #                            # 遮挡段攒满超时在漂移位重锚，vis4 实测 6 次误仲裁→幽灵帧）；恢复=IMU 离开静止

    @staticmethod
    def _point_in_quad(q, x, y):
        """点是否在凸四边形内（叉积同号，任意方向序）。"""
        pos = neg = False
        for i in range(4):
            j = (i + 1) % 4
            cr = (q[j][0] - q[i][0]) * (y - q[i][1]) - (q[j][1] - q[i][1]) * (x - q[i][0])
            if cr > 0:
                pos = True
            elif cr < 0:
                neg = True
            if pos and neg:
                return False
        return True

    @classmethod
    def _containment(cls, q, pts):
        """参考点被四边形包含的比例。"""
        if pts is None or len(pts) == 0:
            return 1.0
        return sum(cls._point_in_quad(q, p[0], p[1]) for p in pts) / len(pts)

    # ---- D1 静止锚（IMU 静止分支）----

    @staticmethod
    def _min_edge_dist(q, pts):
        """参考点到四边形四边的最小距离（px）。真棋盘最外排棋子质心距物理边
        ≈ MARGIN=0.5 格；外扩型伪框（掩码溢到桌面）把边推出棋子外 → 最小边距
        显著增大——这是包含门（单向，只拦内收）拦不住的外扩判别量。"""
        q = np.asarray(q, np.float64).reshape(4, 2)
        pts = np.asarray(pts, np.float64)
        best = float("inf")
        for i in range(4):
            a, b = q[i], q[(i + 1) % 4]
            ab = b - a
            l2 = float(ab @ ab)
            if l2 < 1e-9:
                d = np.linalg.norm(pts - a, axis=1)
            else:
                tt = np.clip(((pts - a) @ ab) / l2, 0, 1)
                proj = a + tt[:, None] * ab
                d = np.linalg.norm(pts - proj, axis=1)
            best = min(best, float(d.min()))
        return best

    @staticmethod
    def _cell_size(c):
        """锚四角 → 画面格宽（canonical 450×500 对角 672.7、格 50 的比例）。"""
        return float(np.hypot(*(c[2] - c[0]))) * (CELL / np.hypot(450.0, 500.0))

    # D1 挪盘兜底（双确认超时）：①连续拒绝 ≥REANCHOR_ASK 且被拒四角相对锚为四角整体
    # 平移（每角 >REANCHOR_MOVE_CELL 格——真挪盘四角全动，手遮挡漂移只变形 1-2 角）
    # → reanchor_candidate() 给出候选；②上层用候选跑 YOLO 做内容验证（新位置棋子质心
    # 包含 ≥95% 且 ≥10 个）→ resolve_reanchor。三层全过才清锚重锁，否则保持。
    REANCHOR_ASK = 60
    REANCHOR_MOVE_CELL = 0.5

    def reanchor_candidate(self):
        """双确认①②：超时 + 几何（整体平移）。满足返回候选四角，否则 None。"""
        if (not self.imu_static or self.anchor is None
                or len(self._anchor_rej) < self.REANCHOR_ASK):
            return None
        cell = self._cell_size(self.anchor)
        cand = self._anchor_rej[-1]
        d = np.linalg.norm(cand - self.anchor, axis=1)
        return cand if float(d.min()) > self.REANCHOR_MOVE_CELL * cell else None

    def resolve_reanchor(self, accept):
        """双确认③结果回写：接受 → 清锚+清历史走重锁；否决 → 重置计数继续攒。"""
        if accept:
            print("    [anchor] 挪盘双确认通过 → 清锚重锁", flush=True)
            self.anchor, self._anchor_buf, self._anchor_win, self._anchor_rej = None, [], [], []
            self.n_reanchor += 1
            self.prev_corners = None
            self.jump_candidate = None
            self.jump_streak = 0
        else:
            self._anchor_rej = []

    def set_imu_mode(self, mode):
        """IMU 三态写入（PC 模拟开关）：static=锚闸 D1+救援仅一级；moving=救援二级
        （无参考重选，移动重锁）；off=UNKNOWN（传感器缺失，行为退回 S7）。离开静止清锚。"""
        if self.imu_static and mode != "static":
            self.anchor, self._anchor_buf, self._anchor_win, self._anchor_rej = None, [], [], []
        self.imu_static = (mode == "static")
        self.imu_moving = (mode == "moving")

    def _anchor_check(self, corners):
        """锚闸前半（detect 内、跳变闸之前）：静止时四角不可能真移——相对锚任一角
        漂移 >0.2 格即伪（手遮挡掩码边界跟手走），拒绝且不污染 EMA/跳变闸基线。
        **静止态永不自动重锚**（各版本仲裁均已证伪，见类常量注释）——锚失效路径唯一：
        IMU 离开静止（set_imu_mode 清锚）或 unlock（reset_tracking_state）。
        返回 False = 拒绝本帧。"""
        if self.anchor is None:
            return True                      # 无锚：不拦（建锚在 _anchor_feed）
        cell = self._cell_size(self.anchor)
        d = float(np.linalg.norm(corners - self.anchor, axis=1).max())
        if d <= self.ANCHOR_CORNER * cell:
            self._anchor_rej = []
            return True
        self._anchor_rej.append(np.asarray(corners, np.float64).copy())
        self.n_anchor_rej += 1
        return False

    def _anchor_feed(self, corners):
        """锚闸后半（跳变闸通过后喂锚）。两阶段：
        无锚 → 建锚：连续 SETTLE 个互偏 <JITTER → 锚 = 组中位数（窗初始化为这组）。
        有锚 → 滚动维护：被接受检测进窗（cap ROLL），锚 = 窗中位数——跟随 SAM 慢漂
        （box 随棋子走子呼吸），中位数抗 ≤3 帧突变；显示取锚（恒定），慢漂不进显示。"""
        c = np.asarray(corners, np.float64).copy()
        if self.anchor is None:
            if self._anchor_buf:
                m = np.median(np.array(self._anchor_buf), 0)
                if float(np.linalg.norm(c - m, axis=1).mean()) > self.ANCHOR_JITTER:
                    self._anchor_buf = []    # 不一致：清零重数
            self._anchor_buf.append(c)
            if len(self._anchor_buf) >= self.ANCHOR_SETTLE:
                self._anchor_win = list(self._anchor_buf)
                self.anchor = np.median(np.array(self._anchor_win), 0)
                self._anchor_buf = []
                print(f"    [anchor] 建锚 {np.round(self.anchor).astype(int).tolist()}", flush=True)
            return
        win = self._anchor_win
        win.append(c)
        if len(win) > self.ANCHOR_ROLL:
            win.pop(0)
        new = np.median(np.array(win), 0)
        if float(np.abs(new - self.anchor).max()) > 2.0:
            print(f"    [anchor] 滚动更新 {np.round(self.anchor).astype(int).tolist()} → "
                  f"{np.round(new).astype(int).tolist()}", flush=True)
        self.anchor = new

    def _select(self, masks, nw, nh, w, h, ref_pts=None):
        """候选级校验与选择（P3-a + C4 + C7①质心包含门，与端上 selectCandidate 同构）。

        C7① 质心包含门（2026-10-11 video_01 诊断引入，**常态过滤**）：手遮挡使 SAM
        掩码沿手碎裂（右半板丢失、TR 角偏 300-500px），残缺四边形能骗过面积带/非退化/
        贴边/形状闸，且手停留期间逐帧一致、连跳变闸 3 帧一致都骗过；但残缺四边形包不住
        右半棋盘的参考点（上帧棋子质心）→ 包含率 <0.85 直接拒 → 迟滞保持上一好角，
        手离开后干净重锁（同时斩断「EMA 污染→仲裁偏好坏候选」的恶性循环）。
        无历史时在达标候选中取面积最小（真棋盘=包住全部棋子的最紧四角）。
        返回 (corners, recon, comp, area) 或 None。
        """
        cb_w, cb_h = nw / 4.0, nh / 4.0      # 图像内容在 256 掩码画布中的边界
        mask_size = masks.shape[1] * masks.shape[2]
        lo, hi = int(mask_size * 0.02), int(mask_size * 0.35)
        cands = []
        for i in range(masks.shape[0]):
            a = self._largest_component_area(masks[i] > 0, masks.shape[2], masks.shape[1])
            if lo <= a <= hi:
                cands.append((a, i))
        cands.sort(reverse=True)                     # 面积降序 = 原优先序
        if not cands:
            return None
        use_ref = ref_pts is not None and len(ref_pts) >= self.MIN_REF_PTS
        chosen = None                                # (corners, recon, comp, area)
        for pass_recon in (False, True):             # P3-a：先全可见候选，后外推复原
            acceptables = []
            for a, i in cands:
                quad_mask, recon, comp = self._extract_corners_ex(masks[i] > 0, cb_w, cb_h)
                if quad_mask is None:
                    continue
                corners = quad_mask * (1024.0 / masks.shape[1])   # mask→1024
                corners = corners * (np.array([w / nw, h / nh]))  # 1024→原图
                corners = corners.astype(np.float32)
                if not self._is_valid_quad(corners, w, h):
                    continue
                if self.shape_gate and not self._has_board_shape(corners):
                    continue
                if use_ref and self._containment(corners, ref_pts) < self.CONTAIN_MIN:
                    continue                         # C7①：残缺/溢出候选包不住参考点，拒
                if pass_recon != recon:
                    continue
                if not recon and self._is_cut_off(corners, w, h):
                    continue                         # 全可见：贴边=掩码被画面裁，拒
                if recon and (corners[:, 0].min() < -w * 0.5 or corners[:, 0].max() > w * 1.5 or
                              corners[:, 1].min() < -h * 0.5 or corners[:, 1].max() > h * 1.5):
                    continue                         # 复原：允许出画面，但不许远于半幅
                acceptables.append((corners, recon, comp, a))
            if acceptables:
                if self.prev_corners is None:
                    if use_ref:
                        # C7①：先包含率降序（外扩型 cont=1.0 但面积大不占优），再面积升序
                        # （残缺型面积小但 cont<0.85 已被门拒）——同时治溢出与内缩。
                        chosen = min(acceptables,
                                     key=lambda c: (-self._containment(c[0], ref_pts),
                                                    self._quad_area(c[0])))
                    else:
                        chosen = acceptables[0]      # 无历史无参考：面积序第一（原行为）
                else:
                    # P3-a 时间稳定性仲裁：棋盘不瞬移，取与上一帧四角平均位移最小者。
                    # 溢出/碎块掩码四角逐帧乱跳，即使面积更大也会输给稳定真棋盘。
                    ref = self.prev_corners
                    chosen = min(acceptables,
                                 key=lambda c: float(np.hypot(*(c[0] - ref).T).mean()))
                break
        return chosen

    def reset_tracking_state(self):
        """C：锁定丢失时清时间状态（与端上同）。"""
        self.prev_corners = None
        self.jump_candidate = None
        self.jump_streak = 0

    # ---- 主检测 ----
    def detect(self, img, ref_pts=None):
        """@param ref_pts 上帧参考点（棋子质心，图像坐标）——C7① 质心包含门用。"""
        h, w = img.shape[:2]
        # 历史失效保护（2026-10-11 v2，video_01 教训修订）：仅在有**移动证据**时清历史——
        # ① 兜底：连续失败 ≥5（真移动长丢失）；② 救援成功时（提示点陈旧=板已移走）。
        # 手遮挡场景（video_01）保持 prev：恢复帧由时间仲裁+跳变闸正常衔接，且 C7①
        # 包含门会拒掉残缺候选——此前「失败≥2 就清」在遮挡场景误清历史，导致恢复帧
        # 走无历史路径直接采纳坏候选，放大角点跳变。
        if self.fail_streak >= 5 and self.prev_corners is not None and self.relock_fixes:
            self.prev_corners = None
            self.jump_candidate = None
            self.jump_streak = 0
        emb, nw, nh, scale = self._encode(img)
        px, py = self.prompt_x * nw, self.prompt_y * nh
        # C7③ box 提示：有参考点（棋子质心/节点，物理位置）时用 box+点混合提示——
        # 暗对暗物理边场景点提示掩码会「改跟」内部高对比线（video_01 TR 内凹根因），
        # box 直接约束物体范围。box 来源必须是物理参考（prev 四角会自我强化下陷）。
        box = self._box_hint(w, h, ref_pts) if self.relock_fixes else None
        if box is not None:
            scores, masks = self._decode_box(emb, box, px, py, w, h)
        else:
            scores, masks = self._decode(emb, px, py)
        chosen = self._select(masks, nw, nh, w, h, ref_pts)
        self._last_rescue = False
        # 帧内中心提示救援（移动重锁修复 A）：连续失败说明回灌提示点已陈旧（棋盘移走
        # → decoder 输出碎片噪声）。候选全拒且非中心提示时，用画面中心重跑 decoder
        # 单帧自愈；救援成功 = 移动证据，清历史（跳变闸基线已陈旧）。
        if (chosen is None and self.fail_streak >= 1 and self.relock_fixes
                and (self.prompt_x != 0.5 or self.prompt_y != 0.5)):
            scores, masks = self._decode(emb, nw / 2.0, nh / 2.0)
            # 分级救援（2026-10-12 修订 + 10-13 挪盘探针）：①带参考选（遮挡期参考
            # 仍有效，包含门挡残缺候选）；②无参考重选仅在两种情形——imu_moving（板已
            # 移走、参考陈旧，移动重锁）或**静止挪盘探针**（锚拒绝攒满 REANCHOR_ASK：
            # 此时若棋盘真被挪，中心提示掩码落在新位置、旧参考包不住 → 只有绕参考才能
            # 拿到新位置候选，它会被锚闸拒绝进入超时候选，再经双确认裁决；手遮挡段的
            # 探针垃圾候选过不了「整体平移」几何确认，无害）。
            chosen = self._select(masks, nw, nh, w, h, ref_pts)
            probe = (self.imu_static and self.anchor is not None
                     and len(self._anchor_rej) >= self.REANCHOR_ASK)
            if chosen is None and (self.imu_moving or probe):
                chosen = self._select(masks, nw, nh, w, h, None)
            if chosen is not None:
                self._last_rescue = True
                if self.prev_corners is not None:
                    self.prev_corners = None
                    self.jump_candidate = None
                    self.jump_streak = 0
                print("    [center-prompt rescue]", flush=True)
        if chosen is None:
            self.fail_streak += 1
            return None
        self.fail_streak = 0
        corners, recon, comp, area = chosen

        # C7② 网格线精修（2026-10-12 video_01 根治手段）：暗桌+深色边框条使物理边
        # 对比度极低，掩码边界常「改跟」内部高对比线（边框内沿/网格线/绕棋子）→
        # 顶部/TR 系统性内凹 40-150px，闸门类手段治不了边界语义错误。网格线高对比、
        # 无二义、直接定义几何：把候选四角 rectify 后用真实网格线重拟合
        # （refine.py 移植复用，2 轮迭代）。rms 未收敛/变差则保留原角（宁缺毋滥）。
        if self.grid_refine:
            try:
                res = refine_corners(
                    img, (corners.astype(np.float64) / np.array([w, h], np.float64)),
                    iters=2, match_radius=30.0, ransac_reproj=6.0)
                if res["grid_rms"] < self.GRID_RMS_MAX and res["n_nodes"] >= 12:
                    corners = (res["corners_norm"] * np.array([w, h], np.float64)).astype(np.float32)
                    self._last_refined = True
                else:
                    self._last_refined = False
            except Exception:
                self._last_refined = False
        else:
            self._last_refined = False

        self._last_recon = recon
        self._last_area = area
        # D1 静止锚闸（前半，跳变闸之前）：静止时相对锚漂移 >0.4 格 = 伪成功，拒。
        # 放 EMA/跳变闸之前 → 拒绝帧不污染 prevCorners 基线；按漏检走迟滞保持。
        if self.imu_static and not self._anchor_check(corners):
            self.fail_streak += 1
            return None
        # P0-a 时间一致性闸
        if self.prev_corners is not None and self._is_jump(corners, self.prev_corners):
            consistent = (self.jump_candidate is not None
                          and not self._is_jump(corners, self.jump_candidate))
            self.jump_streak = self.jump_streak + 1 if consistent else 1
            self.jump_candidate = corners.copy()
            if self.jump_streak < self.JUMP_ACCEPT_STREAK:
                return None
            self.jump_streak, self.jump_candidate = 0, None
        else:
            self.jump_streak, self.jump_candidate = 0, None
        # D1 静止锚闸（后半）：跳变闸也过了的干净四角才喂锚（防坏帧污染锚）。
        if self.imu_static:
            self._anchor_feed(corners)
        # 提示点回灌（复原态用可见掩码质心，更稳）
        if recon:
            ys, xs = np.nonzero(comp)
            cx = xs.mean() * (1024.0 / masks.shape[1]) * (w / nw)
            cy = ys.mean() * (1024.0 / masks.shape[1]) * (h / nh)
            self.prompt_x = float(np.clip(cx / w, 0.02, 0.98))
            self.prompt_y = float(np.clip(cy / h, 0.02, 0.98))
        else:
            cen = corners.mean(0)
            self.prompt_x = float(np.clip(cen[0] / w, 0.02, 0.98))
            self.prompt_y = float(np.clip(cen[1] / h, 0.02, 0.98))
        # 静止锚显示模式（用户标准：静止场景四角应完全一致）：显示=锚（恒定），
        # 检测只是锚的维护者（滚动中位数，见 _anchor_feed）——检测自身的 ±20-40px
        # 分钟级慢漂（box 随棋子走子呼吸）不进显示；prev 同步为锚保持一致性。
        if self.imu_static and self.anchor is not None:
            out = self.anchor.astype(np.float32)
            self.prev_corners = out.astype(np.float64).copy()
            return self._build_info(out)
        smoothed = self._smooth(corners, w, h).astype(np.float32)
        return self._build_info(smoothed)

    def _build_info(self, corners):
        h2i = h_compute(PHYSICAL_CORNERS, corners)      # canonical→image
        return {"corners": corners.astype(np.float32),
                "imageToGrid": h_compute(corners, PHYSICAL_CORNERS),
                "gridToImage": h2i}

    def refine_quad(self, refined, w, h):
        refined = np.asarray(refined, np.float32)
        if not self._is_valid_quad(refined, w, h) or self._is_cut_off(refined, w, h):
            return None
        if self.prev_corners is not None and self._is_jump(refined, self.prev_corners):
            return None
        smoothed = self._smooth(refined, w, h).astype(np.float32)
        return self._build_info(smoothed)


# ================================================================ BoardTracker（C9-T1 视觉帧间传播）

class BoardTracker:
    """LK 光流帧间单应传播（算法梳理 §9 T1；「棋盘不动、视角动」先验）。

    帧间要解的只是相机位姿 = 图像间单应：
      goodFeaturesToTrack（棋盘 ROI 掩码内）→ calcOpticalFlowPyrLK 前向+后向校验
      → RANSAC findHomography → H_g2i ← H_lk @ H_g2i（棋盘系→当前图像系的累积单应）
    每帧输出传播四角 = H_g2i ∙ PHYSICAL_CORNERS（~10ms → 30fps 显示）。
    累积漂移由 EdgeSAM 定周期重锚纠正（REANCHOR_EVERY）；跟踪质量降（内点不足/
    FB 误差大/单应退化/点数耗尽）立即触发重锚。遮挡期间跟踪继续（可见部分特征
    足以拟合全局单应，运动棋子/手部点被 RANSAC 剔为外点）——这正是逐帧分割做不到的。
    """

    REANCHOR_EVERY = 15      # 跟踪健康时的定周期重锚（帧）≈0.5s，纠正累积漂移（25 时斜视角段传播小漂移显效，f426/f476/f560）
    RECHECK_AFTER_FAIL = 5   # 重锚失败后的重试间隔（跟踪仍活着时不背靠背烧 SAM）
    STALE_AGE = 75           # 传播陈旧逃生：超过此帧数未过准入闸时，无条件信 SAM 并重建参考
    CORRECT_EVERY = 10       # 棋锚纠偏周期（帧）：重锚被闸拦下期间用棋子真值钉住 H（2026-10-14）
    CORRECT_BACKING = 30     # 近期纠偏成功过的窗口（帧）：此窗口内传播有棋子背书，不走陈旧逃生
    AGREE_CELL = 1.5         # 准入闸 v3：棋锚背书时 SAM 与传播四角的逐角最大容差（格）
    DBG_CORRECT = False      # 调试：打印每次纠偏尝试的结果（含失败）

    def __init__(self, max_points=250, min_distance=12, fb_max=2.5,
                 min_track_pts=25, min_inliers=35, ransac_reproj=3.0):
        self.max_points = max_points
        self.min_distance = min_distance
        self.fb_max = fb_max
        self.min_track_pts = min_track_pts
        self.min_inliers = min_inliers
        self.ransac_reproj = ransac_reproj
        self.prev_gray = None
        self.pts = None           # (N,1,2) float32，当前帧跟踪点
        self.H = None             # 3×3 棋盘系(canonical)→当前图像
        self.age = 0              # 距上次成功重锚的帧数
        self.last_correct_age = None  # 最近一次棋锚纠偏成功时的 age（seed 清空）
        self.last_inliers = 0     # 诊断
        self.n_track_frames = 0   # 指标：成功传播帧
        self.n_fail = 0           # 指标：传播失败帧

    def seed(self, gray, corners):
        """EdgeSAM 重锚后播种：H 由检测四角计算；特征点在棋盘 ROI 内重提。"""
        c = np.asarray(corners, np.float64).reshape(4, 2)
        self.H = h_compute(PHYSICAL_CORNERS, c)
        h, w = gray.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        # ROI = 四角多边形外扩 ~0.35 格（棋盘内纹理为主，略含边框）
        cell = EdgeSamDetector._cell_size(c) if hasattr(EdgeSamDetector, "_cell_size") else 60.0
        pad = 0.35 * cell
        roi = c.copy()
        mid = c.mean(0)
        roi = (c - mid) * (1 + 2 * pad / max(np.hypot(*(c - mid).T).max(), 1)) + mid
        cv2.fillPoly(mask, [np.round(roi).astype(np.int32)], 1)
        pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.max_points, qualityLevel=0.01,
            minDistance=self.min_distance, blockSize=7, mask=mask)
        self.pts = pts
        self.prev_gray = gray
        self.age = 0
        self.last_correct_age = None

    def corners(self):
        if self.H is None:
            return None
        return h_transform(self.H, PHYSICAL_CORNERS)

    def ref_to_image(self, ref_board):
        """棋盘系参考点（棋子质心）→ 当前图像系（移动场景包含门/box 的关键）。"""
        if self.H is None or ref_board is None or len(ref_board) == 0:
            return None
        return h_transform(self.H, np.asarray(ref_board, np.float64))

    def track(self, gray):
        """逐帧传播。返回 True=成功（corners() 可用）/ False=质量不足（应重锚）。"""
        if self.H is None or self.prev_gray is None or self.pts is None \
                or len(self.pts) < self.min_track_pts:
            self.n_fail += 1
            return False
        lk = dict(winSize=(31, 31), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.pts, None, **lk)
        if p1 is None:
            self.n_fail += 1
            return False
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, p1, None, **lk)
        good = (st.reshape(-1) == 1)
        if st2 is not None:
            good &= (st2.reshape(-1) == 1)
        if good.sum() >= 8:
            fb = np.linalg.norm(p0r[good] - self.pts[good], axis=2).reshape(-1)
            good[np.where(good)[0][fb >= self.fb_max]] = False
        if good.sum() < self.min_track_pts:
            self.pts = None
            self.n_fail += 1
            return False
        src = self.pts[good].reshape(-1, 1, 2)
        dst = p1[good].reshape(-1, 1, 2)
        Hlk, inl = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_reproj)
        inl_n = 0 if inl is None else int(inl.sum())
        det = 0.0 if Hlk is None else float(np.linalg.det(Hlk[:2, :2]))
        if Hlk is None or inl_n < self.min_inliers or not (0.25 <= det <= 4.0) \
                or not np.all(np.isfinite(Hlk)):
            self.pts = None
            self.n_fail += 1
            return False
        self.H = Hlk @ self.H
        c = self.corners()
        h, w = gray.shape[:2]
        if (not np.all(np.isfinite(c)) or c[:, 0].min() < -w * 0.5 or c[:, 0].max() > w * 1.5
                or c[:, 1].min() < -h * 0.5 or c[:, 1].max() > h * 1.5):
            self.H = None
            self.pts = None
            self.n_fail += 1
            return False
        self.pts = dst
        self.prev_gray = gray
        self.age += 1
        self.last_inliers = inl_n
        self.n_track_frames += 1
        return True

    @staticmethod
    def _unique_nearest(ref, proj, det_pts, radius):
        """就近唯一匹配：投影参考 ↔ 最近质心（≤radius），距离升序贪心，一对一。"""
        pairs = []
        for i, p in enumerate(proj):
            dd = np.linalg.norm(det_pts - p, axis=1)
            j = int(dd.argmin())
            if dd[j] <= radius:
                pairs.append((float(dd[j]), i, j))
        pairs.sort()
        used_i, used_j, src, dst = set(), set(), [], []
        for _, i, j in pairs:
            if i in used_i or j in used_j:
                continue
            used_i.add(i)
            used_j.add(j)
            src.append(ref[i])
            dst.append(det_pts[j])
        return src, dst

    def piece_correct(self, yolo, img, ref_board):
        """棋子锚定纠偏（2026-10-14，video_03 f426/f476 诊断）。

        重锚被准入闸连续拦下的快速移动段，LK 累积漂移无人纠正（f426 BR 外扩
        30-40px，f491 漂移 127px→f527 749px）。棋子是独立于 SAM/LK 的第三方
        真值源：ref_board（棋盘系质心，静止物理真值）↔ 当前帧 YOLO 质心 →
        RANSAC 单应（内点 0.35 格）直接钉住 H——视频标注期「棋子锚定 RANSAC」
        同款技术。两级匹配：
          ① 就近唯一匹配（≤0.75 格）：漂移 <0.75 格时的快速路径；
          ② PCA 粗对齐（免对应，video_label_yoloassist 同款）：两团点主轴 8 种
             符号/换轴组合的相似变换对齐后吸附质心——漂移任意大都能恢复
             （track5 教训：就近匹配捕获范围只有 0.75 格，拉近段 300px 漂移
             恰好让它最需要时失灵）。镜像歧义由 det>0 闸拒绝（反射单应 det<0）。
        成功返回纠偏量 px（四角均值位移），失败返回 None（保持传播 H）。
        """
        if self.H is None or not ref_board or len(ref_board) < 12:
            return None
        ref = np.asarray(ref_board, np.float64).reshape(-1, 2)
        corners = self.corners()
        dets, _ = yolo.detect(img, corners)
        if len(dets) < 12:
            # 大漂移时按漂移四角的裁剪会切掉棋子——外扩 1.5 倍重跑一次兜底
            mid = np.asarray(corners, np.float64).reshape(4, 2).mean(0)
            expanded = (np.asarray(corners, np.float64).reshape(4, 2) - mid) * 1.5 + mid
            dets, _ = yolo.detect(img, expanded)
        if len(dets) < 12:
            return None
        cell = float(EdgeSamDetector._cell_size(corners))
        det_pts = np.asarray([[d[0], d[1]] for d in dets], np.float64)
        proj = h_transform(self.H, ref)
        src, dst = self._unique_nearest(ref, proj, det_pts, 0.75 * cell)
        fits = [(np.asarray(src), np.asarray(dst))] if len(src) >= 12 else []
        if not fits:
            # ② PCA 粗对齐候选对（按匹配数降序，逐一试拟合）
            def pca(x):
                m = x.mean(0)
                c = (x - m).T @ (x - m) / len(x)
                w, v = np.linalg.eigh(c)
                o = np.argsort(w)[::-1]
                return m, v[:, o], w[o]
            ms, vs, ws = pca(ref)
            md, vd, wd = pca(det_pts)
            cand = []
            for swap in (False, True):
                vs2 = vs[:, ::-1] if swap else vs
                ws2 = ws[::-1] if swap else ws
                for f0 in (1.0, -1.0):
                    for f1 in (1.0, -1.0):
                        R = vd @ np.diag([f0, f1]) @ vs2.T
                        scale = np.sqrt(np.clip(wd, 1e-6, None)
                                        / np.clip(ws2, 1e-6, None))
                        p = md + (ref - ms) @ R.T * scale
                        s2, d2 = self._unique_nearest(ref, p, det_pts, 1.2 * cell)
                        if len(s2) >= 12:
                            cand.append((len(s2), s2, d2))
            cand.sort(key=lambda t: -t[0])
            fits = [(np.asarray(s2), np.asarray(d2)) for _, s2, d2 in cand]
        for src, dst in fits:
            Hf, inl = cv2.findHomography(src, dst, cv2.RANSAC, 0.35 * cell)
            n_in = 0 if inl is None else int(inl.sum())
            det = 0.0 if Hf is None else float(np.linalg.det(Hf[:2, :2]))
            det0 = float(np.linalg.det(self.H[:2, :2]))
            # 尺度闸（分辨率无关，2026-10-15 真机教训）：相对当前 H 的比值 ∈[0.5,2]——
            # 绝对闸 [0.25,4] 是 1080p 大棋盘标定，480p 手机棋盘 ~200px 合法 det≈0.1。
            if Hf is None or n_in < 10 or not (det0 > 0 and 0.5 <= det / det0 <= 2.0) \
                    or not np.all(np.isfinite(Hf)):
                continue
            # 旋转一致闸（2026-10-14 移植前夜发现）：标准开局点云 180° 旋转对称 →
            # PCA 180° 组合把全部点匹配到镜像对位（count 满分、det>0 闸拦不住），
            # 拟合出翻转 H。棋盘 10 帧纠偏周期内不可能转 40°+ → 轴向夹角闸拦翻转。
            mid = np.array([[np.mean(PHYSICAL_CORNERS[:, 0]),
                             np.mean(PHYSICAL_CORNERS[:, 1])]])
            ax = (h_transform(Hf, mid + np.array([[100.0, 0.0]]))
                  - h_transform(Hf, mid)).ravel()
            ax0 = (h_transform(self.H, mid + np.array([[100.0, 0.0]]))
                   - h_transform(self.H, mid)).ravel()
            cosang = float(np.dot(ax, ax0) / (np.linalg.norm(ax) * np.linalg.norm(ax0) + 1e-9))
            if cosang < np.cos(np.deg2rad(40.0)):
                continue
            shift = float(np.linalg.norm(
                h_transform(Hf, PHYSICAL_CORNERS) - corners, axis=1).mean())
            self.H = Hf
            self.last_correct_age = self.age
            return shift
        return None


# ================================================================ QuadRefiner（QuadRefiner.kt 移植）

class QuadRefinerPy:
    MIN_CENTROIDS, SNAP_CELL, INLIER_CELL = 12, 0.8, 0.35
    RANSAC_ITERS, MAX_MOVE, MIN_INLIERS = 500, 1.5, 16
    SPREAD_F, SPREAD_R = 6, 6

    @staticmethod
    def refine(corners, centroids):
        import random
        corners = np.asarray(corners, np.float32)
        if len(centroids) < QuadRefinerPy.MIN_CENTROIDS:
            return None
        h0 = h_compute(PHYSICAL_CORNERS, corners)
        nodes = h_transform(h0, np.array(
            [[MARGIN + f * CELL, MARGIN + r * CELL] for r in range(10) for f in range(9)],
            np.float64))
        spac = np.sort([np.hypot(*(nodes[r * 9 + f + 1] - nodes[r * 9 + f]))
                        for r in range(10) for f in range(8)])
        cell = spac[len(spac) // 2]
        if cell <= 0:
            return None
        dets = np.asarray(centroids, np.float64)
        pair_node, pair_cent, node_fr = [], [], []
        for c in dets:
            i = int(np.argmin(np.hypot(nodes[:, 0] - c[0], nodes[:, 1] - c[1])))
            if np.hypot(*(nodes[i] - c)) > QuadRefinerPy.SNAP_CELL * cell:
                continue
            pair_node.append([MARGIN + (i % 9) * CELL, MARGIN + (i // 9) * CELL])
            pair_cent.append(c)
            node_fr.append((i % 9, i // 9))
        if len(pair_node) < QuadRefinerPy.MIN_CENTROIDS:
            return None
        S = np.array(pair_node, np.float64)
        D = np.array(pair_cent, np.float64)
        thr = QuadRefinerPy.INLIER_CELL * cell
        rng = random.Random(42)
        n = len(S)
        input_area = EdgeSamDetector._quad_area(corners.astype(np.float64))

        def valid(h):
            pc = h_transform(h, PHYSICAL_CORNERS)
            if np.any(~np.isfinite(pc)):
                return False
            if np.max(np.hypot(*(pc - corners).T)) > QuadRefinerPy.MAX_MOVE * cell:
                return False
            ratio = EdgeSamDetector._quad_area(pc) / input_area
            return 0.7 <= ratio <= 1.4

        def evaluate(h):
            proj = h_transform(h, S)
            dist = np.linalg.norm(proj[:, None, :] - D[None, :, :], axis=2)
            j, near = dist.argmin(1), dist.min(1)
            inl = near < thr
            # 一对一贪心（按距离升序）
            pairs = sorted([(dist[k, j[k]], k, j[k]) for k in range(n) if inl[k]])
            used_n, used_c, fs, rs = set(), set(), set(), set()
            for _, k, jj in pairs:
                if k in used_n or jj in used_c:
                    continue
                used_n.add(k); used_c.add(jj)
                fs.add(node_fr[k][0]); rs.add(node_fr[k][1])
            return len(used_c), inl, j, len(fs), len(rs)

        best_h, best_score = None, -1
        for _ in range(QuadRefinerPy.RANSAC_ITERS):
            idx = rng.sample(range(n), 4)
            try:
                h = h_compute(S[idx], D[idx])
            except Exception:
                continue
            if not np.all(np.isfinite(h)) or not valid(h):
                continue
            sc = evaluate(h)[0]
            if sc > best_score:
                best_h, best_score = h, sc
        if best_h is None or best_score < QuadRefinerPy.MIN_INLIERS:
            return None
        sc, _, _, nf, nr = evaluate(best_h)
        if sc < QuadRefinerPy.MIN_INLIERS or nf < QuadRefinerPy.SPREAD_F or nr < QuadRefinerPy.SPREAD_R:
            return None
        return h_transform(best_h, PHYSICAL_CORNERS).astype(np.float32)


# ================================================================ YOLO 检测器（YoloPieceDetector.kt 移植）

class YoloDetectorPy:
    CONF, IOU, PAD_R, IMG = 0.25, 0.45, 0.06, 640

    def __init__(self):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(YOLO_ONNX, providers=["CPUExecutionProvider"])

    def detect(self, img, corners):
        """返回 (dets[(cx,cy,x1,y1,x2,y2,cls,conf)], grid 10×9 (ordinal,color) or None)。"""
        H, W = img.shape[:2]
        xs, ys = np.asarray(corners)[:, 0], np.asarray(corners)[:, 1]
        minx, maxx = xs.min(), xs.max()
        miny, maxy = ys.min(), ys.max()
        bw, bh = maxx - minx, maxy - miny
        if bw <= 0 or bh <= 0:
            return [], None
        l = int(np.clip(minx - bw * self.PAD_R, 0, W - 1))
        t = int(np.clip(miny - bh * self.PAD_R, 0, H - 1))
        r_ = int(np.clip(maxx + bw * self.PAD_R, 0, W))
        b = int(np.clip(maxy + bh * self.PAD_R, 0, H))
        if r_ - l <= 0 or b - t <= 0:
            return [], None
        crop = img[t:b, l:r_]
        ch, cw = crop.shape[:2]
        rr = self.IMG / max(cw, ch)
        nw, nh = max(1, round(cw * rr)), max(1, round(ch * rr))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.IMG, self.IMG, 3), 114, np.uint8)
        ox, oy = int((self.IMG - nw) / 2), int((self.IMG - nh) / 2)
        canvas[oy:oy + nh, ox:ox + nw] = resized
        x = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))[None]
        out = self.sess.run(None, {"images": x})[0][0]     # [18,8400]
        boxes_raw, classes, scores = [], [], []
        stride = out.shape[1]
        for a in range(stride):
            cl = out[4:, a]
            c = int(cl.argmax())
            if cl[c] <= self.CONF:
                continue
            cx, cy, w2, h2 = out[0, a], out[1, a], out[2, a], out[3, a]
            x1 = (cx - w2 / 2 - ((self.IMG - nw) / 2)) / rr + l
            y1 = (cy - h2 / 2 - ((self.IMG - nh) / 2)) / rr + t
            x2 = (cx + w2 / 2 - ((self.IMG - nw) / 2)) / rr + l
            y2 = (cy + h2 / 2 - ((self.IMG - nh) / 2)) / rr + t
            boxes_raw.append([x1, y1, x2, y2]); classes.append(c); scores.append(float(cl[c]))
        # 按类 NMS（Kotlin 枚举序先到先得）
        keep = [True] * len(boxes_raw)
        for i in range(len(boxes_raw)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(boxes_raw)):
                if not keep[j] or classes[i] != classes[j]:
                    continue
                xx1 = max(boxes_raw[i][0], boxes_raw[j][0]); yy1 = max(boxes_raw[i][1], boxes_raw[j][1])
                xx2 = min(boxes_raw[i][2], boxes_raw[j][2]); yy2 = min(boxes_raw[i][3], boxes_raw[j][3])
                inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                aA = (boxes_raw[i][2] - boxes_raw[i][0]) * (boxes_raw[i][3] - boxes_raw[i][1])
                aB = (boxes_raw[j][2] - boxes_raw[j][0]) * (boxes_raw[j][3] - boxes_raw[j][1])
                union = aA + aB - inter
                if union > 0 and inter / union > self.IOU:
                    keep[j] = False
        dets = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[0], b[1], b[2], b[3], c, s)
                for b, c, s, k in zip(boxes_raw, classes, scores, keep) if k]
        dets.sort(key=lambda d: -d[7])
        return dets, None


def snap_to_grid(dets, info):
    """置信度降序就近取整吸附先到先得（recognize 第 6 步）。"""
    grid = [[None] * 9 for _ in range(10)]
    h = info["imageToGrid"]
    for d in dets:
        p = h_transform(h, [[d[0], d[1]]])[0]
        f = int(round((p[0] - MARGIN) / CELL))
        r = int(round((p[1] - MARGIN) / CELL))
        if 0 <= f <= 8 and 0 <= r <= 9 and grid[r][f] is None:
            c = d[6]
            grid[r][f] = (TYPE_ORDINALS[c % 7], c <= 6)
    return grid


# 标准开局参考点（棋盘系交点真值）：开局时 32 子的棋盘坐标完全已知——
# 用它当 ref_board 可把「约定」钉到真实物理位置（不依赖任何检测四角），
# 纠正种子级偏差（track7 教训：f41 种子 BR 外扩 25-40px 被棋锚纠偏忠实继承
# 到全程）。任意一子走动即产生越位 → matches_std_opening 自动失效，中局回退
# 质心投影约定（= 最近一次被接受检测的约定）。
_STD_CELLS = ([(f, 0) for f in range(9)]              # 黑方底线：车马象士将士象马车（9 列全占）
              + [(1, 2), (7, 2)]                        # 黑炮
              + [(f, 3) for f in (0, 2, 4, 6, 8)]      # 黑卒
              + [(f, 6) for f in (0, 2, 4, 6, 8)]      # 红兵
              + [(1, 7), (7, 7)]                        # 红炮
              + [(f, 9) for f in range(9)])             # 红方底线（9 列全占）
STD_REF_BOARD = [[MARGIN + f * CELL, MARGIN + r * CELL] for f, r in _STD_CELLS]
_STD_OCC_ARR = np.array([[1 if (f, r) in _STD_CELLS else 0
                          for f in range(9)] for r in range(10)])


def matches_std_opening(grid):
    """几何性判据：≥20 个子落在标准开局交点上（extra 容忍 4——biased 四角下
    个别子 snap 偏 1 格 / 幻影框 / 走子都不影响位置拟合：越位子只是 RANSAC
    外点，身份与拟合无关）。走子多了 on_std 自然跌破 20 → 中局自动失效。"""
    occ = np.array([[1 if grid[r][f] else 0 for f in range(9)] for r in range(10)])
    on_std = int(((occ == 1) & (_STD_OCC_ARR == 1)).sum())
    extra = int(((occ == 1) & (_STD_OCC_ARR == 0)).sum())
    return on_std >= 20 and extra <= 4


# ================================================================ 投票 / FEN / 记谱（MainActivity 移植）

def vote_grid(history):
    out = [[None] * 9 for _ in range(10)]
    for r in range(10):
        for f in range(9):
            counts = {}
            for g in reversed(history):     # 新→旧，平票取较新
                p = g[r][f]
                if p is not None:
                    counts[p] = counts.get(p, 0) + 1
            if counts:
                out[r][f] = max(counts.items(), key=lambda kv: kv[1])[0]
    return out


def grid_to_fen(grid):
    parts = []
    for r in range(10):
        s, empty = "", 0
        for f in range(9):
            p = grid[r][f]
            if p is None:
                empty += 1
            else:
                if empty:
                    s += str(empty); empty = 0
                s += FEN_CHAR[p]
        if empty:
            s += str(empty)
        parts.append(s)
    return "/".join(parts) + " w"


def is_illegal(grid):
    red = [0] * 7; black = [0] * 7
    rg = bg = False
    total = 0
    for r in range(10):
        for f in range(9):
            p = grid[r][f]
            if p is None:
                continue
            total += 1
            t, red_side = p
            cnt, = (red if red_side else black),
            cnt[t] += 1
            if red_side:
                if t == 0:
                    if rg or not (3 <= f <= 5 and 7 <= r <= 9):
                        return True
                    rg = True
                elif t == 1 and not (3 <= f <= 5 and 7 <= r <= 9):
                    return True
                elif t == 2 and not (5 <= r <= 9):
                    return True
                elif t == 6 and not (0 <= r <= 6):
                    return True
            else:
                if t == 0:
                    if bg or not (3 <= f <= 5 and 0 <= r <= 2):
                        return True
                    bg = True
                elif t == 1 and not (3 <= f <= 5 and 0 <= r <= 2):
                    return True
                elif t == 2 and not (0 <= r <= 4):
                    return True
                elif t == 6 and not (3 <= r <= 9):
                    return True
    if total > 32 or not rg or not bg:
        return True
    return any(red[i] > MAX_COUNTS[i] or black[i] > MAX_COUNTS[i] for i in range(7))


CN_NUM = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
RED_FILE = ["九", "八", "七", "六", "五", "四", "三", "二", "一"]


def notation(grid, ff, fr, tf, tr):
    t, red = grid[fr][ff]
    name = PT[t]
    col = (lambda f: RED_FILE[f]) if red else (lambda f: str(f + 1))
    if fr == tr:
        verb, num = "平", col(tf)
    else:
        forward = -1 if red else 1
        verb = "进" if (tr - fr) * forward > 0 else "退"
        num = col(tf) if t in (3, 2, 1) else (CN_NUM[abs(tr - fr)] if red else str(abs(tr - fr)))
    src = col(ff)
    for r in range(10):
        if r == fr:
            continue
        p = grid[r][ff]
        if p is not None and p[0] == t and p[1] == red:
            src = "前" if (fr < r if red else fr > r) else "后"
            break
    return (src + name if src in "前后" else name + src) + verb + num


class Engine:
    def __init__(self):
        self.proc = None
        self.cache = {}

    def best(self, fen):
        if fen in self.cache:
            return self.cache[fen]
        if self.proc is None:
            self.proc = subprocess.Popen(
                [JAVA, "-cp", XQW_CP, "xqwlight.XqwDriver"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                text=True, encoding="utf-8")
        self.proc.stdin.write(fen + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline().strip()
        mv = None
        if line and line != "none":
            p = line.split()
            mv = tuple(int(v) for v in p)
        self.cache[fen] = mv
        return mv


# ================================================================ 渲染

BGR = {"r": (60, 60, 255), "b": (255, 200, 60)}


def render(img, info, grid, dets, move, note, frame_no, state_txt):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil)
    if info is not None:
        c = info["corners"].tolist()
        dr.line([tuple(c[0]), tuple(c[1]), tuple(c[2]), tuple(c[3]), tuple(c[0])],
                fill=(0, 255, 0), width=4)
        for p in c:
            dr.ellipse([p[0] - 7, p[1] - 7, p[0] + 7, p[1] + 7], fill=(0, 255, 0))
    for d in dets:
        cx, cy, x1, y1, x2, y2, cls, conf = d
        color = BGR["r" if cls <= 6 else "b"]
        dr.rectangle([x1, y1, x2, y2], outline=color, width=2)
        dr.text((x1, y1 - 24), f"{CLASS_NAMES[cls]}{conf:.2f}", font=FONT_S, fill=color)
    if move is not None and info is not None:
        ff, fr, tf, tr = move
        pts = h_transform(info["gridToImage"], [
            [MARGIN + ff * CELL, MARGIN + fr * CELL],
            [MARGIN + tf * CELL, MARGIN + tr * CELL]])
        p1, p2 = tuple(pts[0]), tuple(pts[1])
        dr.line([p1, p2], fill=(0, 215, 255), width=6)
        dr.ellipse([p1[0] - 12, p1[1] - 12, p1[0] + 12, p1[1] + 12], outline=(0, 215, 255), width=4)
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        n = np.hypot(dx, dy) or 1
        tip = (p2[0] - dx / n * 22, p2[1] - dy / n * 22)
        dr.polygon([tuple(p2),
                    (tip[0] - dy / n * 12, tip[1] + dx / n * 12),
                    (tip[0] + dy / n * 12, tip[1] - dx / n * 12)], fill=(0, 215, 255))
        dr.text((12, 8), f"红方建议：{note}", font=FONT, fill=(0, 215, 255))
    dr.text((12, 44), f"帧 {frame_no}  {state_txt}", font=FONT_S, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ================================================================ 主循环（MainActivity.processFrame 移植）

DETECT_EVERY_N, LOCK_MAX_MISSES, HISTORY_SIZE = 5, 15, 5


def run(video, out_dir, max_frames=0, imu_mode="off", track=False):
    os.makedirs(out_dir, exist_ok=True)
    edge = EdgeSamDetector()
    # 定稿（2026-10-14 用户看图裁决，A/B=compare_seed_ab）：**单种子+棋锚纠偏**胜出。
    # 曾试「跟踪模式旁路 detect 内部包含门」（理由：真值参考下斜视角 SAM 内收
    # 0.83-0.9<0.95 被内部门全拒→重锚断供）——但旁路后 47px 偏差的 f218 种子被
    # 放进来刷新约定，f476 继承其 TL 偏差，用户判 B 版差。内部门保持开启：
    # 斜视角段 detect 返回 None 不再是问题（棋锚纠偏承担传播精度，SAM 只需
    # 好几何时供种）；tracker 死亡时 ref_img=None 走中心提示全新重锁兜底。
    # grid_refine 同理试过又撤（track11 证伪：斜视角锁错线组，P3 陷阱移动场景
    # 同样成立）。种子约定误差问题见 BoardTracker 方案文档 §0。
    edge.set_imu_mode(imu_mode)   # PC 模拟：恒定三态（真机由 monitor 逐帧驱动）
    yolo = YoloDetectorPy()
    refiner = QuadRefinerPy()
    engine = Engine()
    last_legal, last_legal_n = None, 0   # D3 合法参照（最近一次合法投票局面）
    tracker = BoardTracker() if track else None
    ref_board = None      # 棋盘系参考点（YOLO 质心经 imageToGrid 投影，静止于棋盘坐标系）
    recheck_in = 0        # 跟踪活着但重锚失败后的重试倒计时（不背靠背烧 SAM）

    cap = cv2.VideoCapture(video)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = min(n_total, max_frames) if max_frames else n_total

    frame_counter = 0
    lock_misses = 0
    latest_info, latest_grid, latest_dets = None, None, []
    recent = []
    last_centroids = []
    suggest, suggest_note, last_fen = None, None, None
    t0 = time.time()
    n_out = 0
    while True:
        ok, img = cap.read()
        if not ok or (limit and frame_counter >= limit):
            break
        frame_counter += 1
        state = "保持"
        # 失败/未锁定期间加速采样（移动重锁修复 B，与端上同）：每帧都检。
        every_n = 1 if (latest_info is None or lock_misses > 0) else DETECT_EVERY_N
        # C9-T1 跟踪模式：LK 传播优先承担每帧四角（30fps 显示，遮挡期间继续），
        # EdgeSAM 降频为事件触发重锚（无 H / 传播失败 / 定周期 REANCHOR_EVERY）。
        prop_info = None
        if tracker is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            alive = tracker.track(gray) if tracker.H is not None else False
            if alive:
                # 棋锚纠偏：重锚被闸拦下 / SAM 候选不稳的快速移动段，用棋子真值
                # 钉住 H，把 LK 累积漂移按周期清零（f426/f476 的根治项）。
                if BoardTracker.DBG_CORRECT:
                    print(f"    [age-dbg] f{frame_counter}: age={tracker.age} "
                          f"nref={len(ref_board) if ref_board else 0}", flush=True)
                corr = None
                if ref_board and len(ref_board) >= 12 \
                        and tracker.age % tracker.CORRECT_EVERY == 0:
                    corr = tracker.piece_correct(yolo, img, ref_board)
                    if corr is not None and corr > 8:
                        print(f"    [correct] f{frame_counter}: 棋锚纠偏 "
                              f"{corr:.0f}px", flush=True)
                    elif BoardTracker.DBG_CORRECT:
                        print(f"    [correct-dbg] f{frame_counter}: age={tracker.age} "
                              f"corr={corr}", flush=True)
                c = tracker.corners().astype(np.float32)
                prop_info = {"corners": c,
                             "imageToGrid": h_compute(c, PHYSICAL_CORNERS),
                             "gridToImage": h_compute(PHYSICAL_CORNERS, c)}
                latest_info = prop_info
                state = (f"棋锚纠偏{corr:.0f}px" if corr is not None
                         else f"跟踪{tracker.last_inliers}")
            tracker_due = tracker.H is None or not alive or tracker.age >= tracker.REANCHOR_EVERY
            want_detect = tracker_due and recheck_in <= 0
            if recheck_in > 0:
                recheck_in -= 1
        else:
            want_detect = frame_counter % every_n == 0
        if want_detect:
            # 参考点：跟踪模式用棋盘系参考经当前 H 投回图像（移动场景包含门/box 有效的
            # 关键——图像系参考在相机一动就全体作废）；否则用上帧图像系质心。
            ref_img = (tracker.ref_to_image(ref_board) if tracker is not None
                       else (last_centroids if last_centroids else None))
            detected = edge.detect(img, ref_img)
            # 重锚准入闸（2026-10-14，video_03 诊断）：斜视角段 SAM 系统性内收 1 格的
            # 候选会经 MOVING 态救援二级（绕包含门）混入——但跟踪模式下参考是经当前 H
            # 投影的新鲜参考，绕门理由不成立。SAM 四角须包住 ≥0.9 投影参考才播种/更新
            # 显示，否则丢弃本次检测、显示继续走传播；传播陈旧（≥STALE_AGE 未过闸）
            # 时无条件信 SAM 并重建参考（防快甩后传播走歪卡死）。
            if detected is not None and tracker is not None and prop_info is not None:
                cont0 = (EdgeSamDetector._containment(
                    np.asarray(detected["corners"], np.float64), ref_img)
                    if ref_img is not None and len(ref_img) else 1.0)
                med0 = (EdgeSamDetector._min_edge_dist(detected["corners"], ref_img)
                        if ref_img is not None and len(ref_img) else 0.0)
                med0_cell = med0 / EdgeSamDetector._cell_size(
                    np.asarray(detected["corners"], np.float64)) if ref_img is not None else 0.0
                # v2：包含率拦内收（<0.9），最小边距拦外扩（>1.0 格——真值 ~0.5 格，
                # video_03 f138/f476 的外扩伪框实测 0.7-1.5 格）
                if cont0 < 0.9 or med0_cell > 1.0:
                    # 棋锚背书：近期纠偏成功过 = 传播有棋子真值钉着、不是陈旧，
                    # 拦下的必是坏 SAM → 继续拦（防陈旧逃生用坏 SAM 顶掉好传播）。
                    backed = (tracker.last_correct_age is not None
                              and tracker.age - tracker.last_correct_age
                              <= tracker.CORRECT_BACKING)
                    if tracker.age < tracker.STALE_AGE or backed:
                        d0 = float(np.linalg.norm(
                            np.asarray(detected["corners"], np.float64) - tracker.corners(),
                            axis=1).mean())
                        print(f"    [reanchor] f{frame_counter}: 存疑保持（参考包含率 "
                              f"{cont0:.2f}，最小边距 {med0_cell:.2f} 格，"
                              f"SAM-传播偏差 {d0:.0f}px"
                              f"{'，棋锚背书' if backed else ''}）", flush=True)
                        latest_info = prop_info
                        state = "重锚存疑"
                        detected = None
                    else:
                        print(f"    [reanchor] f{frame_counter}: 陈旧逃生（包含率 {cont0:.2f}）"
                              f"→ 信 SAM 重建参考", flush=True)
                        ref_board = None
            if detected is not None and tracker is not None and prop_info is not None \
                    and ref_img is not None and len(ref_img):
                # 准入闸 v3（2026-10-14 track5/6 教训）：棋锚背书时传播就是棋子真值
                # 投影——SAM 须与传播四角逐角一致（≤AGREE_CELL 格）才准播种。
                # track5/6 实测：纠偏静默生效（漂移 ~3.5px/10帧），f320/f365 的
                # SAM quad 偏 296/664px 仍过 v2 闸（min 统计太弱：一个参考贴边即放行）
                # → 坏种子投毒 ref_board → 纠偏把错误约定自洽钉死、detect 再不成功。
                backed = (tracker.last_correct_age is not None
                          and tracker.age - tracker.last_correct_age
                          <= tracker.CORRECT_BACKING)
                if backed:
                    cell = EdgeSamDetector._cell_size(
                        np.asarray(detected["corners"], np.float64))
                    d_max = float(np.linalg.norm(
                        np.asarray(detected["corners"], np.float64) - tracker.corners(),
                        axis=1).max())
                    if d_max > tracker.AGREE_CELL * cell:
                        print(f"    [reanchor] f{frame_counter}: 棋锚否决"
                              f"（SAM-传播最大角偏差 {d_max:.0f}px > "
                              f"{tracker.AGREE_CELL} 格）", flush=True)
                        latest_info = prop_info
                        state = "重锚存疑"
                        detected = None
            if detected is not None:
                lock_misses = 0
                info = detected
                # P0-b 质心精修：参考必须与当前画面同坐标系——跟踪模式用棋盘系参考经
                # 当前 H 投回（图像系 last_centroids 相机一动即全体作废）。
                refine_pts = (tracker.ref_to_image(ref_board)
                              if tracker is not None and ref_board else last_centroids)
                if refine_pts is not None and len(refine_pts) >= 12:
                    refined = refiner.refine(info["corners"], refine_pts)
                    if refined is not None:
                        info = edge.refine_quad(refined, img.shape[1], img.shape[0]) or detected
                dets, _ = yolo.detect(img, info["corners"])
                last_centroids = [[d[0], d[1]] for d in dets]
                if tracker is not None:
                    # 诊断：SAM 重锚 vs 传播四角的偏差 + 对（投影后）参考的包含率
                    prop_c = tracker.corners()
                    cont = (EdgeSamDetector._containment(np.asarray(info["corners"], np.float64), ref_img)
                            if ref_img is not None and len(ref_img) else 1.0)
                    if prop_c is not None:
                        d = float(np.linalg.norm(
                            np.asarray(info["corners"], np.float64) - prop_c, axis=1).mean())
                        print(f"    [reanchor] f{frame_counter}: SAM-传播偏差={d:.0f}px "
                              f"参考包含率={cont:.2f}", flush=True)
                    tracker.seed(gray, info["corners"])
                grid = snap_to_grid(dets, info)
                recent.append(grid)
                while len(recent) > HISTORY_SIZE:
                    recent.pop(0)
                voted = vote_grid(recent)
                n_pc = sum(1 for r in range(10) for f in range(9) if voted[r][f])
                # D3 投票门 v2（vis3 教训：is_illegal 当拒绝条件会因 YOLO 偶缺一帅逐帧
                # 全拒→冻结；vis7 教训：is_illegal 恒真时参照永不建立→门旁路→>32 幽灵
                # 泄漏）。自举：首个体检合格（20≤子数≤32）的投票即可当参照。
                if last_legal is not None and (n_pc > 32 or n_pc < last_legal_n * 0.6):
                    recent.pop()
                    voted, n_pc = last_legal, last_legal_n
                    state = "投票保持"
                else:
                    state = "检测OK"
                # ref_board 重建：准入闸已保证新检测包住 ≥0.9 旧参考（或陈旧逃生
                # 清空重建）——棋盘系质心坐标是静止物理真值，坏检测进不来。
                # 标准开局约定重置（2026-10-14，track7 教训）：占位匹配标准开局时
                # 改用标准交点真值——纠正种子级偏差（f41 BR 外扩 25-40px 被棋锚
                # 纠偏忠实继承到全程）。
                if tracker is not None and last_centroids:
                    occ = np.array([[1 if voted[r][f] else 0 for f in range(9)]
                                    for r in range(10)])
                    _on = int(((occ == 1) & (_STD_OCC_ARR == 1)).sum())
                    _ex = int(((occ == 1) & (_STD_OCC_ARR == 0)).sum())
                    print(f"    [std-dbg] f{frame_counter}: on_std={_on} extra={_ex}",
                          flush=True)
                    if matches_std_opening(voted):
                        ref_board = [list(p) for p in STD_REF_BOARD]
                        print(f"    [ref-reset] f{frame_counter}: "
                              f"标准开局约定重置（32 交点）", flush=True)
                    else:
                        ref_board = h_transform(
                            info["imageToGrid"], np.asarray(last_centroids)).tolist()
                latest_info, latest_grid, latest_dets = info, voted, dets
                if not is_illegal(voted) or (last_legal is None and 20 <= n_pc <= 32):
                    last_legal, last_legal_n = voted, n_pc
                fen = grid_to_fen(voted)
                if fen != last_fen:
                    # 与 App 一致：不设合法性门，直接问引擎（驱动内部兜住无解/异常）。
                    mv = engine.best(fen)
                    if mv is not None and voted[mv[1]][mv[0]] is not None:
                        suggest = mv
                        suggest_note = notation(voted, *mv)
                    else:
                        suggest, suggest_note = None, None
                    last_fen = fen
                print(f"    f{frame_counter}: 检出{len(dets)} 投票后{n_pc}子 "
                      f"建议={suggest_note}", flush=True)
            else:
                if tracker is not None and tracker.H is not None:
                    # 跟踪还活着：显示由传播承担，节流重试重锚（不烧 SAM）
                    recheck_in = tracker.RECHECK_AFTER_FAIL
                    state = "重锚节流"
                    lock_misses = 0
                else:
                    lock_misses += 1
                    state = f"漏检{lock_misses}"
                if lock_misses > LOCK_MAX_MISSES:
                    if edge.imu_static and latest_info is not None:
                        # D2 静止不清屏：IMU 判静止 ⇒ 棋盘没动，长漏检=手遮挡——
                        # 显示保持最后好四角/局面，检测层照常 fast-sampling 重试，
                        # 重锁成功后无缝衔接。真清屏只发生在移动/未知态。
                        state = "静止保持"
                    else:
                        latest_info, latest_grid, latest_dets = None, None, []
                        recent, last_centroids = [], []
                        suggest, suggest_note, last_fen = None, None, None
                        edge.reset_tracking_state()   # 移动重锁修复 C（与端上同）
                        if tracker is not None:
                            tracker.H, tracker.pts, tracker.prev_gray = None, None, None
                        state = "丢失"
                # D1 挪盘兜底（双确认③内容验证）：超时+整体平移候选 → 用它跑 YOLO 取
                # 当前帧棋子质心（棋盘被挪则棋子随行）→ 包含 ≥95% 且 ≥10 个才清锚重锁。
                cand = edge.reanchor_candidate()
                if cand is not None:
                    dets, _ = yolo.detect(img, cand)
                    cents = [[d[0], d[1]] for d in dets]
                    okv = (len(cents) >= EdgeSamDetector.MIN_REF_PTS and
                           EdgeSamDetector._containment(cand, cents) >= EdgeSamDetector.CONTAIN_MIN)
                    edge.resolve_reanchor(okv)
                    if okv:
                        last_centroids = []      # 旧位置参考作废（与端上 resolveReanchor 同）
                        state = "挪盘重锚"
                    else:
                        print(f"    [anchor] 挪盘双确认否决（{len(cents)} 质心）→ 保持", flush=True)
        dets_show = latest_dets
        if latest_info is not None and dets_show:
            q = np.asarray(latest_info["corners"], np.float64).reshape(4, 2)
            dets_show = [d for d in dets_show
                         if EdgeSamDetector._point_in_quad(q, d[0], d[1])]
        vis = render(img, latest_info, latest_grid, dets_show, suggest,
                     suggest_note or "", frame_counter, state)
        cv2.imwrite(os.path.join(out_dir, f"frame_{frame_counter:05d}.jpg"), vis,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        n_out += 1
        if frame_counter % 100 == 0:
            print(f"  {frame_counter}/{limit}  {time.time()-t0:.0f}s  "
                  f"建议={suggest_note}", flush=True)
    cap.release()
    if tracker is not None:
        print(f"[track] 传播成功帧={tracker.n_track_frames} 传播失败帧={tracker.n_fail} "
              f"最近内点={tracker.last_inliers}", flush=True)
    print(f"[done] {n_out} 帧 -> {out_dir}  总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    video = sys.argv[1]
    out = sys.argv[2]
    EdgeSamDetector.QUAD_MODE = opt("--quad-mode", "hull")   # hull（默认，线上行为）| line（C5 边线求交，失败自动回退 hull）
    # --imu static：PC 模拟 IMU 静止（锚闸 D1 + 不清屏 D2 + 投票门 D3 + 仲裁）；
    # moving=模拟移动（救援二级启用）；缺省 off=UNKNOWN（行为与 S7 完全一致）。
    run(video, out, int(opt("--max", "0")), imu_mode=opt("--imu", "off"),
        track=(opt("--track", "off") == "on"))
