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

    # ---- 主检测 ----
    def detect(self, img):
        h, w = img.shape[:2]
        emb, nw, nh, scale = self._encode(img)
        px, py = self.prompt_x * nw, self.prompt_y * nh
        scores, masks = self._decode(emb, px, py)
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
        chosen = None                                # (corners, recon, comp)
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
                if pass_recon != recon:
                    continue
                if not recon and self._is_cut_off(corners, w, h):
                    continue                         # 全可见：贴边=掩码被画面裁，拒
                if recon and (corners[:, 0].min() < -w * 0.5 or corners[:, 0].max() > w * 1.5 or
                              corners[:, 1].min() < -h * 0.5 or corners[:, 1].max() > h * 1.5):
                    continue                         # 复原：允许出画面，但不许远于半幅
                acceptables.append((corners, recon, comp))
            if acceptables:
                if self.prev_corners is None:
                    chosen = acceptables[0]          # 无历史：面积序第一（原行为）
                else:
                    # P3-a 时间稳定性仲裁：棋盘不瞬移，取与上一帧四角平均位移最小者。
                    # 溢出/碎块掩码四角逐帧乱跳，即使面积更大也会输给稳定真棋盘。
                    ref = self.prev_corners
                    chosen = min(acceptables,
                                 key=lambda c: float(np.hypot(*(c[0] - ref).T).mean()))
                break
        if chosen is None:
            return None
        corners, recon, comp = chosen
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

DETECT_EVERY_N, LOCK_MAX_MISSES, HISTORY_SIZE = 5, 3, 5


def run(video, out_dir, max_frames=0):
    os.makedirs(out_dir, exist_ok=True)
    edge = EdgeSamDetector()
    yolo = YoloDetectorPy()
    refiner = QuadRefinerPy()
    engine = Engine()

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
        if frame_counter % DETECT_EVERY_N == 0:
            detected = edge.detect(img)
            if detected is not None:
                lock_misses = 0
                info = detected
                if len(last_centroids) >= 12:
                    refined = refiner.refine(info["corners"], last_centroids)
                    if refined is not None:
                        info = edge.refine_quad(refined, img.shape[1], img.shape[0]) or detected
                dets, _ = yolo.detect(img, info["corners"])
                last_centroids = [[d[0], d[1]] for d in dets]
                grid = snap_to_grid(dets, info)
                recent.append(grid)
                while len(recent) > HISTORY_SIZE:
                    recent.pop(0)
                voted = vote_grid(recent)
                latest_info, latest_grid, latest_dets = info, voted, dets
                state = "检测OK"
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
                n_pc = sum(1 for r in range(10) for f in range(9) if voted[r][f])
                print(f"    f{frame_counter}: 检出{len(dets)} 投票后{n_pc}子 "
                      f"建议={suggest_note}", flush=True)
            else:
                lock_misses += 1
                state = f"漏检{lock_misses}"
                if lock_misses > LOCK_MAX_MISSES:
                    latest_info, latest_grid, latest_dets = None, None, []
                    recent, last_centroids = [], []
                    suggest, suggest_note, last_fen = None, None, None
                    state = "丢失"
        vis = render(img, latest_info, latest_grid, latest_dets, suggest,
                     suggest_note or "", frame_counter, state)
        cv2.imwrite(os.path.join(out_dir, f"frame_{frame_counter:05d}.jpg"), vis,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        n_out += 1
        if frame_counter % 100 == 0:
            print(f"  {frame_counter}/{limit}  {time.time()-t0:.0f}s  "
                  f"建议={suggest_note}", flush=True)
    cap.release()
    print(f"[done] {n_out} 帧 -> {out_dir}  总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    video = sys.argv[1]
    out = sys.argv[2]
    run(video, out, int(opt("--max", "0")))
