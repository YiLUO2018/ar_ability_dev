"""合成数据渲染器（方案 v2 阶段 0）。

用程序化渲染生成「天生已知标签」的象棋棋盘数据，产出两套数据集：

1. ``corner/`` —— 棋盘角点数据集（训练棋盘角点模型）
   - ``images/*.jpg``：含透视棋盘 + 随机背景的场景图（默认 640×640）。
   - ``labels/*.txt``：每行 ``x y``（归一化 [0,1]），顺序 TL, TR, BR, BL。

2. ``classify/`` —— 棋子分类数据集（训练整盘 10×9×15 分类器）
   - ``images/*.jpg``：透视校正后的 canonical 棋盘（默认 450×500）。
   - ``labels/*.txt``：10 行 × 9 列整数（rank-major，grid[rank][file]），取值 0..14，
     类别映射见 ``labels.py``。

用法：:
    python render.py --out data --corner 1000 --classify 1000 --seed 0
    python render.py --out data --corner 10 --classify 10 --visualize 4
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from labels import (
    EMPTY,
    GLYPHS,
    layout_to_grid,
    random_midgame_layout,
    standard_initial_layout,
)

# ---------------------------------------------------------------------------
# 几何常量（canonical 棋盘）
# ---------------------------------------------------------------------------
CELL = 50
MARGIN = 25
BOARD_W = MARGIN * 2 + CELL * 8   # 450
BOARD_H = MARGIN * 2 + CELL * 9   # 500

# canonical 网格四角（file0,rank0 → file8,rank0 → file8,rank9 → file0,rank9）
CANON_CORNERS = np.array(
    [
        [MARGIN, MARGIN],
        [MARGIN + 8 * CELL, MARGIN],
        [MARGIN + 8 * CELL, MARGIN + 9 * CELL],
        [MARGIN, MARGIN + 9 * CELL],
    ],
    dtype=np.float32,
)

# 物体空间四角（cell 单位，居中），用于 3D 透视投影
OBJ_CORNERS = np.array(
    [
        [-4.0, -4.5, 0.0],
        [4.0, -4.5, 0.0],
        [4.0, 4.5, 0.0],
        [-4.0, 4.5, 0.0],
    ],
    dtype=np.float64,
)

# ---------------------------------------------------------------------------
# 字体（跨平台：Windows / WSL / Linux 自动探测）
# ---------------------------------------------------------------------------
_FONT_NAMES = ["simhei.ttf", "simkai.ttf", "simsun.ttc", "msyh.ttc", "Deng.ttf"]


def _discover_fonts():
    dirs = []
    if os.name == "nt":
        windir = os.environ.get("WINDIR", "C:/Windows")
        dirs = [os.path.join(windir, "Fonts")]
    else:
        dirs = ["/mnt/c/Windows/Fonts", "/usr/share/fonts", "/usr/share/fonts/truetype"]

    specs = []
    for d in dirs:
        for name in _FONT_NAMES:
            p = os.path.join(d, name)
            if os.path.exists(p):
                specs.append((p, 0 if name.lower().endswith(".ttc") else None))

    # Linux 兜底：Noto CJK / 文泉驿
    if not specs and os.name != "nt":
        import glob as _glob
        for pat in ("/usr/share/fonts/**/*CJK*.tt[cf]", "/usr/share/fonts/**/wqy*.tt[cf]"):
            for p in _glob.glob(pat, recursive=True):
                specs.append((p, None))
    return specs


_AVAILABLE_FONTS = _discover_fonts()
if not _AVAILABLE_FONTS:
    _AVAILABLE_FONTS = [("", None)]  # 兜底，PIL 默认字体不支持中文，会触发 load_default

_font_cache: dict = {}


def _load_font(path, index, size):
    key = (path, index, size)
    if key not in _font_cache:
        try:
            if path == "":
                _font_cache[key] = ImageFont.load_default()
            elif index is not None:
                _font_cache[key] = ImageFont.truetype(path, size, index=index)
            else:
                _font_cache[key] = ImageFont.truetype(path, size)
        except Exception:
            _font_cache[key] = ImageFont.load_default()
    return _font_cache[key]


def random_font(rng, size):
    path, index = _AVAILABLE_FONTS[int(rng.integers(0, len(_AVAILABLE_FONTS)))]
    return _load_font(path, index, size)


# ---------------------------------------------------------------------------
# 背景与木纹
# ---------------------------------------------------------------------------
def make_background(w: int, h: int, rng) -> np.ndarray:
    """随机桌面背景（纯色 / 渐变 / 木纹 / 噪声）。"""
    mode = int(rng.integers(0, 4))
    if mode == 0:  # 纯色
        color = rng.integers(40, 225, size=3)
        img = np.full((h, w, 3), color, dtype=np.float32)
    elif mode == 1:  # 竖直渐变
        c1 = rng.integers(40, 210, size=3).astype(np.float32)
        c2 = rng.integers(40, 210, size=3).astype(np.float32)
        t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        img = c1[None, None, :] * (1 - t) + c2[None, None, :] * t
        img = np.repeat(img, w, axis=1)
    elif mode == 2:  # 木纹（横向条纹）
        base = rng.integers(120, 200, size=3).astype(np.float32)
        y = np.arange(h, dtype=np.float32)[:, None, None]
        x = np.arange(w, dtype=np.float32)[None, :, None]
        streak = np.sin(y * rng.uniform(0.01, 0.04) + x * rng.uniform(0.0, 0.02))
        img = base[None, None, :] + (streak * rng.uniform(10, 30)).astype(np.float32)
        img = np.repeat(img, w, axis=1) if img.shape[1] == 1 else img
        if img.shape[1] != w:
            img = np.tile(img[:, :1, :], (1, w, 1))
        img = np.resize(img, (h, w, 3))
    else:  # 噪声纹理
        base = rng.integers(60, 180, size=3).astype(np.float32)
        img = base[None, None, :] + rng.normal(0, 18, (h, w, 3)).astype(np.float32)

    img = np.clip(img + rng.normal(0, 4, (h, w, 3)), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def make_board_background(w: int, h: int, rng) -> np.ndarray:
    """棋盘底色（木纹质感）。"""
    base = rng.integers(185, 225, size=3).astype(np.float32)  # 偏暖木色
    y = np.arange(h, dtype=np.float32)[:, None, None]
    x = np.arange(w, dtype=np.float32)[None, :, None]
    grain = (
        np.sin(y * rng.uniform(0.02, 0.05) + x * rng.uniform(0.0, 0.02)) * 8
        + rng.normal(0, 4, (h, w, 3))
    )
    img = np.clip(base[None, None, :] + grain, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


# ---------------------------------------------------------------------------
# 棋盘渲染（canonical）
# ---------------------------------------------------------------------------
def _xy(file, rank):
    return MARGIN + file * CELL, MARGIN + rank * CELL


def render_canonical_board(layout: dict, rng) -> Image.Image:
    """渲染一张 canonical 棋盘（450×500，PIL RGB），含网格/九宫/楚河汉界/棋子。"""
    bg = make_board_background(BOARD_W, BOARD_H, rng)
    img = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    line_color = (int(rng.integers(45, 75)), int(rng.integers(30, 55)), int(rng.integers(15, 40)))
    line_w = 2

    # 外框
    draw.rectangle([6, 6, BOARD_W - 7, BOARD_H - 7], outline=line_color, width=3)

    # 竖线（file 0..8，楚河汉界处断开）
    for f in range(9):
        x = _xy(f, 0)[0]
        draw.line([x, _xy(0, 0)[1], x, _xy(0, 4)[1]], fill=line_color, width=line_w)
        draw.line([x, _xy(0, 5)[1], x, _xy(0, 9)[1]], fill=line_color, width=line_w)

    # 横线（rank 0..9）
    for r in range(10):
        y = _xy(0, r)[1]
        draw.line([_xy(0, r)[0], y, _xy(8, r)[0], y], fill=line_color, width=line_w)

    # 九宫斜线
    def palace(f0, r0, f1, r1):
        draw.line([*_xy(f0, r0), *_xy(f1, r1)], fill=line_color, width=line_w)

    palace(3, 0, 5, 2)
    palace(5, 0, 3, 2)
    palace(3, 7, 5, 9)
    palace(5, 7, 3, 9)

    # 楚河汉界
    river_y = (MARGIN + 4 * CELL + MARGIN + 5 * CELL) // 2
    river_font = random_font(rng, 34)
    river_color = tuple(max(0, int(c * 0.55)) for c in line_color)
    draw.text((_xy(4, 0)[0] + MARGIN + 4 * CELL, river_y), "楚河", font=river_font,
              fill=river_color, anchor="mm")
    draw.text((MARGIN + 2 * CELL, river_y), "汉界", font=river_font,
              fill=river_color, anchor="mm")

    # 棋子
    for (f, r), (color, t) in layout.items():
        cx, cy = _xy(f, r)
        draw_piece(draw, cx, cy, color, t, rng)

    return img


def draw_piece(draw: ImageDraw.ImageDraw, cx: int, cy: int, color: str, t: int, rng):
    """在 (cx, cy) 画一枚棋子。"""
    radius = int(rng.uniform(20, 23))
    # 位置微扰
    cx = cx + rng.uniform(-1.5, 1.5)
    cy = cy + rng.uniform(-1.5, 1.5)

    fill = (
        int(rng.integers(222, 236)),
        int(rng.integers(198, 216)),
        int(rng.integers(158, 182)),
    )
    outline = (int(rng.integers(135, 160)), int(rng.integers(105, 130)), int(rng.integers(62, 88)))

    # 阴影（右下偏移的暗圆）
    draw.ellipse(
        [cx - radius + 2, cy - radius + 2, cx + radius + 2, cy + radius + 2],
        fill=(0, 0, 0),
    )
    # 饼身
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=fill, outline=outline, width=2)
    # 内圈
    draw.ellipse([cx - radius + 3, cy - radius + 3, cx + radius - 3, cy + radius - 3], outline=outline, width=1)

    # 汉字
    glyph = GLYPHS[t][0 if color == "red" else 1]
    glyph_color = (
        (int(rng.integers(175, 205)), int(rng.integers(20, 45)), int(rng.integers(20, 45)))
        if color == "red"
        else (int(rng.integers(15, 35)), int(rng.integers(15, 35)), int(rng.integers(15, 35)))
    )
    font = random_font(rng, int(radius * 1.55))
    draw.text((cx, cy), glyph, font=font, fill=glyph_color, anchor="mm")


# ---------------------------------------------------------------------------
# 透视投影（生成场景四角）
# ---------------------------------------------------------------------------
def _rotation(roll_deg, pitch_deg, yaw_deg):
    roll, pitch, yaw = np.radians([roll_deg, pitch_deg, yaw_deg])
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    return Rz @ Ry @ Rx


def project_scene_corners(rng, scene_w: int, scene_h: int) -> np.ndarray:
    """3D 透视投影得到棋盘在场景中的四角（TL, TR, BR, BL）。

    相机在原点，沿 +z 看向位于 ``(0, 0, dist)`` 的棋盘平面；``f = dist`` 使棋盘中心
    按 1:1 投影（cell 单位 → 图像单位），透视强度仅由 roll/pitch/yaw 决定，最后统一
    缩放到场景尺寸。``dist`` 足够大以保证棋盘完全位于相机前方（z > 0）。
    """
    roll = rng.uniform(0, 360)
    pitch = rng.uniform(8, 45)
    yaw = rng.uniform(-28, 28)
    dist = rng.uniform(14.0, 26.0)
    R = _rotation(roll, pitch, yaw)
    pts = (R @ OBJ_CORNERS.T).T  # (4,3) 居中于原点
    cam = pts + np.array([0.0, 0.0, dist])
    if (cam[:, 2] <= 0.3).any():  # 理论不会发生，兜底
        raise RuntimeError(f"corner behind camera: {cam[:, 2]}")
    u = dist * cam[:, 0] / cam[:, 2]
    v = dist * cam[:, 1] / cam[:, 2]

    w = u.max() - u.min()
    h = v.max() - v.min()
    # 棋盘占场景比例随机（尺度多样性），并保证四周留白
    fill = rng.uniform(0.55, 0.80)
    scale = min(scene_w * fill / w, scene_h * fill / h)
    u *= scale
    v *= scale

    # 居中 + 随机平移（幅度按四边形实际包围盒的安全余量自适应，绝不越界）
    quad = np.stack([u, v], axis=1)
    lo = quad.min(axis=0)
    hi = quad.max(axis=0)
    half_ext = np.maximum(-lo, hi)
    margin = 20.0
    max_jx = max(scene_w / 2 - half_ext[0] - margin, 0.0)
    max_jy = max(scene_h / 2 - half_ext[1] - margin, 0.0)
    jx = rng.uniform(-1, 1) * max_jx
    jy = rng.uniform(-1, 1) * max_jy
    return (quad + np.array([scene_w / 2 + jx, scene_h / 2 + jy])).astype(np.float32)


# ---------------------------------------------------------------------------
# 图像增强
# ---------------------------------------------------------------------------
def _motion_blur(img, rng):
    size = int(rng.integers(5, 10))
    if size % 2 == 0:
        size += 1
    c = size // 2
    angle = rng.uniform(0, np.pi)
    dx, dy = np.cos(angle), np.sin(angle)
    kernel = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        x = int(round(c + (i - c) * dx))
        y = int(round(c + (i - c) * dy))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0
    s = kernel.sum()
    if s == 0:
        return img
    kernel /= s
    return cv2.filter2D(img, -1, kernel)


def _lens_distort(img, k1, k2):
    h, w = img.shape[:2]
    fx, fy = float(w), float(h)
    cx, cy = w / 2.0, h / 2.0
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x = (xs - cx) / fx
    y = (ys - cy) / fy
    r2 = x * x + y * y
    inv = 1.0 + (-k1) * r2 + (-k2) * r2 * r2  # 近似逆
    map_x = (x * inv * fx + cx).astype(np.float32)
    map_y = (y * inv * fy + cy).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _distort_points(pts, k1, k2, w, h):
    """对归一化坐标点施加径向畸变（前向模型，与图像近似一致）。"""
    fx, fy = float(w), float(h)
    cx, cy = w / 2.0, h / 2.0
    x = (pts[:, 0] * w - cx) / fx
    y = (pts[:, 1] * h - cy) / fy
    r2 = x * x + y * y
    scale = 1.0 + k1 * r2 + k2 * r2 * r2
    xd = x * scale * fx + cx
    yd = y * scale * fy + cy
    return np.stack([xd / w, yd / h], axis=1).astype(np.float32)


def augment_photo(img: np.ndarray, rng) -> np.ndarray:
    """光度/噪声/模糊/JPEG 增强（不改变几何，不影响标签）。"""
    # 亮度/对比度
    img = np.clip(img.astype(np.float32) * rng.uniform(0.7, 1.3) + rng.uniform(-30, 30), 0, 255).astype(np.uint8)
    # gamma
    if rng.random() < 0.5:
        g = rng.uniform(0.6, 1.6)
        lut = (255.0 * (np.linspace(0, 1, 256) ** g)).astype(np.uint8)
        img = cv2.LUT(img, lut)
    # 每通道色偏
    if rng.random() < 0.7:
        j = rng.uniform(0.85, 1.15, size=3)
        img = np.clip(img.astype(np.float32) * j[None, None, :], 0, 255).astype(np.uint8)
    # 高斯噪声
    if rng.random() < 0.7:
        sigma = rng.uniform(2, 12)
        img = np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)
    # 椒盐
    if rng.random() < 0.25:
        n = int(img.shape[0] * img.shape[1] * rng.uniform(0.0005, 0.002))
        for _ in range(n):
            yy = int(rng.integers(0, img.shape[0]))
            xx = int(rng.integers(0, img.shape[1]))
            img[yy, xx] = 255 if rng.random() < 0.5 else 0
    # 模糊
    if rng.random() < 0.6:
        kind = int(rng.integers(0, 3))
        if kind == 0:
            img = cv2.GaussianBlur(img, (3, 3), 0)
        elif kind == 1:
            img = _motion_blur(img, rng)
        else:
            img = cv2.medianBlur(img, 3)
    # JPEG 压缩
    if rng.random() < 0.85:
        q = int(rng.integers(40, 95))
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img


# ---------------------------------------------------------------------------
# 采样生成
# ---------------------------------------------------------------------------
def generate_corner_sample(rng, scene_w: int, scene_h: int):
    """返回 (scene_bgr, corners_normalized[4,2])。"""
    layout = random_midgame_layout(rng) if rng.random() < 0.75 else standard_initial_layout()
    board_pil = render_canonical_board(layout, rng)
    board_bgr = cv2.cvtColor(np.array(board_pil), cv2.COLOR_RGB2BGR)

    corners = project_scene_corners(rng, scene_w, scene_h)
    H = cv2.getPerspectiveTransform(CANON_CORNERS, corners)

    background = make_background(scene_w, scene_h, rng)

    # 阴影
    mask = np.full((BOARD_H, BOARD_W, 1), 255, np.uint8)
    shadow_shift = np.array([rng.uniform(6, 16), rng.uniform(8, 20)], dtype=np.float32)
    H_shadow = cv2.getPerspectiveTransform(CANON_CORNERS, corners + shadow_shift)
    shadow = cv2.warpPerspective(mask, H_shadow, (scene_w, scene_h)).astype(np.float32)
    shadow = cv2.GaussianBlur(shadow, (15, 15), 0)[..., None] / 255.0
    scene = (background.astype(np.float32) * (1 - rng.uniform(0.15, 0.35) * shadow)).astype(np.uint8)

    # 棋盘
    warped = cv2.warpPerspective(board_bgr, H, (scene_w, scene_h))
    warped_mask = cv2.warpPerspective(mask, H, (scene_w, scene_h))[..., None] > 0
    scene[warped_mask[..., 0]] = warped[warped_mask[..., 0]]

    # 径向畸变（含标签同步）
    k1 = rng.uniform(-0.15, 0.15)
    k2 = rng.uniform(-0.06, 0.06)
    scene = _lens_distort(scene, k1, k2)
    corners_norm = _distort_points(corners / np.array([scene_w, scene_h], np.float32), k1, k2, scene_w, scene_h)
    corners_norm = np.clip(corners_norm, 0.001, 0.999)

    scene = augment_photo(scene, rng)
    return scene, corners_norm


def generate_corner_negative_sample(rng, scene_w: int, scene_h: int):
    """无棋盘负样本：仅随机背景 + 增强，供角点模型「置信度门控」训练。"""
    scene = make_background(scene_w, scene_h, rng)
    # 偶尔叠加另一层纹理，避免模型把「纯色背景」当过拟合信号
    if rng.random() < 0.5:
        overlay = make_background(max(scene_w // 2, 32), max(scene_h // 2, 32), rng)
        overlay = cv2.resize(overlay, (scene_w, scene_h))
        scene = cv2.addWeighted(scene, 0.6, overlay, 0.4, 0)
    return augment_photo(scene, rng)


def generate_classify_sample(rng):
    """返回 (board_bgr, grid[10,9])，带轻微 warp 抖动模拟透视校正误差。"""
    layout = random_midgame_layout(rng) if rng.random() < 0.7 else standard_initial_layout()
    grid = layout_to_grid(layout)
    board_pil = render_canonical_board(layout, rng)
    board_bgr = cv2.cvtColor(np.array(board_pil), cv2.COLOR_RGB2BGR)

    # 小抖动（模拟 warp 残差）
    if rng.random() < 0.8:
        dst = CANON_CORNERS + rng.uniform(-4.0, 4.0, size=(4, 2)).astype(np.float32)
        H = cv2.getPerspectiveTransform(CANON_CORNERS, dst)
        board_bgr = cv2.warpPerspective(board_bgr, H, (BOARD_W, BOARD_H), borderMode=cv2.BORDER_REPLICATE)

    # 径向畸变（标签不随像素位置变化，安全）
    if rng.random() < 0.4:
        board_bgr = _lens_distort(board_bgr, rng.uniform(-0.2, 0.2), rng.uniform(-0.08, 0.08))

    board_bgr = augment_photo(board_bgr, rng)
    return board_bgr, grid


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def _write_image(img: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])


def _write_corner_labels(path: str, corners_norm: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        for x, y in corners_norm:
            fp.write(f"{x:.6f} {y:.6f}\n")


def _write_negative_label(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("no_board\n")


def _write_classify_labels(path: str, grid: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        for r in range(10):
            fp.write(" ".join(str(int(grid[r, f])) for f in range(9)) + "\n")


def _save_visualize(img: np.ndarray, corners_norm: np.ndarray | None, path: str):
    vis = img.copy()
    if corners_norm is not None:
        h, w = img.shape[:2]
        pts = (corners_norm * np.array([w, h], np.float32)).astype(int)
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
        for (x, y), c in zip(pts, colors):
            cv2.circle(vis, (x, y), 6, c, -1)
        cv2.polylines(vis, [pts], True, (0, 255, 255), 2)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, vis)


def generate_set(kind: str, count: int, out_dir: str, rng, scene_w: int, scene_h: int, visualize: int, neg: int = 0):
    img_dir = os.path.join(out_dir, kind, "images")
    lbl_dir = os.path.join(out_dir, kind, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    prev_dir = os.path.join(out_dir, "preview")
    os.makedirs(prev_dir, exist_ok=True)

    for i in range(count):
        name = f"{i:06d}"
        if kind == "corner":
            img, corners = generate_corner_sample(rng, scene_w, scene_h)
            _write_image(img, os.path.join(img_dir, name + ".jpg"))
            _write_corner_labels(os.path.join(lbl_dir, name + ".txt"), corners)
            if i < visualize:
                _save_visualize(img, corners, os.path.join(prev_dir, f"corner_{name}.png"))
        else:
            img, grid = generate_classify_sample(rng)
            _write_image(img, os.path.join(img_dir, name + ".jpg"))
            _write_classify_labels(os.path.join(lbl_dir, name + ".txt"), grid)
            if i < visualize:
                _save_visualize(img, None, os.path.join(prev_dir, f"classify_{name}.png"))

    # 负样本（仅 corner 有意义）
    for j in range(neg):
        name = f"{count + j:06d}"
        img = generate_corner_negative_sample(rng, scene_w, scene_h)
        _write_image(img, os.path.join(img_dir, name + ".jpg"))
        _write_negative_label(os.path.join(lbl_dir, name + ".txt"))

    print(f"[{kind}] {count} 正 + {neg} 负样本 → {os.path.join(out_dir, kind)}")


def main():
    ap = argparse.ArgumentParser(description="合成象棋棋盘数据渲染器")
    ap.add_argument("--out", default="out", help="输出目录")
    ap.add_argument("--corner", type=int, default=0, help="角点训练样本数")
    ap.add_argument("--corner-val", type=int, default=0, help="角点验证样本数")
    ap.add_argument("--corner-neg", type=int, default=0, help="角点训练负样本（无棋盘）数")
    ap.add_argument("--corner-neg-val", type=int, default=0, help="角点验证负样本数")
    ap.add_argument("--classify", type=int, default=0, help="分类训练样本数")
    ap.add_argument("--classify-val", type=int, default=0, help="分类验证样本数")
    ap.add_argument("--size", type=int, default=640, help="角点场景边长")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--visualize", type=int, default=0, help="每种写 N 张预览图")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    for kind, n, neg in [("corner", args.corner, args.corner_neg),
                         ("classify", args.classify, 0)]:
        if n > 0:
            generate_set(kind, n, os.path.join(args.out, "train"), rng, args.size, args.size, args.visualize, neg)
    for kind, n, neg in [("corner", args.corner_val, args.corner_neg_val),
                         ("classify", args.classify_val, 0)]:
        if n > 0:
            generate_set(kind, n, os.path.join(args.out, "val"), rng, args.size, args.size, args.visualize, neg)

    meta = {
        "board": {"width": BOARD_W, "height": BOARD_H, "cell": CELL, "margin": MARGIN},
        "corner": {"count": args.corner, "val": args.corner_val,
                    "neg": args.corner_neg, "neg_val": args.corner_neg_val,
                    "size": args.size, "corner_order": "TL,TR,BR,BL", "coord": "normalized [0,1]",
                    "negative_label": "no_board"},
        "classify": {"count": args.classify, "val": args.classify_val,
                      "labels": "10x9 rank-major, 0=empty, 1-7=red GENERAL..PAWN, 8-14=black"},
        "seed": args.seed,
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)

    print("完成。meta →", os.path.join(args.out, "meta.json"))


if __name__ == "__main__":
    main()
