"""边线求交提角（C5-P1 原型，2026-10-11）：全可见路径的「先提 4 条边线，相邻求交」。

动机（算法梳理-01 §7）：SAM 掩码四角圆钝，凸包顶点落在圆弧上、系统性内缩；
直边段长而干净，最小二乘拟合边线后延长求交能「穿过圆角」交出真实角（亚像素）。

算法（对原始凸包环做，**不先 simplify**——栅格阶梯点正是边线拟合的样本；
转角分角与弧长排除天然抗阶梯噪声）：
  1. 逐顶点转角：入边/出边方向夹角。直边阶梯 ≈0~15°，角点 30°+；
  2. 转角降序 + 环上弧长分离（≥ min_corner_sep）选 4 个角点；
     第 4 大转角 < min_turn_deg 视为无清晰角点 → 失败；
  3. 相邻角点之间的顶点段 = 边链，剔除距任一端角点弧长 ≤ excl_px 的顶点
     （避开角点圆弧/切角过渡区）；
  4. 每链总体最小二乘（TLS）拟合直线：点数不足或垂距 RMS > rms_max → 失败
     （弯曲/遮挡边不可靠，宁缺毋滥）；
  5. 相邻边线求交 = 四角，按 TL,TR,BR,BL 排序。

任何一步失败返回 None——调用方回退「凸包 + 贪心减点」，保证不劣于现状。
"""
import numpy as np


def _order_quad(pts):
    q = np.asarray(pts, np.float64)
    s, d = q.sum(1), q[:, 1] - q[:, 0]
    return np.array([q[np.argmin(s)], q[np.argmin(d)], q[np.argmax(s)], q[np.argmax(d)]])


def _ring_arc(ring):
    """环上累计弧长（含回到起点的闭合段）。"""
    a = np.asarray(ring, np.float64)
    seg = np.hypot(*(np.roll(a, -1, axis=0) - a).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    return cum, cum[-1] + seg[-1]      # 位置数组 + 总周长


def _turn_angles(ring):
    """环上逐顶点转角（度）：入边方向与出边方向的夹角。直边≈0，角点大。"""
    a = np.asarray(ring, np.float64)
    v1 = a - np.roll(a, 1, axis=0)     # 入边向量 (i-1 → i)
    v2 = np.roll(a, -1, axis=0) - a    # 出边向量 (i → i+1)
    l1 = np.hypot(v1[:, 0], v1[:, 1])
    l2 = np.hypot(v2[:, 0], v2[:, 1])
    ok = (l1 > 1e-9) & (l2 > 1e-9)
    cos = np.ones(len(a))
    cos[ok] = np.clip((v1[ok] * v2[ok]).sum(1) / (l1[ok] * l2[ok]), -1, 1)
    return np.degrees(np.arccos(cos))


def _circular_dist(pos, total, x):
    d = np.abs(pos - x)
    return np.minimum(d, total - d)


def _pick_corners(turns, arc_pos, total, min_sep, win_px=8.0):
    """角点评分 = ±win_px 弧长窗口内的转角累计（孤立阶梯尖峰无支撑，真实角点持续转角）。

    按评分降序贪心选 4 个环上弧长分离 ≥ min_sep 的角点。
    返回 (角点索引环序, 前4评分)。
    """
    n = len(turns)
    score = np.empty(n)
    for i in range(n):
        d = _circular_dist(arc_pos, total, arc_pos[i])
        score[i] = turns[d <= win_px].sum()
    order = np.argsort(score)[::-1]
    picked = []
    for i in order:
        if all(_circular_dist(arc_pos[i], total, arc_pos[j]) >= min_sep for j in picked):
            picked.append(int(i))
            if len(picked) == 4:
                break
    return sorted(picked), score[order[:4]]


def _fit_line_tls(pts):
    """总体最小二乘直线。返回 (center, direction, rms)。"""
    c = pts.mean(0)
    d = pts - c
    _, _, vt = np.linalg.svd(d, full_matrices=False)
    dirv = vt[0]
    normal = vt[1] if len(vt) > 1 else np.array([-dirv[1], dirv[0]])
    resid = d @ normal
    return c, dirv, float(np.sqrt(np.mean(resid ** 2)))


def _intersect(c1, d1, c2, d2):
    """直线 c1+t·d1 与 c2+s·d2 求交。"""
    a = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    if abs(np.linalg.det(a)) < 1e-9:
        return None
    t, _ = np.linalg.solve(a, c2 - c1)
    return c1 + t * d1


def simplify_ring(ring, eps=2.5):
    """保轮廓简化：贪心删除到邻点弦垂距 < eps 的顶点（栅格阶梯毛刺），真角点垂距大得以保留。"""
    poly = [np.asarray(p, np.float64) for p in ring]
    while len(poly) > 4:
        n = len(poly)

        def seg_dist(p, a, b):
            dx, dy = b[0] - a[0], b[1] - a[1]
            l2 = dx * dx + dy * dy
            if l2 == 0:
                return float(np.hypot(p[0] - a[0], p[1] - a[1]))
            t = np.clip(((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2, 0, 1)
            return float(np.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy))

        costs = [seg_dist(poly[i], poly[(i - 1) % n], poly[(i + 1) % n]) for i in range(n)]
        j = int(np.argmin(costs))
        if costs[j] >= eps:
            break
        poly.pop(j)
    return poly


def _points_near_segment(pts, a, b, band, ext_frac=0.0):
    """边界点集中，距线段 a→b 垂距 ≤ band 且沿线段方向投影在 [−ext, 1+ext] 内的点。

    ext_frac=0：严格限制在线段范围内（角点过渡区的切角弦像素不掺入边线样本）。
    """
    d = b - a
    l2 = float(d @ d)
    if l2 < 1e-9:
        return pts[np.linalg.norm(pts - a, axis=1) <= band]
    t = ((pts - a) @ d) / l2
    perp2 = np.sum((pts - a) ** 2, axis=1) - (t * t) * l2
    keep = (t >= -ext_frac) & (t <= 1 + ext_frac) & (perp2 <= band * band)
    return pts[keep]


def line_intersect_quad(
    hull,
    boundary=None,
    min_corner_score_deg=45.0,
    min_corner_sep=20.0,
    excl_px=12.0,
    min_chain_pts=2,
    rms_max=2.0,
    cloud_band=10.0,
):
    """凸包环（有序顶点）→ 边线求交四角（TL,TR,BL,BL 序）。失败返回 None（调用方回退贪心减点）。

    混合管线（P3 教训的运用）：
      - 角点检测在 simplify 后的环上做（贪心删垂距<2.5px 顶点杀阶梯毛刺，
        ±8px 窗口转角累计评分，孤立尖峰无支撑）；
      - 边线样本优先取**原始边界像素云**（距 hull 链段 ≤ cloud_band）——毛刺翻转
        对称无偏，不像 hull 只取外向包络（实测外向偏置被线交放大 ~2px）；
        无边界点集时回退 hull 链顶点；
      - 链两端按弧长排除 excl_px（覆盖圆角/切角过渡区）；
      - TLS 拟合 + RMS 闸（弯曲/遮挡边拒绝），相邻求交。
    """
    ring = np.asarray(hull, np.float64)
    n = len(ring)
    if n < 8:
        return None
    sring = simplify_ring(ring)
    if len(sring) < 4:
        return None

    sturns = _turn_angles(sring)
    sarc_pos, stotal = _ring_arc(sring)
    scorners, top_scores = _pick_corners(sturns, sarc_pos, stotal, min_corner_sep)
    if len(scorners) < 4 or top_scores[3] < min_corner_score_deg:
        return None

    # simplify 只删不加点——简化环顶点必来自原始环，按坐标最近匹配映射回原始索引。
    sarr = np.asarray(sring)
    corners = [int(np.argmin(np.linalg.norm(ring - sarr[c], axis=1))) for c in scorners]

    arc_pos, total = _ring_arc(ring)

    # 每条边 = 相邻角点之间的边线。样本优先取边界像素云（角点连线上、两端各缩
    # excl_px 避开圆弧过渡区）——不依赖 hull 中间顶点，稀疏 hull 也可用；
    # 无边界点集时回退 hull 链中间顶点（须 ≥ min_chain_pts 个）。
    lines = []
    for k in range(4):
        i0 = corners[k]
        i1 = corners[(k + 1) % 4]
        sample = None
        if boundary is not None and len(boundary) >= 32:
            a = ring[i0]
            b = ring[i1]
            seg = b - a
            seg_len = float(np.hypot(*seg))
            if seg_len > 2 * excl_px:
                u = seg / seg_len
                a2 = a + u * excl_px
                b2 = b - u * excl_px
                cloud = _points_near_segment(np.asarray(boundary, np.float64), a2, b2, cloud_band)
                if len(cloud) >= 8:
                    sample = cloud
        if sample is None:
            span = (i1 - i0) % n
            if span <= 2:
                return None
            idx = [(i0 + t) % n for t in range(1, span)]
            keep = [i for i in idx
                    if _circular_dist(arc_pos[i], total, arc_pos[i0]) > excl_px
                    and _circular_dist(arc_pos[i], total, arc_pos[i1]) > excl_px]
            if len(keep) < min_chain_pts:
                return None
            sample = ring[np.array(keep)]
        c, d, rms = _fit_line_tls(sample)
        if rms > rms_max:
            return None                     # 弯曲/遮挡边不可靠
        lines.append((c, d))

    qs = []
    for k in range(4):
        p = _intersect(*lines[k], *lines[(k + 1) % 4])
        if p is None:
            return None
        qs.append(p)
    return _order_quad(qs)


def mask_hull_pixels(binary_mask, min_area_frac=0.02, return_boundary=False):
    """最大连通域 → 边界像素 → Andrew 单调链凸包（**与端上 _mask_hull 同构**）。

    注意：不能用 cv2.convexHull——它压掉共线阶梯点（直边只剩端点）。
    返回有序凸包顶点环（Nx2 float64）或 None；return_boundary=True 时额外返回
    边界像素点集（边线拟合的无偏样本——毛刺翻转对称，不像 hull 只取外向包络）。
    """
    import cv2
    m = (np.asarray(binary_mask) > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    if n <= 1:
        return (None, None) if return_boundary else None
    best = 1 + int(np.argmax(stats[1:, 4]))
    if stats[best, 4] < m.size * min_area_frac:
        return (None, None) if return_boundary else None
    comp = lab == best
    pad = np.pad(comp, 1, constant_values=False)
    up, dn = pad[:-2, 1:-1], pad[2:, 1:-1]
    lf, rt = pad[1:-1, :-2], pad[1:-1, 2:]
    boundary = comp & ~(up & dn & lf & rt)
    ys, xs = np.nonzero(boundary)
    if len(xs) < 4:
        return (None, None) if return_boundary else None
    bpts = np.stack([xs, ys], 1).astype(np.float64)
    pts = np.unique(bpts.astype(np.int64), axis=0).astype(np.float64)
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
    hull = np.array(hull) if len(hull) >= 4 else None
    if return_boundary:
        return hull, bpts
    return hull


def mask_to_quad_line(binary_mask):
    """二值掩码 → 像素级凸包 + 边界点集 → 边线求交。失败 None。便利封装（测试/离线用）。"""
    hull, boundary = mask_hull_pixels(binary_mask, return_boundary=True)
    if hull is None:
        return None
    return line_intersect_quad(hull, boundary)
