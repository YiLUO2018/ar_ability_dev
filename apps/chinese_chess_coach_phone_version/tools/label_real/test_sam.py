"""验证 SAM 粗定位：分割棋盘掩码 → 拟合同四边形 → 对比真实角点。"""
import sys
import time

import cv2
import numpy as np

SAM_ROOT = r"L:\Reinforcement_Learning\segment\segment-anything-main"
sys.path.insert(0, SAM_ROOT)
from segment_anything import sam_model_registry, SamPredictor

IMG = r"L:\projects\ar_ability_dev\apps\chinese_chess_coach_phone_version\debug_frames\board_21.png"
CHECKPOINT = SAM_ROOT + r"\models\sam_vit_b_01ec64.pth"
GT = np.array([[116, 76], [350, 76], [375, 299], [102, 306]], dtype=np.float32)  # TL,TR,BR,BL（SAM 拟合值，旧 TL(166,77) 已弃）


def order_quad(pts):
    """按 TL,TR,BR,BL 排序四点。"""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def quad_from_mask(mask):
    """从二值掩码提取最大轮廓并拟合四边形角点（像素）。"""
    m = (mask.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    # 尝试多种 epsilon 找到 4 顶点
    for eps in [0.01, 0.015, 0.02, 0.03, 0.05]:
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) == 4:
            return order_quad(approx.reshape(4, 2)), area
    # 兜底：凸包 + 用最远四点
    hull = cv2.convexHull(c)
    hull = hull.reshape(-1, 2)
    # 找最靠近四个极值方向的点
    return None, area


def main():
    t0 = time.time()
    print("加载 SAM ViT-B ...")
    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
    sam.to("cpu")
    sam.eval()
    print(f"模型加载 {time.time()-t0:.1f}s")

    img = cv2.imread(IMG)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    print(f"图像 {W}x{H}")

    predictor = SamPredictor(sam)
    predictor.set_image(img_rgb)

    # 棋盘中心（由 GT 估算，仅用于提示点）
    center = GT.mean(axis=0)
    print(f"提示点 = {center}")

    masks, scores, _ = predictor.predict(
        point_coords=np.array([center]),
        point_labels=np.array([1]),
        multimask_output=True,
    )
    print(f"返回 {len(masks)} 个掩码，scores={np.round(scores,3)}")

    for i, (mask, score) in enumerate(zip(masks, scores)):
        quad, area = quad_from_mask(mask)
        frac = mask.mean() * 100
        if quad is not None:
            err = np.mean([np.linalg.norm(quad[j] - GT[j]) for j in range(4)])
            print(f"[mask{i}] score={score:.3f} 面积占比={frac:.1f}% 四角={np.round(quad).astype(int).tolist()} 角点误差={err:.1f}px")
        else:
            print(f"[mask{i}] score={score:.3f} 面积占比={frac:.1f}% 未拟合出四边形")
    print(f"总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
