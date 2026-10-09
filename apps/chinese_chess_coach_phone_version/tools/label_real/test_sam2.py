"""SAM 粗定位 → 几何精修 完整管线验证。"""
import sys
import time

import cv2
import numpy as np

SAM_ROOT = r"L:\Reinforcement_Learning\segment\segment-anything-main"
sys.path.insert(0, SAM_ROOT)
sys.path.insert(0, r"L:\projects\ar_ability_dev\apps\chinese_chess_coach_phone_version\tools\label_real")
from segment_anything import sam_model_registry, SamPredictor
from refine import refine_corners

IMG = r"L:\projects\ar_ability_dev\apps\chinese_chess_coach_phone_version\debug_frames\board_21.png"
CHECKPOINT = SAM_ROOT + r"\models\sam_vit_b_01ec64.pth"
GT = np.array([[116, 76], [350, 76], [375, 299], [102, 306]], dtype=np.float32)  # TL,TR,BR,BL（SAM 拟合值，旧 TL(166,77) 已弃）


def order_quad(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def quad_approx(mask):
    m = (mask.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    for eps in [0.01, 0.015, 0.02, 0.03, 0.05, 0.08]:
        a = cv2.approxPolyDP(c, eps * peri, True)
        if len(a) == 4:
            return order_quad(a.reshape(4, 2))
    return None


def quad_minrect(mask):
    m = (mask.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    return order_quad(cv2.boxPoints(rect))


def main():
    t0 = time.time()
    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
    sam.to("cpu"); sam.eval()
    img = cv2.imread(IMG)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    pred = SamPredictor(sam)
    pred.set_image(img_rgb)
    masks, scores, _ = pred.predict(
        point_coords=np.array([GT.mean(axis=0)]), point_labels=np.array([1]), multimask_output=True)
    # 取面积最接近棋盘(约19%)且分数高的
    mask = masks[1]
    print(f"mask 面积占比={mask.mean()*100:.1f}% score={scores[1]:.3f} 耗时{time.time()-t0:.1f}s")

    qa = quad_approx(mask)
    qm = quad_minrect(mask)
    for name, q in [("approxPolyDP", qa), ("minAreaRect", qm)]:
        if q is None:
            print(f"{name}: 拟合失败"); continue
        err = np.mean([np.linalg.norm(q[j] - GT[j]) for j in range(4)])
        print(f"{name}: 四角={np.round(q).astype(int).tolist()} 误差={err:.1f}px")

    # 用 approxPolyDP 粗角点做 refine_corners 精修
    init = qa
    init_norm = init / np.array([W, H], np.float32)
    res = refine_corners(img, init_norm)
    c = res["corners_norm"] * np.array([W, H])
    err = np.mean([np.linalg.norm(c[j] - GT[j]) for j in range(4)])
    print(f"\nrefine(SAM粗角点): grid_rms={res['grid_rms']:.2f}px n_nodes={res['n_nodes']} ortho={res['ortho_deg']:.1f}° aspect={res['aspect']:.3f}")
    print(f"  精修后四角={np.round(c).astype(int).tolist()} 误差={err:.1f}px")


if __name__ == "__main__":
    main()
