"""在 board_21.png 上可视化 SAM 分割结果 vs GT 角点。"""
import sys
import cv2
import numpy as np

SAM_ROOT = r"L:\Reinforcement_Learning\segment\segment-anything-main"
sys.path.insert(0, SAM_ROOT)
from segment_anything import sam_model_registry, SamPredictor

IMG = r"L:\projects\ar_ability_dev\apps\chinese_chess_coach_phone_version\debug_frames\board_21.png"
OUT = r"L:\projects\ar_ability_dev\apps\chinese_chess_coach_phone_version\debug_frames\board_21_sam_result.png"
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


def main():
    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
    sam.to("cpu"); sam.eval()
    img = cv2.imread(IMG)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    pred = SamPredictor(sam)
    pred.set_image(rgb)
    masks, scores, _ = pred.predict(
        point_coords=np.array([GT.mean(axis=0)]), point_labels=np.array([1]), multimask_output=True)
    mask = masks[1]
    q = quad_approx(mask)

    vis = img.copy()
    # 1) SAM 掩码：绿色半透明覆盖
    overlay = vis.copy()
    overlay[mask] = (0, 180, 0)
    vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)

    # 2) SAM 拟合四角：蓝色实线 + 圆点
    qq = q.astype(int)
    cv2.polylines(vis, [qq], True, (255, 0, 0), 2)
    for i, (x, y) in enumerate(qq):
        cv2.circle(vis, (x, y), 6, (255, 0, 0), -1)

    # 3) GT 角点：红色虚线 + 圆点
    gg = GT.astype(int)
    cv2.polylines(vis, [gg], True, (0, 0, 255), 2)
    for i, (x, y) in enumerate(gg):
        cv2.circle(vis, (x, y), 6, (0, 0, 255), 2)

    names = ["TL", "TR", "BR", "BL"]
    for i, (x, y) in enumerate(qq):
        err = np.linalg.norm(q[i] - GT[i])
        cv2.putText(vis, f"{names[i]} SAM({x},{y})", (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    for i, (x, y) in enumerate(gg):
        cv2.putText(vis, f"{names[i]} GT({x},{y})", (x + 8, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 图例
    cv2.putText(vis, "green = SAM mask", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2)
    cv2.putText(vis, "blue = SAM quad", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(vis, "red = GT corners", (10, 71), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite(OUT, vis)
    print("已保存:", OUT)
    err = np.mean([np.linalg.norm(q[i] - GT[i]) for i in range(4)])
    print(f"SAM 四角误差 = {err:.1f}px")


if __name__ == "__main__":
    main()
