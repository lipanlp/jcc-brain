"""商店小图标的形状检测；不依赖 OCR，也不读取持有英雄来猜可升星状态。"""
from __future__ import annotations

import cv2
import numpy as np

from jccbrain.vision.eye.image import Region


def upgrade_region(card_name: Region) -> Region:
    """相对卡名格定位左上角星标；1774×1136 下裁剪约 69×31，双星间距约 16px。"""
    return Region(card_name.x + .006, card_name.y - .115, .038, .026)


# 五角星几何轮廓，不是从某张截图复制的亮度模板。ROI 按窗口比例裁剪，再归一化；
# 1774×1136 实测 17 个双星格的双星最低相关度 0.836，118 个非双星格最高 0.395。
# 取 0.65，避免亮度阈值漏掉 0057/0075 的暗星，也排除背景和羁绊图案。
STAR_SCORE = .65
_star = np.zeros((13, 13), np.uint8)
cv2.fillPoly(_star, [np.array([[6, 0], [8, 4], [12, 4], [9, 7], [10, 12],
                             [6, 9], [2, 12], [3, 7], [0, 4], [4, 4]], np.int32)], 255)
_star = cv2.GaussianBlur(_star, (3, 3), .5)


def upgrade_stars(patch: np.ndarray | None) -> int | None:
    """确认相邻的两个/三个五角星返回星数；无标记、单星或看不清时返回 None。"""
    if patch is None or not patch.size:
        return None
    # 固定的是归一化图上的几何比例；实际像素区域随窗口宽高缩放。
    normalized = cv2.resize(patch, (69, 31), interpolation=cv2.INTER_AREA)
    response = cv2.matchTemplate(normalized[..., :3].min(axis=2), _star, cv2.TM_CCOEFF_NORMED)
    n, labels = cv2.connectedComponents((response >= STAR_SCORE).astype(np.uint8), connectivity=8)
    positions = set()
    for label in range(1, n):
        y, x = np.unravel_index(np.where(labels == label, response, -1).argmax(), response.shape)
        # 归一化 ROI 上星形左上角实测 x=3~5 / 19~21，y=5~7，留少量抗锯齿余量。
        if 4 <= y < 10:
            if 2 <= x < 9:
                positions.add(0)
            elif 18 <= x < 25:
                positions.add(1)
            elif 34 <= x < 41:
                positions.add(2)
    # 三星目前只有合成排版回归；识别第三颗，避免把它截断成双星。
    return len(positions) if {0, 1} <= positions else None


LOCK_REGION = Region(.83, .775, .05, .045)


def shop_locked(patch: np.ndarray | None) -> bool | None:
    """锁按钮裁剪 → 锁定/未锁/无法判断。暂不接入状态输出：真实样本只有未锁图标。"""
    if patch is None or not patch.size:
        return None
    normalized = cv2.resize(patch, (90, 52), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(normalized, cv2.COLOR_BGR2HSV)
    # 1774×1136 未锁图标：金色主体约 18×24/240px，HSV 的 5%~95% 分位
    # H=19~33、S=42~108、V=88~187。边框不在中心主体内；范围留出抗锯齿余量。
    mask = cv2.inRange(hsv, (10, 35, 75), (40, 255, 255))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for label in range(1, n):
        x, y, w, h, _area = stats[label]
        if not (.1 <= w / 90 <= .4 and .3 <= h / 52 <= .8 and .3 <= (x + w / 2) / 90 <= .7):
            continue
        icon = (labels[y:y + h, x:x + w] == label).astype(np.uint8)
        contours, hierarchy = cv2.findContours(icon, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes = []
        for contour, entry in zip(contours, hierarchy[0]):
            if entry[3] < 0 or cv2.contourArea(contour) < .02 * w * h:
                continue
            _hx, hy, _hw, hh = cv2.boundingRect(contour)
            holes.append((hy + hh / 2) / h)
        # 下半部钥匙孔确认主体；上半部有闭环才是闭合锁梁。真实未锁钥匙孔中心为 0.65h；
        # 合成闭锁的梁孔中心 <0.4h。没有真实闭锁样本，不能把合成验证当作实机验收。
        if any(cy > .5 for cy in holes):
            return any(cy <= .5 for cy in holes)
    return None
