"""人口水印的保守像素补读；正常文字先交给原有OCR。"""
import cv2
import numpy as np

from jccbrain.vision.eye.digits import Glyph, classify


def read_population(cell: np.ndarray | None, win_h: int) -> list[int] | None:
    if cell is None or not cell.size:
        return None
    # 半透明灰字在紫色棋盘上的min通道比背景高；金币的彩色高光大多被排除。
    white = cell.min(axis=2)
    threshold, _ = cv2.threshold(white, 0, 255, cv2.THRESH_OTSU)
    mask = (white > threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    glyphs = []
    for i in range(1, n):
        x, y, w, h, _area = (int(v) for v in stats[i])
        # 1774×1136：水印数字高61～62px(.0537～.0546H)，斜杠75px(.066H)。
        # 相连金币可使左字高72px；保留后用另一侧相同字号裁掉底部干扰。
        if .035 * win_h <= h <= .085 * win_h:
            glyphs.append(Glyph(x, y, x + w, y + h, (labels[y:y + h, x:x + w] == i).astype(np.uint8)))
    glyphs.sort(key=lambda g: g.x0)
    if len(glyphs) != 3:
        return None
    ref = min(glyphs[0].h, glyphs[2].h)
    # 实拍两侧数字顶沿差0px，容许.005H；不接受上下错开的单位血条/名字碎片。
    if abs(glyphs[0].y0 - glyphs[2].y0) > .005 * win_h:
        return None
    for i in (0, 2):
        g = glyphs[i]
        if g.h > ref * 1.1:
            glyphs[i] = Glyph(g.x0, g.y0, g.x1, g.y0 + ref, g.mask[:ref])
    found = [classify(g, ref) for g in glyphs]
    text = ''.join(c for c, _ in found)
    # 原标准+track47帧：灰水印三联最低分.804；0051亮边水印最低分.376，被保守拒绝；
    # 非水印均不能组成对齐的三字形。原有模板不变，拒绝时仍由OCR结果决定。
    if (min(s for _, s in found) < .8 or len(text) != 3 or text[1] != '/'
            or not text[0].isdigit() or not text[2].isdigit()):
        return None
    used, total = int(text[0]), int(text[2])
    return [used, total] if 1 <= total <= 9 and used <= total else None
