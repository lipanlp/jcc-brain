"""顶部回合条：只认可见图标，圆标下方三角短暂消失时沿用同回合的指针。"""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

TEMPLATES = Path(__file__).with_name('phase_icons.npz')
SCENE_TEMPLATES = Path(__file__).with_name('phase_scenes.npz')
# 1774×1136原始27帧：七个图标中心，裁剪全部按窗口比例。
CENTERS = (.411, .440, .470, .499, .529, .558, .588)
# 可见指针的同类相关最低.903，异位最高.560；隐藏指针不参与正例。
CURSOR_SCORE = .8
# 下一图标同类最低.9999、异类最高.6476（含变暗海克斯帧）。
ICON_SCORE = .8
# 当前格包含对手头像。1774×1136原27帧：同类最低.7871、异类最高.6242；
# 取.75，仅当前格使用，不放宽下一格阈值。头像模板只覆盖已核对的旧对局。
CURRENT_ICON_SCORE = .75


def patch(img: np.ndarray, index: int, cursor: bool = False) -> np.ndarray | None:
    h, w = img.shape[:2]
    x = CENTERS[index]
    y0, y1, dx, size = (.064, .071, .007, (24, 8)) if cursor else (.032, .063, .012, (32, 32))
    cell = img[round(y0 * h):round(y1 * h), round((x - dx) * w):round((x + dx) * w)]
    return cv2.resize(cell, size) if cell.size else None


def vector(cell: np.ndarray, cursor: bool = False) -> np.ndarray:
    a = cell.astype(np.float32)
    a = a[..., 2] - a[..., 0] if cursor else a.max(axis=2)
    v = a.ravel() - a.mean()
    return v / max(float(np.linalg.norm(v)), 1e-6)


@lru_cache(maxsize=1)
def templates():
    if not TEMPLATES.exists():
        return None
    with np.load(TEMPLATES, allow_pickle=False) as z:
        return z['cursor'], z['icons'], z['names'].tolist()


@lru_cache(maxsize=1)
def current_templates():
    if not TEMPLATES.exists():
        return None
    with np.load(TEMPLATES, allow_pickle=False) as z:
        if 'current_icons' not in z:
            return None
        return z['current_icons'], z['current_names'].tolist()


def detect_reconnect(_img: np.ndarray) -> bool:
    # TODO: 需真实断线重连截图；回放没有此场景，未标定前保守不识别。
    return False


def dialog_patch(img: np.ndarray) -> np.ndarray | None:
    """只看中央取消/确定按钮，按窗口比例裁剪；目前实拍仅退出录像确认框。"""
    h, w = img.shape[:2]
    cell = img[round(.57*h):round(.65*h), round(.35*w):round(.65*w)]
    return cv2.resize(cell, (128, 36), interpolation=cv2.INTER_AREA) if cell.size else None


@lru_cache(maxsize=1)
def scene_templates():
    if not SCENE_TEMPLATES.exists():
        return None
    with np.load(SCENE_TEMPLATES, allow_pickle=False) as data:
        return data['dialog']


def dialog_score(img: np.ndarray) -> float:
    cell, template = dialog_patch(img), scene_templates()
    return float(vector(cell) @ template) if cell is not None and template is not None else -1.


def detect_scene(img: np.ndarray, round_type: str = 'unknown') -> str | None:
    """瞬时覆盖，不污染准备/战斗的跨帧记忆；确认框>选秀>观战。"""
    from jccbrain.vision.eye.augments import spectate_score

    # 1774×1136：5张退出录像确认框最低.99999；旧27+新其他153张最高.23371。
    # 五张同一按钮布局仅背景不同；未宣称覆盖其他弹窗或登录框。
    if dialog_score(img) >= .8:
        return 'dialog'
    # 当前格图标已通过StagebarReader门限；不能用回合号或下一格的选秀图标猜。
    if round_type == 'carousel':
        return 'carousel'
    # 1774×1136普通观战5帧最低.6808，本人帧最高.5161，取.60。
    if spectate_score(img) >= .60:
        return 'spectate'
    return None


class StagebarReader:
    def __init__(self):
        self.reset()

    def reset(self):
        self.stage = None
        self.index = None

    def read_rounds(self, img: np.ndarray, stage: str | None) -> tuple[str, str]:
        """一次定位，同时读下一格/当前格；当前头像不能用阶段号推断。"""
        next_round = self.read(img, stage)
        data = current_templates()
        if not stage or self.index is None or data is None:
            return next_round, 'unknown'
        cell = patch(img, self.index)
        if cell is None:
            return next_round, 'unknown'
        icons, names = data
        matches = icons @ vector(cell)
        best = int(np.argmax(matches))
        return next_round, names[best] if matches[best] >= CURRENT_ICON_SCORE else 'unknown'

    def read_current(self, img: np.ndarray, stage: str | None) -> str:
        return self.read_rounds(img, stage)[1]

    def read(self, img: np.ndarray, stage: str | None) -> str:
        if stage != self.stage:
            self.stage, self.index = stage, None
        data = templates()
        if not stage or data is None:
            return 'unknown'
        cursor, icons, names = data
        scores = []
        for i in range(len(CENTERS)):
            cell = patch(img, i, True)
            scores.append(float(vector(cell, True) @ cursor) if cell is not None else -1.)
        best = int(np.argmax(scores))
        if scores[best] >= CURSOR_SCORE:
            self.index = best
        # 最后一格的下一回合在当前条外，不能借回合号硬猜。
        if self.index is None or self.index + 1 == len(CENTERS):
            return 'unknown'
        cell = patch(img, self.index + 1)
        if cell is None:
            return 'unknown'
        matches = icons @ vector(cell)
        best = int(np.argmax(matches))
        return names[best] if matches[best] >= ICON_SCORE else 'unknown'
