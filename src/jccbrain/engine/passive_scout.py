"""战斗对手的被动侦察：确认相邻画面稳定后，每回合仅消费一次。"""
from __future__ import annotations

import numpy as np

from jccbrain.core.models import GameState, Phase
from jccbrain.engine.scout import ScoutShot

STABLE_THRESHOLD = 6


def board_thumbnail(image: np.ndarray) -> np.ndarray | None:
    """游戏中央棋盘区（含敌我半场），排除右侧名单/底部商店，原图缩小1/8。

    在原始像素0..255量纲上比较，不把JPEG字节或单位名字变化当成画面差异。
    """
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        return None
    h, w = image.shape[:2]
    roi = image[int(h * .12):int(h * .66):8, int(w * .20):int(w * .85):8]
    return roi.astype(np.float32) if roi.size else None


class PassiveScout:
    def __init__(self, book):
        self.book = book
        self._generation = book.started
        self.reset()

    def reset(self):
        self._done: set[str] = set()
        self._key = None
        self._previous = None

    def needs_image(self, state: GameState) -> bool:
        if self._generation != self.book.started:
            self.reset()
            self._generation = self.book.started
        return bool(state.phase == Phase.COMBAT and state.stage and state.stage not in self._done
                    and not self.book.passive_recorded(state.stage)
                    and state.opponent and state.opponent.get('board'))

    def observe(self, state: GameState, image: np.ndarray | None, threshold=STABLE_THRESHOLD):
        if not self.needs_image(state):
            self._key = self._previous = None
            return None
        opponent = state.opponent
        key = (state.stage, opponent.get('name'))
        thumbnail = board_thumbnail(image)
        previous = self._previous if key == self._key else None
        self._key, self._previous = key, thumbnail
        if thumbnail is None or previous is None or thumbnail.shape != previous.shape:
            return None
        if float(np.mean(np.abs(thumbnail - previous))) >= threshold:
            return None
        # 只使用明确的名字；匿名观察不根据场上单位或匹配阵容猜身份。
        shot = ScoutShot.from_dict({'board': opponent['board']}, name=opponent.get('name'), round=state.stage)
        shot.source = 'passive'
        result = self.book.add(shot)
        self._done.add(state.stage)
        return result
