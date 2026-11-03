"""连续多帧确认换局，容忍缺帧和单帧阶段回跳。"""
from __future__ import annotations

from jccbrain.core.models import GameState, Phase


class GameWatcher:
    def __init__(self):
        self._max_stage: tuple[int, int] | None = None
        self._ended = False
        self._early_frames = 0

    def update(self, state: GameState) -> bool:
        stage = state.stage_tuple
        ended_before = self._ended
        self._ended = self._ended or state.phase == Phase.ENDED
        if stage is None:
            return False
        early = ((self._max_stage is not None and self._max_stage >= (2, 1) and stage <= (1, 4))
                 or (ended_before and stage[0] == 1))
        self._early_frames = self._early_frames + 1 if early else 0
        if self._early_frames >= 3:
            self._max_stage = stage
            self._ended = False
            self._early_frames = 0
            return True
        self._max_stage = max(self._max_stage or stage, stage)
        return False
