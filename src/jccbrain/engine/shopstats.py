"""本局商店来牌计数：连续两帧确认内容，同一快照只计一次。"""
from __future__ import annotations

from collections import Counter

from jccbrain.core.models import GameState


class ShopStats:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.counts: Counter[str] = Counter()
        self._pending: tuple[str | None, ...] | None = None
        self._counted: tuple[str | None, ...] | None = None

    def update(self, state: GameState) -> None:
        if not state.shop_open:
            self._pending = None
            return
        snapshot = tuple(state.shop_names[:5])
        if snapshot == self._pending and snapshot != self._counted:
            # 仅有格子变空是买卡；出现新卡或换卡才算下一次商店。
            if self._counted is None or any(
                    name and name != self._counted[i] for i, name in enumerate(snapshot)):
                self.counts.update(name for name in snapshot if name)
            self._counted = snapshot
        self._pending = snapshot
