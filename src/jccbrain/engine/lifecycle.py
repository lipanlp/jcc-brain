"""辅助模式的原始输入守卫；不识别像素，也不使用稳定器沿用的 HUD 字段。"""
from __future__ import annotations

from jccbrain.core.models import GameState, Phase


def has_live_hud(state: GameState) -> bool:
    """单个 HUD 字段缺失不暂停；残留单位、商店与 phase 不能证明画面有效。"""
    return (state.stage_tuple is not None or state.level is not None or state.gold is not None
            or (state.phase == Phase.AUGMENT
                and any(option.name.strip() or option.desc.strip() for option in state.augments)))


class AssistantLifecycle:
    def __init__(self):
        self.status = 'active'
        self.generation = 0
        self.placement: int | None = None
        self._early_frames = 0

    @property
    def active(self) -> bool:
        return self.status == 'active'

    def _transition(self, status: str) -> None:
        if status != self.status:
            self.status = status
            self.generation += 1

    def observe(self, raw: GameState) -> bool:
        """返回可信新局已确认；调用者须在同一发布锁内使用状态及代次。"""
        if self.status == 'stopped':
            return False
        if raw.phase == Phase.ENDED:
            self.placement = raw.placement or self.placement
            self._early_frames = 0
            self._transition('ended')
        elif self.status == 'ended':
            # 明确结束后，旧回合残帧不能唤回。连续三帧第一阶段且人口/金币
            # 与新局一致才恢复；仅有旧 stage 或稳定后的等级都不算此信号。
            stage = raw.stage_tuple
            early = (stage is not None and stage[0] == 1 and 1 <= stage[1] <= 4
                     and raw.phase in (Phase.PLANNING, Phase.PVE, Phase.COMBAT)
                     and raw.level is not None and 1 <= raw.level <= 4 and raw.gold is not None)
            self._early_frames = self._early_frames + 1 if early else 0
            if self._early_frames >= 3:
                self._early_frames = 0
                self.placement = None
                self._transition('active')
                return True
        else:
            self._transition('active' if has_live_hud(raw) else 'waiting')
        return False

    def invalidate(self) -> None:
        self.generation += 1

    def stop(self) -> None:
        self._transition('stopped')

    def neutral_state(self) -> GameState:
        ended = self.status == 'ended'
        return GameState(phase=Phase.ENDED if ended else Phase.UNKNOWN,
                         placement=self.placement if ended else None)

    @property
    def headline(self) -> str:
        if self.status == 'ended':
            return f'对局结束，第 {self.placement} 名' if self.placement else '对局结束'
        return '等待有效对局画面，建议已暂停'
