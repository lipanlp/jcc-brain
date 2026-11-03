"""全自动托管；海克斯手动接管与自动选项会话均在模式内锁存。"""
from __future__ import annotations

import threading
from uuid import uuid4

from jccbrain.actuator.actions import PickAugment
from jccbrain.core.models import Phase
from jccbrain.modes.base import Mode
from jccbrain.modes.scout import own_view


class AutopilotMode(Mode):
    key = "autopilot"
    title = "全自动托管"
    acts = True

    def __init__(self, ctx, sink):
        super().__init__(ctx, sink)
        self.augment_selection = ctx.settings.autopilot.augment_selection
        self._pause_lock = threading.Lock()
        self._instance = uuid4().hex
        self._pause_serial = 0
        self._paused = False
        self._token = None
        self._stage = None
        self._clear_frames = 0
        self._clear_key = None
        self._last_frame = None
        self._auto_pending = False
        self._can_resume = False
        self._read_serial = 0
        self._observed_serial = -1

    def read(self, fresh=True):
        st = self.ctx.vision.read(fresh)
        with self._pause_lock:
            self._read_serial += 1
        self.sink.on_state(st)
        return st

    @property
    def status(self):
        with self._pause_lock:
            return {'augment_selection': self.augment_selection, 'paused': self._paused,
                    'token': self._token, 'stage': self._stage,
                    'can_resume': self._paused and self._can_resume and not self.stopped
                                  and self._observed_serial == self._read_serial,
                    'reason': ('海克斯由你处理；选择后等待两帧有效己方画面，再确认继续'
                               if self._paused else '等待海克斯选项结束' if self._auto_pending else '')}

    def _on_new_game(self):
        with self._pause_lock:
            self._paused = self._auto_pending = self._can_resume = False
            self._token = None
            self._stage = None
            self._clear_frames = 0
            self._clear_key = None
        self.sink.on_mode_changed()

    @staticmethod
    def _own_frame(st):
        return (bool(st.viewing) and any(st.viewing.get(k) is not None for k in ('slot', 'name'))
                and sum(p.is_self for p in st.players) == 1 and own_view(st) and not st.players_stale)

    @staticmethod
    def _operable(st):
        return (st.phase in (Phase.PLANNING, Phase.COMBAT) and st.shop_open
                and st.stage_tuple is not None and st.stage_tuple[0] >= 1 and 1 <= st.stage_tuple[1] <= 7
                and AutopilotMode._own_frame(st)
                and not st.units_stale and not st.players_stale
                and isinstance(st.gold, int) and not isinstance(st.gold, bool) and st.gold >= 0
                and isinstance(st.level, int) and not isinstance(st.level, bool) and 1 <= st.level <= 10
                and not st.augments)

    def _observe_augments(self, st):
        """每次 fresh read 后撤销旧的可继续状态；候选残留也视为海克斯。"""
        with self._pause_lock:
            before = (self._paused, self._token, self._can_resume, self._auto_pending)
            options = st.phase == Phase.AUGMENT or bool(st.augments)
            if options and self.augment_selection == 'manual' and not self._paused:
                self._paused = True
                self._stage = st.stage
                self._pause_serial += 1
                self._token = f'{self._instance}:{self._pause_serial}'
            if self._paused and self._stage != st.stage:
                # 回合变化使旧按钮授权失效；仍需对新画面重新人工确认。
                self._stage = st.stage
                self._pause_serial += 1
                self._token = f'{self._instance}:{self._pause_serial}'
            clear = self._operable(st)
            key = (st.stage, st.phase, (st.viewing or {}).get('slot'), (st.viewing or {}).get('name'))
            fresh = st.frame is None or st.frame != self._last_frame
            if not clear:
                self._clear_frames = 0
                self._clear_key = None
            elif key != self._clear_key:
                self._clear_key = key
                self._clear_frames = 1
            elif fresh:
                self._clear_frames += 1
            self._last_frame = st.frame
            self._observed_serial = self._read_serial
            self._can_resume = clear and self._clear_frames >= 2
            if self._auto_pending and self._can_resume:
                self._auto_pending = False
            blocked = self._paused or self._auto_pending
            changed = before != (self._paused, self._token, self._can_resume, self._auto_pending)
        if changed:
            self.sink.on_mode_changed()
        return blocked

    def resume(self, token):
        with self._pause_lock:
            if self.stopped:
                raise ValueError('托管已经停止，不能继续')
            if not self._paused or token != self._token:
                raise ValueError('这次暂停已失效，请查看当前托管状态')
            if not self._can_resume or self._observed_serial != self._read_serial:
                raise ValueError('请先完成海克斯选择，并等待两帧有效己方可操作画面')
            self._paused = self._can_resume = False
            self._token = None
        self.sink.on_mode_changed()

    def loop(self) -> str:
        last_sig = None
        repeats = 0
        try:
            while not self.stopped:
                st = self.read()
                if self.stopped:
                    break
                if st.phase == Phase.ENDED:
                    return f"对局结束：第 {st.placement or '?'} 名"
                adv = self.advise(st)
                if self._observe_augments(st):
                    self.sleep(0.4)
                    continue
                plan = self.ctx.planner.plan(st, adv)
                if not any(isinstance(a, PickAugment) for a in plan) and not self._operable(st):
                    plan = []
                if any(isinstance(a, PickAugment) for a in plan) and not self._own_frame(st):
                    plan = []
                if not plan:
                    self.sleep(0.4)
                    continue
                if any(isinstance(a, PickAugment) for a in plan):
                    # 发出前即占用选项会话；画面/阶段抖动不能重选。
                    with self._pause_lock:
                        self._auto_pending = True
                        self._clear_frames = 0
                        self._clear_key = None
                    plan = [next(a for a in plan if isinstance(a, PickAugment))]
                    self.sink.on_mode_changed()
                sig = (st.stage, st.gold, tuple(st.shop_names), tuple(a.describe() for a in plan))
                repeats = repeats + 1 if sig == last_sig else 0
                last_sig = sig
                if repeats >= 2 and not any(isinstance(a, PickAugment) for a in plan):
                    self.sink.on_log("画面没有变化，操作可能没生效（标定不准或被遮挡），暂停 3 秒")
                    self.sleep(3)
                    continue
                self.sink.on_log("执行：" + "，".join(a.describe() for a in plan) + f"（{adv.headline}）")
                self.act(plan)
            return "已停止"
        finally:
            with self._pause_lock:
                self._paused = self._auto_pending = self._can_resume = False
                self._token = None
            self.sink.on_mode_changed()
