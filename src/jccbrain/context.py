"""运行上下文：按设置把各模块组装起来，CLI 和 UI 共用。

各模块之间不互相创建对方，统一由这里注入，替换任何一个实现（比如换视觉后端）只改这里。
"""
from __future__ import annotations

import logging
import threading
from functools import cached_property

from jccbrain.actuator.executor import Executor, create_executor
from jccbrain.actuator.layout import Layout
from jccbrain.actuator.safety import SafetyGuard
from jccbrain.actuator.window import WindowLocator
from jccbrain.core.rules import SetRules, rules_for_season
from jccbrain.core.settings import Settings
from jccbrain.data.db import GameDB, enabled_comp_sources
from jccbrain.engine.advisor import HeuristicAdvisor
from jccbrain.engine.llm import LLMClient
from jccbrain.engine.scout import ScoutBook
from jccbrain.engine.planner import AutoPlanner
from jccbrain.vision import VisionProvider, create_provider

log = logging.getLogger(__name__)


class AppContext:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.load()
        self.selection_lock = threading.RLock()
        self.game_lock_id: str | None = None  # None 未初始化；空字符串表示本局主动解除。
        self.game_watch_ids: list[str] = []   # 本局关注，不持久化；API 最多接受 4 个。
        self.game_augments: list[str] = []    # 视觉已选海克斯的规范化名单，仅在本局内保留。
        self.guard = SafetyGuard(self.settings.actuator.max_actions_per_sec, self.settings.actuator.failsafe_corner)
        self.extra_taken: dict[str, int] = {}     # 手动补充：别人拿走了几张

    def effective_lock_id(self) -> str | None:
        value = (self.settings.assistant.default_lock_comp_id
                 if self.game_lock_id is None else self.game_lock_id)
        return value if value and self.enabled_comp(value) else None

    def selected_lock_id(self) -> str:
        """用户选择的来源身份；暂不可用时仍保留，不能退回全库。"""
        return (self.settings.assistant.default_lock_comp_id
                if self.game_lock_id is None else self.game_lock_id)

    def comp_identity(self, id: str) -> dict:
        original = self.db.comp(id, raw=True)
        canonical = self.enabled_comp(id)
        return {'id': id, 'name': original.name if original else id,
                'canonical_id': canonical.id if canonical and canonical.heroes else None,
                'available': bool(canonical and canonical.heroes)}

    def enabled_comps(self, source: str = 'all'):
        return self.db.comps_ranked(source, enabled_sources=enabled_comp_sources(self.settings.data))

    def enabled_comp_catalog(self, source: str = 'all'):
        return self.db.comps_catalog(source, enabled_sources=enabled_comp_sources(self.settings.data))

    def enabled_comp(self, id: str, *, raw: bool = False):
        return self.db.comp(id, enabled_sources=enabled_comp_sources(self.settings.data), raw=raw)

    def assistant_selection(self):
        from jccbrain.core.comp_presets import snapshot
        with self.selection_lock:
            return snapshot(self)

    def refresh_comp_sources(self) -> list[str]:
        """热重载匹配器；来源关闭仅暂停选择，恢复后继续使用原身份。"""
        messages = []
        if self.game_lock_id and not self.enabled_comp(self.game_lock_id):
            messages.append(f'阵容来源已关闭，本局锁定保留但暂不可用：{self.game_lock_id}')
        unavailable = [id for id in self.game_watch_ids if not self.enabled_comp(id)]
        if unavailable:
            messages.append('部分关注来源已关闭，记录保留但暂不可用：' + '、'.join(unavailable))
        self.advisor.reload()
        for message in messages:
            log.info(message)
        return messages

    @cached_property
    def db(self) -> GameDB:
        return GameDB(season_id=self.settings.data.season_id)

    @cached_property
    def rules(self) -> SetRules:
        return rules_for_season(self.db.season)

    @cached_property
    def scout(self) -> ScoutBook:
        return ScoutBook(self.db, self.enabled_comps)

    @cached_property
    def advisor(self) -> HeuristicAdvisor:
        return HeuristicAdvisor(self.db, self.rules, self.enabled_comps, self.scout,
                                economy_preference_getter=lambda: self.settings.assistant.economy_preference)

    @cached_property
    def layout(self) -> Layout:
        return Layout.load()

    @cached_property
    def locator(self) -> WindowLocator:
        return WindowLocator(self.settings.actuator.window_owner)

    @cached_property
    def planner(self) -> AutoPlanner:
        return AutoPlanner(self.db, self.layout)

    # 以下几个依赖设置，设置变更后调用 reset_runtime() 重建
    @cached_property
    def vision(self) -> VisionProvider:
        return create_provider(self.settings.vision, self.db, self.settings.actuator.window_owner)

    @cached_property
    def llm(self) -> LLMClient:
        return LLMClient(self.settings.llm)

    @cached_property
    def executor(self) -> Executor:
        return create_executor(self.settings.actuator, self.layout, self.guard)

    def reset_runtime(self) -> None:
        """设置改动后重建视觉/大模型/执行器。"""
        if "vision" in self.__dict__:
            self.vision.stop()
        for k in ("vision", "llm", "executor", "layout", "planner", "locator"):
            self.__dict__.pop(k, None)
        self.guard.max_actions_per_sec = self.settings.actuator.max_actions_per_sec
        self.guard.failsafe_corner = self.settings.actuator.failsafe_corner

    def reset_llm(self) -> None:
        """只重建大模型客户端（切换档位时用，不打断正在运行的模式）。"""
        self.__dict__.pop("llm", None)

    def reload_data(self, loaded: GameDB | None = None) -> None:
        if loaded is None:
            self.db.reload()
        else:
            self.db.reload_from(loaded)
        self.advisor.reload()

    def switch_season(self, season_id: str) -> None:
        """切换赛季：保存设置、数据库切到该赛季，并重建依赖赛季数据的模块（规则、建议、视觉词典等）。

        调用前应先停止正在运行的模式。锁定的目标阵容和指定的游戏版本属于旧赛季，一并清掉。
        """
        season_id = str(season_id)
        if season_id != self.db.season_id:
            self.settings.assistant.default_lock_comp_id = ""
            self.game_lock_id = None
            self.game_watch_ids = []
            self.game_augments = []
            self.scout.reset()
            self.settings.data.game_version = ""
            self.settings.data.official_version = ""
        self.settings.data.season_id = season_id
        self.settings.save()
        self.db.use_season(season_id)
        for k in ("rules", "advisor"):
            self.__dict__.pop(k, None)
        self.extra_taken.clear()
        self.reset_runtime()

    def sync(self, details: bool = True, progress=None, season_id: str | None = None,
             db: GameDB | None = None) -> dict:
        """同步一个赛季；官网/微社区共用完整官方快照，来源开关控制消费渠道。"""
        from jccbrain.core.seasons import season
        from jccbrain.data.dataj import DatajClient
        from jccbrain.data.official import OfficialClient
        from jccbrain.data.sync import SyncError, sync_season

        d = self.settings.data
        if not enabled_comp_sources(d):
            raise SyncError('至少要选一个数据源')
        target = db or self.db
        s = target.season if season_id is None else season(season_id, target.catalog())
        same = s.id == d.season_id
        official = OfficialClient() if d.source_official or d.source_ingame else None
        # 赛季没有 dataj 数据时 sync_season 会提示并跳过
        dataj = DatajClient(d.base_url, s.dataj_set or 0, d.request_interval_s) if d.source_dataj else None
        try:
            return sync_season(target, s, official=official, dataj=dataj,
                               official_version=(d.official_version if same else "") or None,
                               dataj_version=(d.game_version if same else "") or None,
                               details=details, progress=progress)
        finally:
            for c in (official, dataj):
                if c:
                    c.close()

    def seasons_needing_sync(self, db: GameDB | None = None) -> list[tuple[str, str | None, str | None]]:
        """需要同步的赛季：没同步过，或官方资料库有新版本。返回 [(season_id, 本地版本, 官方最新版本)]。

        只读官方版本索引，不下载数据文件；官网和微社区均关闭时只返回没同步过的赛季。
        """
        from jccbrain.data.official import OfficialClient, OfficialError

        target = db or self.db
        synced = target.synced_seasons()
        d = self.settings.data
        official = OfficialClient() if d.source_official or d.source_ingame else None
        out = []
        try:
            if official:
                try:
                    target.save_catalog(official.season_tabs())
                except OfficialError:
                    pass
            for s in target.catalog():
                local = (synced.get(s.id) or {}).get("game_version")
                newest = official.newest_version(s) if official else None
                if local is None or (newest and newest != local):
                    out.append((s.id, local, newest))
        finally:
            if official:
                official.close()
        return out

    def sync_all_seasons(self, details: bool = True, progress=None, only_needed: bool = True,
                         db: GameDB | None = None) -> list[str]:
        """同步官网列出的全部赛季（默认只同步没同步过或有新版本的）。单个赛季失败不影响其他赛季。返回同步了的赛季。"""
        say = progress or (lambda m: log.info(m))
        if not enabled_comp_sources(self.settings.data):
            from jccbrain.data.sync import SyncError
            raise SyncError('至少要选一个数据源')
        target = db or self.db
        current = target.season_id
        todo = ([sid for sid, _, _ in self.seasons_needing_sync(target)] if only_needed
                else [s.id for s in target.catalog()])
        if not todo:
            say("所有赛季都是最新的，不需要同步")
        done = []
        for sid in todo:
            try:
                self.sync(details, progress, season_id=sid, db=target)
                done.append(sid)
            except Exception as e:  # noqa: BLE001 - 一个赛季失败不影响其他赛季
                log.warning("赛季 %s 同步失败：%s", sid, e)
                say(f"赛季 {sid} 同步失败：{e}")
        target.use_season(current)
        return done

    def close(self) -> None:
        if "vision" in self.__dict__:
            self.vision.stop()
