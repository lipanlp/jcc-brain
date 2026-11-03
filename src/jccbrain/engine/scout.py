"""每局侦察画像。只消费视觉结果；羁绊优先、棋子细化，不推断未知身份。"""
from __future__ import annotations

import logging
import threading
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from time import time

from jccbrain.core.models import GameState, Player, TraitStatus, Unit
from jccbrain.core.paths import var_dir
from jccbrain.engine.comps import CompMatcher

log = logging.getLogger(__name__)


def round_index(stage: str | None) -> int | None:
    try:
        major, minor = map(int, stage.split('-'))
        return major * 7 + minor if major > 0 and 1 <= minor <= 7 else None
    except (AttributeError, ValueError):
        return None


@dataclass
class ScoutShot:
    slot: int | None = None
    name: str | None = None
    round: str | None = None
    t: float = field(default_factory=time)
    traits: list[TraitStatus] = field(default_factory=list)
    level: int | None = None
    board: list[Unit] = field(default_factory=list)
    bench: list[Unit] = field(default_factory=list)

    source: str = 'active'
    loaded: bool = True  # 旧快照/被动侦察保持兼容；主动扫描明确写入加载判据。
    timing: dict = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict, *, slot=None, name=None, round=None, t=None, loaded=None, timing=None, flags=None):
        state = GameState.from_dict(data)
        viewing = data.get('viewing') or {}
        return cls(viewing.get('slot') if viewing.get('slot') is not None else slot, viewing.get('name') or name, round or state.stage,
                   time() if t is None else t, state.traits, state.level, state.board, state.bench,
                   source=data.get('source', 'active'), loaded=data.get('loaded', True) if loaded is None else loaded,
                   timing=deepcopy(data.get('timing') or {}) if timing is None else deepcopy(timing),
                   flags=list(data.get('flags') or []) if flags is None else list(flags))


@dataclass
class PlayerScout:
    name: str | None
    slot: int | None
    hp: int | None
    round: str | None
    level: int | None
    traits: list[TraitStatus]
    units: list[Unit]
    comp_guess: dict | None
    carries: list[dict]
    stale_rounds: int = 0
    t: float = 0
    source: str = 'active'
    loaded: bool = True
    timing: dict = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


class ScoutBook:
    def __init__(self, db, comps_getter=None, directory: Path | None = None):
        self.db = db
        self.comps_getter = comps_getter or db.comps_ranked
        self.directory = directory
        self._lock = threading.RLock()
        self.reset()

    def reset(self):
        with self._lock:
            self._latest: dict[str, PlayerScout] = {}
            self.history: dict[str, list[PlayerScout]] = {}
            self.players: list[Player] = []
            self._passive_rounds: set[str] = set()
            self.round: str | None = None
            # 微秒区分同一秒重开；只有收到第一张侦察结果才创建文件。
            self.started = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
            self.path: Path | None = None

    def update(self, state: GameState):
        with self._lock:
            if state.stage:
                self.round = state.stage
            if state.players and not state.players_stale:
                self.players = deepcopy(state.players)
            for row in self._latest.values():
                player = self._player(row.slot, row.name)
                if player:
                    row.hp = player.hp
                    if not row.name:
                        row.name = player.name
                now, seen = round_index(self.round), round_index(row.round)
                row.stale_rounds = max(0, now - seen) if now is not None and seen is not None else 0

    def _player(self, slot, name):
        return next((p for p in self.players if (slot is not None and p.slot == slot)
                     or (name and p.name == name)), None)

    def _trait_guess(self, traits, comps):
        observed = {t.name: t.count for t in traits if t.name and t.count and t.count > 0}
        best = None
        for comp in comps:
            expected = Counter()
            for name in {h.name for h in comp.heroes}:
                hero = self.db.hero(name)
                if hero:
                    expected.update(set(hero.traits) & set(comp.traits[:3]))
            if not expected or not (expected.keys() & observed.keys()):
                continue
            # 每条主羁绊等权；人数不足/超过都扣分，完全一致为 1。
            fit = sum(min(observed.get(n, 0), count) / max(observed.get(n, 0), count)
                      for n, count in expected.items()) / len(expected)
            if best is None or fit > best['fit']:
                best = {'id': comp.id, 'name': comp.name, 'fit': fit, 'by': 'traits'}
        return best

    def add(self, shot: ScoutShot) -> PlayerScout | None:
        with self._lock:
            shot = deepcopy(shot)
            if shot.source == 'passive' and shot.round in self._passive_rounds:
                return None
            player = self._player(shot.slot, shot.name)
            if player and player.is_self:
                return None
            shot.name = shot.name or (player.name if player else None)
            if shot.slot is None and player:
                shot.slot = player.slot
            if shot.slot is None and not shot.name and shot.source != 'passive':
                return None
            # 同名主动/被动结果合并；后来的 slot 结果也与已有具名结果关联。
            key = next((k for k, row in self._latest.items()
                        if (shot.slot is not None and row.slot == shot.slot)
                        or (shot.name and row.name == shot.name)),
                       f'slot:{shot.slot}' if shot.slot is not None else f'name:{shot.name}' if shot.name else 'unknown:passive')
            old = self._latest.get(key)
            if old:
                if shot.slot is None:
                    shot.slot = old.slot
                old_round, new_round = round_index(old.round), round_index(shot.round)
                if (old_round is not None and new_round is not None and new_round < old_round) \
                        or (new_round == old_round and shot.t < old.t):
                    return None  # 后台旧任务晚到，不覆盖更新的画像。
            comps = self.comps_getter()
            guess = self._trait_guess(shot.traits, comps)
            trusted_board = shot.loaded and 'board_not_loaded' not in shot.flags
            units = [u for u in shot.board + shot.bench if u.known] if trusted_board else []
            if trusted_board and shot.board and units:
                matches = CompMatcher(comps, self.db).match(GameState(board=shot.board, bench=shot.bench), top=None)
                match = max(matches, key=lambda m: m.fit, default=None)
                if match and match.fit > 0 and (guess is None or match.fit >= guess['fit']):
                    guess = {'id': match.comp.id, 'name': match.name, 'fit': match.fit, 'by': 'units'}
            comp = next((c for c in comps if guess and c.id == guess['id']), None)
            carry_names = {h.name for h in comp.heroes if h.is_carry} if comp else set()
            carries = [{'name': u.name, 'star': u.star} for u in units if u.star >= 2 or u.name in carry_names]
            row = PlayerScout(shot.name, shot.slot, player.hp if player else None, shot.round,
                              shot.level, shot.traits, units, guess, carries, t=shot.t, source=shot.source,
                              loaded=shot.loaded, timing=deepcopy(shot.timing), flags=list(shot.flags))
            self._latest[key] = row
            self.history.setdefault(key, []).append(deepcopy(row))
            self.history[key] = self.history[key][-3:]
            self.update(GameState(stage=self.round))
            if shot.source == 'passive' and shot.round:
                self._passive_rounds.add(shot.round)
            self._persist(shot)
            return deepcopy(row)

    def _persist(self, shot):
        import json
        try:
            directory = self.directory if self.directory is not None else var_dir() / 'scout'
            directory.mkdir(parents=True, exist_ok=True)
            self.path = directory / f'{self.started}.jsonl'
            with self.path.open('a', encoding='utf-8') as out:
                out.write(json.dumps(asdict(shot), ensure_ascii=False) + '\n')
        except Exception:
            log.exception('侦察快照写入失败')

    def latest(self) -> list[PlayerScout]:
        with self._lock:
            return deepcopy(list(self._latest.values()))

    def last_seen_hp(self, slot: int) -> int | None:
        """侦察当时血量，不是update补入的当前血量，用于跳过近期且血量未变者。"""
        with self._lock:
            return next((history[-1].hp for history in self.history.values()
                         if history and history[-1].slot == slot), None)

    def passive_recorded(self, round: str) -> bool:
        with self._lock:
            return round in self._passive_rounds

    def taken(self) -> Counter:
        taken = Counter()
        for row in self.latest():
            if (row.slot is not None or row.name) and (row.hp is None or row.hp > 0):
                for unit in row.units:
                    taken[unit.name] += unit.copies
        return taken

    def pool_basis(self) -> str:
        rounds = list(dict.fromkeys(row.round for row in self.latest() if row.units and row.round
                                   and (row.hp is None or row.hp > 0)))
        return '按自己、侦察（' + ('、'.join(rounds) if rounds else '尚无棋盘数据') + ' 回合）与手动登记估算'

    def view(self, target_id=None) -> list[dict]:
        rows = sorted(self.latest(), key=lambda r: (r.hp is None, -(r.hp or 0)))[:7]
        return [{**asdict(row), 'key': f'slot:{row.slot}' if row.slot is not None else f'name:{row.name}' if row.name else 'unknown:passive',
                 'contested': bool(row.comp_guess and row.comp_guess['id'] == target_id and row.comp_guess['fit'] >= .4)}
                for row in rows]
