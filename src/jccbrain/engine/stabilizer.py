"""状态稳定器：把逐帧识别的噪声压平，再交给规则引擎。

视觉每帧独立识别，常见的抖动有：
- 金币/等级只在商店栏里显示，战斗中商店收起就读不到；
- 「准备阶段」横幅出现的一两帧棋盘被遮住，读成空棋盘；
- 某个数字或某格商店偶尔误读一帧（金币 12 读成 2）；
- 同一个单位这一帧认出来了，下一帧又是「?」，星级时有时无；
- 海克斯刚弹出时名字还没显示。

规则引擎直接吃这些帧，建议就会一闪一闪。这里按「同一局内沿用 + 新值连续两帧才采用」处理，
只用帧数不用时间，离线回放和实时结果一致。操作游戏的模式不要用它（它会沿用旧画面），只给辅助大脑展示用。
"""
from __future__ import annotations

from typing import Any

from jccbrain.core.models import GameState, Phase
from jccbrain.engine.economy import hp_rank
from jccbrain.engine.session import GameWatcher

# 读不到时沿用上次值的字段
CARRY = ("gold", "interest", "level", "xp", "hp", "xp_cost", "refresh_cost", "odds", "streak", "board_count")
# 变化要连续两帧一致才采用的字段（单帧误读）
CONFIRM = ("gold", "level", "xp", "hp", "streak")
IN_GAME = ("planning", "combat", "augment")


def _stage_tuple(s: str | None) -> tuple[int, int] | None:
    try:
        a, b = (s or "").split("-", 1)
        return int(a), int(b)
    except ValueError:
        return None


def _sanitize(d: dict[str, Any]) -> dict[str, Any]:
    """明显不可能的读数当作没读到（经验「9/0」之类）。"""
    xp = d.get("xp")
    if xp is not None and not (isinstance(xp, list) and len(xp) == 2 and 0 <= xp[0] < xp[1]):
        d = {**d, "xp": None}
    return d


class StateStabilizer:
    def __init__(self, empty_frames: int = 6):
        self.empty_frames = empty_frames      # 连续这么多帧读空才认为真的空了
        self.reset()

    def reset(self) -> None:
        self.acc: dict[str, Any] = {}         # 已采用的状态
        self.prev: dict[str, Any] = {}        # 上一帧原始识别
        self.empty_run: dict[str, int] = {}
        self._hp_rank: tuple[int, int] | None = None
        self._pending_hp_rank: tuple[int, int] | None = None
        self._last_stage: tuple[int, int] | None = None
        self._game_watcher = GameWatcher()

    def update(self, d: dict[str, Any]) -> dict[str, Any]:
        d = _sanitize(d)
        out = dict(d)
        stage = d.get("stage")
        acc_stage = self.acc.get("stage")
        new_t, old_t = _stage_tuple(stage), self._last_stage
        new_game = self._game_watcher.update(GameState(stage=stage, phase=Phase.parse(d.get("phase"))))
        if new_game or (new_t and old_t and new_t < (2, 1) <= old_t):
            self.reset()                      # 新的一局
            acc_stage = None
        if new_t:
            self._last_stage = new_t           # 中间暂时读不到阶段，也不能忘记换局边界
        if not stage and acc_stage and d.get("phase") in IN_GAME:
            out["stage"] = stage = acc_stage
        in_game = bool(stage)
        same_round = in_game and stage == acc_stage

        if in_game:
            for k in CARRY:
                v, old = d.get(k), self.acc.get(k)
                if v is None:
                    if old is not None:
                        out[k] = old
                elif k in CONFIRM and (old is not None or k == "streak") and v != old and self.prev.get(k) != v:
                    out[k] = old              # 新值只出现了一帧，先不采用
            self._rank(out)
            self._shop(d, out, same_round)
            for zone in ("board", "bench", "traits"):
                self._zone(zone, d, out)
            if d.get("phase") == "augment":
                self._augments(d, out, same_round)
        else:
            self._pending_hp_rank = None

        # 已选强化是本局事实；漏读/部分列表/非对局帧都不能删掉已确认的名字。
        taken = list(self.acc.get("augments_taken") or [])
        incoming = d.get("augments_taken")
        if isinstance(incoming, list):
            for name in incoming:
                if isinstance(name, str) and name.strip() and name not in taken:
                    taken.append(name)
        out["augments_taken"] = taken

        self.prev = d
        self.acc = out
        return out

    # ------------------------------------------------------------ 细节
    def _rank(self, out: dict[str, Any]) -> None:
        """只稳定名次/存活人数，不改玩家栏原始血量和 stale 标记。"""
        candidate = hp_rank(GameState.from_dict(out))
        if candidate is None:
            self._pending_hp_rank = None       # 不可用时沿用同局最后可信排名
        elif self._hp_rank is None or candidate in (self._hp_rank, self._pending_hp_rank):
            self._hp_rank = candidate          # 首次有效立即采用，后续变化连续两帧才采用
            self._pending_hp_rank = None
        else:
            self._pending_hp_rank = candidate
        out["hp_rank"] = list(self._hp_rank) if self._hp_rank else None

    def _shop(self, d: dict, out: dict, same_round: bool) -> None:
        shop = d.get("shop") or {}
        acc = self.acc.get("shop") or {}
        if not shop.get("open"):
            # 商店收起：同一回合内商店不会变（刷新要先打开），沿用最后看到的
            if same_round and acc.get("slots"):
                out["shop"] = {**acc, "open": True, "stale": True}
            return
        slots = list(shop.get("slots") or [])
        acc_slots = acc.get("slots") or []
        prev_shop = self.prev.get("shop") or {}
        prev_slots = (prev_shop.get("slots") or []) if prev_shop.get("open") else []

        def name(s):
            return (s or {}).get("name")

        for i, s in enumerate(slots):
            a = acc_slots[i] if i < len(acc_slots) else None
            confirmed = i < len(prev_slots) and name(s) == name(prev_slots[i])
            if a is not None and name(s) != name(a) and not confirmed:
                slots[i] = a                  # 这一格只变了一帧
        out["shop"] = {**shop, "slots": slots}

    def _zone(self, zone: str, d: dict, out: dict) -> None:
        units = d.get(zone) or []
        acc = self.acc.get(zone) or []
        if not units and acc:
            self.empty_run[zone] = self.empty_run.get(zone, 0) + 1
            if self.empty_run[zone] < self.empty_frames:
                out[zone] = acc
            return
        self.empty_run[zone] = 0
        if zone == "traits":
            return
        out[zone] = [self._fill_unit(u, acc, zone) for u in units]

    @staticmethod
    def _fill_unit(u: dict, acc: list[dict], zone: str) -> dict:
        """这一帧没认出名字/星级的单位，用同一位置上一帧的结果补上。"""
        name = str(u.get("name") or "?")
        if not name.startswith("?") and u.get("star") is not None:
            return u
        if zone == "bench":
            match = next((a for a in acc if a.get("slot") == u.get("slot") and u.get("slot") is not None), None)
        else:
            match = None
            if u.get("x") is not None and u.get("y") is not None:
                near = [(abs(a["x"] - u["x"]) + abs(a["y"] - u["y"]), a) for a in acc
                        if a.get("x") is not None and a.get("y") is not None]
                near = [n for n in near if n[0] < 0.04]
                match = min(near, key=lambda n: n[0])[1] if near else None
        if not match:
            return u
        u = dict(u)
        m_name = str(match.get("name") or "?")
        if name.startswith("?") and not m_name.startswith("?"):
            u["name"] = m_name
        if u.get("star") is None and match.get("star") is not None and u.get("name") == m_name:
            u["star"] = match["star"]
        return u

    def _augments(self, d: dict, out: dict, same_round: bool) -> None:
        augs = d.get("augments") or []
        acc = self.acc.get("augments") or []
        if not augs and acc and same_round:
            out["augments"] = acc
            return
        fixed = []
        for i, a in enumerate(augs):
            old = acc[i] if i < len(acc) else None
            if not a.get("name") and old and old.get("name") and same_round:
                a = {**a, "name": old["name"]}
            fixed.append(a)
        out["augments"] = fixed
