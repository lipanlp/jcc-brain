"""运营决策：这回合该存钱、升级还是 D 牌。

经验规则（可按自己的理解调整）：
- 平时守 50 利息；多出来的钱优先按运营曲线升级。
- 到了阵容的「D 牌点」才集中 D：reroll 阵容在 Lv5~7 慢 D，fast8 在 4-1/4-2 上 8 后 D。
- 血量危险时提前 D，保血优先于利息。
- 2~3 阶段保持三连以上的连胜/连败；4 阶段起血量倒数三名时少留金币稳血。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from jccbrain.core.models import GameState, Phase
from jccbrain.core.rules import SetRules, stage_key


class EconAction(str, Enum):
    SAVE = "save"           # 存钱吃利息
    LEVEL = "level"         # 买经验
    ROLL = "roll"           # D 牌
    ALL_IN = "all_in"       # 梭哈：血量危急
    NONE = "none"           # 读不到关键信息


@dataclass
class EconDecision:
    action: EconAction
    reason: str
    keep_gold: int = 50          # 本回合花钱时保留的金币
    xp_buys: int = 0             # 建议买几次经验
    roll_budget: int = 0         # 建议用于 D 牌的金币


def reroll_level(comp_carry_cost: int) -> int:
    """reroll 阵容在几级 D：1 费卡 5 级，2 费 6 级，3 费 7 级。"""
    return {1: 5, 2: 6, 3: 7}.get(comp_carry_cost, 8)


def hp_rank(state: GameState) -> tuple[int, int] | None:
    """从当前可信玩家栏计算（名次，存活人数）；零血量和消失的出局者不计入。"""
    players = state.players
    if state.players_stale or not 1 <= len(players) <= 8:
        return None
    if any(p.hp is None or not 0 <= p.hp <= 100 for p in players):
        return None
    own = [p for p in players if p.is_self]
    if len(own) != 1 or state.hp is None or state.hp <= 0 or own[0].hp != state.hp:
        return None
    alive = [p for p in players if p.hp > 0]
    return 1 + sum(p.hp > state.hp for p in alive), len(alive)


def decide(state: GameState, rules: SetRules, style: str = "fast8", carry_cost: int = 4,
           preference: str = "adaptive") -> EconDecision:
    stage, gold, level = state.stage_tuple, state.gold, state.level
    if stage is None or gold is None or level is None:
        return EconDecision(EconAction.NONE, "读不到阶段/金币/等级，暂不给运营建议")
    hp = state.hp if state.hp is not None else 100
    xp_cur, xp_need = state.xp or (0, rules.xp_to_next.get(level, 99))
    if xp_need <= xp_cur:   # 误读（如「9/0」），按规则表算
        xp_cur, xp_need = 0, rules.xp_to_next.get(level, 99)
    xp_cost = state.xp_cost or rules.xp_cost
    sk = stage_key(stage)

    # 1. 血量危急：保命
    if hp <= 30 and sk >= 401:
        return EconDecision(EconAction.ALL_IN, f"血量 {hp}，必须梭哈提升战力", keep_gold=0, roll_budget=gold)

    ranking = (state.hp_rank or hp_rank(state)) if sk >= 401 else None
    rank, alive = ranking or (0, 0)
    # 四人以上看倒数三名；决赛圈只催最后一名，避免领先者也被要求额外花钱。
    cutoff = alive - 2 if alive > 3 else alive
    low_rank = alive > 1 and rank >= cutoff
    rank_hint = f"血量第 {rank}/{alive}，少留金币优先稳血；" if low_rank else ""
    reserve_hp = min(hp, 50) if low_rank else hp
    # 只在可信己方准备画面调整可选支出；低血/低排名仍走原有必要稳血线。
    viewing = state.viewing or {}
    foreign = bool(viewing and not any(p.is_self and (
        p.slot is not None and viewing.get('slot') == p.slot or p.name and viewing.get('name') == p.name)
        for p in state.players))
    adjustable = (state.phase == Phase.PLANNING and not foreign and not state.players_stale
                  and state.hp is not None and 50 < hp <= 100 and 0 <= gold
                  and 1 <= level <= 10 and stage[0] >= 1 and 1 <= stage[1] <= 7 and not low_rank)
    preference = preference if adjustable and preference in ('health', 'economy') else 'adaptive'
    pref_hint = {'health': '保血优先：在既定路线内提前补强；',
                 'economy': '经济优先：减少非必要支出，增加保留；', 'adaptive': ''}[preference]

    # 2. 连败时先攒经济；血量已经偏低时仍按原有曲线提升战力。
    streak = state.streak or {}
    streak_n = streak.get("n")
    early_streak = stage[0] in (2, 3) and isinstance(streak_n, int) and streak_n >= 3
    winning = early_streak and streak.get("kind") == "win"
    if early_streak and streak.get("kind") == "loss" and hp > 50:
        return EconDecision(EconAction.SAVE, f"连败 {streak_n}，存钱吃连败奖励，不为过渡卡花钱",
                            keep_gold=gold)

    # 3. 运营曲线：该升级了；连胜最多在本阶段曲线上限内提前升一级。
    plan = "fast8" if style in ("fast8", "fast9") else "standard"
    if style == "reroll":
        target = min(rules.target_level(stage, "standard") or level, reroll_level(carry_cost))
        if sk >= 402:  # 三星成型后再慢慢上人口
            target = max(target, rules.target_level(stage, "standard") or target)
    else:
        target = rules.target_level(stage, plan) or level
    xp_short = xp_need - xp_cur
    buys = -(-xp_short // rules.xp_per_buy)
    cost = buys * xp_cost
    win_hint = f"连胜 {streak_n}，上人口保连胜；" if winning else ""
    if winning and level >= target:
        ceiling = rules.target_level((stage[0], 7), plan) or level
        if style == "reroll":
            ceiling = min(ceiling, reroll_level(carry_cost))
        keep = 10 if stage[0] == 2 else 20
        if level < ceiling and gold - cost >= keep:
            return EconDecision(EconAction.LEVEL, f"{win_hint}提前升到 {level + 1} 级（差 {xp_short} 经验）",
                                keep_gold=keep, xp_buys=buys)
    if level < target:
        if gold - cost >= 0:
            keep = 0 if hp < 50 else min(gold - cost, 50)
            if low_rank:
                keep = min(keep, 20)
            return EconDecision(EconAction.LEVEL, f"{rank_hint}{win_hint}运营曲线 {state.stage} 应到 {target} 级（差 {xp_short} 经验）",
                                keep_gold=keep, xp_buys=buys)

    # 4. D 牌点；排名靠后时按血量偏低的保留线处理，不触发额外梭哈。
    if style == "reroll":
        roll_from = 301 if carry_cost <= 1 else 302 if carry_cost == 2 else 401
        if level >= reroll_level(carry_cost) and sk >= roll_from:
            # 慢 D：血量健康时只花 50 以上的部分，血量越低保留越少
            keep = 50 if reserve_hp > 60 else 30 if reserve_hp > 40 else 10
            if preference == 'health':
                keep = max(keep - 10, 0)
            elif preference == 'economy':
                keep += 10
            if gold - keep < (state.refresh_cost or rules.refresh_cost):
                return EconDecision(EconAction.SAVE, f"{pref_hint}{rank_hint}reroll 阵容慢 D 阶段：金币超过 {keep} 的部分再 D，现在先存",
                                    keep_gold=keep)
            return EconDecision(EconAction.ROLL, f"{pref_hint}{rank_hint}reroll 阵容：在 {level} 级慢 D，保留 {keep} 金币",
                                keep_gold=keep, roll_budget=max(gold - keep, 0))
    else:
        roll_level = 9 if style == "fast9" else 8
        if level >= roll_level and sk >= 402:
            keep = 10 if reserve_hp > 50 else 0
            if preference == 'health':
                keep = 0
            elif preference == 'economy':
                keep = 20
            return EconDecision(EconAction.ROLL, f"{pref_hint}{rank_hint}{level} 级到位，D 出 4/5 费核心", keep_gold=keep,
                                roll_budget=max(gold - keep, 0))
        if level >= 7 and sk >= 401 and reserve_hp <= 50:
            reason = f"{rank_hint}7 级先 D 稳血" if low_rank else f"血量 {hp} 偏低，7 级先 D 稳血"
            return EconDecision(EconAction.ROLL, reason, keep_gold=20,
                                roll_budget=max(gold - 20, 0))

    # 保血在已到路线要求的 7 级窗口适度提前稳血，不挤占后续关键升级。
    if preference == 'health' and style != 'reroll' and level == 7 and level >= target and sk >= 305 and hp <= 65:
        keep = 30
        if gold - keep >= (state.refresh_cost or rules.refresh_cost):
            return EconDecision(EconAction.ROLL, f"{pref_hint}血量 {hp}，7 级先适度 D 稳血，保留 {keep} 金币",
                                keep_gold=keep, roll_budget=gold - keep)

    # 5. 默认存钱；经济优先仅延后非必要经验支出，不改变前面的关键升级。
    surplus_keep = 60 if preference == 'economy' else 50
    surplus_hint = pref_hint if preference == 'economy' else ''
    if gold >= surplus_keep + xp_cost and level < 9:
        return EconDecision(EconAction.LEVEL, f"{surplus_hint}{win_hint}利息已满，多余金币买经验", keep_gold=surplus_keep,
                            xp_buys=(gold - surplus_keep) // xp_cost)
    nxt = (gold // 10 + 1) * 10
    hint = f"再存 {nxt - gold} 金币多 1 利息" if gold < 50 else "保持 50 利息"
    win_save = f"连胜 {streak_n}，保持战力并攒经济；" if winning else ""
    optional_hint = pref_hint if preference == "economy" and gold >= 50 + xp_cost else ""
    return EconDecision(EconAction.SAVE, f"{optional_hint}{win_save}存钱：{hint}", keep_gold=min((gold // 10) * 10, 50))
