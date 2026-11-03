"""大模型接入。

LLM 不直接看原始画面，而是拿到：结构化状态 + 规则引擎的建议 + 候选阵容/概率，
在此基础上做综合判断（例如：要不要转阵容、这回合是否该保血）。API 由用户在设置里填写。

支持两类协议：
- openai：任何兼容 OpenAI Chat Completions 的服务（OpenAI、DeepSeek、通义、Kimi、本地 Ollama 等）
- anthropic：Claude Messages API
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from jccbrain.core.models import GameState, Phase
from jccbrain.core.settings import LLMSettings
from jccbrain.engine.advisor import Advice

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是金铲铲之战（云顶之弈手游）顶尖玩家的决策助手。
你会收到当前对局的结构化状态、规则引擎给出的初步建议、候选阵容和抽卡概率。
请在初步建议基础上做最终判断，重点考虑：阵容是否该转、本回合运营（存钱/升级/D牌）、买卖哪些卡、海克斯怎么选。
字段为 null 表示这一帧没识别到，不代表数据错误，按已有信息判断即可；「?」开头的单位是没认出名字的棋子。
前期（2-1 之前）不要为了过渡卡把钱花光，只买对子、和现有羁绊相关的卡或能补上场空位的卡。
如果给出 locked_comp，说明玩家已锁定该阵容，switch_comp 必须为空字符串，不要建议转阵容。
如果给出 watched_comps，main_comp 是当前主阵容，其余只比较来牌；目标卡购买只围绕主阵容，不因关注比较而同时囤几套。switch_comp 只能从关注内选或为空。
结合玩家当前装备与已选海克斯判断阵容和合成。
评级S/A/B/C/D仅为资料标签，不得作为强度或推荐依据；比较只使用实际均名、实况契合等事实。comparison_scope给出有效集合，switch_comp只能在allowed_comps内或为空，不可回到全库。
items是全局持有汇总，不是空闲库存；available_item_bench才是装备栏，equipped_items只表示各单位已穿的事实。
item_plan是按实况分配的装备安排：current_unit是当前上场佩戴者，final_to是最终去向，assets只来自装备栏。已穿装备不能当空闲材料；暂留/备料不等于可立即合成或佩戴。转交需拆卸器，只有单位确需下场时才考虑出售回收，不要求卖掉仍需保留的核心。
结合对手阵容判断撞阵、卡池争夺与站位针对。
海克斯回合综合阵容统计与当前局势（装备缺口、经济、血量、阵容短板、运营节奏）选择，不要只看阵容推荐。
augment_options给出每个选项的类别、描述和规则评分，augment_situation是局势画像。
headline必须只写出一个所选海克斯的完整名字，reason写局势依据；无法判断时说明缺少信息，不编造选项。
只输出 JSON：{"headline": "不超过20字的决策", "actions": ["具体操作，最多3条"], "reason": "不超过60字的理由",
"switch_comp": "建议转的阵容名或空字符串"}"""


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class Preset:
    key: str
    title: str
    model: str
    effort: str             # 平时的思考强度
    effort_augment: str     # 海克斯回合的思考强度
    timeout_s: float
    note: str


# 2026-10-22 实测（智谱 GLM-5.3 系列，真实对局状态，中位耗时）：
#   flashx+low 2.6s（最稳）；5.3+low 4.1s；flash+low 4.8s（不比标准版快，未采用）；
#   flash+high 7.3s；5.3+high 9.2s；5.3+max 超过 40s（海克斯只有约 30 秒，不可用）
PRESETS: dict[str, Preset] = {p.key: p for p in (
    Preset("fast", "极速", "glm-5.3-flashx", "low", "low", 15,
           "约 2.5 秒出建议，适合快节奏 D 牌、只需要大方向时"),
    Preset("balanced", "均衡（推荐）", "glm-5.3", "low", "high", 25,
           "平时约 4 秒；海克斯回合加强思考约 10 秒"),
    Preset("deep", "深度", "glm-5.3", "high", "high", 35,
           "约 9 秒，判断更细，适合准备阶段时间充裕时"),
)}


def apply_preset(cfg: LLMSettings, key: str) -> Preset:
    """把预设写进设置（模型、思考强度、超时）。API 地址和 Key 不变。"""
    p = PRESETS[key]
    cfg.preset = key
    cfg.model = p.model
    cfg.reasoning_effort = p.effort
    cfg.reasoning_effort_augment = p.effort_augment
    cfg.timeout_s = p.timeout_s
    cfg.max_tokens = max(cfg.max_tokens, 3000)
    return p


@dataclass
class LLMReply:
    headline: str
    actions: list[str]
    reason: str
    switch_comp: str = ""
    raw: str = ""
    latency_s: float = 0.0


def build_context(state: GameState, advice: Advice) -> dict[str, Any]:
    """压缩成大模型需要的最小上下文（控制 token）。"""
    context = {
        "stage": state.stage, "phase": state.phase.value, "gold": state.gold, "level": state.level,
        "comp_status": advice.comp_status,
        "items": advice.held_items, "augments_taken": advice.augments_taken, "item_plan": advice.item_plan,
        "available_item_bench": [dict(item) for item in state.item_bench],
        "equipped_items": [{'key': f'{zone}:{index}', 'name': unit.name, 'known': unit.known,
                            'occupied_slots': len(unit.items), 'items': [dict(item) for item in unit.items]}
                           for zone, group in [('board', state.board), ('bench', state.bench)]
                           for index, unit in enumerate(group) if unit.items],
        "units_stale": state.units_stale,
        "xp": state.xp, "hp": state.hp, "streak": state.streak,
        "shop": [s.name if s else None for s in state.shop],
        "board": [f"{u.name}{u.star}星" for u in state.board],
        "bench": [f"{u.name}{u.star}星" for u in state.bench],
        "traits": [f"{t.name}{t.count or ''}{'(激活)' if t.active else ''}" for t in state.traits],
        "players_hp": [p.hp for p in state.players],
        "augment_options": [{'name': a.name, 'category': list(a.category), 'desc': a.desc[:80], 'rule_score': a.score}
                            for a in advice.augments] or
                           [{'name': a.name, 'category': [], 'desc': a.desc[:80], 'rule_score': None} for a in state.augments],
        "rule_advice": advice.lines(),
        "candidate_comps": [{
            "name": m.name, "avg_place": m.comp.avg_placement, "fit": round(m.fit, 2),
            "core": [h.name for h in m.comp.heroes if h.is_core or h.is_carry],
            "missing": [h.name for h in m.missing[:5]], "why": m.why,
        } for m in advice.comps[:3]],
        "target_odds": [{
            "name": t.name, "need": t.need, "per_roll": round(t.odds.per_roll, 3) if t.odds else None,
            "gold_for_80": t.odds.gold_for_80 if t.odds else None,
        } for t in advice.targets if t.need],
    }
    if advice.selection:
        context['comparison_scope'] = advice.selection
    context['opponents'] = [{
        'name': r['name'], 'hp': r['hp'], 'round': r['round'], 'level': r['level'],
        'comp': r['comp_guess']['name'] if r['comp_guess'] else None,
        'by': r['comp_guess']['by'] if r['comp_guess'] else None,
        'carries': r['carries'], 'top_traits': sorted(r['traits'], key=lambda t: -(t['count'] or 0))[:3],
    } for r in sorted(advice.opponents, key=lambda r: (r['hp'] is None, -(r['hp'] or 0)))[:7]]
    if state.phase == Phase.AUGMENT:
        context['augment_situation'] = advice.augment_situation
    if advice.locked and advice.target:
        context['locked_comp'] = advice.target.name
    if advice.comp_status == '关注':
        context['main_comp'] = advice.target.name if advice.target else None
        context['watched_comps'] = [
            {'name': m.name, 'seen': m.seen, 'seen_carry': m.seen_carry,
             'owned_copies': m.owned_copies, 'fit': round(m.fit, 2)} for m in advice.comps]
    return context


class LLMClient:
    def __init__(self, cfg: LLMSettings):
        self.cfg = cfg
        self._http = httpx.Client(timeout=cfg.timeout_s)
        self._last_call = 0.0
        self._last_key: str | None = None

    @property
    def api_key(self) -> str:
        return self.cfg.api_key or os.environ.get("JCC_LLM_API_KEY", "")

    @property
    def ready(self) -> bool:
        return self.cfg.enabled and bool(self.api_key) and bool(self.cfg.model)

    @staticmethod
    def call_key(state: GameState, advice: Advice | None = None) -> str | None:
        """这一帧该归到哪一次提问；None 表示信息不够、不该问。

        每回合只问一次（准备和战斗阶段合并）；海克斯阶段等三个名字都读到再问，刷新了海克斯才重问。
        """
        if not state.stage or state.phase not in (Phase.PLANNING, Phase.COMBAT, Phase.AUGMENT):
            return None
        if state.phase == Phase.AUGMENT:
            names = [a.name for a in advice.augments] if advice and advice.augments else [a.name for a in state.augments]
            if not names or not all(a.name for a in state.augments):
                return None
            return f"{state.stage}|augment|{','.join(names)}"
        if state.gold is None or state.level is None:
            return None
        return f"{state.stage}|round"

    def should_call(self, state: GameState, advice: Advice | None = None) -> bool:
        """同一个提问只问一次，并限制最小间隔。"""
        key = self.call_key(state, advice)
        if key is None or key == self._last_key or time.monotonic() - self._last_call < self.cfg.min_interval_s:
            return False
        return True

    def advise(self, state: GameState, advice: Advice) -> LLMReply:
        self._last_key = self.call_key(state, advice)
        self._last_call = time.monotonic()
        user = json.dumps(build_context(state, advice), ensure_ascii=False)
        effort = self.cfg.reasoning_effort
        if state.phase == Phase.AUGMENT and self.cfg.reasoning_effort_augment:
            effort = self.cfg.reasoning_effort_augment
        t0 = time.monotonic()
        text = self._anthropic(user) if self.cfg.provider == "anthropic" else self._openai(user, effort)
        reply = parse_reply(text)
        reply.latency_s = time.monotonic() - t0
        return reply

    def ping(self) -> str:
        """测试连接，返回「耗时 + 模型回复」。"""
        t0 = time.monotonic()
        msg = "只回复两个字：收到"
        if self.cfg.provider == "anthropic":
            text = self._anthropic(msg)
        else:
            text = self._openai(msg, self.cfg.reasoning_effort, system="")
        return f"{time.monotonic() - t0:.1f}s：{text.strip()[:100]}"

    def _openai(self, user: str, effort: str = "", system: str = SYSTEM_PROMPT) -> str:
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        body: dict[str, Any] = {
            "model": self.cfg.model, "temperature": 0.3, "max_tokens": self.cfg.max_tokens,
            "messages": ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}],
        }
        if effort:
            body["reasoning_effort"] = effort
        if self.cfg.json_mode and system:
            body["response_format"] = {"type": "json_object"}
        r = self._http.post(url, headers={"Authorization": f"Bearer {self.api_key}"}, json=body)
        if r.status_code >= 400:
            raise LLMError(f"HTTP {r.status_code}：{r.text[:300]}")
        msg = r.json()["choices"][0]["message"]
        content = msg.get("content") or ""
        if not content.strip():
            raise LLMError("模型只返回了思考过程没有正文，请调大「最大输出 token」或降低思考强度")
        return content

    def _anthropic(self, user: str) -> str:
        base = self.cfg.base_url.rstrip("/") if "anthropic" in self.cfg.base_url else "https://api.anthropic.com"
        r = self._http.post(base + "/v1/messages", headers={
            "x-api-key": self.api_key, "anthropic-version": "2023-06-01",
        }, json={
            "model": self.cfg.model, "max_tokens": self.cfg.max_tokens, "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user}],
        })
        if r.status_code >= 400:
            raise LLMError(f"HTTP {r.status_code}：{r.text[:300]}")
        return "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text")


def parse_reply(text: str) -> LLMReply:
    """容错解析：模型可能在 JSON 外包 ``` 或多说几句。"""
    raw = text
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            d = json.loads(text[start:end + 1])
            return LLMReply(str(d.get("headline", "")), [str(a) for a in d.get("actions", [])],
                            str(d.get("reason", "")), str(d.get("switch_comp", "")), raw)
        except ValueError:
            pass
    return LLMReply(text.strip()[:60], [], text.strip(), "", raw)
