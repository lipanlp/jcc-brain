"""Display-only provenance for assistant buys; no rule or OCR confirmation changes."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from time import monotonic_ns

from jccbrain.core.models import Phase


def new_buy_session() -> int:
    # Microseconds remain exactly representable by JavaScript numbers.
    return monotonic_ns() // 1000


@dataclass(frozen=True)
class BuySource:
    session: int
    frame: int  # monotonic publication order, also increments on invalidation; not camera frame id
    generation: int
    selection: str
    season: str
    stage: str | None
    phase: str
    fingerprint: str
    eligible: bool
    decision: str = ''
    input_epoch: int = 0  # captured BEFORE fresh read; never relabel an in-flight result


def buy_fingerprint(st, *, observed=False) -> str:
    """Only buy dependencies, ignoring timer, coordinates, items and seen counters.

    Shop upgrade is a potential-upgrade marker, not the card's current star.
    Preserve it (and an explicit raw star if supplied) to invalidate either change;
    current rules still simulate each shop card as one copy.
    """
    raw_shop = st.raw.get('shop') or {}
    raw_slots = raw_shop.get('slots') or []
    ranking = st.hp_rank
    if observed and ranking is None:
        # Same public derivation used by the stabilizer, without confirming or
        # mutating anything. Raw player rows and derived (rank, alive) are the
        # same dependency when they express the same current purchase policy.
        from jccbrain.engine.economy import hp_rank
        ranking = hp_rank(st)
    value = {
        'shop': [(s.name, s.cost, s.unsure, s.item, s.upgrade,
                  (raw_slots[i] or {}).get('star', 1) if i < len(raw_slots) else 1)
                 if s else None for i, s in enumerate(st.shop)],
        'open': st.shop_open, 'stale': bool(raw_shop.get('stale')),
        'gold': st.gold, 'level': st.level, 'xp': st.xp, 'hp': st.hp,
        'streak': st.streak, 'xp_cost': st.xp_cost, 'refresh_cost': st.refresh_cost,
        'odds': st.odds, 'ranking': ranking or sorted(((p.hp, p.is_self) for p in st.players), key=repr),
        'players_stale': st.players_stale,
        'viewing': {k: (st.viewing or {}).get(k) for k in ('slot', 'name', 'self')},
        'self_players': sorted(((p.name, p.slot) for p in st.players if p.is_self), key=repr),
        'units_stale': st.units_stale, 'board_count': st.board_count,
        'owned': sorted((u.name, u.star, u.zone) for u in st.units),
        'traits': sorted((t.name, t.count or 0) for t in st.traits),
    }
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def buy_eligible(st) -> bool:
    return (st.phase in (Phase.PLANNING, Phase.COMBAT) and st.shop_open
            and not (st.raw.get('shop') or {}).get('stale') and not st.units_stale
            and st.gold is not None and st.level is not None
            and not (st.viewing and st.viewing.get('self') is False))


def same_buy_context(source: BuySource, current: BuySource) -> bool:
    return source.eligible and current.eligible and all(
        getattr(source, key) == getattr(current, key) for key in
        ('session', 'input_epoch', 'generation', 'selection', 'season', 'stage', 'phase', 'fingerprint'))


def source_matches_state(source: BuySource, st) -> bool:
    current = getattr(st, 'buy_source', None)
    return bool(current and same_buy_context(source, current)
                and source.stage == st.stage and source.phase == st.phase.value
                and buy_eligible(st) and source.fingerprint == buy_fingerprint(st))


def buy_order(source):
    """Publication barrier order; usable for typed sources and serialized payloads."""
    get = source.get if isinstance(source, dict) else lambda key, default=0: getattr(source, key, default)
    return get('session', 0), get('input_epoch', 0), get('frame', 0)
