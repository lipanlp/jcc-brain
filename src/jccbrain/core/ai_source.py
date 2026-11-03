"""AI display provenance, separate from shop permission and sample counters."""
from dataclasses import dataclass, replace
from hashlib import sha256
import json

from jccbrain.core.buy_source import buy_fingerprint
from jccbrain.core.models import Phase


@dataclass(frozen=True)
class AiSource:
    session: int
    revision: int = 0             # monotonic semantic generation, including A -> B -> A
    sequence: int = 0             # publication order, not a capture/read timestamp
    generation: int = 0
    selection: str = ''
    season: str = ''
    input_epoch: int = 0
    stage: str | None = None
    phase: str = 'unknown'
    fingerprint: str = ''
    ready: bool = False


@dataclass(frozen=True)
class AiProjection:
    source: AiSource
    valid: bool = False
    text: str = ''
    augment: str | None = None
    options: tuple[str, ...] = ()
    buy_frame: int = 0  # actual request provenance, not relabelled on ordinary reads


def ai_order(source):
    if isinstance(source, dict):
        return tuple(source[k] for k in ('session', 'revision', 'sequence'))
    return source.session, source.revision, source.sequence


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def raw_semantics(state, *, observed=False):
    # buy_fingerprint already normalizes raw/derived hp_rank and ignores timer,
    # coordinates and sample IDs. Add LLM inventory and actual option content.
    # In nonpurchase scenes the stabilizer intentionally carries the last shop
    # as stale history. It is not a currently visible shop or an AI buy permit.
    normalized = state
    if state.phase in (Phase.AUGMENT, Phase.CAROUSEL):
        normalized = replace(state, shop_open=False, shop=[None] * 5,
                             raw={**state.raw, 'shop': {'stale': False}})
    return digest([state.stage, state.phase.value, buy_fingerprint(normalized, observed=observed),
                   [(a.name, a.desc[:80]) for a in state.augments], state.augments_taken,
                   semantic_value(state.item_bench), [(u.name, u.star, u.known, semantic_value(u.items)) for u in state.units]])


def ai_scene(state):
    return (bool(state.stage) and state.phase in (Phase.PLANNING, Phase.COMBAT, Phase.AUGMENT, Phase.CAROUSEL)
            and not state.units_stale and not (state.viewing and state.viewing.get('self') is False))


def semantic_value(value):
    """Ignore item/unit sampling coordinates, never change the prompt itself."""
    if isinstance(value, list):
        return [semantic_value(v) for v in value]
    if isinstance(value, dict):
        ignored = {'x', 'y', 'bbox', 'rect', 'frame_id', 'timestamp'} if 'name' in value else set()
        return {k: semantic_value(v) for k, v in value.items() if k not in ignored}
    return value


def context_semantics(context):
    result = semantic_value(context)
    # This unlabeled list is a health distribution, not player identity/order.
    result['players_hp'] = sorted(result.get('players_hp', []), key=repr)
    return result


def same_ai_context(candidate, current):
    """Strategy data may reuse identical semantics; AI replies still use revision."""
    if not candidate or not current:
        return False
    def get(value, key):
        return value.get(key) if isinstance(value, dict) else getattr(value, key)
    return (get(candidate, 'ready') and get(current, 'ready')
            and get(candidate, 'revision') <= get(current, 'revision')
            and all(get(candidate, key) == get(current, key) for key in
                    ('session', 'generation', 'selection', 'season', 'input_epoch', 'stage', 'phase', 'fingerprint')))
