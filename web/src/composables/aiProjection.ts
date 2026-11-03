import { shallowRef } from 'vue';
import { confirmedBuyProjection, currentAiContext } from './buyProjection';
import { bus } from '../bridge/events';
import type { Advice, AiProjection, GameState, Status } from '../types';
const current = shallowRef<AiProjection | null>(null);
let otherMode = false;
const order = (a: AiProjection['source'], b: AiProjection['source']) => a.session - b.session || a.revision - b.revision || a.sequence - b.sequence;
function accept(next?: AiProjection | null) {
    if (next && (!current.value || order(next.source, current.value.source) > 0
        || order(next.source, current.value.source) === 0 && !next.valid)) current.value = next;
}
bus.on('ai_projection', payload => accept(payload as AiProjection));
bus.on('status', payload => {
    const status = payload as Status;
    otherMode = !!status.mode && status.mode.key !== 'assistant';
    accept(status.ai_projection);
});
export function semanticAi(advice: Advice | null, state: GameState | null) {
    const p = current.value;
    if (otherMode && !advice?.buy_source) return undefined;
    if (!p) return undefined; // older/nonassistant producers keep their existing display contract
    const own = state?.buy_source;
    const s = p.source;
    const full = advice?.buy_source;
    if (otherMode || !p.valid || !s.ready || !own || !state || !advice
        || s.input_epoch !== (own.input_epoch ?? 0)
        || !(['session', 'generation', 'selection', 'season', 'stage', 'phase'] as const).every(k => s[k] === own[k])
        || !full || full.fingerprint !== own.fingerprint
        || (full.input_epoch ?? 0) !== s.input_epoch
        || !(['session', 'generation', 'selection', 'season', 'stage', 'phase'] as const).every(k => s[k] === full[k])
        || !currentAiContext(advice.ai_source)
        || advice.stage !== s.stage || advice.phase !== s.phase
        || s.phase === 'augment' && JSON.stringify(advice.augments.map(a => a.name)) !== JSON.stringify(p.options)) return null;
    if (s.phase === 'planning' || s.phase === 'combat') {
        const buys = confirmedBuyProjection();
        if (!buys || p.buy_frame < buys.source.frame) return null;
    }
    return p;
}
export function semanticAiEnabled() { return !!current.value; }

export function visibleHudAugments(frame: import('../types').Hud | null) {
    const p = current.value;
    if (!p || otherMode && !frame?.buy_source) return frame?.augments ?? [];
    const own = frame?.buy_source;
    const valid = !otherMode && p.valid && p.source.ready && own
        && p.source.input_epoch === (own.input_epoch ?? 0)
        && (['session', 'generation', 'selection', 'season', 'stage', 'phase'] as const).every(k => p.source[k] === own[k]);
    return (frame?.augments ?? []).map(a => {
        const ai = !!valid && a.name === p.augment && p.options.includes(a.name);
        return { ...a, ai_best: ai, text: (ai ? 'AI 推荐 · ' : '') + (a.rule_text ?? a.text.replace('AI 推荐 · ', '')) };
    });
}
