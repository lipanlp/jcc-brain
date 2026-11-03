import { ref, shallowRef } from 'vue';
import { bus } from '../bridge/events';
import type { Advice, BuyProjection, BuySource, GameState, Hud, Status } from '../types';

// One projection channel per webview, independent of full strategy / preview.
const projection = shallowRef<BuyProjection | null>(null);
const context = shallowRef<BuySource | null>(null);
let aiContext: import('../types').AiProjection['source'] | null = null;
function acceptAiContext(next?: import('../types').AiProjection | null) {
    const s = next?.source;
    if (s && (!aiContext || s.session > aiContext.session || s.session === aiContext.session
        && (s.revision > aiContext.revision || s.revision === aiContext.revision && s.sequence >= aiContext.sequence))) aiContext = s;
}
bus.on('ai_projection', p => acceptAiContext(p as import('../types').AiProjection));
export function currentAiContext(source?: import('../types').AiProjection['source']) {
    return !!source?.ready && !!aiContext?.ready && source.revision <= aiContext.revision
        && (['session', 'generation', 'selection', 'season', 'input_epoch', 'stage', 'phase', 'fingerprint'] as const)
            .every(k => source[k] === aiContext![k]);
}
let session = 0;
const enabled = ref(false);
let otherMode = false;
const epoch = (s: BuySource) => s.input_epoch ?? 0;
function order(a: BuySource, b: BuySource) {
    return a.session - b.session || epoch(a) - epoch(b) || a.frame - b.frame;
}
function allowed(source: BuySource, barrier?: BuySource | null) {
    return !barrier || order(source, barrier) > 0
        || order(source, barrier) === 0 && (!source.eligible || barrier.eligible);
}
function same(a: BuySource, b: BuySource) {
    return a.eligible && b.eligible && epoch(a) === epoch(b) && (['session', 'generation', 'selection', 'season', 'stage', 'phase', 'fingerprint'] as const)
        .every(key => a[key] === b[key]);
}
function stream(source: BuySource) {
    if (source.session < session) return false;
    if (source.session > session) {
        session = source.session;
        projection.value = context.value = null;
    }
    enabled.value = !otherMode;
    return true;
}
function acceptContext(source?: BuySource) {
    if (!source || !stream(source) || !allowed(source, projection.value?.source)) return;
    if (!allowed(source, context.value)) return;
    context.value = source;
    const current = projection.value;
    if (current && order(source, current.source) >= 0 && !same(current.source, source))
        projection.value = { source, valid: false, buys: [] };
}
function acceptProjection(next?: BuyProjection | null) {
    if (!next || !stream(next.source) || !allowed(next.source, projection.value?.source)
        || !allowed(next.source, context.value) && !(context.value && same(next.source, context.value))) return;
    const current = projection.value;
    if (current?.valid && !next.valid && next.source.eligible && order(next.source, current.source) === 0) return;
    projection.value = next;
}
// All current-shop text uses this same authority as lists and highlights.
export function confirmedBuyProjection() {
    const current = projection.value;
    return enabled.value && current?.valid && context.value && same(current.source, context.value) ? current : null;
}
export function legacyBuyPresentation(advice: Advice) {
    return !enabled.value && !advice.buy_source;
}
export function visibleBuys(advice: Advice | null) {
    if (!enabled.value) return advice?.buy_source ? [] : advice?.buys ?? [];
    const current = projection.value;
    return current?.valid && context.value && same(current.source, context.value) ? current.buys : [];
}
export function visibleShop(frame: Hud | null) {
    if (!enabled.value) return frame?.shop ?? [];
    const current = projection.value;
    return frame?.buy_source && current?.valid && context.value && same(current.source, context.value)
        && same(current.source, frame.buy_source) ? frame.shop.flatMap(slot => {
            const buy = current.buys.find(b => b.slot === slot.key);
            if (buy) return [{ ...slot, kind: 'buy' as const, text: '买 · ' + buy.reason }];
            return slot.kind === 'target' && frame.buy_projection?.source.decision === current.source.decision ? [slot] : [];
        }) : [];
}
bus.on('state', payload => acceptContext((payload as GameState).buy_source));
bus.on('buy_projection', payload => acceptProjection(payload as BuyProjection));
bus.on('hud', payload => {
    const frame = payload as Hud;
    acceptContext(frame.buy_source);
    acceptProjection(frame.buy_projection);
});
bus.on('status', payload => {
    const status = payload as Status;
    acceptAiContext(status.ai_projection);
    otherMode = !!status.mode && status.mode.key !== 'assistant';
    if (otherMode) { enabled.value = false; context.value = null; }
    else if (status.buy_projection) {
        acceptContext(status.buy_projection.source);
        acceptProjection(status.buy_projection);
    }
});

export function currentBuyState(source?: BuySource) {
    return !source || source.session >= session && allowed(source, projection.value?.source)
        && allowed(source, context.value);
}
// A nonpurchase phase has eligible=false by design. Its fresh full explanation
// is authorized by an accepted state snapshot, not by a valid buy projection.
export function currentNonPurchaseAi(advice: Advice, state: GameState | null) {
    const source = advice.buy_source;
    const own = state?.buy_source;
    const latest = context.value;
    const barrier = projection.value?.source;
    const matches = (a: BuySource, b: BuySource) => epoch(a) === epoch(b)
        && (['session', 'frame', 'generation', 'selection', 'season', 'stage', 'phase', 'fingerprint'] as const)
            .every(key => a[key] === b[key]);
    return enabled.value && !!state && !!source && !!own && !!latest
        && (state.phase === 'augment' || state.phase === 'carousel')
        && advice.phase === state.phase && source.phase === state.phase
        && !!state.stage && advice.stage === state.stage && source.stage === state.stage
        && matches(source, own) && matches(source, latest)
        && currentBuyState(source)
        // An equal-frame nonpurchase publication is not a revocation of its
        // matching state. A newer clear (including input/stop) always wins.
        && (!barrier || order(source, barrier) > 0 || matches(source, barrier));
}
export function currentBuyHud(frame: Hud) {
    const nested = frame.buy_projection?.source;
    // Scout/status updates may combine a cached own state with a fresh clear.
    // Accept their notice, while visibleShop continues to require matching state.
    const source = nested && (!frame.buy_source || order(nested, frame.buy_source) >= 0) ? nested : frame.buy_source;
    return currentBuyState(source);
}
export function currentFullAdvice(next: Advice, current: Advice | null) {
    const source = next.buy_source;
    if (!source) return true;
    const barrier = projection.value?.source;
    const semantic = !source.eligible && (next.phase === 'augment' || next.phase === 'carousel')
        && currentAiContext(next.ai_source)
        && (!barrier || source.stage === barrier.stage && source.phase === barrier.phase);
    // Full strategy / population preview stays paired with its original frame.
    // Its text may precede unchanged dynamic buys, but never an input epoch.
    return source.session >= session && (!barrier || source.session > barrier.session
        || source.session === barrier.session && epoch(source) === epoch(barrier)
        && source.generation === barrier.generation && source.selection === barrier.selection
        && source.season === barrier.season && (barrier.eligible || order(source, barrier) >= 0 || semantic)
        && (order(source, barrier) !== 0 || allowed(source, barrier)))
        && allowed(source, current?.buy_source);
}
