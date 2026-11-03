import type { Advice, BuySource, GameState } from '../types';
import { semanticAi } from './aiProjection';
import { confirmedBuyProjection, currentNonPurchaseAi, legacyBuyPresentation, visibleBuys } from './buyProjection';

// Full strategy can precede a dynamic shop update. Only reuse its nonpurchase
// summary when the economic / main decision still matches the confirmed source.
function decisionMatches(a: BuySource | undefined, b: BuySource) {
    return !!a && a.eligible && b.eligible && (a.input_epoch ?? 0) === (b.input_epoch ?? 0)
        && (['session', 'generation', 'selection', 'season', 'stage', 'phase', 'decision'] as const)
            .every(key => a[key] === b[key]);
}
function shopMatches(advice: Advice, source: BuySource) {
    return decisionMatches(advice.buy_source, source) && advice.buy_source?.fingerprint === source.fingerprint;
}
export function headlineSummary(advice: Advice | null) {
    if (!advice) return '正在分析当前画面…';
    return advice.headline_summary ?? (advice.headline.split('；').filter(part => !part.startsWith('买 ')).join('；') || '观望');
}
export function purchaseHeadline(advice: Advice | null) {
    if (!advice) return '';
    const current = confirmedBuyProjection();
    const buys = legacyBuyPresentation(advice) ? visibleBuys(advice)
        : current && decisionMatches(advice.buy_source, current.source) ? current.buys : [];
    return buys.length ? '买 ' + buys.map(buy => buy.name).join('、') : '';
}
export function visibleCompReasons(advice: Advice, comp: Advice['comps'][number]) {
    const current = confirmedBuyProjection();
    const showShop = legacyBuyPresentation(advice) || !!(current?.buys.length && shopMatches(advice, current.source));
    const parts = comp.why_parts ?? comp.why.map(text => ({ kind: text.startsWith('商店有 ') ? 'shop' : 'strategy', text }));
    return parts.filter(part => part.kind !== 'shop' || showShop).map(part => part.text);
}
export function visibleAiText(advice: Advice | null, state: GameState | null) {
    if (!advice) return '';
    const semantic = semanticAi(advice, state);
    if (semantic !== undefined) return semantic?.text ?? '';
    if (legacyBuyPresentation(advice) || currentNonPurchaseAi(advice, state)) return advice.llm_text ?? '';
    const current = confirmedBuyProjection();
    // Free text has no reliable clause categories. Never reuse it after a shop
    // update, even if the old fingerprint or an empty buy list happens to match.
    return current && shopMatches(advice, current.source) && advice.buy_source!.frame >= current.source.frame
        ? advice.llm_text ?? '' : '';
}

export function visibleAiAugment(advice: Advice | null, state: GameState | null) {
    const semantic = semanticAi(advice, state);
    return semantic === undefined ? advice?.ai_augment : semantic?.augment;
}
