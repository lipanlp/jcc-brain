import type { InjectionKey, Ref } from 'vue';

export const layoutKey = 'jcc-live-card-layout';
export const cardDefinitions = [
    { id: 'targets', title: '阵容与概率', span: 2, height: 460 },
    { id: 'comps', title: '推荐阵容', span: 2, height: 340 },
    { id: 'opponents', title: '对手', span: 2, height: 420 },
    { id: 'suggestions', title: '本回合建议', span: 1, height: 360 },
    { id: 'units', title: '对局数据', span: 1, height: 360 }
] as const;
export type CardId = typeof cardDefinitions[number]['id'];
export interface CardLayout { id: CardId; span: 1 | 2; height: number; }
export const minHeight = 220;
export const maxHeight = 900;
export function defaults(): CardLayout[] { return cardDefinitions.map(({ id, span, height }) => ({ id, span, height })); }
export function boundedHeight(value: number) { return Math.max(minHeight, Math.min(maxHeight, Math.round(value / 20) * 20)); }
export function loadLayout(): CardLayout[] {
    try {
        const raw: unknown = JSON.parse(localStorage.getItem(layoutKey) ?? 'null');
        if (!raw || typeof raw !== 'object' || !('version' in raw) || raw.version !== 1 || !('cards' in raw) || !Array.isArray(raw.cards)) return defaults();
        const seen = new Set<CardId>();
        const cards: CardLayout[] = [];
        for (const value of raw.cards) {
            if (!value || typeof value !== 'object') continue;
            const definition = cardDefinitions.find(card => card.id === value.id);
            if (!definition || seen.has(definition.id)) continue;
            seen.add(definition.id);
            cards.push({ id: definition.id, span: value.span === 1 || value.span === 2 ? value.span : definition.span,
                height: typeof value.height === 'number' && Number.isFinite(value.height) ? boundedHeight(value.height) : definition.height });
        }
        return [...cards, ...defaults().filter(card => !seen.has(card.id))];
    }
    catch { return defaults(); }
}
export interface LayoutControls {
    cards: Ref<CardLayout[]>;
    start: (event: PointerEvent, id: CardId, kind: 'drag' | 'resize') => void;
    move: (id: CardId, direction: number) => void;
    size: (id: CardId, span: number, height: number) => void;
    dragKey: (event: KeyboardEvent, id: CardId) => void;
    resizeKey: (event: KeyboardEvent, id: CardId) => void;
}
export const controlsKey: InjectionKey<LayoutControls> = Symbol('live-card-layout');
