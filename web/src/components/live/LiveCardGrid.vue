<script setup lang="ts">
import { onBeforeUnmount, provide, ref } from 'vue';
import { MoveDiagonal2, RotateCcw } from 'lucide-vue-next';
import { boundedHeight, cardDefinitions, controlsKey, defaults, layoutKey, loadLayout, type CardId } from './layout';
defineOptions({ inheritAttrs: false });
const cards = ref(loadLayout());
const root = ref<HTMLElement>();
const active = ref<CardId | null>(null);
const target = ref<{ id: CardId; after: boolean } | null>(null);
const ghost = ref<{ x: number; y: number; title: string } | null>(null);
const notice = ref('');
const storageFailed = ref(false);
let finishInteraction: ((cancel?: boolean) => void) | undefined;
function save() {
    try { localStorage.setItem(layoutKey, JSON.stringify({ version: 1, cards: cards.value })); storageFailed.value = false; }
    catch { storageFailed.value = true; }
}
function move(id: CardId, direction: number) {
    const index = cards.value.findIndex(card => card.id === id);
    const next = Math.max(0, Math.min(cards.value.length - 1, index + direction));
    if (next === index) return;
    const [card] = cards.value.splice(index, 1);
    cards.value.splice(next, 0, card!);
    notice.value = `${name(id)}已移到第 ${next + 1} 位`;
    save();
}
function size(id: CardId, span: number, height: number) {
    const card = cards.value.find(card => card.id === id)!;
    card.span = span === 1 ? 1 : 2;
    if (Number.isFinite(height)) card.height = boundedHeight(height);
    notice.value = `${name(id)}：${card.span === 1 ? '半宽' : '全宽'}，高度 ${card.height}`;
    save();
}
function name(id: CardId) { return cardDefinitions.find(card => card.id === id)!.title; }
function dragKey(event: KeyboardEvent, id: CardId) {
    const directions: Record<string, number> = { ArrowUp: -1, ArrowDown: 1, Home: -cards.value.length, End: cards.value.length };
    if (!(event.key in directions)) return;
    event.preventDefault();
    move(id, directions[event.key]!);
}
function resizeKey(event: KeyboardEvent, id: CardId) {
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
    event.preventDefault();
    const card = cards.value.find(card => card.id === id)!;
    size(id, event.key === 'ArrowLeft' ? 1 : event.key === 'ArrowRight' ? 2 : card.span,
        card.height + (event.key === 'ArrowDown' ? 20 : event.key === 'ArrowUp' ? -20 : 0));
}
function start(event: PointerEvent, id: CardId, kind: 'drag' | 'resize') {
    if (event.button !== 0 || !event.isPrimary || !root.value) return;
    finishInteraction?.(true);
    const handle = event.currentTarget as HTMLElement;
    const element = root.value.querySelector<HTMLElement>(`[data-card-id="${id}"]`)!;
    const scroller = root.value.closest<HTMLElement>('.live-page')!;
    const original = cards.value.map(card => ({ ...card }));
    const card = cards.value.find(card => card.id === id)!;
    const initial = element.getBoundingClientRect();
    const startX = event.clientX, startY = event.clientY, scrollTop = scroller.scrollTop;
    const initialSpan = card.span, initialHeight = card.height;
    let x = startX, y = startY, moved = false, frame = 0;
    const oldCursor = document.body.style.cursor, oldSelect = document.body.style.userSelect;
    event.preventDefault();
    handle.focus({ preventScroll: true });
    handle.setPointerCapture(event.pointerId);
    function locateTarget() {
        const source = element.getBoundingClientRect();
        if (x >= source.left && x <= source.right && y >= source.top && y <= source.bottom) { target.value = null; return; }
        const candidates = [...root.value!.querySelectorAll<HTMLElement>('[data-card-id]')].filter(node => node.dataset.cardId !== id);
        let nearest: HTMLElement | undefined, distance = Infinity;
        for (const node of candidates) {
            const rect = node.getBoundingClientRect();
            const dx = Math.max(rect.left - x, 0, x - rect.right), dy = Math.max(rect.top - y, 0, y - rect.bottom);
            const score = dx * dx + dy * dy;
            if (score < distance) { distance = score; nearest = node; }
        }
        if (nearest) {
            const rect = nearest.getBoundingClientRect();
            target.value = { id: nearest.dataset.cardId as CardId, after: y > rect.top + rect.height / 2 };
        }
    }
    function update() {
        if (kind === 'drag') {
            ghost.value = { x: Math.max(8,Math.min(x+14,innerWidth-190)), y: Math.max(8,Math.min(y+14,innerHeight-50)), title: name(id) };
            locateTarget();
        }
        else {
            const gridWidth = root.value!.getBoundingClientRect().width;
            // Single-column presentation leaves the saved desktop width intact.
            const singleColumn = getComputedStyle(root.value!).gridTemplateColumns.split(' ').length === 1;
            const width = initial.width + x - startX;
            card.span = singleColumn ? initialSpan : width < gridWidth * .75 ? 1 : 2;
            card.height = boundedHeight(initialHeight + y - startY + scroller.scrollTop - scrollTop);
        }
    }
    function tick() {
        if (moved) {
            const rect = scroller.getBoundingClientRect();
            const top = Math.max(0, rect.top), bottom = Math.min(innerHeight, rect.bottom);
            const speed = y < top + 64 ? -Math.min(18, (top + 64 - y) / 3) : y > bottom - 64 ? Math.min(18, (y - bottom + 64) / 3) : 0;
            if (speed) { scroller.scrollTop += speed; update(); }
        }
        frame = requestAnimationFrame(tick);
    }
    function pointerMove(next: PointerEvent) {
        if (next.pointerId !== event.pointerId) return;
        x = next.clientX; y = next.clientY;
        if (!moved && Math.hypot(x - startX, y - startY) < 5) return;
        moved = true; active.value = id;
        document.body.style.cursor = kind === 'drag' ? 'grabbing' : 'nwse-resize';
        document.body.style.userSelect = 'none';
        update();
    }
    function end(cancel = false) {
        cancelAnimationFrame(frame);
        window.removeEventListener('pointermove', pointerMove);
        window.removeEventListener('pointerup', pointerUp);
        window.removeEventListener('pointercancel', pointerCancel);
        window.removeEventListener('keydown', escape);
        window.removeEventListener('blur', blur);
        handle.removeEventListener('lostpointercapture', lostCapture);
        if (handle.hasPointerCapture(event.pointerId)) handle.releasePointerCapture(event.pointerId);
        document.body.style.cursor = oldCursor; document.body.style.userSelect = oldSelect;
        if (cancel) cards.value = original;
        else if (moved) {
            if (kind === 'drag' && target.value) {
                const remaining = cards.value.filter(card => card.id !== id);
                const insertion = remaining.findIndex(card => card.id === target.value!.id) + (target.value.after ? 1 : 0);
                remaining.splice(insertion, 0, card);
                cards.value = remaining;
                notice.value = `${name(id)}已移到第 ${insertion + 1} 位`;
            }
            else notice.value = `${name(id)}大小已调整`;
            save();
        }
        active.value = null; target.value = null; ghost.value = null; finishInteraction = undefined;
    }
    function pointerUp(next: PointerEvent) { if (next.pointerId === event.pointerId) end(); }
    function pointerCancel(next: PointerEvent) { if (next.pointerId === event.pointerId) end(true); }
    function escape(next: KeyboardEvent) { if (next.key === 'Escape') { next.preventDefault(); end(true); } }
    function blur() { end(true); }
    function lostCapture() { end(true); }
    finishInteraction = end;
    window.addEventListener('pointermove', pointerMove);
    window.addEventListener('pointerup', pointerUp);
    window.addEventListener('pointercancel', pointerCancel);
    window.addEventListener('keydown', escape);
    window.addEventListener('blur', blur);
    handle.addEventListener('lostpointercapture', lostCapture);
    frame = requestAnimationFrame(tick);
}
function reset() { finishInteraction?.(true); cards.value = defaults(); save(); notice.value = '已恢复默认顺序和大小'; }
provide(controlsKey, { cards, start, move, size, dragKey, resizeKey });
onBeforeUnmount(() => finishInteraction?.(true));
</script>
<template>
  <div class="live-layout" v-bind="$attrs">
  <div class="layout-toolbar">
    <span>拖动标题旁把手排序 · 拖动右下角调整大小</span>
    <button type="button" @click="reset"><RotateCcw :size="13" />恢复默认布局</button>
  </div>
  <p v-if="storageFailed" class="storage-notice" role="status">布局暂时仅在当前页面保留，浏览器存储不可用。</p>
  <p class="sr-only" role="status" aria-live="polite">{{ notice }}</p>
  <div ref="root" class="live-card-grid" :class="{'is-interacting':active}">
    <div v-for="card in cards" :key="card.id" class="live-card-frame" :class="{ 'is-active':active===card.id, 'insert-before':target?.id===card.id && !target.after, 'insert-after':target?.id===card.id && target.after }"
         :data-card-id="card.id" :data-span="card.span" :style="{gridColumn:`span ${card.span}`,height:`${card.height}px`}">
      <slot :name="card.id" />
      <button type="button" class="resize-handle" :aria-label="`调整${name(card.id)}大小`" title="拖动调整大小；左右方向键切换宽度，上下方向键调整高度"
              @pointerdown="start($event,card.id,'resize')" @keydown="resizeKey($event,card.id)"><MoveDiagonal2 :size="16" /></button>
    </div>
  </div>
  </div>
  <Teleport to="body"><div v-if="ghost" class="layout-ghost" :style="{left:`${ghost.x}px`,top:`${ghost.y}px`}">{{ ghost.title }} · 拖动排序</div></Teleport>
</template>
<style scoped>
.layout-toolbar { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; color: var(--secondary); font-size: var(--font-xs); }
.layout-toolbar button { display: flex; align-items: center; gap: 5px; background: var(--hover); color: var(--text); border-radius: var(--radius-sm); padding: 6px 8px; }
.live-card-grid { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: var(--space-4); align-items: start; }
.live-card-frame { position: relative; min-width: 0; container: live-card / inline-size; }
.live-card-frame :deep(.u-card) { height: 100%; display: flex; flex-direction: column; overflow: visible; }
.live-card-frame :deep(.u-card > header) { flex: 0 0 auto; flex-wrap: wrap; border-radius: var(--radius-lg) var(--radius-lg) 0 0; }
.live-card-frame :deep(.card-heading) { min-width: 0; max-width: 100%; }
.live-card-frame :deep(.card-heading > div) { min-width: 0; overflow-wrap: anywhere; }
.live-card-frame :deep(.card-actions) { flex-wrap: wrap; min-width: 0; max-width: 100%; }
.live-card-frame :deep(.card-actions > *) { max-width: 100%; white-space: normal; overflow-wrap: anywhere; }
.live-card-frame :deep(.card-content) { flex: 1; min-height: 0; overflow: auto; border-radius: 0 0 var(--radius-lg) var(--radius-lg); padding-bottom: 28px; overflow-wrap: anywhere; }
.live-card-frame :deep(.u-table) { min-width: 620px; }
.live-card-frame :deep(.comps-card .u-table) { min-width: 900px; }
.live-card-frame :deep(.target-table table) { min-width: 340px; }
.live-card-frame:focus-within { z-index: 3; }
.live-card-frame.is-active { opacity: .6; }
.live-card-frame.insert-before::before,.live-card-frame.insert-after::after { content: ''; position: absolute; left: 0; right: 0; height: 4px; border-radius: 2px; background: var(--accent); z-index: 20; pointer-events: none; }
.live-card-frame.insert-before::before { top: -10px; }
.live-card-frame.insert-after::after { bottom: -10px; }
.resize-handle { position: absolute; right: 2px; bottom: 2px; width: 28px; height: 26px; display: flex; justify-content: center; align-items: center; cursor: nwse-resize; touch-action: none; background: var(--surface-raised); color: var(--secondary); border-radius: var(--radius-sm); z-index: 2; }
.resize-handle:hover { color: var(--text); }
.layout-ghost { position: fixed; z-index: 1000; pointer-events: none; background: var(--surface-raised); color: var(--text); border: 1px solid var(--accent); border-radius: var(--radius-md); padding: 10px 12px; box-shadow: var(--shadow-card); font-size: 12px; }
.storage-notice { color: var(--orange-text); font-size: var(--font-xs); margin-bottom: 8px; }
.sr-only { position: absolute; width: 1px; height: 1px; padding: 0; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
@media(max-width:1199px) { .live-card-grid { grid-template-columns: minmax(0,1fr); } .live-card-frame { grid-column: 1 / -1 !important; } }
</style>
