<script setup lang="ts">
import { visibleHudAugments } from '../composables/aiProjection';
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import { visibleShop } from '../composables/buyProjection';
import { hudFrame } from '../composables/hud';
import type { Hud } from '../types';
const position = (x: number, y: number) => ({ transform: `translate(${x * 100}%, ${y * 100}%)` });
const host = ref<HTMLElement | null>(null);
const size = ref({ width: 1, height: 1 });
let observer: ResizeObserver | undefined;
onMounted(() => {
  observer = new ResizeObserver(entries => {
    const box = entries[0]?.contentRect;
    if (box) size.value = { width: box.width, height: box.height };
  });
  if (host.value) observer.observe(host.value);
});
onBeforeUnmount(() => observer?.disconnect());
const shop = computed(() => visibleShop(hudFrame.value));
const scale = computed(() => (hudFrame.value?.font_size ?? 12) / 12);
const colors = { ok: '#30d158', misplaced: '#ff9f0a', bench: '#64b5ff', missing: '#a1a1aa' };
const cells = computed(() => (hudFrame.value?.positions ?? []).map(p => {
  const x = p.x * size.value.width, y = p.y * size.value.height;
  const w = p.w * size.value.width, h = p.h * size.value.height;
  return { ...p, x, y, w, h, color: colors[p.state],
    points: [[0, -.5], [.5, -.25], [.5, .25], [0, .5], [-.5, .25], [-.5, -.25]]
      .map(([dx, dy]) => `${x + dx! * w},${y + dy! * h}`).join(' '),
    labelWidth: Math.max(w, p.name.length * 11 * scale.value + 8) };
}));
function insetArrow(a: Hud['arrows'][number]) {
  const target = cells.value.find(p => p.key === a.key);
  if (!target) return [];
  const x1 = a.x1 * size.value.width, y1 = a.y1 * size.value.height;
  const x2 = a.x2 * size.value.width, y2 = a.y2 * size.value.height;
  const dx = x2 - x1, dy = y2 - y1, distance = Math.hypot(dx, dy);
  // 两端各让出格半径的 40%，短到无法画线时不画反向箭头。
  const inset = target.h / 2 * .4;
  if (distance <= inset * 2) return [];
  return [{ ...a, x1: x1 + dx / distance * inset, y1: y1 + dy / distance * inset,
    x2: x2 - dx / distance * inset, y2: y2 - dy / distance * inset }];
}
const arrows = computed(() => (hudFrame.value?.arrows ?? []).flatMap(insetArrow));
</script>
<template>
  <div ref="host" class="hud-overlay" :style="{ opacity: hudFrame?.opacity ?? 1, fontSize: `${hudFrame?.font_size ?? 12}px` }" aria-label="游戏 HUD">
    <template v-if="hudFrame">
      <svg class="position-guide" :viewBox="`0 0 ${size.width} ${size.height}`" aria-label="目标站位指引">
        <defs>
          <marker v-for="kind in (['move', 'place'] as const)" :id="`position-arrow-${kind}`" :key="kind"
            viewBox="0 0 10 10" refX="9" refY="5" :markerWidth="9 * scale" :markerHeight="9 * scale" markerUnits="userSpaceOnUse" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 z" :fill="kind === 'move' ? colors.misplaced : colors.bench" />
          </marker>
        </defs>
        <line v-for="arrow in arrows" :key="arrow.key" class="guide-arrow" :data-kind="arrow.kind"
          :x1="arrow.x1" :y1="arrow.y1" :x2="arrow.x2" :y2="arrow.y2" :stroke-width="2.5 * scale"
          :stroke="arrow.kind === 'move' ? colors.misplaced : colors.bench" :marker-end="`url(#position-arrow-${arrow.kind})`" />
        <g v-for="cell in cells" :key="cell.key" class="guide-cell" :data-state="cell.state" :data-name="cell.name">
          <polygon :points="cell.points" fill="rgba(0,0,0,.06)" :stroke="cell.color"
            :stroke-width="(cell.state === 'ok' ? 1.2 : 2) * scale"
            :stroke-dasharray="cell.state === 'bench' || cell.state === 'missing' ? `${5 * scale} ${4 * scale}` : undefined" />
          <rect :x="cell.x - cell.labelWidth / 2" :y="cell.y + cell.h * .15" :width="cell.labelWidth"
            :height="15 * scale" :rx="3 * scale" fill="rgba(0,0,0,.72)" />
          <text class="guide-name" :x="cell.x" :y="cell.y + cell.h * .15 + 11.5 * scale"
            :font-size="11 * scale" text-anchor="middle" fill="white">{{ cell.name }}</text>
          <g v-if="cell.role" :transform="`translate(${cell.x - cell.w * .4} ${cell.y - cell.h * .28})`">
            <circle :r="8 * scale" :fill="cell.role === 'carry' ? '#f7c65b' : '#85b7d8'" stroke="#222" />
            <text :y="3.3 * scale" :font-size="10 * scale" text-anchor="middle" fill="#171717" font-weight="700">{{ cell.role === 'carry' ? 'C' : 'T' }}</text>
          </g>
        </g>
      </svg>
      <div v-for="unit in hudFrame.units" :key="unit.key" class="hud-position unit-position" :data-key="unit.key" :style="position(unit.x, unit.y_bottom)">
        <div class="hud-unit" :class="{ target: unit.target, unknown: unit.unknown }">
          <div class="hud-unit-title" :style="{ color: unit.unknown ? '#ddd' : `var(--cost-${unit.cost}, #ddd)` }">{{ unit.title }}</div>
          <div v-if="unit.traits" class="hud-unit-traits" :style="{ fontSize: `${Math.max((hudFrame.font_size ?? 12) - 2, 9)}px` }">{{ unit.traits }}</div>
        </div>
      </div>
      <div v-for="slot in shop" :key="slot.key" class="hud-position shop-position" :data-slot="slot.key" :style="position(slot.x, slot.y)">
        <div class="hud-shop" :class="slot.kind" :style="{ width: `${slot.w * 100}%`, height: `${slot.h * 100}%` }">
          <span class="hud-shop-tag">{{ slot.text }}</span>
        </div>
      </div>
      <div v-for="augment in visibleHudAugments(hudFrame)" :key="augment.key" class="hud-position augment-position" :style="position(augment.x, augment.y)">
        <div class="hud-augment" :class="{ best: augment.best, 'ai-best': augment.ai_best }">{{ augment.text }}</div>
      </div>
    </template>
  </div>
</template>
<style scoped>
.hud-overlay {
  position: relative;
  width: 100%;
  height: 100%;
  pointer-events: none;
  color: #fff;
}
.hud-position {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  transition: transform 120ms var(--ease);
  z-index: 1;
}
.position-guide {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}
.hud-unit {
  position: absolute;
  width: max-content;
  max-width: 240px;
  transform: translate(-50%, -100%);
  padding: 3px 6px;
  border-radius: 6px;
  background: rgba(0,0,0,.62);
  backdrop-filter: blur(12px);
  text-align: center;
  line-height: 1.35;
}
.hud-unit.target {
  outline: 1.5px solid #ffd60a;
}
.hud-unit-title {
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.hud-unit-traits {
  color: #ccc;
  white-space: nowrap;
}
.hud-shop {
  position: absolute;
  border: 3px solid var(--hud-mark);
  border-radius: 6px;
}
.hud-shop.buy {
  --hud-mark: #30d158;
}
.hud-shop.target {
  --hud-mark: #ffd60a;
}
.hud-shop-tag {
  position: absolute;
  left: 50%;
  top: -5px;
  transform: translate(-50%, -100%);
  max-width: calc(100% + 6px);
  padding: 2px 5px;
  border-radius: 4px;
  background: var(--hud-mark);
  color: #000;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  font-size: .85em;
  font-weight: 600;
}
.hud-augment {
  position: absolute;
  width: 180px;
  transform: translateX(-50%);
  padding: 4px 6px;
  border-radius: 5px;
  background: rgba(0,0,0,.62);
  text-align: center;
}
.hud-augment.best {
  background: #30d158;
  color: #000;
  font-weight: 600;
}
.hud-augment.ai-best { outline: 3px solid #c69cff; outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) {
  .hud-position {
    transition-duration: 0s;
  }
}
</style>
