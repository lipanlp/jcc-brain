<script setup lang="ts">
import { computed, onBeforeUnmount, ref, useId, watch } from 'vue';
import type { Advice, BoardCell, BoardPlan } from '../types';
import { iconImages, requestIcons } from '../composables/icons';
import UPopover from './ui/UPopover.vue';

const props = defineProps<{ plan: BoardPlan; selected: string | null; targets: Advice['targets'] }>();
const emit = defineEmits<{ select: [name: string] }>();
const id = useId().replace(/[^a-zA-Z0-9_-]/g, '');
const W = 58, H = W * 2 / Math.sqrt(3), GAP = W * .06, STEP = W + GAP;
const WIDTH = W * 7 + GAP * 6 + STEP / 2 + 16, HEIGHT = H * 3.25 + 20;
const hovered = ref<string | null>(null);
const focused = ref<string | null>(null);
let hideTimer: ReturnType<typeof setTimeout> | undefined;
const cells = computed(() => {
    const byPos = new Map(props.plan.cells.map(c => [`${c.row}:${c.col}`, c]));
    return Array.from({ length: 28 }, (_, i) => {
        const row = Math.floor(i / 7) + 1, col = i % 7 + 1;
        const x = 8 + W / 2 + (col - 1) * STEP + (row % 2===0 ? STEP / 2 : 0);
        const y = 12 + H / 2 + (row - 1) * H * .75;
        return { row, col, x, y, hero: byPos.get(`${row}:${col}`), key: `${row}-${col}` };
    });
});
const heroes = computed(() => cells.value.filter(c => c.hero).map(c => ({ ...c, hero: c.hero! })));
watch(() => props.plan, plan => {
    requestIcons(plan.cells.flatMap(c => [`hero:${c.name}`, ...c.items.map(n => `item:${n}`),
        ...c.item_details.flatMap(d => d.components.map(n => `item:${n}`))]));
}, { immediate: true });
function icon(kind: string, name: string) { return iconImages.value[`${kind}:${name}`] || undefined; }
function points(x: number, y: number, w = W) {
    const h = w * 2 / Math.sqrt(3);
    return `${x},${y-h/2} ${x+w/2},${y-h/4} ${x+w/2},${y+h/4} ${x},${y+h/2} ${x-w/2},${y+h/4} ${x-w/2},${y-h/4}`;
}
function show(name: string) { clearTimeout(hideTimer); hovered.value = name; }
function hide() { hideTimer = setTimeout(() => { hovered.value = null; }, 120); }
function updateOpen(open: boolean, name: string) { if (open) show(name); else hovered.value = null; }
function target(name: string) { return props.targets.find(t => t.name===name); }
function percent(value: number | null | undefined) { return value==null ? '未知' : `${(value*100).toFixed(1)}%`; }
function held(detail: BoardCell['item_details'][number], index: number) {
    const name = detail.components[index];
    return detail.held.filter(n => n===name).length > detail.components.slice(0,index).filter(n => n===name).length;
}
function ready(hero: BoardCell, name: string) { return hero.item_details.some(d => d.name===name && d.ready); }
onBeforeUnmount(() => clearTimeout(hideTimer));
</script>

<template>
  <div class="board-mini">
    <span class="row-label">前排</span>
    <div class="board-drawing">
      <svg :viewBox="`0 0 ${WIDTH} ${HEIGHT}`" preserveAspectRatio="xMidYMid meet" role="img" aria-label="目标阵容站位，第1行为前排，第4行为后排">
        <defs>
          <filter :id="`${id}-gray`"><feColorMatrix type="saturate" values="0" /></filter>
          <clipPath v-for="c in heroes" :id="`${id}-${c.key}`" :key="c.key"><polygon :points="points(c.x,c.y)" /></clipPath>
        </defs>
        <g v-for="c in cells" :key="c.key" :data-row="c.row" :data-col="c.col" :data-hero="c.hero?.name" class="board-cell"
           :class="{ highlighted: c.hero && (c.hero.name===selected || c.hero.name===hovered || c.hero.name===focused) }"
           :style="{ transformOrigin: `${c.x}px ${c.y}px` }">
          <polygon class="hero-hex" :class="{ empty: !c.hero, selected: c.hero?.name===selected, focused: c.hero?.name===focused }"
                   :points="points(c.x,c.y)" :style="{ '--cell-cost': c.hero?.cost ? `var(--cost-${c.hero.cost})` : 'var(--separator)' }" />
          <template v-if="c.hero">
            <g :clip-path="`url(#${id}-${c.key})`" :opacity="c.hero.owned_star===null ? .45 : 1" :filter="c.hero.owned_star===null ? `url(#${id}-gray)` : undefined">
              <image v-if="icon('hero',c.hero.name)" class="hero-portrait" :href="icon('hero',c.hero.name)"
                     :x="c.x-W/2+2" :y="c.y-H/2+2" :width="W-4" :height="H-4" preserveAspectRatio="xMidYMid slice" />
              <template v-else>
                <polygon :points="points(c.x,c.y,W-4)" :fill="`var(--cost-${c.hero.cost})`" opacity=".3" />
                <text class="portrait-fallback" :x="c.x" :y="c.y+4">{{ c.hero.name.slice(0,2) }}</text>
              </template>
            </g>
            <g :clip-path="`url(#${id}-${c.key})`">
              <rect class="name-band" :x="c.x-W/2" :y="c.y+H*.06" :width="W" :height="H*.22" />
              <foreignObject :x="c.x-W/2" :y="c.y+H*.06" :width="W" :height="H*.22">
                <div xmlns="http://www.w3.org/1999/xhtml" class="hero-name">{{ c.hero.name }}</div>
              </foreignObject>
            </g>
            <text v-if="c.hero.owned_star" class="owned-stars" :x="c.x" :y="c.y-H/2-2">{{ '★'.repeat(c.hero.owned_star) }}</text>
            <g v-else class="missing-mark"><rect :x="c.x+W/2-13" :y="c.y-11" width="13" height="12" rx="3" /><text :x="c.x+W/2-6.5" :y="c.y-2">缺</text></g>
            <g v-if="c.hero.role" class="role-mark" :class="c.hero.role">
              <polygon :points="points(c.x-W/2+5,c.y-H/4,18)" />
              <text :x="c.x-W/2+5" :y="c.y-H/4+4">{{ c.hero.role==='carry' ? 'C' : 'T' }}</text>
            </g>
            <g v-for="(name, index) in c.hero.items" :key="name+index" class="equipped-icon" :class="{ ready: ready(c.hero,name) }"
               :transform="`translate(${c.x-(c.hero.items.length*(W*.24+2)-2)/2+index*(W*.24+2)},${c.y+H/2-W*.24*.68})`">
              <rect :width="W*.24" :height="W*.24" rx="2" />
              <image v-if="icon('item',name)" :href="icon('item',name)" x="1" y="1" :width="W*.24-2" :height="W*.24-2" />
              <text v-else :x="W*.12" :y="W*.17">{{ name.slice(0,1) }}</text>
            </g>
          </template>
          <polygon v-if="c.hero" class="selection-outline" :points="points(c.x,c.y,W+6)" />
        </g>
      </svg>
      <div v-for="c in heroes" :key="c.key" class="hero-hit"
           :style="{ left: `${(c.x-W/2)/WIDTH*100}%`, top: `${(c.y-H/2)/HEIGHT*100}%`, width: `${W/WIDTH*100}%`, height: `${H/HEIGHT*100}%` }">
        <UPopover :model-value="hovered===c.hero.name" :width="360" :auto-focus="false" :label="`${c.hero.name}站位详情`" @update:model-value="updateOpen($event,c.hero.name)">
          <template #trigger>
            <button type="button" class="hero-target" :data-hero="c.hero.name" :aria-label="`${c.hero.name}，第${c.row}行第${c.col}列`"
                    :aria-pressed="selected===c.hero.name" @mouseenter="show(c.hero.name)" @mouseleave="hide"
                    @focus="focused=c.hero.name;show(c.hero.name)" @blur="focused=null;hide()" @click="emit('select',c.hero.name)" />
          </template>
          <div class="hero-tooltip" @mouseenter="show(c.hero.name)" @mouseleave="hide">
            <h3>{{ c.hero.name }} <span>{{ c.hero.cost }}费</span><b v-if="c.hero.role">{{ c.hero.role==='carry' ? '主C' : '主T' }}</b></h3>
            <p>{{ c.hero.traits.join(' · ') || '暂无羁绊' }}</p>
            <p v-if="c.hero.owned_star">已有 {{ c.hero.owned_star }}★ · {{ c.hero.on_board ? '在场' : '备战席' }}</p>
            <p v-else>还缺 {{ target(c.hero.name)?.need ?? '未知' }} 张 · 单次刷出 {{ percent(target(c.hero.name)?.per_roll) }}</p>
            <div v-for="detail in c.hero.item_details" :key="detail.name" class="recipe-line">
              <div class="recipe-title"><img v-if="icon('item',detail.name)" :src="icon('item',detail.name)" alt="" /><strong>{{ detail.name }}</strong><span v-if="detail.ready" class="craft-ready">现在就能合</span></div>
              <div class="components"><span>=</span><template v-for="(name,index) in detail.components" :key="index"><span v-if="index">+</span><span class="component" :class="{ held: held(detail,index) }"><img v-if="icon('item',name)" :src="icon('item',name)" alt="" />{{ name }}<b v-if="held(detail,index)">✓</b></span></template><span v-if="!detail.components.length">暂无合成配方</span></div>
            </div>
          </div>
        </UPopover>
      </div>
    </div>
    <span class="row-label rear">后排</span>
  </div>
</template>

<style scoped>
.board-mini { position: relative; width: 100%; max-width: 660px; padding-left: 24px; margin-inline: auto; }
.row-label { position: absolute; top: 12%; left: 0; color: var(--secondary); font-size: 10px; }
.rear { top: auto; bottom: 12%; }
.board-drawing { position: relative; width: 100%; }
svg { display: block; width: 100%; height: auto; overflow: visible; }
.hero-hex { fill: var(--surface-raised); stroke: var(--cell-cost); stroke-width: 3px; stroke-linejoin: round; }
.hero-hex.empty { fill: var(--hover); fill-opacity: .55; stroke: var(--separator); stroke-width: 1px; }
.board-cell { transition: transform var(--duration-fast); }
.board-cell.highlighted { transform: scale(1.06); }
.board-cell.highlighted .hero-portrait { filter: brightness(1.15); }
.selection-outline { fill: none; stroke: #fff; stroke-width: 3px; stroke-linejoin: round; opacity: 0; pointer-events: none; }
.highlighted .selection-outline { opacity: 1; filter: drop-shadow(0 0 4px rgb(255 255 255 / 40%)); }
text { text-anchor: middle; fill: var(--text); pointer-events: none; }
.portrait-fallback { font-size: 16px; font-weight: 600; }
.owned-stars { fill: var(--yellow-text); font-size: 10px; letter-spacing: 1px; }
.missing-mark rect { fill: #181b22; fill-opacity: .85; }
.missing-mark text { fill: #fff; font-size: 9px; }
.role-mark polygon { stroke: var(--surface); stroke-width: 1.5px; }
.role-mark.carry polygon { fill: #d9b45c; }
.role-mark.tank polygon { fill: #718da6; }
.role-mark text { fill: #141a20; font-size: 12px; font-weight: 800; }
.equipped-icon rect { fill: var(--surface-raised); stroke: var(--secondary); stroke-width: 1px; }
.equipped-icon.ready rect { stroke: var(--green-text); stroke-width: 2px; }
.equipped-icon text { font-size: 10px; }
.name-band { fill: #11151e; fill-opacity: .8; }
.hero-name { font-size: 10px; line-height: 14px; text-align: center; color: #fff; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.hero-hit { position: absolute; }
.hero-hit :deep(.popover-anchor) { width: 100%; height: 100%; display: block; }
.hero-target { display: block; width: 100%; height: 100%; padding: 0; background: transparent; border: 0; outline: 0; cursor: pointer; clip-path: polygon(50% 0,100% 25%,100% 75%,50% 100%,0 75%,0 25%); }
.hero-tooltip h3 { display: flex; align-items: center; gap: 8px; font-size: 15px; font-weight: 600; }
.hero-tooltip h3 span { color: var(--secondary); font-size: 12px; font-weight: 400; }
.hero-tooltip h3 b { margin-left: auto; color: var(--yellow-text); font-size: 12px; }
.hero-tooltip p { font-size: 12px; color: var(--secondary); margin-top: 7px; }
.recipe-line { margin-top: 12px; padding-top: 10px; border-top: 1px solid var(--separator); }
.recipe-title,.components,.component { display: flex; align-items: center; gap: 5px; }
.recipe-title { font-size: 12px; }
.hero-tooltip img { width: 21px; height: 21px; object-fit: cover; border-radius: 3px; }
.components { flex-wrap: wrap; font-size: 11px; margin-top: 7px; color: var(--secondary); }
.component.held,.craft-ready { color: var(--green-text); }
.craft-ready { margin-left: auto; font-size: 11px; }
</style>
