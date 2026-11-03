<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue';
import type { Advice, TraitView } from '../types';
import { iconImages, requestIcons } from '../composables/icons';
import UPopover from './ui/UPopover.vue';

const props = defineProps<{ view?: Advice['traits_view'] }>();
const rows = computed(() => [{ key: 'final', label: '成型', traits: props.view?.final ?? [] },
    { key: 'current', label: '当前', traits: props.view?.current ?? [] }]);
const opened = ref('');
let timer: ReturnType<typeof setTimeout> | undefined;
function show(key: string) { clearTimeout(timer); opened.value = key; }
function hide() { timer = setTimeout(() => { opened.value = ''; }, 120); }
function tone(t: TraitView) {
    if (t.tier_index < 0 || t.active===false) return 'inactive';
    const n = t.breakpoints.length;
    const palette = n<=1 ? ['gold'] : n===2 ? ['bronze', 'gold'] : n===3 ? ['bronze', 'silver', 'gold'] : ['bronze', 'silver', 'gold', 'prismatic'];
    return palette[Math.round(t.tier_index / Math.max(1,n-1) * (palette.length-1))];
}
function icon(key: string) { return iconImages.value[key] || undefined; }
function held(t: TraitView, index: number) {
    const e = t.emblem!;
    const name = e.components[index];
    return e.held.filter(n => n===name).length > e.components.slice(0,index).filter(n => n===name).length;
}
watch(rows, value => requestIcons(value.flatMap(r => r.traits.flatMap(t => [t.picture_key,
    ...(t.emblem ? [t.emblem.name, ...t.emblem.components].map(n => `item:${n}`) : [])]))), { immediate: true });
onBeforeUnmount(() => clearTimeout(timer));
</script>

<template>
  <div class="traits-view">
    <div v-for="row in rows" :key="row.key" class="trait-row" :data-row="row.key">
      <span class="row-label">{{ row.label }}</span>
      <div class="trait-strip">
        <UPopover v-for="t in row.traits" :key="t.name" :model-value="opened===`${row.key}:${t.name}`" :width="340" :auto-focus="false" :label="`${t.name}羁绊详情`"
                  @update:model-value="$event ? show(`${row.key}:${t.name}`) : opened=''">
          <template #trigger>
            <button class="trait-chip" :class="[tone(t), { 'in-comp': row.key==='current' && t.in_comp, dim: row.key==='current' && !t.in_comp }]"
                    :data-trait="t.name" :aria-label="`${row.label} ${t.name} ${t.count ?? '未知'}人`"
                    @mouseenter="show(`${row.key}:${t.name}`)" @mouseleave="hide" @focus="show(`${row.key}:${t.name}`)" @blur="hide"
                    @click="show(`${row.key}:${t.name}`)">
              <img v-if="icon(t.picture_key)" :src="icon(t.picture_key)" alt="" />
              <span v-else class="trait-fallback">{{ t.name.slice(0,1) }}</span>
              <span class="trait-name">{{ t.name }}</span><b>{{ t.count ?? '—' }}</b>
            </button>
          </template>
          <div class="trait-tooltip" @mouseenter="show(`${row.key}:${t.name}`)" @mouseleave="hide">
            <h3>{{ t.name }} <small>{{ t.count ?? '未知' }} 人<span v-if="t.next_breakpoint"> · 下档 {{ t.next_breakpoint }} 人</span></small></h3>
            <p v-if="t.description">{{ t.description }}</p>
            <ol class="trait-levels">
              <li v-for="(count,index) in t.breakpoints" :key="index" :class="{ reached: index===t.tier_index && t.active!==false }">
                <b>({{ count }})</b> {{ t.level_desc[index] || '暂无该档说明' }} <span v-if="index===t.tier_index && t.active!==false">✓ 当前档</span>
              </li>
            </ol>
            <div v-if="t.emblem" class="emblem-recipe">
              <strong><img v-if="icon(`item:${t.emblem.name}`)" :src="icon(`item:${t.emblem.name}`)" alt="" />{{ t.emblem.name }}<small v-if="t.emblem.owned">已持有纹章</small></strong>
              <div class="components"><span>=</span><template v-for="(part,index) in t.emblem.components" :key="index"><span v-if="index">+</span><span class="component" :class="{held:held(t,index)}"><img v-if="icon(`item:${part}`)" :src="icon(`item:${part}`)" alt="" />{{ part }}<small>{{ held(t,index) ? '✓ 已持有' : '未持有' }}</small></span></template><span v-if="!t.emblem.components.length">暂无合成配方</span></div>
            </div>
            <p v-else>暂无关联转职纹章</p>
          </div>
        </UPopover>
        <span v-if="!row.traits.length" class="empty-traits">{{ row.key==='final' ? '暂无成型羁绊' : '等待羁绊识别' }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.traits-view { padding: 8px 12px 0; display: grid; gap: 4px; }
.trait-row { display: flex; align-items: center; gap: 8px; min-width: 0; }
.row-label,.empty-traits { font-size: 11px; color: var(--secondary); white-space: nowrap; }
.trait-strip { display: flex; align-items: center; gap: 6px; overflow-x: auto; min-width: 0; padding: 1px 0; }
.trait-chip { display: inline-flex; flex-shrink: 0; align-items: center; gap: 4px; padding: 1px 5px 1px 1px; border: 1px solid var(--trait-color,var(--separator)); border-radius: 5px; background: var(--hover); font-size: 11px; white-space: nowrap; }
.trait-chip img,.trait-fallback { width: 24px; height: 24px; object-fit: contain; border-radius: 3px; background: var(--trait-color,var(--hover)); }
:root[data-theme="light"] .trait-chip img { background: rgba(0,0,0,.55); border: 1px solid rgba(0,0,0,.25); border-radius: 50%; padding: 2px; box-sizing: border-box; }
@media (prefers-color-scheme: light) {
  :root:not([data-theme]) .trait-chip img { background: rgba(0,0,0,.55); border: 1px solid rgba(0,0,0,.25); border-radius: 50%; padding: 2px; box-sizing: border-box; }
}
.trait-fallback { display: grid; place-items: center; color: var(--text); }
.trait-chip b { color: var(--trait-color,var(--secondary)); }
.bronze { --trait-color: #b68c6b; }.silver { --trait-color: #b4c5d7; }.gold { --trait-color: #e4bc58; }.prismatic { --trait-color: #c2aff7; background: linear-gradient(110deg,#75bfbc26,#c2aff72b,#e4bc5820); }
.dim { opacity: .48; }.inactive { filter: grayscale(1); }.in-comp { box-shadow: inset 0 0 0 1px rgb(255 255 255 / 16%); }
.trait-chip:focus-visible { outline: 2px solid #fff; outline-offset: -2px; }
.trait-tooltip { font-size: 12px; line-height: 1.55; background: var(--surface-raised); padding: 8px; border-radius: 6px; }
.trait-tooltip h3 { font-size: 14px; }.trait-tooltip h3 small { font-weight: 400; color: var(--secondary); font-size: 11px; }
.trait-tooltip p { margin-top: 8px; color: var(--secondary); white-space: pre-line; }
.trait-levels { list-style: none; padding: 0; margin-top: 8px; display: grid; gap: 4px; }
.trait-levels li { padding: 4px 6px; color: var(--secondary); border-radius: 4px; }.trait-levels .reached { background: var(--hover); color: var(--yellow-text); }
.emblem-recipe { border-top: 1px solid var(--separator); padding-top: 8px; margin-top: 10px; }
.emblem-recipe strong,.components,.component { display: flex; gap: 5px; align-items: center; }.components { flex-wrap: wrap; margin-top: 8px; }
.emblem-recipe img { width: 21px; height: 21px; object-fit: contain; border-radius: 3px; }.emblem-recipe small { font-size: 10px; font-weight: 400; color: var(--secondary); }.held,.held small,.emblem-recipe strong small { color: var(--green-text); }
</style>
