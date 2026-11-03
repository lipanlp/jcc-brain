<script setup lang="ts">
import { computed, ref, useId, watch } from 'vue';
import type { CompStage } from '../types';
import { iconImages, requestIcons } from '../composables/icons';
import UTooltip from './ui/UTooltip.vue';
const props = defineProps<{ stage: CompStage; selected: string | null }>();
defineEmits<{ select: [key: string, trigger: HTMLElement] }>();
const id = useId().replace(/[^a-zA-Z0-9_-]/g,'');
const W=58, H=W*2/Math.sqrt(3), STEP=W*1.06, WIDTH=STEP*7+STEP/2+12, HEIGHT=H*3.25+34;
const failed=ref<Record<string,boolean>>({});
const cells=computed(()=>Array.from({length:28},(_,index)=>{
  const row=Math.floor(index/7)+1,col=index%7+1;
  return {row,col,key:`${row}-${col}`,x:8+W/2+(col-1)*STEP+(row%2===0 ? STEP/2 : 0),y:24+H/2+(row-1)*H*.75,
    unit:props.stage.units.find(unit=>unit.position?.[0]===row && unit.position?.[1]===col)};
}));
const placed=computed(()=>cells.value.filter(c=>c.unit).map(c=>({...c,unit:c.unit!})));
const unplaced=computed(()=>props.stage.units.filter(unit=>!unit.position));
watch(()=>props.stage,stage=>requestIcons(stage.units.flatMap(unit=>[...(unit.is_hero ? [`hero:${unit.name}`] : []),...unit.items.map(name=>`item:${name}`)])),{immediate:true});
function icon(key:string){return failed.value[key] ? undefined : iconImages.value[key] || undefined;}
function points(x:number,y:number,w=W){const h=w*2/Math.sqrt(3);return `${x},${y-h/2} ${x+w/2},${y-h/4} ${x+w/2},${y+h/4} ${x},${y+h/2} ${x-w/2},${y+h/4} ${x-w/2},${y-h/4}`;}
</script>
<template>
  <div class="comp-board" :data-stage="stage.id">
    <div class="static-board-drawing">
      <span class="board-direction">前排 · 第 1 行</span>
      <svg :viewBox="`0 0 ${WIDTH} ${HEIGHT}`" role="img" aria-label="静态攻略站位，4 行 7 列，上方为前排">
        <defs><clipPath v-for="cell in placed" :id="`${id}-${cell.key}`" :key="cell.key"><polygon :points="points(cell.x,cell.y,W-4)" /></clipPath></defs>
        <g v-for="cell in cells" :key="cell.key" class="board-cell" :class="{occupied:cell.unit,selected:cell.unit?.key===selected}" :data-row="cell.row" :data-col="cell.col" :data-unit="cell.unit?.name">
          <polygon class="unit-hex" :points="points(cell.x,cell.y)" :style="{'--unit-color':cell.unit?.cost ? `var(--cost-${cell.unit.cost})` : 'var(--separator)'}" />
          <template v-if="cell.unit">
            <g :clip-path="`url(#${id}-${cell.key})`">
              <image v-if="cell.unit.is_hero && icon(`hero:${cell.unit.name}`)" :href="icon(`hero:${cell.unit.name}`)" :x="cell.x-W/2+2" :y="cell.y-H/2+2" :width="W-4" :height="H-4" preserveAspectRatio="xMidYMid slice" @error="failed[`hero:${cell.unit.name}`]=true" />
              <text v-else class="portrait-fallback" :x="cell.x" :y="cell.y">{{ cell.unit.is_hero ? cell.unit.name.slice(0,2) : '其他' }}</text>
              <rect class="name-band" :x="cell.x-W/2" :y="cell.y+3" :width="W" height="15" />
              <text class="unit-name" :x="cell.x" :y="cell.y+14" :textLength="cell.unit.name.length>4 ? W-6 : undefined" lengthAdjust="spacingAndGlyphs">{{ cell.unit.name }}</text>
            </g>
            <text v-if="cell.unit.target_star!=null" class="target-stars" :x="cell.x" :y="cell.y-H/2-3">{{ cell.unit.target_star }}★</text>
            <text v-if="cell.unit.role==='主C'" class="carry-mark" :x="cell.x-W/2+8" :y="cell.y-H/4">C</text>
            <g v-for="(name,index) in cell.unit.items" :key="index" :transform="`translate(${cell.x-cell.unit.items.length*8+index*16},${cell.y+H/2-15})`">
              <rect class="item-frame" width="15" height="15" rx="2" />
              <image v-if="icon(`item:${name}`)" :href="icon(`item:${name}`)" x="1" y="1" width="13" height="13" @error="failed[`item:${name}`]=true" />
              <text v-else class="item-letter" x="7.5" y="11">{{ name.slice(0,1) }}</text>
            </g>
          </template>
        </g>
      </svg>
      <div v-for="cell in placed" :key="cell.key" class="unit-hit" :style="{left:`${(cell.x-W/2)/WIDTH*100}%`,top:`${(cell.y-H/2)/HEIGHT*100}%`,width:`${W/WIDTH*100}%`,height:`${H/HEIGHT*100}%`}">
        <UTooltip :text="`${cell.unit.name} · ${cell.row} 行 ${cell.col} 列${cell.unit.target_star!=null ? ` · 目标 ${cell.unit.target_star}★` : ''}\n${cell.unit.items.join('、') || '未提供装备'}`">
          <button type="button" class="unit-target" :data-unit-key="cell.unit.key" :aria-label="`${cell.unit.name}，${cell.row}行${cell.col}列${cell.unit.role ? '，'+cell.unit.role : ''}，查看静态详情`" :aria-pressed="selected===cell.unit.key" @click="$emit('select',cell.unit.key,$event.currentTarget as HTMLElement)" />
        </UTooltip>
      </div>
      <span class="board-direction rear">后排 · 第 4 行</span>
    </div>
    <div v-if="unplaced.length" class="unplaced">
      <strong>未定位单位</strong>
      <button v-for="unit in unplaced" :key="unit.key" type="button" :data-unit-key="unit.key" :aria-pressed="selected===unit.key" @click="$emit('select',unit.key,$event.currentTarget as HTMLElement)">{{ unit.name }} · {{ unit.position_note }}</button>
    </div>
  </div>
</template>
<style scoped>
.static-board-drawing { position: relative; width: 100%; margin-bottom: 16px; }
svg { display: block; width: 100%; overflow: visible; }
.board-direction { position: absolute; top: 0; left: 0; font-size: var(--font-xs); color: var(--secondary); }
.rear { top: auto; bottom: -16px; left: auto; right: 0; }
.unit-hex { fill: var(--surface-raised); stroke: var(--unit-color); stroke-width: 3; stroke-linejoin: round; }
.board-cell:not(.occupied) .unit-hex { fill: var(--hover); stroke: var(--separator); stroke-width: 1; }
.board-cell.selected .unit-hex { stroke: var(--accent); stroke-width: 5; }
text { text-anchor: middle; fill: var(--text); pointer-events: none; }
.portrait-fallback { font-size: 14px; font-weight: 600; }
.name-band { fill: var(--tooltip); }
.unit-name { fill: var(--tooltip-text); font-size: 10px; }
.target-stars,.carry-mark { fill: var(--yellow-text); font-size: 10px; font-weight: 800; }
.carry-mark { stroke: var(--surface); paint-order: stroke; stroke-width: 3px; }
.item-frame { fill: var(--surface); stroke: var(--secondary); }
.item-letter { font-size: 9px; }
.unit-hit { position: absolute; }
.unit-hit :deep(.tooltip-anchor) { width: 100%; height: 100%; display: block; }
.unit-target { width: 100%; height: 100%; display: block; background: var(--transparent); padding: 0; clip-path: polygon(50% 0,100% 25%,100% 75%,50% 100%,0 75%,0 25%); }
.unit-target:focus-visible { background: var(--focus); }
.unplaced { display: grid; gap: var(--space-2); margin-top: var(--space-3); font-size: var(--font-xs); }
.unplaced button { background: var(--hover); border-radius: var(--radius-sm); padding: var(--space-2); text-align: left; overflow-wrap: anywhere; }
</style>
