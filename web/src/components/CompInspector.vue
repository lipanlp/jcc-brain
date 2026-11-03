<script setup lang="ts">
import { computed, nextTick, ref, useId, watch } from 'vue';
import { Lock, Copy } from 'lucide-vue-next';
import type { CompDetail, TableColumn } from '../types';
import { iconImages, requestIcons } from '../composables/icons';
import { costColor, tierTone, decimal, rate, count, copyText } from '../composables/presentation';
import { toast } from '../composables/toast';
import UBadge from './ui/UBadge.vue';
import UButton from './ui/UButton.vue';
import UTable from './ui/UTable.vue';
import CompBoard from './CompBoard.vue';
import CompTraits from './CompTraits.vue';
import CompItem from './CompItem.vue';
const props=defineProps<{ comp: CompDetail; locked: boolean; defaultLocked: boolean; locking: boolean; watched: boolean; watching: boolean; watchFull: boolean }>();
defineEmits<{lock:[];unlock:[];defaultLock:[];defaultUnlock:[];watch:[]}>();
const root=ref<HTMLElement>();
const unitDetail=ref<HTMLElement>();
const detailId=useId();
let returnTarget:HTMLElement|undefined;
const copying=ref(false), phase=ref(''), selected=ref<string|null>(null);
const failed=ref<Record<string,boolean>>({});
const visual=computed(()=>props.comp.visualization);
const stage=computed(()=>visual.value.stages.find(s=>s.id===phase.value) ?? visual.value.stages[0]!);
const focused=computed(()=>stage.value.units.find(unit=>unit.key===selected.value));
const focusHero=computed(()=>focused.value ? visual.value.heroes[focused.value.name] : undefined);
const units=computed(()=>[...stage.value.units].sort((a,b)=>['主C','副C','核心',''].indexOf(a.role)-['主C','副C','核心',''].indexOf(b.role)));
watch(()=>props.comp,()=>{ phase.value=visual.value.stages[0]?.id ?? ''; selected.value=null; returnTarget=undefined; failed.value={}; },{immediate:true});
watch(()=>props.comp.id,()=>{ const panel=root.value?.closest('.split-inspector');if(panel)panel.scrollTop=0; },{flush:'post'});
watch(phase,()=>{selected.value=null;returnTarget=undefined;});
watch(visual,value=>requestIcons([...Object.keys(value.heroes).map(n=>`hero:${n}`),...Object.keys(value.items).map(n=>`item:${n}`)]),{immediate:true});
const augments=computed(()=>props.comp.augments.map((row,index)=>({...row,key:`${row.name}#${index}`})));
const augmentColumns:TableColumn[]=[{key:'name',title:'海克斯',width:'36%',wrap:true},{key:'rating',title:'评级',width:'14%'},{key:'avg_placement',title:'均名',width:'18%',align:'right',format:decimal},{key:'placement_delta',title:'变化',width:'16%',align:'right'},{key:'play_rate',title:'选取率',width:'16%',align:'right',format:rate}];
const metrics=computed(()=>[{label:'均名',value:decimal(props.comp.avg_placement)},{label:'前四',value:rate(props.comp.top4_rate)},{label:'登顶',value:rate(props.comp.top_rate)},{label:'出场',value:rate(props.comp.pick_rate)},{label:'样本',value:count(props.comp.sample_count)}]);
const popularity=computed(()=>[{label:'使用',value:props.comp.guide?.use_count},{label:'周使用',value:props.comp.guide?.week_use},{label:'收藏',value:props.comp.guide?.collect_count},{label:'评论',value:props.comp.guide?.comment_count}].filter(m=>m.value!=null));
const statDate=computed(()=>props.comp.guide?.stat_date?.replace(/^(\d{4})(\d{2})(\d{2})$/,'$1-$2-$3') ?? '未知');
function portrait(name:string){return failed.value[name] ? undefined : iconImages.value[`hero:${name}`] || undefined;}
function position(unit:typeof stage.value.units[number]){return unit.position ? `${unit.position[0]} 行 ${unit.position[1]} 列` : unit.position_note;}
function revealUnitElement(element:HTMLElement, alignStart=false){
  const panel=root.value?.closest<HTMLElement>('.split-inspector');
  if(!panel?.contains(element))return;
  const bounds=panel.getBoundingClientRect();
  const toolbar=panel.querySelector<HTMLElement>(':scope > .inspector-toolbar');
  const gap=parseFloat(getComputedStyle(panel).getPropertyValue('--space-2')) || 8;
  const top=Math.max(bounds.top+panel.clientTop,toolbar?.getBoundingClientRect().bottom ?? bounds.top)+gap;
  const bottom=bounds.top+panel.clientTop+panel.clientHeight-gap;
  const rect=element.getBoundingClientRect();
  const delta=alignStart || rect.top<top ? rect.top-top : rect.bottom>bottom ? rect.bottom-bottom : 0;
  // 只移动检查器，给窄窗口的 sticky 工具栏留空间；即时滚动也适用于减弱动态效果。
  panel.scrollTo({top:panel.scrollTop+delta,behavior:'instant'});
}
async function closeUnit(){
  const trigger=returnTarget;
  selected.value=null;
  returnTarget=undefined;
  await nextTick();
  if(trigger?.isConnected){trigger.focus({preventScroll:true});revealUnitElement(trigger);}
}
async function selectUnit(key:string, trigger:HTMLElement){
  if(selected.value===key){await closeUnit();return;}
  returnTarget=trigger;
  selected.value=key;
  await nextTick();
  if(selected.value!==key || !unitDetail.value)return;
  unitDetail.value.focus({preventScroll:true});
  revealUnitElement(unitDetail.value,true);
}
async function copyCode(){if(!props.comp.game_code)return;copying.value=true;try{await copyText(props.comp.game_code);toast.success('已复制阵容码');}catch{toast.error('复制失败，请重试');}finally{copying.value=false;}}
function delta(value:unknown){return typeof value==='number' ? `${value>=0 ? '+' : ''}${value.toFixed(2)}` : '—';}
</script>
<template>
  <article ref="root" class="inspector-content comp-inspector" :data-comp-id="comp.id">
    <div class="inspector-heading"><UBadge :tone="tierTone(comp.tier)">{{ comp.tier || '—' }}</UBadge><h1>{{ comp.name }}</h1></div>
    <div class="inspector-chips"><UBadge>{{ comp.source_label }}</UBadge><UBadge>{{ comp.style_label }}</UBadge><span class="inspector-muted">版本 {{ comp.game_version || '未提供' }}</span></div>
    <div class="inspector-actions">
      <UButton variant="primary" :icon="Lock" :loading="locking" @click="locked ? $emit('unlock') : $emit('lock')">{{ locked ? '解除本局锁定' : '本局锁定' }}</UButton>
      <UButton :disabled="locking" @click="defaultLocked ? $emit('defaultUnlock') : $emit('defaultLock')">{{ defaultLocked ? '取消默认锁定' : '设为默认锁定' }}</UButton>
      <UButton :loading="watching" :disabled="!watched&&watchFull" @click="$emit('watch')">{{ watched ? '取消关注' : '加入本局关注' }}</UButton>
      <UButton v-if="comp.game_code" :icon="Copy" :loading="copying" @click="copyCode">复制阵容码</UButton>
    </div>
    <div v-if="comp.source==='dataj'" class="comp-metrics"><div v-for="metric in metrics" :key="metric.label"><span>{{ metric.label }}</span><strong>{{ metric.value }}</strong></div></div>
    <p v-else class="official-note inspector-muted">{{ comp.source_label }}攻略 · 暂无胜率统计</p>

    <section class="formation-section">
      <div class="section-heading"><h2>阵容站位</h2><span class="inspector-muted">静态攻略 · 目标配置</span></div>
      <div v-if="visual.stages.length>1" class="stage-options" role="group" aria-label="攻略阶段"><button v-for="option in visual.stages" :key="option.id" type="button" :data-stage-option="option.id" :aria-pressed="stage.id===option.id" @click="phase=option.id">{{ option.label }}</button></div>
      <p v-if="stage.origin==='guide'" class="stage-origin">关联攻略阶段 · {{ stage.source_name }}。并非 dataj 自身成型。</p>
      <p class="stage-count inspector-muted"><b v-if="stage.population!=null">{{ stage.population }} 人口阶段 · </b>{{ stage.hero_count }} 名英雄<span v-if="stage.other_count"> · {{ stage.other_count }} 个召唤物/其他单位</span> · 点击棋子查看详情</p>
      <CompBoard :stage="stage" :selected="selected" @select="selectUnit" />
      <div v-if="focused" :id="detailId" ref="unitDetail" class="unit-focus" :data-selected-unit="focused.name" role="region" :aria-labelledby="`${detailId}-title`" tabindex="-1">
        <div class="section-heading"><h3 :id="`${detailId}-title`">{{ focused.name }} <small>{{ focused.is_hero ? (focused.cost ? `${focused.cost} 费` : '费用未知') : `其他单位 · ${focused.unit_type}` }}</small></h3><UButton size="sm" variant="plain" aria-label="关闭英雄详情" @click="closeUnit">关闭</UButton></div>
        <p>{{ focused.role || (focused.is_hero ? '来源未标注定位' : '召唤物/其他单位') }} · {{ focused.target_star!=null ? `目标 ${focused.target_star}★` : '目标星级未提供' }} · {{ position(focused) }}</p>
        <p v-if="focused.source_star!=null&&focused.source_star!==focused.target_star" class="inspector-muted">来源棋子星级 {{ focused.source_star }}★；目标按攻略追三星名单显示。</p>
        <p v-if="focusHero?.traits.length">{{ focusHero.traits.join(' · ') }}<span v-if="focusHero.attack_range!=null"> · 射程 {{ focusHero.attack_range }}</span></p>
        <h4 v-if="focusHero?.skill_name">{{ focusHero.skill_name }}</h4><p class="inspector-prose">{{ focusHero?.skill_desc || '来源未提供此单位技能资料' }}</p>
        <div class="equipment-list"><CompItem v-for="(name,index) in focused.items" :key="index" :name="name" :items="visual.items" /></div><p v-if="!focused.items.length" class="inspector-muted">来源未提供装备配置</p>
      </div>
    </section>

    <section class="stage-units"><h2>当前阶段英雄与装备</h2>
      <div v-for="unit in units" :key="unit.key" class="comp-hero" :class="{selected:unit.key===selected}" :data-unit-card="unit.name">
        <button type="button" class="unit-heading" :aria-label="`查看${unit.name}英雄详情`" :aria-expanded="selected===unit.key" :aria-controls="selected===unit.key ? detailId : undefined" @click="selectUnit(unit.key,$event.currentTarget as HTMLElement)">
          <img v-if="unit.is_hero&&portrait(unit.name)" :src="portrait(unit.name)" alt="" :style="{borderColor:costColor(unit.cost)}" @error="failed[unit.name]=true" /><span v-else class="unit-symbol">{{ unit.name.slice(0,2) }}</span>
          <span><strong>{{ unit.name }}</strong><small>{{ unit.is_hero ? (unit.cost ? `${unit.cost} 费` : '费用未知') : `召唤物/其他单位 · ${unit.unit_type}` }} · {{ unit.role || '定位未标注' }} · {{ unit.target_star!=null ? `目标 ${unit.target_star}★` : '星级未提供' }}<span v-if="unit.source_star!=null&&unit.source_star!==unit.target_star"> · 来源 {{ unit.source_star }}★</span></small><small>{{ position(unit) }}</small></span>
        </button>
        <div v-if="unit.items.length" class="equipment-list hero-items"><CompItem v-for="(name,index) in unit.items" :key="index" :name="name" :items="visual.items" /></div><p v-else class="inspector-muted">来源未提供装备配置</p>
      </div>
    </section>
    <section><h2>当前阶段羁绊</h2><CompTraits :traits="stage.traits" /></section>

    <section v-if="comp.guide&&visual.guide" class="official-guide">
      <h2>阵容攻略</h2><p v-if="comp.guide.author" class="inspector-muted">作者 · {{ comp.guide.author }}<span v-if="comp.source==='dataj'"> · 关联攻略：{{ visual.guide.name }}</span></p>
      <div class="guide-sections"><article v-for="part in visual.guide.sections" :key="part.key" :data-guide-section="part.key"><h3>{{ part.title }}</h3><p class="inspector-prose">{{ part.text }}</p></article><p v-if="!visual.guide.sections.length" class="inspector-muted">来源未提供文字思路</p></div>
      <div v-if="visual.guide.carry_items.length||visual.guide.carry_items_backup.length||comp.guide.equipment_order.length" class="guide-group"><h3>装备选择</h3>
        <h4 v-if="visual.guide.carry_items.length">主C 首选</h4><div class="equipment-list"><CompItem v-for="(name,index) in visual.guide.carry_items" :key="index" :name="name" :items="visual.items" /></div>
        <h4 v-if="visual.guide.carry_items_backup.length">主C 备选</h4><div class="equipment-list"><CompItem v-for="(name,index) in visual.guide.carry_items_backup" :key="index" :name="name" :items="visual.items" /></div>
        <h4 v-if="comp.guide.equipment_order.length">散件优先顺序</h4><ol class="equipment-priority"><li v-for="(name,index) in comp.guide.equipment_order" :key="index"><CompItem :name="name" :items="visual.items" /></li></ol>
      </div>
      <div v-if="comp.guide.recommended_augments.length||visual.guide.alternative_augments.length" class="guide-group"><h3>海克斯选择</h3>
        <div v-for="group in [{label:'首选',names:comp.guide.recommended_augments},{label:'备选',names:visual.guide.alternative_augments}]" :key="group.label"><h4 v-if="group.names.length">{{ group.label }}</h4><details v-for="name in group.names" :key="name" class="text-detail"><summary>{{ name }} <span class="inspector-muted">{{ visual.augments[name]?.tier ? `${visual.augments[name]?.tier} 级` : '等级未提供' }}</span></summary><p class="inspector-prose">{{ visual.augments[name]?.description || '资料库暂未收录描述' }}</p></details></div>
      </div>
      <div v-if="visual.guide.adventures.length" class="guide-group"><h3>自然仙灵选择</h3><div v-for="group in visual.guide.adventures" :key="group.stage"><h4>来源阶段 {{ group.stage }}</h4><details v-for="name in group.names" :key="name" class="text-detail"><summary>{{ name }}</summary><p class="inspector-prose">{{ visual.mechanics[name]?.description || '资料库暂未收录描述' }}</p><p v-for="(upgrade,index) in visual.mechanics[name]?.upgrades" :key="index" class="inspector-prose">升级 {{ index+1 }} · {{ upgrade }}</p></details></div></div>
      <div v-if="popularity.length" class="guide-popularity guide-group"><h3>微社区热度</h3><p class="inspector-muted">统计日期 {{ statDate }} · 非胜率</p><div class="inspector-chips"><UBadge v-for="metric in popularity" :key="metric.label">{{ metric.label }} {{ count(metric.value) }}</UBadge></div></div>
      <details class="text-detail original-text"><summary>完整攻略原文</summary><p class="inspector-prose guide-text">{{ comp.guide.text }}</p></details>
    </section>
    <section v-else class="guide-empty"><h2>文字攻略</h2><p class="inspector-muted">该来源未提供可用的文字攻略与关联阶段，以上为阵容本身的数据。</p></section>
    <section v-if="comp.augments.length"><details class="augment-stats"><summary>dataj 海克斯统计 · {{ comp.augments.length }} 项</summary><UTable class="augment-table" :columns="augmentColumns" :rows="augments" row-key="key"><template #cell-rating="{value}"><UBadge :tone="tierTone(value as string)">{{ value || '—' }}</UBadge></template><template #cell-placement_delta="{value}">{{ delta(value) }}</template></UTable></details></section>
  </article>
</template>
<style scoped>
.comp-inspector { min-width: 0; }
.comp-metrics { display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:var(--space-1);margin-top:var(--space-3);padding:var(--space-2);background:var(--hover);border-radius:var(--radius-md); }
.comp-metrics>div {display:flex;flex-direction:column;align-items:center;gap:var(--space-1);font-size:var(--font-sm);}.comp-metrics span,small {color:var(--secondary);font-size:var(--font-xs);}
.official-note {margin-top:var(--space-2);}.section-heading {display:flex;justify-content:space-between;gap:var(--space-2);align-items:center;}.section-heading h2 {margin:0;}
.stage-options {display:flex;flex-wrap:wrap;gap:var(--space-1);margin-top:var(--space-3);}.stage-options button {background:var(--hover);border:1px solid var(--separator);border-radius:var(--radius-sm);font-size:var(--font-xs);padding:var(--space-1) var(--space-2);}.stage-options button[aria-pressed=true] {border-color:var(--accent);background:var(--accent-soft);color:var(--blue-text);}
.stage-count,.stage-origin {margin:var(--space-2) 0;}.stage-origin {font-size:var(--font-xs);color:var(--orange-text);line-height:1.6;}
.unit-focus,.comp-hero,.guide-sections article {border:1px solid var(--separator);border-radius:var(--radius-md);padding:var(--space-3);background:var(--surface-raised);margin-top:var(--space-2);}.unit-focus {border-color:var(--accent);}.unit-focus p {margin-top:var(--space-2);}.unit-focus h4 {margin-top:var(--space-3);}
.unit-heading {display:flex;align-items:center;gap:var(--space-3);width:100%;padding:0;background:var(--transparent);text-align:left;}.unit-heading>span:last-child {min-width:0;}.unit-heading strong {font-size:var(--font-md);overflow-wrap:anywhere;}.unit-heading small {display:block;line-height:1.7;}.unit-heading img,.unit-symbol {width:42px;height:46px;object-fit:cover;flex-shrink:0;border:2px solid var(--separator);border-radius:var(--radius-sm);}.unit-symbol {display:grid;place-items:center;background:var(--hover);font-size:var(--font-xs);}.comp-hero.selected {border-color:var(--accent);}
.equipment-list {display:grid;gap:var(--space-2);margin-top:var(--space-2);}.comp-hero>p {margin-top:var(--space-2);}.guide-group {margin-top:var(--space-5);padding-top:var(--space-4);border-top:1px solid var(--separator);}.guide-group h3,.guide-sections h3 {font-size:var(--font-md);font-weight:600;}.guide-group h4 {font-size:var(--font-xs);color:var(--secondary);font-weight:500;margin:var(--space-3) 0 var(--space-2);}.guide-sections p {margin-top:var(--space-2);}.equipment-priority {padding-left:var(--space-5);}.equipment-priority li {margin-top:var(--space-2);}
.text-detail {border-bottom:1px solid var(--separator);padding:var(--space-2) 0;font-size:var(--font-sm);}.text-detail summary,.augment-stats>summary {cursor:pointer;font-size:var(--font-sm);line-height:1.7;}.text-detail p {padding:var(--space-2) 0;}.original-text {margin-top:var(--space-4);}.guide-popularity .inspector-chips {margin-top:var(--space-2);}
.augment-table :deep(th),.augment-table :deep(td) {font-size:var(--font-xs);padding:var(--space-1);}.augment-table :deep(td:first-child) {white-space:normal;}
</style>
