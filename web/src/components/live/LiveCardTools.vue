<script setup lang="ts">
import { computed, inject, ref } from 'vue';
import UPopover from '../ui/UPopover.vue';
import { GripVertical, SlidersHorizontal } from 'lucide-vue-next';
import { cardDefinitions, controlsKey, maxHeight, minHeight, type CardId } from './layout';
const props = defineProps<{ id: CardId }>();
const settingsOpen = ref(false);
const controls = inject(controlsKey)!;
const card = computed(() => controls.cards.value.find(card => card.id === props.id)!);
const index = computed(() => controls.cards.value.findIndex(card => card.id === props.id));
const title = cardDefinitions.find(card => card.id === props.id)!.title;
</script>
<template>
  <div class="live-card-tools">
    <button type="button" class="drag-handle" :aria-label="`拖动${title}排序`" :title="`拖动${title}；也可用上下方向键排序，Home / End 移到首尾`"
            @pointerdown="controls.start($event,id,'drag')" @keydown="controls.dragKey($event,id)"><GripVertical :size="17" /></button>
    <UPopover v-model="settingsOpen" :width="230" :label="`${title}布局`">
      <template #trigger="{toggle}">
        <button type="button" class="layout-settings" :aria-label="`调整${title}布局`" :aria-expanded="settingsOpen" title="排序与大小" @click="toggle"><SlidersHorizontal :size="14" /></button>
      </template>
      <div class="layout-options">
        <div class="move-options">
          <button type="button" :disabled="index===0" @click="controls.move(id,-1)">前移</button>
          <button type="button" :disabled="index===controls.cards.value.length-1" @click="controls.move(id,1)">后移</button>
        </div>
        <label>宽度 <select :aria-label="`${title}宽度`" :value="card.span" @change="controls.size(id,Number(($event.target as HTMLSelectElement).value),card.height)"><option :value="1">半宽</option><option :value="2">全宽</option></select></label>
        <label>高度 <input type="number" :aria-label="`${title}高度`" :min="minHeight" :max="maxHeight" step="20" :value="card.height" @change="controls.size(id,card.span,Number(($event.target as HTMLInputElement).value))" /></label>
        <small>高度 {{ minHeight }}–{{ maxHeight }} px。窄屏自动单列。</small>
      </div>
    </UPopover>
  </div>
</template>
<style scoped>
.live-card-tools { display: flex; gap: 2px; align-items: center; flex: 0 0 auto; }
button { display: flex; align-items: center; justify-content: center; width: 26px; height: 28px; border-radius: var(--radius-sm); color: var(--secondary); background: transparent; }
button:hover { color: var(--text); background: var(--hover); }
.drag-handle { cursor: grab; touch-action: none; }
.drag-handle:active { cursor: grabbing; }
.layout-options { display: flex; flex-direction: column; gap: 10px; font-size: var(--font-xs); }
.move-options { display: flex; gap: 6px; }
.move-options button { background: var(--hover); width: auto; padding-inline: 12px; }
button:disabled { opacity: .4; }
label { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
select,input { width: 100px; min-width: 0; color: var(--text); background: var(--surface); border: 1px solid var(--separator); border-radius: var(--radius-sm); padding: 4px; }
small { color: var(--secondary); line-height: 1.5; }
</style>
