<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import type { CompItemInfo } from '../types';
import { iconImages, requestIcons } from '../composables/icons';
const props = defineProps<{ name: string; items: Record<string, CompItemInfo> }>();
const item = computed(() => props.items[props.name]);
const failed = ref<Record<string, boolean>>({});
function icon(name: string) { return failed.value[name] ? undefined : iconImages.value[`item:${name}`] || undefined; }
watch(item, value => requestIcons([props.name, ...(value?.components.map(c => c.name) ?? [])].map(name => `item:${name}`)), { immediate: true });
</script>
<template>
  <details class="comp-item" :data-item="name">
    <summary :aria-label="`${name}装备详情`">
      <img v-if="icon(name)" :src="icon(name)" alt="" @error="failed[name]=true" />
      <span v-else class="item-fallback" aria-hidden="true">{{ name.slice(0,1) }}</span>
      <span>{{ name }}</span>
    </summary>
    <div class="item-info">
      <p>{{ item?.description || (item?.known ? '来源未提供装备描述' : '资料库暂未收录此装备') }}</p>
      <div v-if="item?.components.length" class="recipe" :aria-label="`${name}合成配方`">
        <span>配方</span>
        <template v-for="(part,index) in item.components" :key="index">
          <span v-if="index">＋</span>
          <span class="recipe-part"><img v-if="icon(part.name)" :src="icon(part.name)" alt="" @error="failed[part.name]=true" />{{ part.name }}<small v-if="!part.known">（资料缺失）</small></span>
        </template>
      </div>
      <p v-else class="item-muted">{{ item?.known ? '未提供合成配方' : '配方未知' }}</p>
    </div>
  </details>
</template>
<style scoped>
.comp-item { border: 1px solid var(--separator); border-radius: var(--radius-sm); background: var(--surface-raised); min-width: 0; font-size: var(--font-sm); }
summary { display: flex; align-items: center; gap: var(--space-2); padding: var(--space-1) var(--space-2); cursor: pointer; list-style: none; overflow-wrap: anywhere; }
summary::-webkit-details-marker { display: none; }
summary::after { content: '⌄'; margin-left: auto; color: var(--secondary); }
[open]>summary::after { content: '⌃'; }
img,.item-fallback { width: 24px; height: 24px; flex-shrink: 0; border-radius: var(--radius-sm); object-fit: cover; }
.item-fallback { display: grid; place-items: center; background: var(--hover); color: var(--secondary); }
.item-info { padding: var(--space-2); border-top: 1px solid var(--separator); }
.item-info p { white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.6; margin: 0; }
.recipe,.recipe-part { display: flex; align-items: center; flex-wrap: wrap; gap: var(--space-1); }
.recipe { margin-top: var(--space-2); }
.item-muted,small { color: var(--secondary); font-size: var(--font-xs); }
</style>
