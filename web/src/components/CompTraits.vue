<script setup lang="ts">
import { ref, watch } from 'vue';
import type { CompTrait } from '../types';
import { iconImages, requestIcons } from '../composables/icons';
import UPopover from './ui/UPopover.vue';
const props = defineProps<{ traits: CompTrait[] }>();
const opened = ref('');
const failed = ref<Record<string, boolean>>({});
watch(() => props.traits, values => { opened.value=''; requestIcons(values.map(t => t.picture_key)); }, { immediate: true });
</script>
<template>
  <div class="comp-traits-view">
    <div class="trait-chips">
      <UPopover v-for="trait in traits" :key="trait.name" :model-value="opened===trait.name" :width="340" :label="`${trait.name}静态羁绊详情`" @update:model-value="opened=$event ? trait.name : ''">
        <template #trigger>
          <button type="button" class="static-trait" :class="{reached:trait.tier && trait.tier>0}" :aria-label="`${trait.name} ${trait.count}名英雄`" @click="opened=opened===trait.name ? '' : trait.name">
            <img v-if="iconImages[trait.picture_key] && !failed[trait.name]" :src="iconImages[trait.picture_key]!" alt="" @error="failed[trait.name]=true" />
            <span v-else class="trait-symbol">{{ trait.name.slice(0,1) }}</span>
            <span>{{ trait.name }} <b>{{ trait.count }}</b><small>{{ trait.activation==='special' ? '特殊条件' : trait.breakpoints.join(' / ') || '档位未知' }}</small></span>
          </button>
        </template>
        <div class="trait-copy">
          <h3>{{ trait.name }} · {{ trait.count }} 名英雄</h3>
          <p v-if="trait.activation==='special'" class="trait-note">特殊条件触发，不能仅凭人数判断激活。</p>
          <p>{{ trait.description || '来源未提供羁绊描述' }}</p>
          <ul v-if="trait.levels.length">
            <li v-for="(level,index) in trait.levels" :key="index"><b>{{ level.count==null ? '特殊条件' : `(${level.count})` }}</b> {{ level.text }}</li>
          </ul>
          <ul v-else><li v-for="(text,index) in trait.level_desc" :key="index">{{ text }}</li></ul>
          <p v-if="trait.tier && trait.tier>0" class="trait-note">英雄基础人数达到第 {{ trait.tier }} 档。</p>
        </div>
      </UPopover>
      <span v-if="!traits.length" class="trait-note">来源未提供可统计的英雄羁绊</span>
    </div>
    <p class="trait-note">按不同英雄的基础羁绊统计；转职、加成与其他触发条件以来源为准。</p>
  </div>
</template>
<style scoped>
.trait-chips { display: flex; flex-wrap: wrap; gap: var(--space-2); }
.static-trait { display: flex; align-items: center; gap: var(--space-2); padding: var(--space-1) var(--space-2); background: var(--hover); border: 1px solid var(--separator); border-radius: var(--radius-sm); text-align: left; font-size: var(--font-sm); }
.static-trait.reached { border-color: var(--accent); }
.static-trait img,.trait-symbol { width: 26px; height: 26px; object-fit: contain; background: var(--tooltip); border-radius: var(--radius-pill); padding: var(--space-inset); }
.trait-symbol { display: grid; place-items: center; color: var(--tooltip-text); }
small { display: block; color: var(--secondary); font-size: var(--font-xs); margin-top: var(--space-half); }
b { margin-left: var(--space-1); }
.trait-note { color: var(--secondary); font-size: var(--font-xs); line-height: 1.6; margin-top: var(--space-2); }
.trait-copy h3 { font-size: var(--font-lg); }
.trait-copy p,.trait-copy li { font-size: var(--font-sm); line-height: 1.7; white-space: pre-wrap; overflow-wrap: anywhere; margin-top: var(--space-2); }
.trait-copy ul { padding-left: var(--space-4); }
</style>
