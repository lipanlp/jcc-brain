<script setup lang="ts">
import type { DatabaseKind, DatabaseEntry } from '../types';
import { costColor, tierTone } from '../composables/presentation';
import UBadge from './ui/UBadge.vue';
const props = defineProps<{
    kind: DatabaseKind;
    entry: DatabaseEntry;
}>();
function roundsText() {
    return props.entry.rounds?.map(([a, b]) => `${a}~${b ?? '结束'}`).join('、') || '—';
}
</script>
<template>
  <article class="inspector-content database-inspector" :data-entry-name="entry.name">
    <h1>{{ entry.name }}</h1>
    <template v-if="kind==='heroes'">
      <div class="inspector-chips entry-badges">
        <UBadge :style="{color:costColor(entry.cost)}">{{ entry.cost ?? '—' }} 费</UBadge>
        <UBadge v-for="name in entry.traits" :key="name" tone="blue">{{ name }}</UBadge>
      </div>
      <p class="entry-summary inspector-muted">射程 · {{ entry.attack_range ?? '—' }}</p>
      <section>
        <h2>{{ entry.skill_name || '技能' }}</h2>
        <p class="inspector-prose">{{ entry.skill_desc || '暂无技能描述' }}</p>
      </section>
    </template>
    <template v-else-if="kind==='traits'">
      <div class="entry-badges">
        <UBadge>{{ entry.kind }}</UBadge>
      </div>
      <section v-if="entry.levels?.length">
        <h2>羁绊效果</h2>
        <div v-for="(level,index) in entry.levels" :key="`${level.count}#${index}`" class="trait-level">
          <UBadge tone="blue">{{ level.count }}</UBadge>
          <div>
            <p class="inspector-prose">{{ level.text || '—' }}</p>
            <p v-if="level.values.length" class="trait-values numeric">数值 · {{ level.values.join(' / ') }}</p>
          </div>
        </div>
      </section>
      <section v-else>
        <h2>效果</h2>
        <p class="inspector-prose">{{ entry.description || entry.level_desc?.join('\n') || '暂无描述' }}</p>
      </section>
    </template>
    <template v-else-if="kind==='items'">
      <div class="entry-badges">
        <UBadge>{{ entry.kind || '装备' }}</UBadge>
      </div>
      <section v-if="entry.recipe?.length">
        <h2>合成公式</h2>
        <div class="inspector-chips">
          <template v-for="(name,index) in entry.recipe" :key="`${name}#${index}`">
            <span v-if="index" class="inspector-muted">＋</span>
            <UBadge>{{ name }}</UBadge>
          </template>
        </div>
      </section>
      <section v-if="entry.trait_name">
        <h2>关联羁绊</h2>
        <UBadge tone="blue">{{ entry.trait_name }}</UBadge>
      </section>
      <section>
        <h2>描述</h2>
        <p class="inspector-prose">{{ entry.description || '暂无描述' }}</p>
      </section>
    </template>
    <template v-else-if="kind==='augments'">
      <div class="entry-badges">
        <UBadge :tone="entry.tier===1 ? 'gray' : entry.tier===2 ? 'yellow' : entry.tier===3 ? 'purple' : 'gray'">{{ entry.tier_label || '—' }}</UBadge>
      </div>
      <section>
        <h2>强化效果</h2>
        <p class="inspector-prose">{{ entry.description || '暂无描述' }}</p>
      </section>
    </template>
    <template v-else-if="kind==='mechanics'">
      <div class="entry-badges">
        <UBadge>{{ entry.category || '赛季机制' }}</UBadge>
      </div>
      <dl class="entry-facts">
        <div>
          <dt>价格</dt>
          <dd class="numeric">{{ entry.price ?? '—' }}</dd>
        </div>
        <div>
          <dt>出现回合</dt>
          <dd class="numeric">{{ roundsText() }}</dd>
        </div>
      </dl>
      <section>
        <h2>描述</h2>
        <p class="inspector-prose">{{ entry.description || '暂无描述' }}</p>
      </section>
      <section v-if="entry.upgrades?.length">
        <h2>升级</h2>
        <ol class="upgrade-list">
          <li v-for="(name,index) in entry.upgrades" :key="`${name}#${index}`">
            <span v-if="index" class="upgrade-arrow">→</span>
            <UBadge>{{ name }}</UBadge>
          </li>
        </ol>
      </section>
    </template>
    <template v-else-if="kind==='encounters'">
      <section>
        <h2>奇遇效果</h2>
        <p class="inspector-prose">{{ entry.description || '暂无描述' }}</p>
      </section>
    </template>
    <template v-else-if="kind==='guides'">
      <div class="entry-badges">
        <UBadge :tone="tierTone(entry.quality)">{{ entry.quality || '—' }}</UBadge>
      </div>
      <section>
        <h2>成型阵容</h2>
        <div class="inspector-chips">
          <UBadge v-for="(name,index) in entry.heroes" :key="`${name}#${index}`">{{ name }}</UBadge>
        </div>
      </section>
      <section>
        <h2>追三星</h2>
        <div class="inspector-chips">
          <UBadge v-for="(name,index) in entry.three_star" :key="`${name}#${index}`" tone="yellow">{{ name }}</UBadge>
          <span v-if="!entry.three_star?.length" class="inspector-muted">无</span>
        </div>
      </section>
      <section>
        <h2>攻略</h2>
        <p class="inspector-prose">{{ entry.text || '暂无攻略正文' }}</p>
      </section>
    </template>
  </article>
</template>
<style scoped>
.entry-badges {
  margin-top: var(--space-3);
}
.entry-summary {
  margin-top: var(--space-4);
}
.trait-level {
  display: flex;
  align-items: flex-start;
  gap: var(--space-3);
}
.trait-level+.trait-level {
  margin-top: var(--space-4);
}
.trait-level>.u-badge {
  min-width: 28px;
  justify-content: center;
  flex-shrink: 0;
}
.trait-values {
  margin-top: var(--space-1);
  font-size: var(--font-xs);
  color: var(--secondary);
}
.entry-facts {
  font-size: var(--font-sm);
  margin: var(--space-5) 0 0;
}
.entry-facts>div {
  display: flex;
  gap: var(--space-3);
  margin-top: var(--space-2);
}
.entry-facts dt {
  color: var(--secondary);
  min-width: 60px;
}
.entry-facts dd {
  margin: 0;
}
.upgrade-list {
  list-style: none;
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2);
  padding: 0;
  margin: 0;
}
.upgrade-list li {
  display: flex;
  gap: var(--space-2);
  align-items: center;
}
.upgrade-arrow {
  color: var(--secondary);
}
</style>
