<script setup lang="ts">
import { useAppStore } from '../stores/app';
import UButton from './ui/UButton.vue';
import USpinner from './ui/USpinner.vue';
const store = useAppStore();
</script>
<template>
  <section v-if="!store.connected" class="startup-notice" role="status" aria-live="polite" aria-label="启动状态">
    <USpinner v-if="store.startupPhase!=='failed'" :size="14" />
    <div>
      <p v-if="store.startupPhase==='failed'">暂时无法读取本地数据，请重试。</p>
      <p v-else-if="store.startupSlow">准备时间较长，请稍候…</p>
      <p v-else>{{ store.startupPhase==='retrying' ? '正在重新读取本地数据…' : '正在准备本地数据…' }}</p>
      <small>你仍可切换页面或打开设置。</small>
    </div>
    <UButton v-if="store.startupPhase==='failed'" size="sm" @click="store.retryStartup">重试</UButton>
  </section>
</template>
<style scoped>
.startup-notice { display:flex;align-items:center;gap:var(--space-3);padding:var(--space-2) var(--space-5);border-bottom:1px solid var(--separator);background:var(--surface);flex-shrink:0;min-width:0; }
.startup-notice>div { flex:1;min-width:0;overflow-wrap:anywhere; }
p { font-size:var(--font-sm);line-height:1.6; }
small { color:var(--secondary);font-size:var(--font-xs);line-height:1.6; }
</style>
