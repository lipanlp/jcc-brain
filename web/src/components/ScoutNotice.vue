<script setup lang="ts">
import { nextTick, onBeforeUnmount, ref, watch } from 'vue';
import { call } from '../bridge/api';
import type { ScoutNotice } from '../types';
const props = defineProps<{ notice?: ScoutNotice | null }>();
const host = ref<HTMLElement | null>(null);
const paint = () => new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
let generation = 0;
let retry: ReturnType<typeof setTimeout> | undefined;
watch(() => props.notice?.id, async id => {
  const current = ++generation;
  clearTimeout(retry);
  if (!id) return;
  await nextTick();
  await paint();
  await paint();
  async function ack() {
    const node = host.value;
    if (current !== generation || id !== props.notice?.id || !node || document.visibilityState !== 'visible') return;
    const box = node.getBoundingClientRect();
    const style = getComputedStyle(node);
    if (!box.width || !box.height || style.visibility !== 'visible' || Number(style.opacity) <= 0) return;
    try {
      if (await call<boolean>('scout_notice_shown', id, { silent: true })) return;
    } catch { /* 没有有效回执就没有自动操作。 */ }
    if (current === generation) retry = setTimeout(() => { void ack(); }, 100);
  }
  await ack();
}, { immediate: true });
onBeforeUnmount(() => { ++generation; clearTimeout(retry); });
</script>
<template>
  <div v-if="notice" ref="host" class="scout-notice" role="status" :data-notice-id="notice.id">
    <strong>{{ notice.seconds }}秒后自动侦察</strong>
    <small>移动鼠标或按键取消本次</small>
  </div>
</template>
<style scoped>
.scout-notice { position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
  display: flex; align-items: center; flex-direction: column; gap: 6px; box-sizing: border-box;
  width: max-content; max-width: calc(100% - 24px); padding: 14px 22px; border-radius: 12px;
  border: 1px solid rgba(255,255,255,.3); background: rgba(24,24,28,.9); color: #fff;
  font-size: clamp(14px, 3vw, 20px); text-align: center; pointer-events: none; }
.scout-notice small { font-size: 12px; overflow-wrap: anywhere; }
</style>
