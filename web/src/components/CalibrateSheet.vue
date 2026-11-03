<script setup lang="ts">
import { computed, onBeforeUnmount, ref, shallowRef, watch } from 'vue';
import { calibrationOpen } from '../composables/tools';
import { call } from '../bridge/api';
import { toast } from '../composables/toast';
import type { CalibrationPoint } from '../types';
import USheet from './ui/USheet.vue';
import UButton from './ui/UButton.vue';
const points = shallowRef<CalibrationPoint[]>([]);
const got = ref<Record<string, [
    number,
    number
]>>({});
const running = ref(false), saving = ref(false);
const error = ref(''), elapsed = ref(0);
const current = computed(() => points.value[Object.keys(got.value).length]);
const count = computed(() => Math.max(0, 3 - Math.floor(elapsed.value / 1000)));
const complete = computed(() => points.value.length > 0 && Object.keys(got.value).length === points.value.length);
const circumference = 2 * Math.PI * 28;
let generation = 0, frame = 0;
let cancelCountdown: (() => void) | undefined;
function stop() {
    generation++;
    cancelCountdown?.();
    cancelCountdown = undefined;
    cancelAnimationFrame(frame);
    running.value = false;
}
watch(calibrationOpen, async (open) => {
    stop();
    if (!open)
        return;
    const ticket = generation;
    got.value = {};
    error.value = '';
    elapsed.value = 0;
    points.value = [];
    try {
        const value = await call<CalibrationPoint[]>('calibrate_points');
        if (ticket === generation)
            points.value = value;
    }
    catch { /* call reports */
    }
});
function countdown(ticket: number): Promise<boolean> {
    const start = performance.now();
    elapsed.value = 0;
    return new Promise(resolve => {
        cancelCountdown = () => resolve(false);
        const tick = () => {
            if (ticket !== generation) {
                resolve(false);
                return;
            }
            elapsed.value = Math.min(3000, performance.now() - start);
            if (elapsed.value >= 3000) {
                cancelCountdown = undefined;
                resolve(true);
            }
            else
                frame = requestAnimationFrame(tick);
        };
        frame = requestAnimationFrame(tick);
    });
}
async function start() {
    stop();
    const ticket = generation;
    got.value = {};
    error.value = '';
    running.value = true;
    try {
        for (const point of points.value) {
            if (!await countdown(ticket))
                return;
            const value = await call<{
                x: number;
                y: number;
            }>('calibrate_capture');
            if (ticket !== generation)
                return;
            got.value[point.key] = [value.x, value.y];
        }
    }
    catch (reason) {
        if (ticket === generation)
            error.value = reason instanceof Error ? reason.message : String(reason);
    }
    finally {
        if (ticket === generation)
            running.value = false;
    }
}
function close() {
    if (saving.value)
        return;
    stop();
    calibrationOpen.value = false;
}
async function save() {
    if (!complete.value)
        return;
    saving.value = true;
    try {
        await call('calibrate_save', JSON.parse(JSON.stringify(got.value)));
        toast.success('已保存到 layout.json');
        calibrationOpen.value = false;
    }
    catch { /* call reports */
    }
    finally {
        saving.value = false;
    }
}
onBeforeUnmount(stop);
</script>
<template>
  <USheet :model-value="calibrationOpen" title="坐标标定" :width="600" @update:model-value="close">
    <p class="calibration-instructions">
      点「开始」后，按提示在 3 秒内把鼠标移到游戏中对应位置的中心（不要点击）。
      <br />
      棋盘格看不清时，可以先拖起一个棋子让格子高亮。
    </p>
    <div class="calibration-prompt">
      <div v-if="running" class="calibration-ring">
        <svg viewBox="0 0 68 68" aria-hidden="true">
          <circle class="ring-track" cx="34" cy="34" r="28" />
          <circle class="ring-progress" cx="34" cy="34" r="28" :stroke-dasharray="circumference" :stroke-dashoffset="circumference*elapsed/3000" />
        </svg>
        <strong>{{ count }}</strong>
      </div>
      <div>
        <h2>{{ complete ? '完成，点「保存」写入布局' : current?.label || '准备标定' }}</h2>
        <p>{{ running ? '把鼠标移到对应位置，保持不动' : `共 ${points.length} 个点位，每个点位倒计时 3 秒` }}</p>
      </div>
    </div>
    <p v-if="error" class="calibration-error">{{ error }} · 可以重新开始</p>
    <div class="calibration-results">
      <div v-for="point in points" :key="point.key">
        <span>{{ point.label }}</span>
        <span class="numeric">{{ got[point.key] ? `(${got[point.key]![0].toFixed(3)}, ${got[point.key]![1].toFixed(3)})` : '—' }}</span>
      </div>
    </div>
    <template #footer>
      <UButton :disabled="saving" @click="close">取消</UButton>
      <UButton :disabled="running||saving||!points.length" @click="start">{{ Object.keys(got).length||error ? '重新开始' : '开始' }}</UButton>
      <UButton variant="primary" :disabled="!complete||running" :loading="saving" @click="save">保存</UButton>
    </template>
  </USheet>
</template>
<style scoped>
.calibration-instructions {
  font-size: var(--font-sm);
  line-height: 1.8;
  color: var(--secondary);
}
.calibration-prompt {
  display: flex;
  gap: var(--space-4);
  align-items: center;
  min-height: 110px;
  padding: var(--space-4) 0;
}
.calibration-prompt h2 {
  font-size: var(--font-xl);
  font-weight: 600;
}
.calibration-prompt p {
  font-size: var(--font-xs);
  color: var(--secondary);
  margin-top: var(--space-1);
}
.calibration-ring {
  width: 68px;
  height: 68px;
  position: relative;
  flex-shrink: 0;
}
.calibration-ring svg {
  width: 100%;
  height: 100%;
  transform: rotate(-90deg);
}
.calibration-ring circle {
  fill: none;
  stroke-width: 4;
}
.ring-track {
  stroke: var(--accent-soft);
}
.ring-progress {
  stroke: var(--accent);
  stroke-linecap: round;
}
.calibration-ring strong {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  font-size: var(--font-xl);
  font-variant-numeric: tabular-nums;
}
.calibration-error {
  font-size: var(--font-sm);
  color: var(--red-text);
  margin-bottom: var(--space-3);
}
.calibration-results {
  font-size: var(--font-sm);
}
.calibration-results>div {
  display: flex;
  justify-content: space-between;
  gap: var(--space-4);
  padding: var(--space-2) 0;
  border-bottom: 1px solid var(--separator);
}
.calibration-results .numeric {
  color: var(--secondary);
}
@media (prefers-reduced-motion:reduce) {
  .ring-progress {
    visibility: hidden;
  }
}
</style>
