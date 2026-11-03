import { computed } from 'vue';
import { call } from '../bridge/api';
import { useAppStore } from '../stores/app';
export function useScouting() {
    const store = useAppStore();
    const progress = computed(() => store.status?.scout);
    const busy = computed(() => !!(progress.value?.scanning || progress.value?.processing || progress.value?.pending || progress.value?.notice));
    const unavailable = computed(() => store.status?.scout_unavailable_reason ?? (store.status?.mode?.key !== 'assistant' ? '请先启动辅助大脑' : store.state?.phase !== 'planning' ? '仅在准备阶段可用' : store.state.timer === null || store.state.timer < 5 ? '倒计时至少 5 秒才能侦察' : ''));
    async function scout(stop = false) { try { await call(stop ? 'scout_stop' : 'scout_now'); } catch { /* bridge reports */ } }
    return { progress, busy, unavailable, scout };
}
