import { computed, onBeforeUnmount, ref, shallowRef, watch } from 'vue';
import type { useAppStore } from '../stores/app';
import type { PopulationPreview, Unit } from '../types';
import { call } from '../bridge/api';

type PreviewRequest = { key: string; level: number; target: string; season: string; revision?: string };
const unitsKey = (units: Unit[] | undefined) => (units ?? []).filter(unit => unit.known)
    .map(unit => [unit.name, unit.star] as const).sort((a, b) => a[0].localeCompare(b[0]) || a[1] - b[1]);

export function usePopulationPreview(store: ReturnType<typeof useAppStore>) {
    const population = ref('auto');
    const preview = shallowRef<PopulationPreview | null>(null);
    const loading = ref(false);
    const error = ref('');
    const manual = computed(() => population.value !== 'auto');
    const valid = computed(() => !!store.advice?.target && !!store.state && store.state.phase !== 'ended'
        && store.state.phase === store.advice.phase && store.state.stage === store.advice.stage
        && (store.state.stage != null || store.state.level != null || store.state.gold != null));
    const identity = computed(() => manual.value && valid.value && store.status && store.advice?.target
        ? JSON.stringify([population.value, store.advice.target.id, store.status.season.id, store.advice.selection?.revision]) : null);
    const dataKey = computed(() => identity.value ? JSON.stringify([
        // 只比较站位/持有量/目标/卡池相关输入；计时器、金币等每帧变化不触发预览。
        unitsKey(store.state?.board), unitsKey(store.state?.bench),
        (store.state?.item_bench ?? []).map(item => [item.name, item.kind]).sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b))),
        store.advice?.board, store.advice?.targets, store.advice?.opponents,
        store.status?.game_version, store.status?.sources,
    ]) : null);
    let generation = 0;
    let stopped = false;
    let inFlight = false;
    let desired: PreviewRequest | null = null;
    let completedKey: string | null = null;

    async function refresh() {
        if (stopped || inFlight || !desired || desired.key === completedKey) return;
        const request = desired;
        const epoch = generation;
        inFlight = true;
        loading.value = true;
        error.value = '';
        try {
            const result = await call<PopulationPreview | null>('population_preview', request.level, request.target, request.season, request.revision ?? null, { silent: true });
            if (stopped || epoch !== generation) return;
            // 同一上下文可先呈现这次完整快照，再合并刷新最新输入，避免快更新下永远空白。
            if (result === null) preview.value = null;
            else if (result.population === request.level && result.target_id === request.target && result.season_id === request.season
                && (!result.selection_revision || result.selection_revision === request.revision))
                preview.value = result;
        }
        catch (problem) {
            if (!stopped && epoch === generation) error.value = problem instanceof Error ? problem.message : '预览加载失败';
        }
        finally {
            inFlight = false;
            if (epoch === generation) completedKey = request.key;
            loading.value = false;
            // 至多一个在途请求和一份最新期望；中间帧不排队。
            if (!stopped) void refresh();
        }
    }
    watch([identity, dataKey], ([nextIdentity, key], [previousIdentity]) => {
        if (nextIdentity !== previousIdentity) {
            generation++;
            preview.value = null;
            error.value = '';
            completedKey = null;
        }
        const target = store.advice?.target?.id;
        const season = store.status?.season.id;
        desired = nextIdentity && key && target && season
            ? { key, level: Number(population.value), target, season, revision: store.advice?.selection?.revision } : null;
        loading.value = !!desired && (inFlight || desired.key !== completedKey);
        void refresh();
    }, { immediate: true, flush: 'sync' });
    onBeforeUnmount(() => { stopped = true; generation++; desired = null; });
    return { population, manual, loading, error,
        board: computed(() => valid.value ? (manual.value ? preview.value?.board : store.advice?.board) : null),
        targets: computed(() => valid.value ? (manual.value ? preview.value?.targets ?? [] : store.advice?.targets ?? []) : []),
    };
}
