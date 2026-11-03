import { computed, ref } from 'vue';
import { call } from '../bridge/api';
import { useAppStore } from '../stores/app';

const watching = ref(false);

export function useWatchComps() {
    const store = useAppStore();
    const watchIds = computed(() => (store.status?.watch ?? []).map(comp => comp.id));
    const isWatched = (id: string) => watchIds.value.includes(id);
    // 目录按原始来源id选择，推荐行按算法id显示；取消推荐行只移除一个明确来源。
    const comparisonSelection = (id: string) => (store.status?.watch ?? [])
        .find(comp => (comp.canonical_id === undefined ? comp.id : comp.canonical_id) === id);
    const isCompared = (id: string) => !!comparisonSelection(id);
    async function setWatch(ids: string[]) {
        if (watching.value) return;
        watching.value = true;
        try { await call('set_watch_comps', ids); }
        catch { /* call reports errors */ }
        finally { watching.value = false; }
    }
    function toggleWatch(id: string) {
        return setWatch(isWatched(id) ? watchIds.value.filter(cid => cid!==id) : [...watchIds.value, id]);
    }
    function toggleCompared(id: string) {
        return toggleWatch(comparisonSelection(id)?.id ?? id);
    }
    return { watching, watchIds, isWatched, setWatch, toggleWatch, comparisonSelection, isCompared, toggleCompared };
}
