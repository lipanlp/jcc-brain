import { shallowRef } from 'vue';
import { call } from '../bridge/api';
import { bus } from '../bridge/events';

export const iconImages = shallowRef<Record<string, string | null>>({});
const active = new Map<string, number>();
const refreshAfter = new Set<string>();
let season: string | undefined;
let generation = 0;

export function requestIcons(keys: string[], refresh = false) {
    const version = generation;
    const wanted = [...new Set(keys)].filter(key => {
        if (active.has(key)) {
            if (refresh) refreshAfter.add(key);
            return false;
        }
        return refresh || !(key in iconImages.value);
    });
    if (!wanted.length) return;
    wanted.forEach(key => active.set(key, version));
    void call<Record<string, string | null>>('icons', wanted, { silent: true }).then(values => {
        if (version===generation) iconImages.value = { ...iconImages.value, ...values };
    }).catch(() => {}).finally(() => {
        wanted.forEach(key => { if (active.get(key)===version) active.delete(key); });
        if (version!==generation) return;
        const again = wanted.filter(key => refreshAfter.delete(key));
        requestIcons(again, true);
    });
}
const unsubscribe = bus.on('icons_ready', keys => requestIcons(keys as string[], true));
const unsubscribeStatus = bus.on('status', payload => {
    const next = (payload as { season: { id: string } }).season.id;
    if (season && next!==season) {
        generation++;
        iconImages.value = {};
        active.clear();
        refreshAfter.clear();
    }
    season = next;
});
if (import.meta.hot) import.meta.hot.dispose(() => { unsubscribe(); unsubscribeStatus(); });
