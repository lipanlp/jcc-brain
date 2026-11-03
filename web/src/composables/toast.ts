import { readonly, ref } from 'vue';
export type ToastTone = 'success' | 'error' | 'info' | 'warning';
export interface ToastItem {
    id: number;
    text: string;
    tone: ToastTone;
}
const items = ref<ToastItem[]>([]);
let sequence = 0;
const timers = new Map<number, ReturnType<typeof setTimeout>>();
export function dismissToast(id: number) {
    clearTimeout(timers.get(id));
    timers.delete(id);
    items.value = items.value.filter(item => item.id !== id);
}
function show(tone: ToastTone, text: string) {
    const id = ++sequence;
    items.value.push({ id, text, tone });
    timers.set(id, setTimeout(() => dismissToast(id), tone === 'error' ? 6000 : 3500));
    return id;
}
export const toast = {
    success: (text: string) => show('success', text), error: (text: string) => show('error', text),
    info: (text: string) => show('info', text), warning: (text: string) => show('warning', text),
};
export const toastItems = readonly(items);
