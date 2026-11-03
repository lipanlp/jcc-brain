import { nextTick, onBeforeUnmount, watch, type Ref } from 'vue';
const stack: symbol[] = [];
const focusable = 'button:not(:disabled), [href], input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex]:not([tabindex="-1"])';
export function useOverlay(open: Ref<boolean>, panel: Ref<HTMLElement | undefined>, close: () => void, trap = false, autoFocus = trap) {
    const id = Symbol('overlay');
    let previous: HTMLElement | null = null;
    function remove() {
        const index = stack.indexOf(id);
        if (index >= 0)
            stack.splice(index, 1);
    }
    function keydown(event: KeyboardEvent) {
        if (!open.value || stack.at(-1) !== id)
            return;
        if (event.key === 'Escape') {
            event.preventDefault();
            event.stopImmediatePropagation();
            close();
        }
        if (event.key === 'Tab' && trap) {
            const nodes = [...(panel.value?.querySelectorAll<HTMLElement>(focusable) ?? [])].filter(node => node.getClientRects().length);
            const first = nodes[0];
            const last = nodes.at(-1);
            if (!first) {
                event.preventDefault();
                panel.value?.focus();
            }
            else if (event.shiftKey && (document.activeElement === first || !panel.value?.contains(document.activeElement))) {
                event.preventDefault();
                last?.focus();
            }
            else if (!event.shiftKey && (document.activeElement === last || !panel.value?.contains(document.activeElement))) {
                event.preventDefault();
                first.focus();
            }
        }
    }
    function containFocus(event: FocusEvent) {
        if (trap && open.value && stack.at(-1) === id && !panel.value?.contains(event.target as Node))
            (panel.value?.querySelector<HTMLElement>(focusable) ?? panel.value)?.focus();
    }
    document.addEventListener('keydown', keydown, true);
    document.addEventListener('focusin', containFocus);
    watch(open, async (value) => {
        remove();
        if (value) {
            previous = document.activeElement as HTMLElement;
            stack.push(id);
            await nextTick();
            if (autoFocus && open.value)
                (panel.value?.querySelector<HTMLElement>(focusable) ?? panel.value)?.focus();
        }
        else if (previous?.isConnected)
            previous.focus();
    }, { immediate: true, flush: 'sync' });
    onBeforeUnmount(() => {
        remove();
        document.removeEventListener('keydown', keydown, true);
        document.removeEventListener('focusin', containFocus);
        if (open.value && previous?.isConnected)
            previous.focus();
    });
}
