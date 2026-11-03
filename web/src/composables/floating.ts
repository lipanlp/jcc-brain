import { nextTick, onBeforeUnmount, ref, watch, type Ref } from 'vue';
export function useFloating(open: Ref<boolean>, anchor: Ref<HTMLElement | undefined>, panel: Ref<HTMLElement | undefined>, matchWidth = false) {
    const style = ref<Record<string, string>>({ visibility: 'hidden' });
    function update() {
        if (!open.value || !anchor.value || !panel.value)
            return;
        const a = anchor.value.getBoundingClientRect();
        // Entrance transforms change the visual bounds, not the final layout size.
        const p = { width: panel.value.offsetWidth, height: panel.value.offsetHeight };
        const margin = 10;
        const width = matchWidth ? Math.min(a.width, innerWidth - margin * 2) : p.width;
        const below = a.bottom + 7;
        const top = below + p.height > innerHeight - margin && a.top > p.height + margin ? a.top - p.height - 7 : Math.min(below, innerHeight - p.height - margin);
        style.value = { left: `${Math.max(margin, Math.min(a.left, innerWidth - width - margin))}px`, top: `${Math.max(margin, top)}px`, visibility: 'visible', ...(matchWidth ? { width: `${width}px` } : {}) };
    }
    const observer = new ResizeObserver(update);
    watch(open, async (value) => {
        observer.disconnect();
        if (value) {
            style.value = { visibility: 'hidden' };
            await nextTick();
            update();
            if (panel.value)
                observer.observe(panel.value);
            if (anchor.value)
                observer.observe(anchor.value);
        }
    }, { flush: 'post' });
    window.addEventListener('resize', update);
    window.addEventListener('scroll', update, true);
    onBeforeUnmount(() => {
        observer.disconnect();
        window.removeEventListener('resize', update);
        window.removeEventListener('scroll', update, true);
    });
    return style;
}
