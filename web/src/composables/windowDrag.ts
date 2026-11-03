import type { ObjectDirective } from 'vue';
import { call } from '../bridge/api';

type WindowKey = 'main' | 'hud_panel';
type Rect = { x: number; y: number; w: number; h: number };
const interactive = 'button, input, select, textarea, a, [role=button], [data-no-drag]';
const elements = new Map<HTMLElement, WindowKey>();
const cleanups = new WeakMap<HTMLElement, () => void>();
const dirty = new Set<WindowKey>();
let frame: number | undefined;

function rect(el: Element): Rect | undefined {
  const r = el.getBoundingClientRect();
  if (r.width <= 0 || r.height <= 0 || getComputedStyle(el).visibility === 'hidden') return;
  return { x: r.x, y: r.y, w: r.width, h: r.height };
}

/** Called after route DOM updates as well as by the directive's observers. */
export function refreshWindowDragRegions() {
  for (const key of elements.values()) dirty.add(key);
  schedule();
}

function schedule() {
  if (frame !== undefined) return;
  frame = requestAnimationFrame(() => {
    frame = undefined;
    for (const key of dirty) {
      const regions: Rect[] = [], exclusions: Rect[] = [];
      for (const [el, owner] of elements) {
        if (owner !== key) continue;
        const area = rect(el);
        if (!area) continue;
        regions.push(area);
        // Include the root too if the directive is ever placed on a control.
        const controls = [...el.querySelectorAll(interactive)];
        if (el.matches(interactive)) controls.push(el);
        for (const control of controls) {
          const area = rect(control);
          if (area) exclusions.push(area);
        }
      }
      void call('window_drag_regions', key, regions, exclusions, { silent: true }).catch(() => {});
    }
    dirty.clear();
  });
}

function bind(el: HTMLElement, key: WindowKey): () => void {
  elements.set(el, key);
  const changed = () => { dirty.add(key); schedule(); };
  const resize = new ResizeObserver(changed);
  let observed = new Set<Element>();
  const observe = () => {
    const next = new Set<Element>([el, ...el.querySelectorAll(interactive)]);
    for (const target of observed) if (!next.has(target)) resize.unobserve(target);
    for (const target of next) if (!observed.has(target)) resize.observe(target);
    observed = next;
  };
  observe();
  const mutations = new MutationObserver(() => { observe(); changed(); });
  mutations.observe(el, { subtree: true, childList: true, characterData: true, attributes: true });
  window.addEventListener('resize', changed);
  changed();
  return () => {
    resize.disconnect();
    mutations.disconnect();
    window.removeEventListener('resize', changed);
    elements.delete(el);
    // Also send an empty snapshot when the last region is removed.
    changed();
  };
}

export const vWindowDrag: ObjectDirective<HTMLElement, WindowKey> = {
  mounted(el, binding) { cleanups.set(el, bind(el, binding.value)); },
  updated(_el) { refreshWindowDragRegions(); },
  unmounted(el) { cleanups.get(el)?.(); cleanups.delete(el); },
};
