export type BadgeTone = 'gray' | 'blue' | 'green' | 'orange' | 'red' | 'yellow' | 'purple';
export function tierTone(tier: string | null | undefined): BadgeTone {
    return ({ S: 'red', A: 'orange', B: 'yellow', C: 'green', D: 'blue' } as const)[tier as 'S' | 'A' | 'B' | 'C' | 'D'] ?? 'gray';
}
export function costColor(cost: number | null | undefined): string {
    return cost && cost >= 1 && cost <= 5 ? `var(--cost-${cost})` : 'var(--secondary)';
}
export function targetsByCost<T extends { cost?: unknown }>(rows: readonly T[]): T[] {
    const rank = (cost: unknown) => typeof cost === 'number' && Number.isInteger(cost) && cost >= 1 && cost <= 5 ? cost : 0;
    // 稳定排序展示副本；同费和未知费用沿用上游顺序。
    return [...rows].sort((a, b) => rank(b.cost) - rank(a.cost));
}
export function decimal(value: unknown): string {
    return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(2) : '—';
}
// dataj rates already use percentage units in the Python API.
export function rate(value: unknown): string {
    return typeof value === 'number' && Number.isFinite(value) ? `${value.toFixed(1)}%` : '—';
}
export function count(value: unknown): string {
    return typeof value === 'number' && Number.isFinite(value) ? value.toLocaleString('zh-CN') : '—';
}
export async function copyText(text: string): Promise<void> {
    try {
        if (!navigator.clipboard?.writeText)
            throw new Error('clipboard unavailable');
        await navigator.clipboard.writeText(text);
    }
    catch {
        const previous = document.activeElement as HTMLElement | null;
        const input = document.createElement('textarea');
        input.value = text;
        input.className = 'clipboard-copy selectable';
        document.body.append(input);
        input.select();
        try {
            if (!document.execCommand('copy'))
                throw new Error('无法写入剪贴板');
        }
        finally {
            input.remove();
            previous?.focus();
        }
    }
}
