import { shallowRef } from 'vue';
export interface ConfirmOptions {
    title: string;
    message: string;
    confirmText?: string;
    cancelText?: string;
    destructive?: boolean;
    alertOnly?: boolean;
}
interface Request extends ConfirmOptions {
    resolve: (value: boolean) => void;
}
export const activeConfirm = shallowRef<Request | null>(null);
const queue: Request[] = [];
export function confirm(options: ConfirmOptions): Promise<boolean> {
    return new Promise(resolve => {
        queue.push({ ...options, resolve });
        if (!activeConfirm.value)
            activeConfirm.value = queue.shift()!;
    });
}
export function settleConfirm(value: boolean) {
    const current = activeConfirm.value;
    activeConfirm.value = queue.shift() ?? null;
    current?.resolve(value);
}
export async function alert(options: {
    title: string;
    message: string;
}): Promise<void> {
    await confirm({ ...options, confirmText: "好", alertOnly: true });
}
