import { toast } from '../composables/toast';
type ApiResult<T> = {
    ok: true;
    data: T;
} | {
    ok: false;
    error: string;
};
type ApiMethod = (...args: unknown[]) => Promise<ApiResult<unknown>>;
export interface CallOptions {
    silent: boolean;
}
declare global {
    interface Window {
        pywebview?: {
            api?: Record<string, ApiMethod>;
        };
    }
}
let ready: Promise<void> | undefined;
let cancelReady: (() => void) | undefined;
function hasBridge() {
    return !!window.pywebview?.api && Object.keys(window.pywebview.api).length > 0;
}
export function waitReady(): Promise<void> {
    if (hasBridge())
        return Promise.resolve();
    return ready ??= new Promise((resolve, reject) => {
        let settled = false;
        function finish(error?: Error) {
            if (settled) return;
            settled = true;
            clearTimeout(timer);
            window.removeEventListener('pywebviewready', onReady);
            ready = undefined;
            cancelReady = undefined;
            if (error) reject(error);
            else resolve();
        }
        function onReady() {
            if (hasBridge()) finish();
        }
        const timer = setTimeout(() => finish(new Error('连接本地服务超时，请重试')), 5000);
        cancelReady = () => finish(new Error('本地服务连接已取消'));
        window.addEventListener('pywebviewready', onReady);
    });
}
export function cancelWaiting() { cancelReady?.(); }
window.addEventListener('pagehide', cancelWaiting);
if (import.meta.hot) import.meta.hot.dispose(() => {
    cancelWaiting();
    window.removeEventListener('pagehide', cancelWaiting);
});
/** A trailing {silent: true} controls error presentation; all other arguments pass through. */
export async function call<T>(name: string, ...args: unknown[]): Promise<T> {
    const last = args.at(-1);
    const options = last && typeof last === 'object' && Object.keys(last).length === 1 && 'silent' in last && typeof last.silent === 'boolean' ? args.pop() as CallOptions : undefined;
    try {
        await waitReady();
        const method = window.pywebview?.api?.[name];
        if (!method)
            throw new Error(`后端方法不存在：${name}`);
        const result = await method(...args) as ApiResult<T>;
        if (!result.ok)
            throw new Error(result.error);
        return result.data;
    }
    catch (error) {
        if (!options?.silent)
            toast.error(error instanceof Error ? error.message : String(error));
        throw error;
    }
}
