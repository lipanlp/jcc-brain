export interface BridgeEvent {
    type: string;
    payload: unknown;
}
type Handler = (payload: unknown) => void;
const listeners = new Map<string, Set<Handler>>();
export const bus = {
    on(type: string, handler: Handler): () => void {
        let handlers = listeners.get(type);
        if (!handlers) {
            handlers = new Set();
            listeners.set(type, handlers);
        }
        handlers.add(handler);
        return () => {
            handlers.delete(handler);
            if (handlers.size === 0)
                listeners.delete(type);
        };
    },
    emit(type: string, payload: unknown): void {
        for (const handler of [...(listeners.get(type) ?? [])]) {
            handler(payload);
        }
    },
};
