"""线程安全的 Python → JavaScript 事件推送及模式/日志适配。"""
from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from time import perf_counter
from typing import Any

from jccbrain.modes.base import ModeSink
from jccbrain.core.buy_source import buy_order
from jccbrain.core.ai_source import ai_order, AiSource, same_ai_context
from jccbrain.webui import serialize

log = logging.getLogger(__name__)


class EventBus:
    """100ms 合并状态与日志；JS 执行期间不持状态锁，关闭时避免等待已销毁的页面。"""

    def __init__(self, evaluate: Callable[[str], None] | None = None):
        self._evaluators = {'main': evaluate} if evaluate is not None else {}
        self._lock = threading.RLock()
        self._delivery_lock = threading.RLock()
        self._stopped = threading.Event()
        self._latest: dict[str, Any] = {}
        self._buy_fence = None
        self._last_buy_payload = None
        self._last_ai_payload = None
        self._logs: list[Any] = []
        self._queued: list[tuple[str, Any]] = []
        self._thread = threading.Thread(target=self._run, name="webui-events", daemon=True)
        self._thread.start()

    def attach(self, evaluate: Callable[[str], None], key: str = 'main') -> None:
        with self._lock:
            if self._stopped.is_set():
                return
            self._evaluators[key] = evaluate
            queued, self._queued = self._queued, []
            if self._last_buy_payload is not None:
                self._latest['buy_projection'] = self._last_buy_payload
            if self._last_ai_payload is not None:
                self._latest['ai_projection'] = self._last_ai_payload
        for event_type, payload in queued:
            self._send(event_type, payload)

    def detach(self, key: str) -> None:
        with self._lock:
            self._evaluators.pop(key, None)

    def emit(self, event_type: str, payload: Any) -> None:
        with self._lock:
            if self._stopped.is_set():
                return
            if not self._accept_buy_event(event_type, payload):
                return
            if event_type == 'ai_projection':
                self._last_ai_payload = payload
            if event_type == 'buy_projection' and isinstance(payload, dict) and payload.get('source'):
                self._buy_fence = payload['source']
                self._last_buy_payload = payload
            if event_type in ("state", "advice", "hud", "buy_projection", "ai_projection"):
                self._latest[event_type] = payload
                return
            if event_type == "log":
                self._logs.append(payload)
                return
            if not self._evaluators:
                self._queued.append((event_type, payload))
                return
        self._send(event_type, payload)

    @staticmethod
    def _event_source(kind, payload):
        if not isinstance(payload, dict):
            return None
        source = payload.get('source') if kind == 'buy_projection' else payload.get('buy_source')
        projection = payload.get('buy_projection')
        if isinstance(projection, dict) and projection.get('source'):
            nested = projection['source']
            if source is None or buy_order(nested) >= buy_order(source):
                source = nested  # fresh clear + old own state still carries a current scout toast
        return source

    def _accept_buy_event(self, kind, payload):
        if kind == 'ai_projection':
            previous = self._last_ai_payload
            return not previous or ai_order(payload['source']) >= ai_order(previous['source'])
        source = self._event_source(kind, payload)
        fence = self._buy_fence
        if source is None or fence is None:
            return True
        if (kind == 'buy_projection' and self._last_buy_payload and self._last_buy_payload.get('valid')
                and not payload.get('valid') and source.get('eligible', False)
                and buy_order(source) == buy_order(fence)):
            return False
        semantic_advice = (kind == 'advice' and fence.get('phase') in ('augment', 'carousel')
            and not source.get('eligible') and self._last_ai_payload
            and same_ai_context(payload.get('ai_source'), self._last_ai_payload['source'])
            and all(source.get(k) == fence.get(k) for k in
                    ('session', 'input_epoch', 'generation', 'selection', 'season', 'stage', 'phase')))
        if semantic_advice:
            return True  # retain its original paired snapshot, never re-label its frame
        return not (buy_order(source) < buy_order(fence)
                    or buy_order(source) == buy_order(fence) and not fence.get('eligible', False)
                    and source.get('eligible', False))

    def close(self) -> None:
        self._stopped.set()
        with self._lock:
            self._evaluators.clear()
            self._latest.clear()
            self._logs.clear()
            self._queued.clear()
            self._buy_fence = self._last_buy_payload = self._last_ai_payload = None
        if threading.current_thread() is not self._thread:
            # WebKit 关闭后 evaluate_js 可能无法返回；守护线程不能阻塞窗口退出。
            self._thread.join(timeout=1)

    def _run(self) -> None:
        while not self._stopped.wait(0.1):
            with self._lock:
                if not self._evaluators or self._stopped.is_set():
                    continue
                latest, self._latest = self._latest, {}
                logs, self._logs = self._logs, []
            # Semantic authority must arrive before a full snapshot that uses it.
            ai_payload = latest.pop("ai_projection", None)
            if ai_payload is not None:
                self._send("ai_projection", ai_payload)
            for event_type, payload in latest.items():
                self._send(event_type, payload)
            if logs:
                self._send("log", logs)

    def _send(self, event_type: str, payload: Any) -> None:
        with self._delivery_lock:
            with self._lock:
                if self._stopped.is_set():
                    return
                keys = list(self._evaluators)
            try:
                message = json.dumps({"type": event_type, "payload": payload}, ensure_ascii=False).replace("</", "<\\/")
            except (TypeError, ValueError):
                log.exception("序列化界面事件 %s 失败", event_type)
                return
            for key in keys:
                with self._lock:
                    evaluate = self._evaluators.get(key)
                    if not self._accept_buy_event(event_type, payload):
                        return
                    if self._stopped.is_set():
                        return
                if evaluate is None:
                    continue
                started = perf_counter()
                try:
                    evaluate(f"window.__jcc_emit({message})")
                except Exception:
                    log.exception("推送界面事件 %s 到 %s 失败", event_type, key)
                finally:
                    log.debug('界面事件 %s → %s evaluate_js 耗时 %.1fms',
                              event_type, key, (perf_counter() - started) * 1000)


def _log_entry(text: str, level: str = 'info', source: str = 'app', created: float | None = None) -> dict:
    from datetime import datetime
    timestamp = datetime.now() if created is None else datetime.fromtimestamp(created)
    return {'time': timestamp.strftime('%H:%M:%S'), 'text': text, 'level': level, 'source': source}


class WebSink(ModeSink):
    def __init__(self, bus, db_getter, *, on_state=None, on_advice=None, on_finished=None,
                 status_getter=None, hud_callback=None, hud_visible=None, on_buy_projection=None, on_ai_projection=None, toast_capable=None):
        self._bus = bus
        self._db_getter = db_getter
        self._state_callback = on_state
        self._advice_callback = on_advice
        self._buy_callback = on_buy_projection
        self._ai_callback = on_ai_projection
        self._hud_callback = hud_callback
        self._finished_callback = on_finished
        self._status_getter = status_getter
        self._hud_visible = hud_visible
        self._toast_capable = toast_capable

    def on_state(self, state) -> None:
        if self._state_callback and self._state_callback(state) is False:
            return
        self._bus.emit('state', serialize.state(state, self._db_getter()))
        if self._hud_callback:
            self._hud_callback()

    def on_advice(self, advice) -> None:
        result = self._advice_callback(advice) if self._advice_callback else None
        if result is False:
            return
        self._bus.emit('advice', serialize.advice_with_ai(advice, result if isinstance(result, AiSource) else None))
        if self._hud_callback:
            self._hud_callback()

    def on_ai_projection(self, projection) -> None:
        if self._ai_callback and self._ai_callback(projection) is False:
            return
        self._bus.emit('ai_projection', serialize.ai_projection(projection))

    def on_buy_projection(self, projection) -> None:
        if self._buy_callback and self._buy_callback(projection) is False:
            return
        self._bus.emit('buy_projection', serialize.buy_projection(projection))
        if self._hud_callback and projection.valid:
            self._hud_callback()

    def on_lock_changed(self) -> None:
        if self._status_getter:
            self._bus.emit('status', self._status_getter())

    def on_watch_changed(self) -> None:
        if self._status_getter:
            status = self._status_getter()
            self._bus.emit('watch_changed', {'watch': status['watch']})
            self._bus.emit('status', status)

    def on_action(self, action, real: bool) -> None:
        text = action.describe()
        self._bus.emit('action', {'text': text, 'real': real})
        self.on_log(('▶ ' if real else '▷ [演练] ') + text)

    def on_scout_changed(self) -> None:
        if self._status_getter:
            self._bus.emit('status', self._status_getter())
        if self._hud_callback:
            self._hud_callback()

    def scout_toast_capable(self) -> bool:
        return bool(self._toast_capable and self._toast_capable())

    def scout_hud_visible(self, window=None) -> bool:
        return bool(self._hud_visible and self._hud_visible(window))

    def on_augments_changed(self) -> None:
        if self._status_getter:
            self._bus.emit('status', self._status_getter())

    def on_mode_changed(self) -> None:
        if self._status_getter:
            self._bus.emit('status', self._status_getter())

    def on_log(self, text: str) -> None:
        self._bus.emit('log', _log_entry(text, source='mode'))

    def on_finished(self, mode: str, reason: str) -> None:
        if self._finished_callback:
            self._finished_callback(mode, reason)
        self.on_log(f'[{mode}] 结束：{reason}')
        self._bus.emit('mode_finished', {'mode': mode, 'reason': reason})
        self._bus.emit('status', self._status_getter() if self._status_getter else {})


class WebLogHandler(logging.Handler):
    def __init__(self, bus):
        super().__init__(logging.INFO)
        self._bus = bus

    def emit(self, record: logging.LogRecord) -> None:
        # 推送失败本身不能再回流到同一条推送链。
        if record.levelno < logging.INFO or record.name == __name__:
            return
        level = 'error' if record.levelno >= logging.ERROR else 'warning' if record.levelno >= logging.WARNING else 'info'
        try:
            text = f'{record.name}: {record.getMessage()}'
            self._bus.emit('log', _log_entry(text, level, 'logger', record.created))
        except Exception:
            self.handleError(record)
