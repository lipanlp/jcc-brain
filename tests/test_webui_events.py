"""Web 界面通信骨架测试，不创建原生窗口。"""
from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest

from jccbrain import __version__
from jccbrain.webui.api import Api, expose
from jccbrain.webui.events import EventBus


@pytest.fixture
def bus():
    messages = []
    delivered = threading.Event()

    def evaluate(script):
        messages.append(script)
        delivered.set()

    event_bus = EventBus(evaluate)
    yield event_bus, messages, delivered
    event_bus.close()


def decode(script):
    prefix = "window.__jcc_emit("
    assert script.startswith(prefix) and script.endswith(")")
    return json.loads(script[len(prefix):-1])


@pytest.mark.parametrize("event_type", ["state", "advice", "hud"])
def test_latest_payload_is_coalesced(bus, event_type):
    event_bus, messages, delivered = bus
    for n in range(10):
        event_bus.emit(event_type, {"n": n})
    assert not messages
    assert delivered.wait(1)
    time.sleep(0.15)
    assert [decode(script) for script in messages] == [{"type": event_type, "payload": {"n": 9}}]


def test_logs_are_batched(bus):
    event_bus, messages, delivered = bus
    for n in range(3):
        event_bus.emit("log", f"日志 {n}")
    assert not messages
    assert delivered.wait(1)
    assert [decode(script) for script in messages] == [
        {"type": "log", "payload": ["日志 0", "日志 1", "日志 2"]},
    ]


def test_other_events_are_immediate(bus):
    event_bus, messages, _ = bus
    event_bus.emit("finished", {"reason": "完成"})
    assert [decode(script) for script in messages] == [{"type": "finished", "payload": {"reason": "完成"}}]


def test_events_wait_for_attach():
    event_bus = EventBus()
    messages = []
    try:
        event_bus.emit("tick", {"n": 1})
        event_bus.emit("tick", {"n": 2})
        event_bus.emit("state", {"n": 1})
        event_bus.emit("state", {"n": 2})
        event_bus.emit("log", "等待连接")
        time.sleep(0.15)
        assert not messages
        delivered = threading.Event()

        def evaluate(script):
            messages.append(script)
            if decode(script)["type"] == "log":
                delivered.set()

        event_bus.attach(evaluate)
        assert [decode(script) for script in messages[:2]] == [
            {"type": "tick", "payload": {"n": 1}},
            {"type": "tick", "payload": {"n": 2}},
        ]
        assert delivered.wait(1)
        assert [decode(script) for script in messages[2:]] == [
            {"type": "state", "payload": {"n": 2}},
            {"type": "log", "payload": ["等待连接"]},
        ]
    finally:
        event_bus.close()


def test_script_closing_tag_is_escaped(bus):
    event_bus, messages, _ = bus
    event_bus.emit("notice", {"text": "中文 </script><script>alert(1)</script>"})
    assert "</script>" not in messages[0]
    assert "中文" in messages[0]
    assert decode(messages[0])["payload"]["text"] == "中文 </script><script>alert(1)</script>"


def test_close_stops_delivery(bus):
    event_bus, messages, _ = bus
    event_bus.emit("state", {"n": 1})
    event_bus.close()
    event_bus.emit("tick", {"n": 1})
    event_bus.attach(messages.append)
    assert not messages


def test_app_info(db, bus):
    event_bus, _, _ = bus
    ctx = SimpleNamespace(db=db, settings=SimpleNamespace(actuator=SimpleNamespace(dry_run=True)))
    api = Api(ctx, event_bus)
    assert api.app_info() == {
        "ok": True,
        "data": {"version": __version__, "season": db.season.label, "dry_run": True},
    }
    assert all(callable(getattr(api, name)) for name in dir(api) if not name.startswith("_"))


def test_expose_returns_error_and_logs(caplog):
    @expose
    def broken():
        raise ValueError("操作失败：测试错误")

    assert broken() == {"ok": False, "error": "操作失败：测试错误"}
    assert "broken" in caplog.text
    assert "ValueError" in caplog.text


def test_close_does_not_block_on_destroyed_webkit():
    entered, release = threading.Event(), threading.Event()
    def evaluate(script):
        entered.set()
        release.wait(3)
    event_bus = EventBus(evaluate)
    try:
        event_bus.emit('state', {'n': 1})
        assert entered.wait(1)
        start = time.monotonic()
        event_bus.close()
        assert time.monotonic() - start < 1.5
    finally:
        release.set()
        event_bus._thread.join(1)
    assert not event_bus._thread.is_alive()


def test_log_entry_uses_machine_local_time(monkeypatch):
    from datetime import datetime
    from jccbrain.webui.events import _log_entry

    # Force a zone different from Asia/Taipei so a hardcoded zone cannot pass.
    try:
        with monkeypatch.context() as patch:
            patch.setenv('TZ', 'UTC')
            time.tzset()
            created = 1234567890.0
            assert _log_entry('本机时间', created=created)['time'] == datetime.fromtimestamp(created).strftime('%H:%M:%S')
            before = datetime.now().strftime('%H:%M:%S')
            current = _log_entry('现在')['time']
            after = datetime.now().strftime('%H:%M:%S')
            assert current in (before, after)
    finally:
        time.tzset()


def test_multi_window_broadcast_detach_and_replace(bus):
    event_bus, main, _ = bus
    overlay, panel, replacement = [], [], []
    event_bus.attach(overlay.append, key='hud')
    event_bus.attach(panel.append, key='hud_panel')
    event_bus.emit('status', {'n': 1})
    assert main == overlay == panel and len(main) == 1
    event_bus.detach('hud')
    event_bus.attach(replacement.append, key='hud_panel')
    event_bus.emit('status', {'n': 2})
    assert len(main) == 2 and len(overlay) == len(panel) == len(replacement) == 1
    assert decode(replacement[0])['payload'] == {'n': 2}


def test_failed_window_does_not_interrupt_other_windows(bus, caplog):
    event_bus, main, _ = bus
    panel = []
    def broken(_):
        raise RuntimeError('测试窗口已经关闭')
    event_bus.attach(broken, key='hud')
    event_bus.attach(panel.append, key='hud_panel')
    event_bus.emit('status', {'n': 1})
    assert len(main) == len(panel) == 1
    assert '测试窗口已经关闭' in caplog.text


def test_coalesced_events_broadcast_to_each_window(bus):
    event_bus, main, delivered = bus
    overlay = []
    event_bus.attach(overlay.append, key='hud')
    event_bus.emit('hud', {'n': 1})
    event_bus.emit('hud', {'n': 2})
    event_bus.emit('log', '批量日志')
    assert delivered.wait(1)
    time.sleep(.12)
    assert main == overlay
    assert [decode(item) for item in main] == [
        {'type': 'hud', 'payload': {'n': 2}}, {'type': 'log', 'payload': ['批量日志']},
    ]


def test_bad_payload_does_not_stop_event_thread(bus, caplog):
    event_bus, messages, delivered = bus
    event_bus.emit('hud', object())
    time.sleep(.2)
    assert not messages and event_bus._thread.is_alive()
    event_bus.emit('hud', {'mode': 'idle'})
    assert delivered.wait(1)
    assert decode(messages[-1])['payload'] == {'mode': 'idle'}
    assert '序列化界面事件 hud 失败' in caplog.text


@pytest.mark.parametrize('seconds, warns', [(0.299, False), (0.300, False), (0.301, True)])
def test_expose_warns_only_above_300ms(monkeypatch, caplog, seconds, warns):
    from jccbrain.webui import api
    ticks = iter((0.0, seconds))
    monkeypatch.setattr(api, 'perf_counter', lambda: next(ticks))

    @expose
    def measured():
        return '完成'

    assert measured() == {'ok': True, 'data': '完成'}
    assert ('界面 API measured 耗时' in caplog.text) == warns
    if warns:
        assert '301ms' in caplog.text


def test_failed_slow_api_also_reports_duration(monkeypatch, caplog):
    from jccbrain.webui import api
    ticks = iter((0.0, 0.5))
    monkeypatch.setattr(api, 'perf_counter', lambda: next(ticks))

    @expose
    def measured_failure():
        raise ValueError('测量错误路径')

    assert measured_failure() == {'ok': False, 'error': '测量错误路径'}
    assert '界面 API measured_failure 调用失败' in caplog.text
    assert '界面 API measured_failure 耗时 500ms' in caplog.text
