import json

import pytest

from scripts import request_capture as client


@pytest.mark.parametrize('ok', [True, False])
def test_returns_only_matching_complete_result_without_starting_daemon(tmp_path, ok):
    wanted = dict(id='wanted', ok=ok, file='image.png' if ok else None, backend='cli', error=None if ok else '失败')
    (tmp_path / 'capture-results.jsonl').write_text(json.dumps({'id': 'another', 'ok': True}) + '\n' + json.dumps(wanted) + '\n{"id":')
    assert client.request_capture(tmp_path, 'heroes', '2-3', '面板确认', request_id='wanted') == wanted
    request = json.loads((tmp_path / 'capture-requests.jsonl').read_text())
    assert request == dict(id='wanted', scene='heroes', stage='2-3', notes='面板确认')


def test_timeout_never_resends_and_keeps_id_for_later_lookup(tmp_path, monkeypatch):
    clock = iter([0., 0., 10.])
    monkeypatch.setattr(client.time, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(client.time, 'sleep', lambda _: None)
    (tmp_path / 'capture-results.jsonl').write_text('{"id":"wanted","ok":true}')
    with pytest.raises(TimeoutError, match='id=wanted'):
        client.request_capture(tmp_path, 'digits', '3-2', '7/8/9', request_id='wanted')
    assert len((tmp_path / 'capture-requests.jsonl').read_text().splitlines()) == 1


@pytest.mark.parametrize('scene,timeout', [('bad', 10), ('heroes', 11), ('heroes', 0)])
def test_invalid_request_does_not_touch_queue(tmp_path, scene, timeout):
    with pytest.raises(ValueError):
        client.request_capture(tmp_path, scene, '', '', timeout=timeout)
    assert not list(tmp_path.iterdir())
