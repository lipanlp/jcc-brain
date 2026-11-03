import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from jccbrain.vision.eye.capture import CaptureTimeout, load_image
from scripts.capture_daemon import CaptureDaemon, append_json


@pytest.fixture
def capture():
    return Mock(return_value=(np.full((20, 30, 3), (11, 55, 180), np.uint8), SimpleNamespace(window_id=123)))


def request(root, id='one', scene='heroes'):
    append_json(root / 'capture-requests.jsonl', dict(id=id, scene=scene, stage='2-3', notes='信息面板'))


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_saves_png_manifest_result_and_restart_skips_completed_ids(tmp_path, capture):
    request(tmp_path)
    with CaptureDaemon(tmp_path, capture, lambda: 'sc') as daemon:
        result, = daemon.process_pending()
        assert result['ok'] and result['backend'] == 'sc' and result['error'] is None
        from pathlib import Path
        path = Path(result['file'])
        assert path == tmp_path / 'replay-samples/heroes/0001.png'
        assert np.max(np.abs(load_image(path).astype(int) - capture.return_value[0].astype(int))) <= 1
        manifest, = rows(path.parent / 'manifest.jsonl')
        assert (manifest['id'], manifest['stage'], manifest['notes'], manifest['window_id']) == ('one', '2-3', '信息面板', 123)
        assert daemon.process_pending() == []
    request(tmp_path)  # 相同ID即使追加了新行也不重截。
    request(tmp_path, 'two')
    with CaptureDaemon(tmp_path, capture, lambda: 'cg') as daemon:
        result, = daemon.process_pending()
        assert result['id'] == 'two' and result['file'].endswith('/0002.png')
    assert capture.call_count == 2
    assert [r['id'] for r in rows(tmp_path / 'capture-results.jsonl')] == ['one', 'two']


def test_partial_line_is_not_processed_until_newline_and_failures_continue(tmp_path, capture):
    path = tmp_path / 'capture-requests.jsonl'
    path.write_text('{"id":"one","scene":"heroes"}')
    capture.side_effect = [CaptureTimeout(), capture.return_value]
    with CaptureDaemon(tmp_path, capture, lambda: 'cg') as daemon:
        assert daemon.process_pending() == [] and not capture.called
        with path.open('a') as file: file.write('\n')
        request(tmp_path, 'two')
        a, b = daemon.process_pending()
        assert not a['ok'] and a['file'] is None and a['error'] == '截图超时'
        assert b['ok'] and capture.call_count == 2


@pytest.mark.parametrize('line', ['bad json', '[]', '{"id":""}', '{"id":"one","scene":"../outside"}',
                                  '{"id":"one","scene":"heroes","stage":1}'])
def test_invalid_requests_are_reported_without_capture(tmp_path, capture, line):
    (tmp_path / 'capture-requests.jsonl').write_text(line + '\n')
    with CaptureDaemon(tmp_path, capture, lambda: 'cg') as daemon:
        result, = daemon.process_pending()
        assert not result['ok'] and result['error'] and not capture.called
        assert daemon.process_pending() == []


def test_restart_after_interrupted_capture_never_captures_again(tmp_path, capture):
    request(tmp_path)
    capture.side_effect = KeyboardInterrupt()
    with CaptureDaemon(tmp_path, capture, lambda: 'sc') as daemon:
        with pytest.raises(KeyboardInterrupt):
            daemon.process_pending()
    capture.reset_mock(side_effect=True)
    with CaptureDaemon(tmp_path, capture, lambda: 'cg') as daemon:
        assert not capture.called and daemon.process_pending() == []
    result, = rows(tmp_path / 'capture-results.jsonl')
    assert not result['ok'] and '中断' in result['error'] and result['backend'] == 'sc'


def test_restart_after_image_saved_recovers_result_without_duplicate_capture(tmp_path, capture, monkeypatch):
    request(tmp_path)
    with CaptureDaemon(tmp_path, capture, lambda: 'sc') as daemon:
        monkeypatch.setattr(daemon, '_finish', Mock(side_effect=KeyboardInterrupt()))
        with pytest.raises(KeyboardInterrupt): daemon.process_pending()
    (tmp_path / 'capture-results.jsonl').write_text('{"id":')
    with CaptureDaemon(tmp_path, capture, lambda: 'cg') as daemon:
        assert daemon.process_pending() == []
    assert capture.call_count == 1
    result, = rows(tmp_path / 'capture-results.jsonl')
    assert result['ok'] and result['backend'] == 'sc'
    assert len(rows(tmp_path / 'replay-samples/heroes/manifest.jsonl')) == 1


def test_second_daemon_cannot_process_concurrently(tmp_path, capture):
    with CaptureDaemon(tmp_path, capture, lambda: 'cg'):
        with pytest.raises(BlockingIOError):
            with CaptureDaemon(tmp_path, capture, lambda: 'cg'):
                pytest.fail('不应拿到第二把锁')


def test_results_committed_before_offset_and_replaced_requests_skip_completed(tmp_path, capture, monkeypatch):
    request(tmp_path)
    with CaptureDaemon(tmp_path, capture, lambda: 'cg') as daemon:
        monkeypatch.setattr(daemon, '_advance', Mock(side_effect=KeyboardInterrupt()))
        with pytest.raises(KeyboardInterrupt): daemon.process_pending()
    with CaptureDaemon(tmp_path, capture, lambda: 'cg') as daemon:
        assert daemon.process_pending() == []
        path = tmp_path / 'capture-requests.jsonl'
        path.unlink()
        request(tmp_path)
        request(tmp_path, 'two')
        assert [r['id'] for r in daemon.process_pending()] == ['two']
    assert capture.call_count == 2
    assert [r['id'] for r in rows(tmp_path / 'capture-results.jsonl')] == ['one', 'two']
