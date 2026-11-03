"""识别性能选项和阶段节流的行为契约。"""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from jccbrain.core.models import GameState, Phase
from jccbrain.vision.eye import rapid
from jccbrain.vision.pyeye import PyEyeProvider, frame_interval


@pytest.mark.parametrize('threads', [None, 0, 1, 2, 4])
def test_real_ort_sessions_disable_spinning_set_threads_and_restore_descriptor(threads):
    from rapidocr_onnxruntime.utils.infer_engine import OrtInferSession

    original = vars(OrtInferSession)['_init_sess_opts']
    ocr = rapid.RapidOCR(ort_threads=threads)
    try:
        for session in [ocr.det.infer.session, ocr.rec.session.session]:
            opts = session.get_session_options()
            assert opts.intra_op_num_threads == (1 if threads is None else threads)
            for kind in ['intra', 'inter']:
                assert opts.get_session_config_entry(f'session.{kind}_op.allow_spinning') == '0'
        assert vars(OrtInferSession)['_init_sess_opts'] is original
    finally:
        ocr.pool.shutdown(wait=True)


@pytest.mark.parametrize('target', ['rapidocr_onnxruntime.ch_ppocr_det.TextDetector',
                                    'rapidocr_onnxruntime.ch_ppocr_rec.TextRecognizer'])
def test_ort_descriptor_restored_when_model_construction_fails(monkeypatch, target):
    from rapidocr_onnxruntime.utils.infer_engine import OrtInferSession

    original = vars(OrtInferSession)['_init_sess_opts']
    def fail(_config):
        assert vars(OrtInferSession)['_init_sess_opts'] is not original
        raise RuntimeError('构造失败')

    monkeypatch.setattr(target, fail)
    with pytest.raises(RuntimeError, match='构造失败'):
        rapid.RapidOCR()
    assert vars(OrtInferSession)['_init_sess_opts'] is original


def test_concurrent_model_builds_keep_independent_options_and_restore_original():
    from concurrent.futures import ThreadPoolExecutor
    from rapidocr_onnxruntime.utils.infer_engine import OrtInferSession

    original = vars(OrtInferSession)['_init_sess_opts']
    def build(config):
        threads, spin = config
        ocr = rapid.RapidOCR(ort_threads=threads, spin=spin)
        try:
            opts = ocr.det.infer.session.get_session_options()
            return opts.intra_op_num_threads, opts.get_session_config_entry('session.intra_op.allow_spinning')
        finally:
            ocr.pool.shutdown(wait=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(build, [(1, True), (4, False)])) == [(1, '1'), (4, '0')]
    assert vars(OrtInferSession)['_init_sess_opts'] is original


@pytest.mark.parametrize('phase,base,expected', [('planning', .25, .25), ('planning', 1.5, 1.5),
    ('combat', .25, 1.), ('augment', .25, 1.), ('carousel', .25, 1.), ('ended', .25, 1.),
    ('new-phase', .25, 1.), ('combat', 1.5, 1.5), ('unknown', .25, 2.),
    (None, .25, 2.), ('', .25, 2.), ('unknown', 3., 3.), (Phase.PLANNING, .25, .25)])
def test_frame_interval_preserves_planning_and_slows_other_phases(phase, base, expected):
    assert frame_interval(phase, base) == expected


def provider(monkeypatch, tmp_path, phase='combat'):
    from jccbrain.vision import pyeye

    clock = SimpleNamespace(now=10.)
    sleeps = []
    monkeypatch.setattr(pyeye.time, 'monotonic', lambda: clock.now)
    def sleep(seconds):
        sleeps.append(seconds)
        clock.now += seconds
    monkeypatch.setattr(pyeye.time, 'sleep', sleep)
    p = PyEyeProvider(gallery_dir=tmp_path)
    p.start = lambda: None
    p._capture = SimpleNamespace(grab=Mock(return_value=(np.zeros((1, 1, 3)), None)))
    p._reader = SimpleNamespace(read=Mock(return_value={'phase': 'planning'}), last_ocr_cells=0)
    p._last = GameState.from_dict({'phase': phase})
    p._last_t = 9.8
    return p, clock, sleeps


def test_provider_uses_previous_phase_but_fresh_false_cache_window_is_unchanged(monkeypatch, tmp_path):
    p, clock, sleeps = provider(monkeypatch, tmp_path)
    previous = p._last
    assert p.read(fresh=False) is previous
    assert not sleeps and p._capture.grab.call_count == 0
    clock.now = 10.1
    assert p.read(fresh=True).phase == Phase.PLANNING
    assert sleeps == pytest.approx([.7])
    p.read(fresh=True)
    assert sleeps == pytest.approx([.7, .25])


def test_failed_capture_waits_unknown_interval_without_changing_cached_state(monkeypatch, tmp_path):
    from jccbrain.vision.base import VisionError
    from jccbrain.vision.eye.capture import CaptureError

    p, clock, sleeps = provider(monkeypatch, tmp_path, 'planning')
    p._capture.grab.side_effect = [CaptureError('没有游戏窗口'), (np.zeros((1, 1, 3)), None)]
    previous = p._last
    with pytest.raises(VisionError, match='没有游戏窗口'):
        p.read()
    assert p._last is previous
    p.read()
    assert sleeps == pytest.approx([.05, 2.])
