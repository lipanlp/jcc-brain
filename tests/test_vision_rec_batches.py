from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import pytest

from jccbrain.vision.eye.rapid import RapidOCR


def test_parallel_rec_keeps_native_batches_padding_and_output_order():
    calls = []

    class Recognizer:
        rec_batch_num = 3

        def __call__(self, images):
            calls.append([int(image[0, 0, 0]) for image in images])
            return [(str(int(image[0, 0, 0])), 1.) for image in images], 0

    engine = RapidOCR.__new__(RapidOCR)
    engine._stats_lock = threading.Lock()
    engine.reset_stats()
    engine.rec = Recognizer()
    widths = [40, 12, 60, 25, 18, 90, 35, 50]
    images = [np.full((10, width, 3), i, np.uint8) for i, width in enumerate(widths)]
    with ThreadPoolExecutor(max_workers=3) as engine.pool:
        assert engine._recognize_batches(images) == [(str(i), 1.) for i in range(8)]
    order = np.argsort(np.array(widths) / 10)
    assert sorted(calls) == sorted([list(order[i:i + 3]) for i in range(0, 8, 3)])


def test_parallel_rec_failure_drains_other_batches():
    entered, release, finished = (threading.Event() for _ in range(3))

    class Recognizer:
        rec_batch_num = 1

        def __call__(self, images):
            if images[0].shape[1] == 10:
                assert entered.wait(5)
                raise ValueError('识别失败')
            entered.set()
            assert release.wait(5)
            finished.set()
            return [('字', 1.)], 0

    engine = RapidOCR.__new__(RapidOCR)
    engine._stats_lock = threading.Lock()
    engine.reset_stats()
    engine.rec = Recognizer()
    with ThreadPoolExecutor(max_workers=2) as engine.pool, ThreadPoolExecutor(max_workers=1) as caller:
        future = caller.submit(engine._recognize_batches, [np.zeros((10, w, 3), np.uint8) for w in (10, 20)])
        try:
            assert entered.wait(5)
            assert not future.done()
        finally:
            release.set()
        with pytest.raises(ValueError, match='识别失败'):
            future.result(timeout=5)
    assert finished.is_set()


def test_internal_model_stats_accumulate_threads_and_reset_without_clearing_cache():
    engine = RapidOCR.__new__(RapidOCR)
    engine._stats_lock = threading.Lock()
    engine.cache = {'kept': '文字'}
    engine.reset_stats()
    import time
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: engine._record('rec', time.perf_counter(), 2), range(20)))
    assert engine.last_counts['rec_lines'] == 40 and engine.last_ms['rec'] >= 0
    engine.reset_stats()
    assert not any(engine.last_counts.values()) and not any(engine.last_ms.values())
    assert engine.cache == {'kept': '文字'}
