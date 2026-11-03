"""手写向量验证回合留出、不同名字间隔和选参，不依赖截图或图库。"""
import numpy as np
import pytest

from scripts.eval_gallery_loo import (MARGINS, THRESHOLDS, Sample, decision, fold_key, loo_neighbors,
                                      memory_add, nearest, read_cache, save_cache, score, select_parameters)


def sample(name, value, fold=('.', '2-1')):
    return Sample('frame.jpg', fold, name, 'bench', 0,
                  np.array([value], dtype=np.float32) if value is not None else None)


def test_fold_uses_full_relative_path_truth_then_reader_stage():
    truth = {'0021.jpg': {'stage': '2-2'}}
    assert fold_key('0021.jpg', {'stage': 'wrong'}, truth) == ('.', '2-2')
    assert fold_key('track/0021.jpg', {'stage': '2-3'}, truth) == ('track', '2-3')
    assert fold_key('0022.jpg', {'stage': '2-2'}, truth) == ('.', '2-2')
    assert fold_key('missing.jpg', {}, truth) == ('.', '<missing>')


def test_leave_one_fold_excludes_all_adjacent_frames_but_keeps_other_directory():
    samples = [sample('甲', 0), sample('甲', 0), sample('乙', .2, ('track', '2-1')),
               sample('丙', .5, ('.', '2-2'))]
    neighbors = loo_neighbors(samples)
    assert [n['name'] for n in neighbors[:2]] == ['乙', '乙']
    assert neighbors[0]['d1'] == pytest.approx(.2)
    assert neighbors[0]['reference_available'] is False
    assert neighbors[2]['name'] == '甲'
    assert neighbors[2]['d2'] == pytest.approx(.3)


def test_margin_uses_nearest_different_name_and_inclusive_gap():
    n = nearest(sample('甲', 0), [sample('甲', .125), sample('甲', .13), sample('乙', .25)])
    assert n['d2'] == .25  # 不能拿同名第二张图 .13 当 d2。
    assert decision(n, .5, .125) == '甲'
    assert decision(n, .5, .126) is None
    assert decision(n, .125, 0) is None  # d1 < t，等于门限拒绝。
    assert decision(nearest(sample('甲', 0), [sample('甲', .125)]), .5, .1) == '甲'


def test_missing_features_and_missing_reference_are_kept_in_denominator():
    queries = [sample('甲', 0), sample('乙', None), sample('丙', .1)]
    neighbors = [nearest(q, [sample('甲', 0), sample('乙', 1)]) for q in queries]
    result = score(queries, neighbors, .5, 0)
    assert (result['samples'], result['correct'], result['named']) == (3, 1, 2)
    assert result['accuracy'] == .5
    assert result['coverage'] == pytest.approx(1 / 3)
    assert result['per_hero']['乙']['samples'] == 1
    assert result['per_hero']['丙']['reference_available'] == 0


def test_parameter_selection_requires_accuracy_and_breaks_ties_by_threshold_then_margin():
    def row(t, m, accuracy, coverage):
        return dict(threshold=t, margin=m, accuracy=accuracy, coverage=coverage)

    expected = row(.5, .1, .98, .7)
    results = [row(.9, 0, .979, .99), row(.4, .1, 1., .6), row(.6, .1, 1., .7),
               row(.5, 0, 1., .7), expected]
    assert select_parameters(results) is expected
    assert select_parameters([row(.3, .1, None, 0), row(.9, 0, .979, .99)]) is None
    assert (len(THRESHOLDS), THRESHOLDS[0], THRESHOLDS[-1]) == (25, .3, .9)
    assert MARGINS == [0, .03, .05, .08, .1]


def test_memory_learning_does_not_write_any_gallery_file(tmp_path):
    from jccbrain.vision.eye.units import Gallery

    gallery = Gallery(tmp_path, pool={'甲'})
    crop = np.zeros((2, 3, 3), dtype=np.uint8)
    fp = np.array([0., 1.])
    assert memory_add(gallery, '甲', fp, crop)
    assert not memory_add(gallery, '非卡池', fp, crop)
    assert gallery.names == {'甲': 1}
    assert gallery.items[0][1] is fp
    assert list(tmp_path.iterdir()) == []


def test_cache_preserves_exact_cached_features_crops_and_fold(tmp_path):
    samples = [sample('甲', .25), sample('乙', None)]
    samples[0].crop = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    provenance = {'hashes': {'input': 'hash'}}
    frames = [{'frame': 'frame.jpg', 'stale': False}]
    save_cache(tmp_path, samples, frames, provenance)
    restored, restored_frames = read_cache(tmp_path, provenance)
    assert restored_frames == frames
    assert restored[0].fold == samples[0].fold
    assert np.array_equal(restored[0].fp, samples[0].fp)
    assert np.array_equal(restored[0].crop, samples[0].crop)
    assert restored[1].fp is None and restored[1].crop is None
    with pytest.raises(ValueError, match='缓存来源不符'):
        read_cache(tmp_path, {'hashes': {'input': 'changed'}})
