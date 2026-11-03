"""单位名卡池边界与分赛季图库：临时目录、手写卡池，不依赖主目录。"""
import hashlib
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jccbrain.vision.eye.units import Gallery, TrackedUnit, UnitTracker

VEC = np.array([1., 0.], dtype=np.float32)
OTHER = np.array([0., 1.], dtype=np.float32)
IMAGE = np.zeros((1, 1, 3), dtype=np.uint8)


def write_gallery(directory, names):
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, name in enumerate(names):
        file = f'{name}_{i}.npy'
        np.save(directory / file, VEC)
        # 仅检查原样复制，不解码这份合成 jpg。
        (directory / file).with_suffix('.jpg').write_bytes(b'synthetic-jpg-' + bytes([i]))
        rows.append({'name': name, 'file': file})
    (directory / 'index.json').write_text(json.dumps(rows, ensure_ascii=False), 'utf-8')
    return rows


def hashes(directory, recursive=True):
    files = directory.rglob('*') if recursive else directory.iterdir()
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files if p.is_file()}


def unit(zone='bench', slot=0, name=None, group=None, fp=VEC):
    return TrackedUnit(zone, slot if zone == 'bench' else None, .25 + .063 * slot, .65 if zone == 'bench' else .3,
                       1, name=name, group=group, fp=fp.copy(), hist=np.array([1., 0.]),
                       fp_prepared=True, crop=IMAGE.copy())


def observe(tracker, units, shop=None, phase='planning'):
    return tracker.observe(IMAGE, phase, shop, prepared=({}, units))


def test_flat_migration_filters_pool_preserves_originals_and_runs_once(tmp_path, monkeypatch):
    rows = write_gallery(tmp_path, ['英雄甲', '英雄乙', '旧赛季', '旧赛季'])
    before = hashes(tmp_path, recursive=False)
    gallery = Gallery(tmp_path, 'test', {'英雄甲', '英雄乙'})
    assert gallery.dir == tmp_path / 'stest'
    assert gallery.names == {'英雄甲': 1, '英雄乙': 1}
    assert json.loads((gallery.dir / 'index.json').read_text()) == rows[:2]
    assert json.loads((gallery.dir / 'migration.json').read_text()) == {
        'from': 'flat', 'copied': 2, 'skipped': {'旧赛季': 2}}
    assert hashes(tmp_path, recursive=False) == before
    assert set(hashes(gallery.dir)) == {'index.json', 'migration.json'} | {
        f'{name}_{i}.{ext}' for i, name in enumerate(['英雄甲', '英雄乙']) for ext in ['npy', 'jpg']}
    for row in rows[:2]:
        for file in [row['file'], str(Path(row['file']).with_suffix('.jpg'))]:
            assert (tmp_path / file).read_bytes() == (gallery.dir / file).read_bytes()
    migrated = hashes(gallery.dir)
    monkeypatch.setattr(Gallery, '_migrate_flat', lambda self: pytest.fail('重复迁移'))
    assert Gallery(tmp_path, 'test', {'英雄甲', '英雄乙'}).names == gallery.names
    assert hashes(gallery.dir) == migrated
    assert hashes(tmp_path, recursive=False) == before


def test_existing_season_directory_without_index_is_not_migrated(tmp_path):
    write_gallery(tmp_path, ['英雄甲'])
    (tmp_path / 'stest').mkdir()
    gallery = Gallery(tmp_path, 'test', {'英雄甲'})
    assert gallery.names == {}
    assert list(gallery.dir.iterdir()) == []


def test_two_seasons_have_independent_samples_even_with_the_same_pool(tmp_path):
    pool = {'英雄甲', '英雄乙'}
    first = Gallery(tmp_path, 'first', pool)
    second = Gallery(tmp_path, 'second', pool)
    assert first.add('英雄甲', VEC, None) is True
    assert second.add('英雄乙', VEC, None) is True
    assert Gallery(tmp_path, 'first', pool).match(VEC) == ('英雄甲', 0.)
    assert Gallery(tmp_path, 'second', pool).match(VEC) == ('英雄乙', 0.)
    assert not (tmp_path / 'index.json').exists()


def test_no_season_reads_flat_directly_without_migration(tmp_path):
    write_gallery(tmp_path, ['英雄甲'])
    before = hashes(tmp_path)
    gallery = Gallery(tmp_path, season=None, pool={'英雄甲'})
    assert gallery.dir == tmp_path
    assert gallery.match(VEC) == ('英雄甲', 0.)
    assert hashes(tmp_path) == before


def write_refs(directory):
    directory.mkdir()
    rows = [{'key': str(i), 'name': name} for i, name in enumerate(['英雄甲', '旧赛季'])]
    (directory / 'unit_refs.json').write_text(json.dumps(
        {'version': 1, 'season': 'test', 'entries': rows}), 'utf-8')
    np.savez_compressed(directory / 'stest.npz', **{'0': VEC, '1': OTHER})


def test_bundled_refs_are_pool_filtered_and_never_written_with_user_labels(tmp_path):
    refs = tmp_path / 'refs'
    write_refs(refs)
    before = hashes(refs)
    gallery = Gallery(tmp_path / 'user', 'test', {'英雄甲', '英雄乙'}, refs_root=refs)
    assert [name for name, _ in gallery.refs] == ['英雄甲']
    assert gallery.items == gallery.index == []
    assert gallery.match(VEC) == ('英雄甲', 0.)
    with pytest.raises(ValueError):
        gallery.refs[0][1][0] = 2
    assert gallery.add('英雄乙', OTHER, None)
    assert gallery.match(OTHER) == ('英雄乙', 0.)
    assert gallery.names == {'英雄乙': 1}
    assert hashes(refs) == before
    assert [row['name'] for row in json.loads((gallery.dir / 'index.json').read_text())] == ['英雄乙']


@pytest.mark.parametrize('season,pool', [('other', {'英雄甲'}), (None, {'英雄甲'}), ('test', None)])
def test_refs_do_not_cross_seasons_or_enable_unknown_pool(tmp_path, season, pool):
    refs = tmp_path / 'refs'
    write_refs(refs)
    assert Gallery(tmp_path / 'user', season, pool, refs_root=refs).refs == ()


def test_refs_dimension_mismatch_and_invalid_resource_fail_closed(tmp_path):
    refs = tmp_path / 'refs'
    write_refs(refs)
    gallery = Gallery(tmp_path / 'user', 'test', {'英雄甲'}, refs_root=refs)
    assert gallery.match(np.zeros(3, dtype=np.float32)) is None
    (refs / 'stest.npz').write_bytes(b'corrupt')
    assert Gallery(tmp_path / 'user', 'test', {'英雄甲'}, refs_root=refs).refs == ()


def test_name_margin_compares_different_names_across_user_and_reference_gallery(tmp_path):
    gallery = Gallery(tmp_path, 'test', {'英雄甲', '英雄乙'}, threshold=.8, margin=.1)
    gallery.items = [('英雄甲', np.array([.5], np.float32)), ('英雄甲', np.array([.51], np.float32))]
    gallery.refs = (('英雄乙', np.array([.55], np.float32)),)
    assert gallery.match(np.zeros(1, np.float32)) is None
    gallery.refs = (('英雄乙', np.array([.7], np.float32)),)
    assert gallery.match(np.zeros(1, np.float32)) == ('英雄甲', .5)
    gallery.margin = 0.
    gallery.refs = (('英雄乙', np.array([.5], np.float32)),)
    assert gallery.match(np.zeros(1, np.float32)) == ('英雄甲', .5)


def test_match_threshold_remains_strict_and_does_not_change_unknown_grouping(tmp_path):
    from jccbrain.vision.eye.units import GROUP_THRESHOLD
    gallery = Gallery(tmp_path, 'test', {'英雄甲'}, threshold=.5, margin=.1)
    gallery.items = [('英雄甲', np.array([.5], np.float32))]
    assert gallery.match(np.zeros(1, np.float32)) is None
    gallery.threshold = .8
    assert gallery.match(np.zeros(1, np.float32)) == ('英雄甲', .5)
    assert GROUP_THRESHOLD == .55


def test_loading_drops_outside_pool_entries_before_reading_vectors(tmp_path, caplog):
    directory = tmp_path / 'stest'
    rows = write_gallery(directory, ['英雄甲', '旧赛季', '旧赛季'])
    (directory / rows[1]['file']).write_bytes(b'invalid-vector')
    with caplog.at_level(logging.INFO):
        gallery = Gallery(tmp_path, 'test', {'英雄甲'})
    assert gallery.index == rows[:1]
    assert gallery.names == {'英雄甲': 1}
    assert '丢弃卡池外条目 2 条' in caplog.text
    assert '旧赛季' in caplog.text
    # 即使外部向内存列表注入条目，匹配也不能绕过卡池。
    gallery.items = [('旧赛季', VEC), ('英雄甲', OTHER)]
    assert gallery.match(VEC) is None


def test_add_rejects_outside_pool_without_writing_any_file(tmp_path, caplog):
    gallery = Gallery(tmp_path, 'test', {'英雄甲'})
    before = hashes(tmp_path)
    with caplog.at_level(logging.WARNING):
        assert gallery.add('旧赛季', VEC, IMAGE, copy_jpg=tmp_path / 'missing.jpg') is False
    assert hashes(tmp_path) == before
    assert gallery.items == gallery.index == []
    assert '拒绝:旧赛季(不在当前赛季卡池)' in caplog.text


def test_default_gallery_disables_names_and_leaves_existing_files_alone(tmp_path):
    write_gallery(tmp_path, ['英雄甲'])
    before = hashes(tmp_path)
    gallery = Gallery(tmp_path)
    assert gallery.names == {} and gallery.items == gallery.index == []
    assert gallery.add('英雄甲', VEC, IMAGE) is False
    gallery.items.append(('英雄甲', VEC))
    assert gallery.match(VEC) is None and gallery.names == {}
    assert hashes(tmp_path) == before


def test_no_pool_start_does_not_prevent_later_migration(tmp_path):
    write_gallery(tmp_path, ['英雄甲'])
    gallery = Gallery(tmp_path, 'test', None)
    assert not gallery.dir.exists()
    assert not gallery.add('英雄甲', VEC, None)
    assert Gallery(tmp_path, 'test', {'英雄甲'}).names == {'英雄甲': 1}


def test_game_eye_import_filters_before_decoding_and_only_writes_season(tmp_path, monkeypatch):
    source, root = tmp_path / 'legacy', tmp_path / 'gallery'
    source.mkdir()
    rows = [{'name': name, 'file': f'{i}.fp'} for i, name in enumerate(['英雄甲', '旧赛季'])]
    (source / 'index.json').write_text(json.dumps(rows), 'utf-8')
    (source / '0.fp').write_bytes(b'fake-archive')
    (source / '0.jpg').write_bytes(b'fake-jpg')
    before = hashes(source)
    decoded = []

    def read_data(file):
        decoded.append(Path(file).name)
        return Path(file).read_bytes()  # 非卡池 1.fp 不存在；若读取则测试失败。

    foundation = SimpleNamespace(
        NSData=SimpleNamespace(dataWithContentsOfFile_=read_data),
        NSKeyedUnarchiver=SimpleNamespace(unarchivedObjectOfClass_fromData_error_=lambda *_: (
            SimpleNamespace(data=lambda: VEC.tobytes()), None)))
    monkeypatch.setitem(sys.modules, 'Foundation', foundation)
    monkeypatch.setitem(sys.modules, 'Vision', SimpleNamespace(VNFeaturePrintObservation=object()))
    gallery = Gallery(root, 'test', {'英雄甲'}, seed_from=source)
    assert decoded == ['0.fp'] and gallery.names == {'英雄甲': 1}
    assert gallery.dir == root / 'stest'
    assert len(list(gallery.dir.glob('*.npy'))) == len(list(gallery.dir.glob('*.jpg'))) == 1
    assert next(gallery.dir.glob('*.jpg')).read_bytes() == b'fake-jpg'
    assert not list(root.glob('*.npy')) and not (root / 'index.json').exists()
    assert hashes(source) == before
    Gallery(root, 'test', {'英雄甲'}, seed_from=source)
    assert decoded == ['0.fp']


@pytest.mark.parametrize('pool', [None, set(), {'英雄甲'}], ids=['no-pool', 'empty-pool', 'current-pool'])
def test_manual_label_rejects_bench_and_board_without_mutation(tmp_path, pool):
    tracker = UnitTracker(Gallery(tmp_path, 'test', pool))
    bench, board = unit(name='英雄甲', group=2), unit('board', name='英雄甲', group=3)
    tracker.last_units = [bench, board]
    tracker.group_name = {2: '英雄甲', 3: '英雄甲'} if pool else {}
    before = hashes(tmp_path)
    result = tracker.label({0: '旧赛季'}, [{'x': board.x, 'y': board.y, 'name': '旧赛季'}])
    assert result == ['拒绝:旧赛季(不在当前赛季卡池)'] * 2
    assert bench.name == board.name == '英雄甲'
    assert tracker.group_name == ({2: '英雄甲', 3: '英雄甲'} if pool else {})
    assert hashes(tmp_path) == before


def test_manual_label_accepts_current_pool_on_bench_and_board(tmp_path):
    tracker = UnitTracker(Gallery(tmp_path, 'test', {'英雄甲'}))
    bench, board = unit(group=2), unit('board', group=3)
    tracker.last_units = [bench, board]
    result = tracker.label({0: '英雄甲'}, [{'x': board.x, 'y': board.y, 'name': '英雄甲'}])
    assert result == ['bench0=英雄甲', f'board({board.x},{board.y})=英雄甲']
    assert bench.name == board.name == '英雄甲'
    assert tracker.group_name == {2: '英雄甲', 3: '英雄甲'}
    assert tracker.gallery.names == {'英雄甲': 2}


@pytest.mark.parametrize('pool,bought,accepted', [({'英雄甲'}, '英雄甲', True),
                                                ({'英雄甲'}, '旧赛季', False), (None, '英雄甲', False)])
def test_shop_disappearance_only_records_current_pool_candidate(tmp_path, pool, bought, accepted):
    tracker = UnitTracker(Gallery(tmp_path, 'test', pool))
    observe(tracker, [unit()], shop=[bought, None])
    new = unit(slot=1, fp=OTHER)
    state = observe(tracker, [unit(), new], shop=[None, None])
    assert 'auto_labeled' not in state
    assert bool(tracker.learning_candidates) is accepted
    if accepted:
        assert tracker.learning_candidates[0]['candidate'] == bought
    assert tracker.gallery.names == {} and new.name is None
    assert not list(tmp_path.rglob('*.npy'))
    assert state['bench'][1]['name'].startswith('?')


def test_two_missing_shop_cards_cannot_be_mistaken_for_one_valid_purchase(tmp_path):
    tracker = UnitTracker(Gallery(tmp_path, 'test', {'英雄甲'}))
    observe(tracker, [unit()], shop=['英雄甲', '旧赛季'])
    state = observe(tracker, [unit(), unit(slot=1, fp=OTHER)], shop=[None, None])
    assert 'auto_labeled' not in state and tracker.gallery.names == {}


@pytest.mark.parametrize('fast_hist_path', [False, True])
def test_temporal_anchor_does_not_inherit_outside_pool_name(tmp_path, fast_hist_path):
    tracker = UnitTracker(Gallery(tmp_path, 'test', {'英雄甲'}))
    previous = unit(name='旧赛季', group=4)
    tracker.prev = [previous]
    current = unit()
    if not fast_hist_path:
        current.hist = np.array([0., 1.])
    state = observe(tracker, [current])
    assert current.name is None and state['bench'][0]['name'] == '?4'
    assert tracker.group_name == {} and tracker.gallery.names == {}


def test_bench_to_board_rejects_outside_pool_anchor(tmp_path):
    tracker = UnitTracker(Gallery(tmp_path, 'test', {'英雄甲'}))
    tracker.prev = [unit(name='旧赛季', group=4)]
    current = unit('board', fp=OTHER)
    state = observe(tracker, [current])
    assert current.name is None and state['board'][0]['name'] != '?4'
    assert tracker.group_name == {} and tracker.gallery.names == {}


def test_group_name_only_records_and_propagates_current_pool(tmp_path):
    tracker = UnitTracker(Gallery(tmp_path, 'test', {'英雄甲'}))
    tracker.group_name = {2: '旧赛季', 3: '英雄甲'}
    bad, good = unit(group=2, name='旧赛季'), unit(slot=1, group=3, fp=OTHER)
    state = observe(tracker, [bad, good])
    assert tracker.group_name == {3: '英雄甲'}
    assert [u['name'] for u in state['bench']] == ['?2', '英雄甲']
    assert bad.name is None and good.name == '英雄甲'


@pytest.mark.parametrize('pool', [None, set(), {'英雄甲'}], ids=['no-pool', 'empty-pool', 'current-pool'])
def test_stale_output_checks_pool_again_without_changing_positions(tmp_path, pool):
    tracker = UnitTracker(Gallery(tmp_path, 'test', pool))
    bench, board = unit(name='旧赛季', group=2), unit('board', name='英雄甲', group=3)
    tracker.last_live = ([bench], [board])
    state = observe(tracker, [], phase='combat')
    assert state['units_stale'] is True
    assert state['bench'] == [{'slot': 0, 'star': 1, 'unsure': True}]
    assert state['board'] == [{'x': .25, 'y': .3, 'star': 1, 'name': '英雄甲' if pool else '?3'}]


@pytest.mark.parametrize('group', [None, 7])
@pytest.mark.parametrize('pool', [None, set(), {'英雄甲'}], ids=['no-pool', 'empty-pool', 'current-pool'])
def test_to_json_guards_manually_injected_outside_pool_name(group, pool):
    current = unit(name='旧赛季', group=group)
    out = current.to_json(pool)
    assert out.get('name') == (f'?{group}' if group is not None else None)
    assert out['slot'] == 0 and out['star'] == 1


def test_to_json_requires_explicit_pool_even_for_a_valid_name():
    current = unit(name='英雄甲', group=7)
    assert current.to_json()['name'] == '?7'
    assert current.to_json({'英雄甲'})['name'] == '英雄甲'


@pytest.fixture
def reader_builder(tmp_path, monkeypatch):
    from jccbrain.core import paths
    from jccbrain.vision.eye import items, ocr
    from jccbrain.vision.pyeye import build_reader

    original = items.ItemCatalog.from_db.__func__
    monkeypatch.setattr(items.ItemCatalog, 'from_db', classmethod(
        lambda cls, db=None: original(cls, db, directory=tmp_path / 'icons')))
    monkeypatch.setattr(paths, 'legacy_gallery_dir', lambda: None)
    monkeypatch.setattr(ocr, 'create_engine', lambda _: object())
    return lambda db: build_reader(db, tmp_path / 'gallery')


def fake_db(**season):
    hero = SimpleNamespace(name='英雄甲', cost=1, traits=[])
    return SimpleNamespace(is_empty=False, items={}, traits=[], augments={}, pool_heroes=lambda: [hero], **season)


@pytest.mark.parametrize('fields,expected', [({'season_id': '18_S19'}, '18_S19'),
                                           ({'set_id': 18}, '18_S19'),
                                           ({'season_id': '16_S19', 'set_id': 18}, '16_S19')])
def test_build_reader_uses_db_pool_and_item_season_rule(tmp_path, reader_builder, fields, expected):
    reader = reader_builder(fake_db(**fields))
    assert reader.units.gallery.pool == {'英雄甲'}
    assert reader.units.gallery.dir == tmp_path / 'gallery' / ('s' + expected)
    assert reader.items.catalog.season_id == expected


@pytest.mark.parametrize('db', [None, SimpleNamespace(is_empty=True, season_id='18_S19', items={})],
                         ids=['missing-db', 'empty-db'])
def test_build_reader_without_db_disables_names_and_warns_once(reader_builder, db, caplog):
    with caplog.at_level(logging.WARNING):
        reader = reader_builder(db)
    assert reader.units.gallery.pool is None
    assert caplog.messages.count('没有当前赛季卡池，单位名字暂不输出') == 1
    current = unit(name='英雄甲', group=2)
    assert observe(reader.units, [current])['bench'][0]['name'] == '?2'
    assert not reader.units.gallery.add('英雄甲', VEC, IMAGE)


def test_provider_status_returns_actual_season_directory_before_and_after_start(tmp_path, reader_builder):
    from jccbrain.vision.pyeye import PyEyeProvider

    db = fake_db(set_id=18)
    provider = PyEyeProvider(db=db, gallery_dir=tmp_path / 'gallery')
    assert provider.status()['gallery'] == str(tmp_path / 'gallery' / 's18_S19')
    assert not (tmp_path / 'gallery').exists()
    provider._reader = reader_builder(db)
    assert provider.status()['gallery'] == str(provider._reader.units.gallery.dir)


@pytest.mark.parametrize('same,different,expected', [(.4, .8, .6), (.8, .6, .54), (.6, .6, .54)])
def test_stale_bench_threshold_uses_midpoint_or_conservative_tier(same, different, expected):
    from jccbrain.vision.eye.units import stale_bench_threshold

    assert stale_bench_threshold(same, different) == pytest.approx(expected)


def test_conservative_stale_assignment_rejects_handwritten_negative_distances():
    from jccbrain.vision.eye.units import assign_stale_bench, stale_bench_threshold

    threshold = stale_bench_threshold(.9, .6)
    assert assign_stale_bench([[.2, .7], [.6, .8]], threshold) == {0: 0}
    assert assign_stale_bench([[threshold]], threshold) == {}


def test_stale_assignment_is_greedy_and_one_to_one():
    from jccbrain.vision.eye.units import assign_stale_bench

    assert assign_stale_bench([[.4, .2], [.3, .1]], .5) == {1: 1, 0: 0}
    assert assign_stale_bench([[.1], [.2]], .5) == {0: 0}


@pytest.fixture
def stale_tracker(tmp_path, monkeypatch):
    from jccbrain.vision.eye import units

    tracker = UnitTracker(Gallery(tmp_path, 'test', {'英雄甲', '英雄乙'}))
    a, b = unit(name='英雄甲', group=1), unit(slot=1, name='英雄乙', group=2, fp=OTHER)
    b.star = 2
    for current in [a, b]:
        current.bar = units.Bar(current.slot, current.slot + 1, 0, 1, current.x, current.y)
    tracker.last_live = ([a, b], [])
    crops = {0: VEC.reshape(1, 1, 2), 1: OTHER.reshape(1, 1, 2)}
    monkeypatch.setattr(units, 'unit_crop', lambda _img, bar: crops.get(bar.min_x))
    monkeypatch.setattr(units, 'feature_print', lambda crop: crop.ravel().copy())
    return tracker, crops, a, b


def test_stale_same_slots_keep_identity_and_star(stale_tracker):
    tracker, _crops, a, b = stale_tracker
    state = observe(tracker, [], phase='combat')
    assert state['units_stale']
    assert state['bench'] == [a.to_json(tracker.gallery.pool), b.to_json(tracker.gallery.pool)]
    assert tracker.last_stale_fp_count == 2
    assert [(r['slot'], r['old_slot']) for r in tracker.last_stale_assignments] == [(0, 0), (1, 1)]


def test_stale_swap_moves_name_group_and_star_without_mutating_anchor(stale_tracker):
    tracker, crops, a, b = stale_tracker
    # 使用未知名字检验组号也随身份移动，不只覆盖已知名字。
    b.name = None
    crops[0], crops[1] = crops[1], crops[0]
    state = observe(tracker, [], phase='combat')
    assert state['bench'] == [{'slot': 0, 'star': 2, 'name': '?2'},
                              {'slot': 1, 'star': 1, 'name': '英雄甲'}]
    assert [(r['slot'], r['old_slot']) for r in tracker.last_stale_assignments] == [(0, 1), (1, 0)]
    assert tracker.last_live == ([a, b], [])
    assert a.name == '英雄甲' and b.group == 2 and a.star == 1 and b.star == 2


@pytest.mark.parametrize('replacement', [None, np.array([[[-1., 0.]]], dtype=np.float32)],
                         ids=['occluded', 'different-unit'])
def test_stale_unmatched_slot_has_no_name_or_group_and_is_unsure(stale_tracker, replacement):
    tracker, crops, a, _b = stale_tracker
    a.items = [{'name': '已有装备'}]
    crops[0] = replacement
    state = observe(tracker, [], phase='augment')
    assert state['bench'][0] == {'slot': 0, 'star': 1, 'items': a.items, 'unsure': True}
    assert 'name' not in state['bench'][0]
    assert state['bench'][1]['name'] == '英雄乙'


def test_stale_missing_old_feature_does_not_restore_cached_name(stale_tracker):
    tracker, _crops, a, _b = stale_tracker
    a.fp = None
    state = observe(tracker, [], phase='combat')
    assert state['bench'][0].get('unsure') is True and 'name' not in state['bench'][0]


def test_stale_unassigned_uses_current_gallery_without_learning(stale_tracker):
    tracker, crops, a, _b = stale_tracker
    replacement = -VEC
    crops[0] = replacement.reshape(1, 1, 2)
    tracker.gallery.items = [('英雄乙', replacement), ('旧赛季', replacement)]
    before = hashes(tracker.gallery.dir)
    state = observe(tracker, [], phase='combat')
    assert state['bench'][0] == {'slot': 0, 'star': 1, 'name': '英雄乙'}
    assert tracker.last_live[0][0] is a and a.name == '英雄甲'
    assert tracker.last_stale_fp_count == 2
    assert hashes(tracker.gallery.dir) == before


def test_stale_live_bench_has_priority_and_uses_normal_gallery_flow(stale_tracker):
    tracker, _crops, _a, _b = stale_tracker
    tracker.gallery.items = [('英雄乙', VEC)]
    current = unit(name=None, fp=VEC)
    current.star = 3
    tracker.prev_shop = ['英雄甲']
    state = observe(tracker, [current], phase='combat', shop=[None])
    assert state['bench'][0] == {'slot': 0, 'star': 3, 'name': '英雄乙'}
    assert tracker.last_stale_assignments[0] == {'slot': 0, 'old_slot': None, 'live': True}
    assert 'auto_labeled' not in state and tracker.prev_shop is None and not tracker.prev


def test_stale_board_is_unchanged_even_when_current_board_and_bench_differ(stale_tracker):
    tracker, crops, a, b = stale_tracker
    board = unit('board', name='英雄甲', group=7)
    board.items = [{'name': '棋盘装备'}]
    old_output = board.to_json(tracker.gallery.pool)
    tracker.last_live = ([a, b], [board])
    crops[0], crops[1] = crops[1], crops[0]
    detected = unit('board', name='英雄乙', fp=OTHER)
    detected.x, detected.star = .7, 3
    state = observe(tracker, [detected], phase='combat')
    assert state['board'] == [old_output]
    assert board.to_json(tracker.gallery.pool) == old_output


def test_shop_ocr_missing_cannot_overwrite_different_known_name_or_propagate(tmp_path):
    tracker=UnitTracker(Gallery(tmp_path,'test',{'英雄甲','英雄乙'}))
    observe(tracker,[unit()],shop=['英雄甲',None])
    new=unit(slot=1,name='英雄乙',fp=OTHER,group=7)
    state=observe(tracker,[unit(),new],shop=[None,None])
    assert new.name=='英雄乙' and tracker.group_name[7]=='英雄乙'
    assert state['bench'][1]['name']=='英雄乙' and 'auto_labeled' not in state
    assert tracker.learning_candidates[0]['candidate']=='英雄甲'
    assert tracker.gallery.names=={} and not list(tmp_path.rglob('*.npy'))


@pytest.mark.parametrize('same,conflict',[(False,False),(True,False),(True,True)])
def test_bench_to_board_needs_current_feature_and_never_writes(tmp_path,same,conflict):
    tracker=UnitTracker(Gallery(tmp_path,'test',{'英雄甲','英雄乙'}))
    tracker.prev=[unit(name='英雄甲',group=9)]
    current=unit('board',name='英雄乙' if conflict else None,fp=VEC if same else OTHER)
    observe(tracker,[current])
    assert current.name==('英雄乙' if conflict else '英雄甲' if same else None)
    assert (current.group==9) is (same and not conflict)
    assert tracker.gallery.names=={} and not list(tmp_path.rglob('*.npy'))


def test_conflicting_group_is_quarantined_instead_of_last_name_winning(tmp_path):
    tracker=UnitTracker(Gallery(tmp_path,'test',{'英雄甲','英雄乙'}))
    tracker.group_name={7:'英雄甲'}
    a=unit(name='英雄乙',group=7);b=unit(slot=1,group=7,fp=OTHER)
    observe(tracker,[a,b])
    assert tracker.group_name=={} and tracker.conflicted_groups=={7}
    assert a.name=='英雄乙' and b.name is None
    tracker.prev=[]
    c=unit(slot=2,group=7,name='英雄甲');c.name_source='group'
    observe(tracker,[c])
    assert c.name is None and tracker.group_name=={}
