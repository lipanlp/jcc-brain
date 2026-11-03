"""T0 装备：真实小格、官方图标、未知/赛季隔离、缓存下载和完整输出。"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from jccbrain.vision.eye.items import (FRAME_LIMIT, GOLD_LIMIT, KINDS, ItemCatalog, ItemReader,
                                      badge_number, bench_slots, frame_score, gold_frame, icon_vector, unit_slots)
from jccbrain.vision.eye.units import Bar

FIXTURES = Path(__file__).parent / 'fixtures'
DATA = json.loads((FIXTURES / 'item_cells.json').read_text('utf-8'))
mac = pytest.mark.skipif(sys.platform != 'darwin', reason='需要 CoreGraphics')


@pytest.fixture(scope='module')
def pixels():
    with np.load(FIXTURES / 'item_cells.npz') as arrays:
        return {key: np.cumsum(arrays[key], axis=1, dtype=np.uint8) for key in arrays.files}


@pytest.fixture
def catalog(tmp_path, pixels):
    result = ItemCatalog(DATA['season_id'], DATA['icons'], tmp_path)
    result.loaded = DATA['icons']
    result.matrix = np.stack([icon_vector(pixels[row['array']], 0) for row in result.loaded])
    return result


def image_case(case, pixels):
    w, h = case.get('window', DATA['window'])
    image = np.zeros((h, w, 3), np.uint8)
    x0, y0, x1, y1 = case['rect']
    pad = case['pad']
    image[y0 - pad:y1 + pad, x0 - pad:x1 + pad] = pixels[case['array']]
    return image, image[y0:y1, x0:x1]


@pytest.mark.parametrize('case', DATA['cases'], ids=lambda c: c['array'] + ':' + c['file'])
def test_real_cells_reject_background_and_never_assign_wrong_names(case, pixels, catalog):
    image, patch = image_case(case, pixels)
    truth = case['label']
    bench = case['zone'] == 'item_bench'
    present = gold_frame(patch) >= GOLD_LIMIT if bench else frame_score(image, case['rect']) >= FRAME_LIMIT
    if not truth:
        assert not present
        return
    if bench:
        assert present
    if not present:  # 模糊/遮挡保守漏检在全量专项中统计，不把漏检写成应有行为。
        return
    found = ItemReader(catalog).identify(patch, case['rect'], image.shape, truth['slot'], bench)
    # 旧小格元数据沿用采样时的 ?；53帧整帧真值和用户确认已补上拆卸器名字。
    name = '装备拆卸器' if bench and 'badge' in truth else truth['name']
    if name == '装备拆卸器':
        assert found['name'] == name and found['kind'] == '通用道具'
        assert found['count'] == truth['badge']
    elif name == '?':
        assert found['name'].startswith('?i')
    elif not found['name'].startswith('?i'):
        assert found['name'] == name
        entry = next(r for r in catalog.entries if r['name'] == name)
        assert found['kind'] == KINDS.get(entry['kind'], '特殊')
    assert 0 < found['x'] < 1 and 0 < found['y'] < 1
    if 'badge' in truth:
        assert found['badge'] == truth['badge']


def test_remover_keeps_name_and_count_when_badge_and_brightness_change(pixels, catalog):
    reader = ItemReader(catalog)
    cases = [c for c in DATA['cases'] if c['zone'] == 'item_bench' and (c['label'] or {}).get('name') == '?']
    assert {c['label']['badge'] for c in cases} == {4, 5}
    for case in cases:
        image, patch = image_case(case, pixels)
        found = reader.identify(patch, case['rect'], image.shape, 0, True)
        assert found['name'] == '装备拆卸器' and found['kind'] == '通用道具'
        assert found['count'] == found['badge'] == case['label']['badge']
    assert not reader.unknown
    reader.reset()
    assert not reader.unknown and reader.catalog is catalog and len(catalog.loaded) == 16


def test_same_name_database_item_supersedes_extra_templates(pixels, tmp_path):
    case = DATA['cases'][0]
    image, patch = image_case(case, pixels)
    row = dict(id='db-remover', name='装备拆卸器', kind='特殊装备')
    catalog = ItemCatalog(DATA['season_id'], [row], tmp_path)
    catalog.loaded, catalog.matrix = [row], np.stack([icon_vector(patch)])
    reader = ItemReader(catalog)
    assert not reader.extra_loaded and reader.extra_matrix.shape == (0, 768)
    found = reader.identify(patch, case['rect'], image.shape, 0, True)
    assert found['name'] == '装备拆卸器' and found['kind'] == '特殊'
    assert found['count'] == found['badge'] == 4


def test_same_name_database_item_without_cached_icon_does_not_load_extra(pixels, tmp_path):
    image, patch = image_case(DATA['cases'][0], pixels)
    catalog = ItemCatalog(DATA['season_id'], [dict(id='db-remover', name='装备拆卸器')], tmp_path)
    reader = ItemReader(catalog)
    assert not reader.extra_loaded
    found = reader.identify(patch, DATA['cases'][0]['rect'], image.shape, 0, True)
    assert found['name'] == '?i1' and found['kind'] is None and 'count' not in found
    assert found['badge'] == 4


def test_any_confident_database_match_is_tried_before_extra(pixels, catalog):
    image, patch = image_case(DATA['cases'][0], pixels)
    catalog.loaded = [dict(id='db-art', name='数据库匹配的道具', kind='基础装备')]
    catalog.matrix = np.stack([icon_vector(patch)])
    reader = ItemReader(catalog)
    assert reader.extra_loaded  # 同名没有屏蔽时仍应先接受数据库匹配。
    found = reader.identify(patch, DATA['cases'][0]['rect'], image.shape, 0, True)
    assert found['name'] == '数据库匹配的道具' and found['kind'] == '基础'
    assert 'count' not in found


def test_extra_templates_never_name_other_known_or_unknown_item_cells(pixels, tmp_path):
    reader = ItemReader(ItemCatalog(DATA['season_id'], [], tmp_path))
    for case in DATA['cases']:
        if not case['label'] or 'badge' in case['label']:
            continue
        image, patch = image_case(case, pixels)
        found = reader.identify(patch, case['rect'], image.shape, case['label']['slot'],
                                case['zone'] == 'item_bench')
        assert found['name'].startswith('?i') and 'count' not in found


def test_remover_without_badge_omits_count(pixels, catalog):
    image, patch = image_case(DATA['cases'][0], pixels)
    patch = patch.copy()
    h, w = patch.shape[:2]
    patch[:round(.36 * h), :round(.36 * w)] = 0
    found = ItemReader(catalog).identify(patch, DATA['cases'][0]['rect'], image.shape, 0, True)
    assert found['name'] == '装备拆卸器' and found['kind'] == '通用道具'
    assert 'badge' not in found and 'count' not in found


def test_extra_match_does_not_consume_unknown_group_numbers(pixels, catalog):
    reader = ItemReader(catalog)
    for index in [0, 13, 5, 13]:
        case = DATA['cases'][index]
        image, patch = image_case(case, pixels)
        found = reader.identify(patch, case['rect'], image.shape, case['label']['slot'], index != 13)
        assert found['name'] == ('?i1' if index == 13 else '装备拆卸器')
    assert len(reader.unknown) == 1
    reader.reset()
    assert not reader.unknown and len(reader.extra_loaded) == 2


def test_numeric_badge_does_not_appear_on_regular_art(pixels):
    for case in DATA['cases']:
        if case['zone'] != 'item_bench':
            continue
        _, patch = image_case(case, pixels)
        assert badge_number(patch) == (case['label'] or {}).get('badge')


def test_inventory_gap_does_not_hide_later_item_and_coordinates_are_window_relative(pixels, catalog):
    case = next(c for c in DATA['cases'] if (c['label'] or {}).get('name') == '女神之泪')
    _, patch = image_case(case, pixels)
    image = np.zeros((1136, 1774, 3), np.uint8)
    x0, y0, x1, y1 = bench_slots(image)[5]
    image[y0:y1, x0:x1] = cv2.resize(patch, (x1 - x0, y1 - y0))
    reader = ItemReader(catalog)
    reader.catalog.prepare = Mock(side_effect=AssertionError('热路径不得下载图标'))
    found = reader.bench(image)
    assert len(found) == 1 and found[0]['slot'] == 5 and found[0]['name'] == '女神之泪'
    assert found[0]['x'] == pytest.approx((x0 + x1) / 2 / image.shape[1])
    assert found[0]['y'] == pytest.approx((y0 + y1) / 2 / image.shape[0])
    reader.catalog.prepare.assert_not_called()


@pytest.mark.parametrize('kind,name', [('基础', '金铲铲'), ('成型', '水银'), ('光明', '光明版无尽之刃'),
                                       ('神器', '大亨之铠'), ('纹章', '永恒之森纹章'), ('特殊', '生命值补剂')])
def test_official_icon_classification_in_synthetic_frame(kind, name, pixels, catalog):
    row = next(r for r in DATA['icons'] if r['name'] == name)
    icon = pixels[row['array']]
    framed = cv2.copyMakeBorder(cv2.resize(icon, (28, 28)), 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=0)
    found = ItemReader(catalog).identify(framed, (10, 10, 44, 44), (100, 100, 3), 0)
    assert found['name'] == name and found['kind'] == kind


def test_unit_equipment_order_is_left_to_right_and_limited_to_three(pixels, catalog):
    image = np.full((1136, 1774, 3), 140, np.uint8)
    bar = Bar(700, 769, 400, 405, .414, .354)
    names = ['水银', '反曲之弓', '巨人腰带']
    for rect, name in zip(unit_slots(bar), names):
        x0, y0, x1, y1 = rect
        icon = pixels[next(r['array'] for r in DATA['icons'] if r['name'] == name)]
        image[y0:y1, x0:x1] = 0
        image[y0 + 2:y1 - 2, x0 + 2:x1 - 2] = cv2.resize(icon, (x1 - x0 - 4, y1 - y0 - 4))
    found = ItemReader(catalog).on_unit(image, bar)
    assert [r['name'] for r in found] == names
    assert [r['slot'] for r in found] == [0, 1, 2]
    assert found[0]['x'] < found[1]['x'] < found[2]['x']


def test_ambiguous_official_art_does_not_choose_arbitrary_name(catalog, pixels):
    row = next(r for r in catalog.loaded if r['name'] == '水银')
    vector = icon_vector(pixels[row['array']], 0)
    catalog.loaded = [dict(row), dict(row, name='另一个当前赛季装备')]
    catalog.matrix = np.stack([vector, vector])
    patch = cv2.copyMakeBorder(pixels[row['array']], 2, 2, 2, 2, cv2.BORDER_CONSTANT)
    found = ItemReader(catalog).identify(patch, (0, 0, 20, 20), (100, 100, 3), 0)
    assert found['name'].startswith('?i')


def test_catalog_uses_selected_database_names_and_only_same_season_url_fallback(tmp_path):
    item = SimpleNamespace(id='1001', name='数据库当前名字', kind='基础装备')
    legacy = ItemCatalog.from_db(SimpleNamespace(set_id=18, items={'1001': item}), tmp_path)
    assert legacy.season_id == '18_S19'
    assert [r['name'] for r in legacy.entries] == ['数据库当前名字']
    assert legacy.entries[0]['picture'].endswith('/1001.png')
    other = ItemCatalog.from_db(SimpleNamespace(season_id='16_S19', set_id=18, items={'1001': item}), tmp_path)
    assert other.season_id == '16_S19' and other.entries[0]['picture'] is None
    assert other.directory != legacy.directory


@mac
def test_cached_icons_from_another_season_never_participate(tmp_path, pixels):
    row = next(r for r in DATA['icons'] if r['id'] == '1001')
    original = ItemCatalog('18_S19', [row], tmp_path)
    original.directory.mkdir(parents=True)
    cv2.imwrite(str(original.file(row)), pixels[row['array']])
    original.load_cached()
    assert len(original.loaded) == 1
    other = ItemCatalog('16_S19', [row], tmp_path)
    assert not other.loaded and other.matrix.shape == (0, 768)


@mac
def test_download_is_startup_only_atomic_and_retries_bad_image(tmp_path, pixels, monkeypatch):
    row = next(r for r in DATA['icons'] if r['id'] == '1001')
    row = dict(row, picture='https://example.invalid/icon.png')
    catalog = ItemCatalog('18_S19', [row], tmp_path)
    good = cv2.imencode('.png', pixels[row['array']])[1].tobytes()
    response = Mock()
    response.content = b'bad image'
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.get.return_value = response
    monkeypatch.setattr('httpx.Client', Mock(return_value=client))
    catalog.prepare()
    assert not catalog.file(row).exists() and not list(catalog.directory.glob('*.tmp'))
    assert not catalog.loaded
    response.content = good
    catalog.prepare()
    assert catalog.file(row).is_file() and len(catalog.loaded) == 1
    catalog.prepare()
    assert client.get.call_count == 2


def test_offline_startup_does_not_retry_network_for_every_icon(tmp_path, monkeypatch):
    import httpx

    rows = [dict(id=str(i), name=f'装备{i}', picture='https://example.invalid/item.png') for i in range(50)]
    catalog = ItemCatalog('18_S19', rows, tmp_path)
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.get.side_effect = httpx.ConnectError('offline')
    monkeypatch.setattr('httpx.Client', Mock(return_value=client))
    catalog.prepare()
    assert 1 <= client.get.call_count <= 4
    assert not catalog.loaded and not list(catalog.directory.glob('*.tmp'))


def test_only_live_provider_downloads_icons_offline_replay_does_not(tmp_path, monkeypatch):
    from jccbrain.vision import pyeye

    prepare, warmup = Mock(), Mock()
    reader = SimpleNamespace(items=SimpleNamespace(catalog=SimpleNamespace(prepare=prepare)),
                             ocr=SimpleNamespace(warmup=warmup))
    monkeypatch.setattr(pyeye, 'build_reader', Mock(return_value=reader))
    (tmp_path / 'frame.jpg').touch()
    pyeye.ImageReplayProvider(tmp_path).start()
    prepare.assert_not_called()
    pyeye.PyEyeProvider().start()
    prepare.assert_called_once()
    assert warmup.call_count == 2


def test_new_equipment_fields_parse_without_changing_unit_or_engine_counts():
    from jccbrain.core.models import GameState

    item = {'slot': 0, 'name': '水银', 'kind': '成型', 'x': .4, 'y': .3}
    data = {'board': [{'name': '蔚', 'star': 2, 'x': .4, 'y': .3, 'items': [item]}],
            'bench': [{'name': '蔚', 'slot': 1, 'items': [dict(item, name='?i1', kind=None)]}],
            'item_bench': [dict(item, name='?i2', badge=5)]}
    state = GameState.from_dict(data)
    assert state.owned_copies()['蔚'] == 4 and state.bench_free == 8
    assert state.board[0].items == [item] and state.bench[0].items[0]['name'] == '?i1'
    assert state.item_bench[0]['badge'] == 5
    assert GameState.from_dict({}).item_bench == []
    assert GameState.from_dict({'board': [{'name': '蔚'}]}).board[0].items == []


@mac
def test_true_frame_reader_outputs_equipment_and_combat_keeps_preparation_items(tmp_path, catalog):
    from jccbrain.vision.eye.capture import load_image
    from jccbrain.vision.pyeye import build_reader

    reader = build_reader(None, tmp_path / 'gallery')
    reader.items.catalog = catalog
    image = load_image(FIXTURES / 'frames/0003.jpg')
    try:
        state = reader.read(image)
        assert [r['name'] for r in state['item_bench']] == ['装备拆卸器', '日炎斗篷', '反曲之弓', '女神之泪']
        assert state['item_bench'][0]['count'] == state['item_bench'][0]['badge'] == 4
        equipped = [u for u in state['board'] if u['items']]
        assert [[i['name'] for i in u['items']] for u in equipped] == [['水银', '汲取剑']]
        prepared = state['board']
        reader.phase, reader.last_timer = 'combat', None
        # 只测战斗记忆路径，单位坐标/装备不能用本帧身体背景重新覆盖。
        reader.units.observe(np.zeros_like(image), 'combat', None)
        stale = reader.units.observe(np.zeros_like(image), 'combat', None)
        assert stale['units_stale'] and stale['board'] == prepared
        reader._new_game()
        assert not reader.items.unknown and reader.units.last_live is None
    finally:
        reader.ocr.pool.shutdown()


@mac
def test_lossless_fixture_pixel_round_trip(pixels):
    from jccbrain.vision.eye.capture import load_image

    case = next(c for c in DATA['cases'] if c['file'] == '0000.jpg')
    # 0000 全帧未入夹具；用已有0003的同一物品不能代替像素校验，核对可用的原始样本。
    source = Path(__file__).resolve().parents[1] / 'game-eye' / 'samples' / case['file']
    if not source.exists():
        pytest.skip('原始样本不在当前机器，其他裁剪回归仍执行')
    image = load_image(source)
    x0, y0, x1, y1 = case['rect']
    assert np.array_equal(image[y0:y1, x0:x1], pixels[case['array']])


@mac
def test_concurrent_first_decode_after_warmup_never_fails():
    """新进程里 8 线程同时首次解码；预热后不应再出现 KeyError('符号名')。"""
    import subprocess

    root = Path(__file__).parents[1]
    result = subprocess.run([sys.executable, str(root / 'scripts/repro_decode_race.py'),
                             str(FIXTURES / 'frames/0003.jpg'), '--runs', '5', '--warm'],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


@mac
def test_download_failure_logs_tell_network_from_local_decode(tmp_path, monkeypatch, caplog):
    import httpx

    rows = [dict(id='1', name='网络坏', picture='https://example.invalid/a.png'),
            dict(id='2', name='解码坏', picture='https://example.invalid/b.png')]
    catalog = ItemCatalog('18_S19', rows, tmp_path)
    bad = Mock(content=b'bad image')

    def get(url):
        if url.endswith('a.png'):
            raise httpx.HTTPStatusError('404', request=Mock(), response=Mock())
        return bad
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.get.side_effect = get
    monkeypatch.setattr('httpx.Client', Mock(return_value=client))
    warm = Mock()
    monkeypatch.setattr('jccbrain.vision.eye.items.warm_decoder', warm)
    with caplog.at_level('WARNING'):
        catalog.prepare()
    warm.assert_called_once()
    messages = [r.getMessage() for r in caplog.records]
    assert any('（网络）网络坏' in m for m in messages)
    assert any('（本地写入/解码）解码坏' in m for m in messages)
    assert not catalog.loaded and not list(catalog.directory.glob('*.tmp'))
