"""图标缓存仅用临时目录与假 HTTP，不依赖网络/运行目录/原生解码器。"""
import base64
import builtins
import threading
import time
from dataclasses import replace
from unittest.mock import Mock

import httpx
import pytest

from jccbrain.data.icons import IconCache, _safe_id

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aEioAAAAASUVORK5CYII=')


def wait_for(predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    assert predicate(), 'background icon operation timed out'


@pytest.fixture
def cache(db, tmp_path, monkeypatch):
    for name in ('奥恩', '绯红树怪'):
        hero = replace(db.hero(name), picture=f'https://icons.example/{name}.png')
        db.heroes[hero.id] = hero
        db.heroes_by_name[name] = hero
    db.items['2010'] = replace(db.items['2010'], picture='https://icons.example/item.png')
    monkeypatch.setattr(httpx, 'get', Mock(side_effect=AssertionError('unexpected network')))
    ready = Mock()
    icons = IconCache(lambda: db, ready, root=tmp_path / 'icons-test')
    yield icons, ready
    icons.close()
    icons._pool.shutdown(wait=True, cancel_futures=True)


def response(content=PNG):
    return httpx.Response(200, content=content, request=httpx.Request('GET', 'https://icons.example/icon.png'))


def test_cache_hit_returns_data_uri_without_http(cache, db):
    icons, ready = cache
    path = icons._root / 'icons' / db.season_id / 'hero' / f'{db.hero("奥恩").id}.png'
    path.parent.mkdir(parents=True)
    path.write_bytes(PNG)
    assert icons.get(['hero:奥恩']) == {'hero:奥恩': 'data:image/png;base64,' + base64.b64encode(PNG).decode()}
    httpx.get.assert_not_called()
    ready.assert_not_called()


def test_reuses_visual_item_cache_read_only(cache, db):
    icons, _ = cache
    path = icons._root / 'item_icons' / db.season_id / '2010.png'
    path.parent.mkdir(parents=True)
    path.write_bytes(PNG)
    assert icons.get(['item:无尽之刃'])['item:无尽之刃'].startswith('data:image/png;base64,')
    assert path.read_bytes() == PNG and not (icons._root / 'icons').exists()
    httpx.get.assert_not_called()


def test_missing_download_publishes_atomically_and_emits_ready(cache, db, monkeypatch):
    icons, ready = cache
    get = Mock(return_value=response())
    monkeypatch.setattr(httpx, 'get', get)
    assert icons.get(['hero:奥恩']) == {'hero:奥恩': None}
    wait_for(lambda: ready.called)
    ready.assert_called_once_with(['hero:奥恩'])
    get.assert_called_once_with(db.hero('奥恩').picture, timeout=10, follow_redirects=True)
    assert icons.get(['hero:奥恩'])['hero:奥恩'].startswith('data:image/png;base64,')
    assert not list(icons._root.rglob('*.tmp'))


def test_invalid_magic_never_published_and_failure_not_retried(cache, monkeypatch):
    icons, ready = cache
    get = Mock(return_value=response(b'<html>not an image</html>'))
    monkeypatch.setattr(httpx, 'get', get)
    icons.get(['item:无尽之刃'])
    wait_for(lambda: not icons._pending)
    assert not list(icons._root.rglob('*.png')) and not list(icons._root.rglob('*.tmp'))
    assert icons.get(['item:无尽之刃']) == {'item:无尽之刃': None}
    get.assert_called_once()
    ready.assert_not_called()


def test_network_error_stops_remaining_jobs(db, tmp_path, monkeypatch):
    for name in ('奥恩', '绯红树怪', '苍蓝哨戒'):
        hero = replace(db.hero(name), picture='https://icons.example/x.png')
        db.heroes_by_name[name] = hero
    get = Mock(side_effect=httpx.ConnectError('offline'))
    monkeypatch.setattr(httpx, 'get', get)
    icons = IconCache(lambda: db, Mock(), root=tmp_path, workers=1)
    try:
        icons.get(['hero:奥恩', 'hero:绯红树怪', 'hero:苍蓝哨戒'])
        wait_for(lambda: not icons._pending)
        icons.get(['hero:苍蓝哨戒'])
        assert get.call_count == 1
    finally:
        icons.close()
        icons._pool.shutdown(wait=True, cancel_futures=True)


def test_duplicate_key_is_not_downloaded_twice(cache, monkeypatch):
    icons, ready = cache
    entered, release = threading.Event(), threading.Event()
    def get(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return response()
    download = Mock(side_effect=get)
    monkeypatch.setattr(httpx, 'get', download)
    try:
        icons.get(['hero:奥恩', 'hero:奥恩'])
        assert entered.wait(2)
        for _ in range(5):
            icons.get(['hero:奥恩'])
        download.assert_called_once()
    finally:
        release.set()
    wait_for(lambda: ready.called)


def test_download_does_not_import_quartz_or_coregraphics(cache, monkeypatch):
    icons, ready = cache
    original = builtins.__import__
    forbidden = []
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in ('Quartz', 'CoreGraphics'):
            forbidden.append(name)
            raise AssertionError('native decoding forbidden')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    monkeypatch.setattr(httpx, 'get', Mock(return_value=response()))
    icons.get(['hero:奥恩'])
    wait_for(lambda: ready.called)
    assert forbidden == []


def test_unknown_keys_missing_urls_and_invalid_arguments(cache, db):
    icons, _ = cache
    db.heroes_by_name['奥恩'] = replace(db.hero('奥恩'), picture=None)
    assert icons.get(['other:x', 'hero:不存在', 'hero:奥恩']) == dict.fromkeys(['other:x', 'hero:不存在', 'hero:奥恩'])
    with pytest.raises(ValueError, match='字符串列表'):
        icons.get('hero:奥恩')
    httpx.get.assert_not_called()


def test_safe_id_does_not_allow_directory_traversal():
    assert _safe_id('18_S19') == '18_S19'
    assert '/' not in _safe_id('../../x') and '.' not in _safe_id('../../x')
    assert _safe_id('a/b') != _safe_id('a:b')


def test_jpeg_cache_uses_matching_mime(cache, db):
    icons, _ = cache
    path = icons._root / 'icons' / db.season_id / 'item' / '2010.png'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'\xff\xd8\xffpayload')
    assert icons.get(['item:无尽之刃'])['item:无尽之刃'].startswith('data:image/jpeg;base64,')


def test_prefetch_covers_comp_heroes_items_and_components(cache, db, monkeypatch):
    icons, _ = cache
    get = Mock(return_value=response())
    monkeypatch.setattr(httpx, 'get', get)
    icons.prefetch(db.comp('104'))
    wait_for(lambda: not icons._pending)
    assert get.call_count == 2  # 绯红树怪及无尽之刃有 URL；其他合成数据没有 URL。
    assert icons.get(['hero:绯红树怪', 'item:无尽之刃'])['item:无尽之刃'] is not None


def test_background_downloads_never_exceed_four_workers(cache, db, monkeypatch):
    icons, _ = cache
    names = list(db.heroes_by_name)
    for name in names:
        db.heroes_by_name[name] = replace(db.hero(name), picture='https://icons.example/x.png')
    lock, release = threading.Lock(), threading.Event()
    active = peak = 0
    def get(*args, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            assert release.wait(2)
            return response()
        finally:
            with lock:
                active -= 1
    monkeypatch.setattr(httpx, 'get', get)
    try:
        icons.get([f'hero:{name}' for name in names])
        wait_for(lambda: peak == 4)
        assert peak == 4
    finally:
        release.set()
    wait_for(lambda: not icons._pending)
    assert peak == 4


def test_trait_download_then_offline_cache(cache, db, monkeypatch):
    icons, ready = cache
    trait = replace(db.trait('峡谷野怪'), picture='https://icons.example/trait.png')
    db.traits[trait.name] = trait
    get = Mock(return_value=response())
    monkeypatch.setattr(httpx, 'get', get)
    assert icons.get(['trait:峡谷野怪']) == {'trait:峡谷野怪': None}
    wait_for(lambda: ready.called)
    get.assert_called_once_with(trait.picture, timeout=10, follow_redirects=True)
    icons._network_failed.set()
    assert icons.get(['trait:峡谷野怪'])['trait:峡谷野怪'].startswith('data:image/png;base64,')
    assert get.call_count == 1
