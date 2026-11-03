"""界面图标缓存：只读取数据库 URL，后台下载，不依赖图像解码框架。"""
from __future__ import annotations

import base64
import hashlib
import logging
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from jccbrain.core import paths

log = logging.getLogger(__name__)


def _safe_id(value: str) -> str:
    safe = re.sub(r'[^a-zA-Z0-9_-]', '_', value)
    return safe if safe == value and safe else (safe[:64] or 'icon') + '-' + hashlib.sha256(value.encode()).hexdigest()[:12]


def _mime(data: bytes) -> str | None:
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png'
    if data.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg'
    return None


class IconCache:
    def __init__(self, db_getter, emit_ready, *, root: Path | None = None, workers: int = 4):
        self._db = db_getter
        self._emit_ready = emit_ready
        self._root = root if root is not None else paths.var_dir()
        self._pool = ThreadPoolExecutor(max_workers=min(4, max(1, workers)), thread_name_prefix='jcc-icons')
        self._lock = threading.RLock()
        self._pending: set[tuple[str, str]] = set()
        self._failed: set[tuple[str, str]] = set()
        self._network_failed = threading.Event()
        self._closed = threading.Event()

    def get(self, keys: list[str]) -> dict[str, str | None]:
        if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
            raise ValueError('图标 keys 必须是字符串列表')
        db = self._db()
        season = _safe_id(str(db.season_id))
        items = {i.name: i for i in db.items.values()}
        result = {}
        for key in dict.fromkeys(keys):
            result[key] = None
            kind, sep, name = key.partition(':')
            entity = db.hero(name) if kind == 'hero' else items.get(name) if kind == 'item' else db.trait(name) if kind == 'trait' else None
            if not sep or entity is None:
                continue
            id = _safe_id(str(entity.id))
            dest = self._root / 'icons' / season / kind / f'{id}.png'
            candidates = ([self._root / 'item_icons' / season / f'{id}.png'] if kind == 'item' else []) + [dest]
            for path in candidates:
                try:
                    data = path.read_bytes()
                except OSError:
                    continue
                mime = _mime(data)
                if mime:
                    result[key] = f'data:{mime};base64,' + base64.b64encode(data).decode('ascii')
                    break
            if result[key] is not None:
                continue
            token = (season, key)
            with self._lock:
                if (self._closed.is_set() or self._network_failed.is_set() or token in self._failed
                        or token in self._pending):
                    continue
                if not entity.picture:
                    self._failed.add(token)
                    continue
                self._pending.add(token)
                self._pool.submit(self._download, token, entity.picture, dest)
        return result

    def prefetch(self, comp, board=None) -> None:
        db = self._db()
        names = {h.name for h in comp.heroes}
        item_names = {name for h in comp.heroes for name in h.items}
        if comp.guide:
            item_names.update(comp.guide.carry_items)
        if board:
            names.update(c.name for c in board.cells)
            item_names.update(name for c in board.cells for name in c.items)
        by_name = {i.name: i for i in db.items.values()}
        for name in list(item_names):
            item = by_name.get(name)
            if item:
                item_names.update(db.item_name(id) for id in item.recipe)
        self.get([f'hero:{name}' for name in sorted(names)] + [f'item:{name}' for name in sorted(item_names)])

    def _download(self, token, url: str, dest: Path) -> None:
        temp = dest.with_name(dest.name + '.' + uuid.uuid4().hex + '.tmp')
        success = False
        try:
            if self._closed.is_set() or self._network_failed.is_set():
                return
            response = httpx.get(url, timeout=10, follow_redirects=True)
            response.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp.write_bytes(response.content)
            if not _mime(temp.read_bytes()):
                raise ValueError('不是 PNG/JPEG 图标')
            if self._closed.is_set():
                return
            temp.replace(dest)
            success = True
        except httpx.RequestError:
            # 首次断网后，本次应用运行不再发起剩余排队下载；已有缓存仍可读。
            self._network_failed.set()
        except Exception as exc:
            log.debug('图标 %s 下载失败：%s', token[1], type(exc).__name__)
        finally:
            temp.unlink(missing_ok=True)
            with self._lock:
                self._pending.discard(token)
                if not success:
                    self._failed.add(token)
            if success and not self._closed.is_set():
                self._emit_ready([token[1]])

    def close(self) -> None:
        self._closed.set()
        self._pool.shutdown(wait=False, cancel_futures=True)
