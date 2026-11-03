"""Opt-in unit learning: capture receipts, evidence queue, verified vectors and revocation.

No clicks, capture threads, training labels, or implicit user-directory writes.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .image import crop
from .unit_panel import PanelReader, PANEL_REGION, bind_selection
from .units import MATCH_THRESHOLD, MATCH_MARGIN, TRACK_THRESHOLD, feature_print, fp_distance

log = logging.getLogger(__name__)


def image_hash(image):
    return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class StoreCorrupt(ValueError):
    pass


class LearningCancelled(RuntimeError):
    """A caller's stop/input/scene gate closed; never a verified identity."""


class UnitLearningStore:
    """Verified entries are immutable directories committed by one atomic index.

    Process lock serializes read-modify-write. The bounded review queue is physically
    separate; pruning it cannot remove the copied evidence of an accepted sample.
    Orphans from a crash before index commit are never valid matching entries.
    """
    def __init__(self, root, season, pool, *, max_pending=64, max_bytes=16*1024*1024, read_only=False):
        self.root = Path(root) / ('s' + str(season))
        self.season, self.pool = str(season), frozenset(pool or ())
        self.max_pending, self.max_bytes = max(1, max_pending), max(1, max_bytes)
        self.verified = self.root / 'verified'
        self.queue = self.root / 'review'
        self.read_only = read_only
        if not read_only:
            for directory in (self.verified, self.queue):
                directory.mkdir(parents=True, exist_ok=True)
        self._thread_lock = threading.RLock()
        # Only currently indexed, eligible rows are cached. The writer caps each
        # hero at three vectors; reject oversized external indexes rather than
        # allowing matching to create an unbounded cache of historical rows.
        self._cache_limit = 3 * len(self.pool)
        self._cache_index_stat = None
        self._cache_rows = ()
        self._cache_index_error = None
        self._vector_cache = {}
        self._active = ()
        self._revision = hashlib.sha256(b'[]').hexdigest()
        self.last_error = None
        self.failure_count = 0
        self.last_failure = None

    @contextlib.contextmanager
    def locked(self):
        import fcntl
        with self._thread_lock, (self.root / '.lock').open('rb' if self.read_only else 'a+b') as stream:
            fcntl.flock(stream, fcntl.LOCK_SH if self.read_only else fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _index(self, directory):
        path = directory / 'index.json'
        if not path.exists():
            if directory == self.verified and any(p.is_dir() and len(p.name) == 32
                                                  for p in directory.iterdir()):
                raise StoreCorrupt('Missing verified index; preserving existing evidence')
            return []
        try:
            value = json.loads(path.read_text('utf-8'))
            if value['season'] != self.season or value['version'] != 1 or not isinstance(value['entries'], list):
                raise ValueError('invalid season/schema')
            rows = value['entries']
            if any(not isinstance(r,dict) or not isinstance(r.get('id'),str) for r in rows):
                raise ValueError('invalid row')
            if len({r['id'] for r in rows}) != len(rows):
                raise ValueError('duplicate id')
            if any(len(r['id']) != 32 or any(c not in '0123456789abcdef' for c in r['id']) for r in rows):
                raise ValueError('invalid id')
            if directory == self.verified:
                files = {'vector.npy','evidence.json','unit.png','before_unit.png','panel.png','previous_panel.png',
                         'selection.png','previous_selection.png'}
                for r in rows:
                    if (not isinstance(r.get('name'),str) or not isinstance(r.get('source'),str)
                            or not isinstance(r.get('revoked'),bool) or not isinstance(r.get('hashes'),dict)
                            or 'vector.npy' not in r['hashes'] or 'evidence.json' not in r['hashes']
                            or any(k not in files or not isinstance(v,str) or len(v)!=64 for k,v in r['hashes'].items())):
                        raise ValueError('invalid verified entry')
            return rows
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise StoreCorrupt(f'{type(exc).__name__}: {exc}; Invalid learning index: {path}') from exc

    def _save(self, directory, rows):
        atomic_json(directory / 'index.json', dict(version=1, season=self.season, entries=rows))

    def _evidence(self, directory, sid):
        p = directory / sid
        try:
            metadata = json.loads((p / 'evidence.json').read_text('utf-8'))
            if metadata.get('season') != self.season:
                raise ValueError('cross-season evidence')
            return metadata
        except (ValueError, OSError) as exc:
            raise StoreCorrupt(f'Invalid evidence: {p}') from exc

    def _commit_dir(self, directory, sid, metadata, images, fp=None):
        staging = directory / ('.' + sid)
        staging.mkdir(exist_ok=False)
        try:
            for key, image in images.items():
                if key not in ('unit', 'before_unit', 'panel', 'previous_panel', 'selection', 'previous_selection'):
                    raise ValueError('unsupported evidence image')
                if image is not None and not cv2.imwrite(str(staging / (key+'.png')), image):
                    raise OSError('Failed to save evidence image')
            if fp is not None:
                with (staging / 'vector.npy').open('wb') as f:
                    np.save(f, fp, allow_pickle=False)
                    f.flush()
                    os.fsync(f.fileno())
            atomic_json(staging / 'evidence.json', metadata)
            hashes = {p.name: file_hash(p) for p in staging.iterdir() if p.is_file()}
            os.replace(staging, directory / sid)
            return hashes
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def record(self, metadata, images, *, status='pending'):
        if self.read_only:
            raise PermissionError('read-only verified reference')
        if status not in ('pending', 'rejected', 'conflict'):
            raise ValueError('invalid review status')
        if metadata.get('season') not in (None,self.season):
            raise ValueError('cross-season evidence')
        image = images.get('unit')
        ch = image_hash(image) if image is not None else image_hash(images['panel'])
        metadata = dict(metadata, season=self.season, crop_hash=ch, status=status)
        if metadata.get('source') not in ('panel', 'manual', 'legacy', 'record'):
            raise ValueError('invalid provenance')
        # Reason/name are included so a later confirmed event isn't lost behind an
        # earlier pending record of the same crop. Frame-only repeats deduplicate.
        dedup = [ch, metadata.get('source'), metadata.get('name'), metadata.get('reason')]
        with self.locked():
            rows = self._index(self.queue)
            # A crash after committing the queue index but before deleting evicted
            # blobs can leave unreferenced review dirs. They are never verified.
            indexed = {r['id'] for r in rows}
            for path in self.queue.iterdir():
                if path.is_dir() and path.name not in indexed:
                    shutil.rmtree(path)
            for row in rows:
                if row['dedup'] == dedup:
                    return row['id']
            sid = uuid.uuid4().hex
            self._commit_dir(self.queue, sid, metadata, images)
            rows.append(dict(id=sid, status=status, name=metadata.get('name'), dedup=dedup))
            def size():
                return sum(p.stat().st_size for row in rows for p in (self.queue/row['id']).rglob('*') if p.is_file())
            removed = []
            while rows and (len(rows) > self.max_pending or size() > self.max_bytes):
                removed.append(rows.pop(0)['id'])
            try:
                self._save(self.queue, rows)
            except BaseException:
                shutil.rmtree(self.queue / sid, ignore_errors=True)
                raise
            for old in removed:
                shutil.rmtree(self.queue / old, ignore_errors=True)
            return sid if any(r['id'] == sid for r in rows) else None

    @staticmethod
    def _stat(path):
        s = path.stat()
        return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_mode)

    def _load_vector(self, row):
        """Load a new/changed sample or retry a failed read, with the process lock."""
        directory = self.verified / row['id']
        if any(file_hash(directory/name) != digest for name, digest in row['hashes'].items()):
            raise StoreCorrupt(f'Changed verified evidence: {row["id"]}')
        metadata = self._evidence(self.verified, row['id'])
        if (metadata.get('name') != row['name'] or metadata.get('source') != row['source']
                or metadata.get('crop_hash') != row.get('crop_hash')
                or metadata.get('status') != 'verified' or not metadata.get('confirmation')):
            raise StoreCorrupt(f'Inconsistent verified identity: {row["id"]}')
        fp = np.load(directory/'vector.npy', allow_pickle=False)
        if not isinstance(fp, np.ndarray) or fp.ndim != 1 or not fp.size or not np.isfinite(fp).all():
            raise StoreCorrupt(f'Invalid verified vector: {row["id"]}')
        # A bytes-backed array cannot have WRITEABLE restored by a caller.
        return np.frombuffer(fp.tobytes(), dtype=fp.dtype)

    def _matching_rows(self, rows):
        eligible = [r for r in rows if not r.get('revoked') and r.get('name') in self.pool
                    and r.get('source') in ('panel', 'manual')]
        if len(eligible) > self._cache_limit:
            raise StoreCorrupt('Verified index exceeds current pool capacity')
        return eligible

    def _load_vectors(self, rows):
        loaded, effective, cache, errors = [], [], {}, []
        eligible = self._matching_rows(rows)
        for row in eligible:
            directory = self.verified / row['id']
            try:
                signature = (json.dumps(row, sort_keys=True), tuple(
                    (name, self._stat(directory/name)) for name in sorted(row['hashes'])))
                old = self._vector_cache.get(row['id'])
                if old is not None and old[0] == signature:
                    _, fp, error = old
                else:
                    fp, error = None, None
                    try:
                        fp = self._load_vector(row)
                        if signature[1] != tuple((name, self._stat(directory/name)) for name in sorted(row['hashes'])):
                            raise StoreCorrupt('Evidence changed while loading')
                    except Exception as exc:
                        fp, error = None, f'{type(exc).__name__}: {exc}'
                if fp is not None:
                    # Cache successful validation only. A transient read/load
                    # failure can disappear without changing any asset stat.
                    cache[row['id']] = signature, fp, error
                    loaded.append((row['name'], fp))
                    effective.append((row['id'], row['name'], row['hashes']['vector.npy']))
                if error:
                    errors.append(error)
            except OSError as exc:
                # Missing/unreadable assets immediately remove the vector; never
                # fall back to its previous cached identity.
                errors.append(f'{type(exc).__name__}: {exc}')
        self._vector_cache = cache
        self._active = tuple(loaded)
        self._revision = hashlib.sha256(json.dumps(effective, sort_keys=True).encode()).hexdigest()
        self.last_error = '; '.join(errors) or None
        return list(self._active)

    def _refresh(self):
        try:
            with self.locked():
                path = self.verified / 'index.json'
                try:
                    fingerprint = self._stat(path)
                except FileNotFoundError:
                    fingerprint = ('missing',)
                if fingerprint != self._cache_index_stat:
                    # Only mark an index loaded after the whole operation
                    # succeeds, including any wrapped or unexpected read error.
                    self._cache_index_stat = None
                    self._cache_rows, self._cache_index_error = (), None
                    try:
                        # Do not keep historical revoked/out-of-season rows in
                        # the hot cache, even though the audit index retains them.
                        self._cache_rows = tuple(self._matching_rows(self._index(self.verified)))
                        if fingerprint != (('missing',) if not path.exists() else self._stat(path)):
                            raise StoreCorrupt('Index changed while loading')
                        self._cache_index_stat = fingerprint
                    except (ValueError, OSError) as exc:
                        # StoreCorrupt may wrap an I/O cause. Retrying all failed
                        # index reads preserves recovery without classifying that
                        # cause or requiring users to touch a healthy file.
                        self._cache_index_error = f'{type(exc).__name__}: {exc}'
                if self._cache_index_error:
                    raise StoreCorrupt(self._cache_index_error)
                self._load_vectors(self._cache_rows)
                if self.last_error:
                    self.failure_count += 1
                    self.last_failure = self.last_error
        except Exception as exc:
            self._vector_cache.clear()
            self._active = ()
            self._revision = hashlib.sha256(b'[]').hexdigest()
            self.last_error = f'{type(exc).__name__}: {exc}'
            self.failure_count += 1
            self.last_failure = self.last_error

    def active_vectors(self):
        with self._thread_lock:
            self._refresh()
            # Fresh array metadata over immutable bytes: callers cannot change
            # cached data, shape or dtype, even through ndarray.base.
            return [(name, np.frombuffer(fp.base, dtype=fp.dtype)) for name, fp in self._active]

    def revision(self):
        # Include effective vector loss from damaged evidence, not only index
        # edits, so UnitTracker also invalidates its remembered/group identities.
        with self._thread_lock:
            self._refresh()
            return self._revision

    def promote(self, record_id, fp, *, existing=(), check=None):
        if self.read_only:
            raise PermissionError('read-only verified reference')
        fp = np.asarray(fp, dtype=np.float32)
        if fp.ndim != 1 or not fp.size or not np.isfinite(fp).all():
            raise ValueError('invalid feature')
        with self.locked():
            if check:
                check()
            review = self._index(self.queue)
            if not any(r['id'] == record_id for r in review):
                return {'status': 'rejected', 'reason': 'missing_review_record'}
            metadata = self._evidence(self.queue, record_id)
            name = metadata.get('name')
            if (name not in self.pool or metadata.get('source') not in ('panel', 'manual')
                    or not metadata.get('confirmation') or metadata['status'] != 'pending'):
                return {'status': 'rejected', 'reason': 'unverified_evidence'}
            rows = self._index(self.verified)
            active = [r for r in rows if not r.get('revoked')]
            def reject(reason):
                metadata.update(status='rejected',reason=reason)
                atomic_json(self.queue/record_id/'evidence.json',metadata)
                for entry in review:
                    if entry['id']==record_id:entry['status']='rejected'
                self._save(self.queue,review)
                return {'status':'rejected','reason':reason}
            if any(r.get('revoked') and r['crop_hash']==metadata['crop_hash'] for r in rows):
                return reject('revoked_crop')
            for row in active:
                if row['crop_hash'] == metadata['crop_hash']:
                    if row['name'] == name:
                        return {'status':'duplicate','reason':'crop_already_verified','id':row['id']}
                    metadata.update(status='conflict',reason='crop_already_verified',conflicting_sample=row['id'])
                    atomic_json(self.queue/record_id/'evidence.json',metadata)
                    for review_row in review:
                        if review_row['id']==record_id:review_row['status']='conflict'
                    self._save(self.queue,review)
                    return {'status':'conflict','reason':'crop_already_verified','id':row['id']}
            # Treat legacy vectors only as a veto, never as a source of new labels.
            # Reuse Gallery's strict threshold and different-name margin. A
            # vector outside the match threshold cannot be an existing identity.
            verified = self._load_vectors(rows)
            if self.last_error:
                raise StoreCorrupt(self.last_error)
            vectors = (*tuple(existing), *verified)
            same = min((fp_distance(fp,v) for n,v in vectors if n == name and v.shape == fp.shape),
                       default=float('inf'))
            conflicts = [dict(name=n, distance=fp_distance(fp, v)) for n,v in vectors
                         if n != name and fp.shape == v.shape and fp_distance(fp,v) < MATCH_THRESHOLD
                         and fp_distance(fp,v)-same < MATCH_MARGIN]
            if conflicts:
                metadata.update(status='conflict', reason='different_name_feature_conflict', conflicts=conflicts)
                atomic_json(self.queue/record_id/'evidence.json', metadata)
                for row in review:
                    if row['id'] == record_id:
                        row['status'] = 'conflict'
                self._save(self.queue, review)
                return {'status': 'conflict', 'reason': metadata['reason'], 'conflicts': conflicts}
            if sum(r['name'] == name for r in active) >= 3:
                return reject('hero_capacity')
            sid = uuid.uuid4().hex
            source = self.queue / record_id
            images = {}
            from .capture import load_image
            for key in ('unit', 'before_unit', 'panel', 'previous_panel', 'selection', 'previous_selection'):
                if (source/(key+'.png')).exists():
                    images[key] = load_image(source/(key+'.png'))
            metadata = dict(metadata, status='verified', reason='panel_confirmed' if metadata['source']=='panel' else 'manual_confirmed',
                            original_record=record_id)
            hashes = self._commit_dir(self.verified, sid, metadata, images, fp)
            rows.append(dict(id=sid, name=name, source=metadata['source'], crop_hash=metadata['crop_hash'],
                             revoked=False, hashes=hashes))
            if check:
                check()  # after image I/O, immediately before the atomic commit
            self._save(self.verified, rows)
            return {'status': 'verified', 'id': sid, 'reason': metadata['reason']}

    def revoke(self, sid):
        if self.read_only:
            raise PermissionError('read-only verified reference')
        with self.locked():
            rows = self._index(self.verified)
            for row in rows:
                if row['id'] == sid:
                    row['revoked'] = True
                    self._save(self.verified, rows)
                    return True
            return False

    def entries(self, kind='verified'):
        with self.locked():
            return self._index(self.verified if kind == 'verified' else self.queue)


@dataclass(frozen=True, slots=True)
class CaptureWindow:
    pid: int
    x: float
    y: float
    width: float
    height: float
    window_id: int


@dataclass(frozen=True)
class LearningFrame:
    image: np.ndarray = field(repr=False, compare=False)
    capture_id: str
    monotonic: float
    window_id: int
    phase: str
    viewing: str | None
    stage: str | None
    pixel_hash: str
    source_hash: str | None
    receipt: object = field(repr=False, compare=False)
    sequence: int
    window: object | None = field(default=None, repr=False, compare=False)

    @property
    def size(self):
        return self.image.shape[1], self.image.shape[0]

    def metadata(self):
        return dict(capture_id=self.capture_id, monotonic=self.monotonic, window_id=self.window_id,
                    size=self.size, phase=self.phase, viewing=self.viewing, stage=self.stage,
                    pixel_hash=self.pixel_hash, source_hash=self.source_hash, sequence=self.sequence,
                    window=asdict(self.window) if self.window is not None else None)


@dataclass(frozen=True)
class LearningSample:
    frame: LearningFrame
    state: dict
    timings: dict = field(default_factory=dict, compare=False)


class FrameSource:
    """Receipts are issued once per capture invocation, or verified archived event.

    The controller rejects changed IDs/times/images on an existing receipt. A new
    paused-game capture may legitimately contain identical pixels. Offline callers
    must use sealed image/event files, not fabricate timestamps for a cached image.
    """
    def __init__(self):
        self._token = object()
        self._issued = OrderedDict()
        self._sequence = 0
        self._last_time = float('-inf')
        self._used_archive_ids = set()
        self._pending_captures = {}

    def _issue(self, image, capture_id, monotonic, window_id, phase, viewing, stage, source_hash=None, window=None):
        if monotonic <= self._last_time or not window_id or capture_id in self._issued:
            raise ValueError('non-fresh capture receipt')
        image = image.copy()
        image.setflags(write=False)
        self._sequence += 1
        frame = LearningFrame(image, capture_id, monotonic, int(window_id), phase, viewing, stage,
                              image_hash(image), source_hash, self._token, self._sequence, window)
        self._issued[capture_id] = frame
        while len(self._issued) > 8:
            self._issued.popitem(last=False)
        self._last_time = monotonic
        return frame

    def grab(self, capturer):
        """Call the existing capturer once. Arbitrary cached images/IDs aren't inputs."""
        image, window = capturer.grab()
        return image, window, self._capture_ticket(image, window)

    def _capture_ticket(self, image, window):
        """Trusted successful-grab boundary, also used by PyEye's sole capturer.

        Kept separate so receipt failure cannot discard an already captured main
        image. Callers cannot supply receipt IDs or sampling timestamps.
        """
        ticket = object()
        self._pending_captures.clear()
        self._pending_captures[ticket] = (image.copy(), image_hash(image), time.monotonic(), window.window_id,
                                         self._window(window))
        return ticket

    def reset(self):
        """Invalidate all outstanding receipts after a learning failure."""
        self._issued.clear()
        self._pending_captures.clear()

    def frame_from_capture(self, ticket, phase, viewing=None, stage=None):
        try:
            image, digest, observed_at, window_id, window = self._pending_captures.pop(ticket)
        except KeyError as exc:
            raise ValueError('no fresh capture invocation') from exc
        if digest != image_hash(image):
            raise ValueError('capture pixels changed')
        return self._issue(image, uuid.uuid4().hex, observed_at, window_id, phase, viewing, stage, window=window)

    @staticmethod
    def _window(window):
        names = ('pid','x','y','width','height','window_id')
        if isinstance(window,dict):
            values = {k:window.get(k) for k in names}
        else:
            values = {k:getattr(window,k,None) for k in names}
        return CaptureWindow(**values) if all(v is not None for v in values.values()) else None

    def archived(self, directory, event, phase, viewing=None, stage=None):
        from .capture import load_image
        directory = Path(directory).resolve()
        events_path = directory / 'events.jsonl'
        rows = [json.loads(l) for l in events_path.read_text('utf-8').splitlines()]
        if event not in rows or event.get('kind') != 'frame':
            raise ValueError('no archived capture receipt')
        path = (directory / event['file']).resolve()
        if path.parent != directory or file_hash(path) != event['sha256']:
            raise ValueError('archive path/hash mismatch')
        archive_id = file_hash(events_path) + ':' + event['file']
        if archive_id in self._used_archive_ids:
            raise ValueError('archived frame reused')
        self._used_archive_ids.add(archive_id)
        image = load_image(path)
        if list(image.shape[:2][::-1]) != [int(event['window']['width']), int(event['window']['height'])]:
            raise ValueError('archive window mismatch')
        return self._issue(image, archive_id, float(event['monotonic']), event['window']['window_id'],
                           phase, viewing, stage, event['sha256'], window=self._window(event['window']))

    def validates(self, frame):
        return (frame.receipt is self._token and self._issued.get(frame.capture_id) is frame
                and frame.pixel_hash == image_hash(frame.image))


@dataclass(frozen=True)
class ClickTransaction:
    before: LearningFrame
    target: tuple[float, float]
    clicked_at: float
    window_id: int


def isolated_crop(unit, units, image):
    """Veto overlapping detected-unit crops or panel pixels, without changing
    the existing unit_crop feature geometry. No nearest-unit assumption.
    """
    h,w=image.shape[:2]
    def bounds(u):
        # Mirror unit_crop's existing offsets and dimensions, including its
        # legacy six-pixel top offset; this never changes the feature itself.
        b=u.bar
        x0=math.floor(b.min_x-.05*b.width);y0=b.max_y+6
        return max(0,x0),y0,min(w,math.ceil(b.min_x-.05*b.width+1.15*b.width)),min(h,math.ceil(y0+1.3*b.width))
    if unit.bar is None or unit.crop is None:return False
    x0,y0,x1,y1=bounds(unit)
    if x1>PANEL_REGION.x*w:return False
    for other in units:
        if other is unit or other.bar is None:continue
        a,b,c,d=bounds(other)
        if min(x1,c)>max(x0,a) and min(y1,d)>max(y0,b):return False
    return True


class UnitLearningController:
    def __init__(self, reader, store, source, *, mode='record'):
        if mode not in ('record', 'panel'):
            raise ValueError('unsupported non-click mode')
        self.reader, self.store, self.source, self.mode = reader, store, source, mode
        self.panel = PanelReader(reader.dict, reader.ocr)
        self.previous = None
        self._last_processed = 0
        self._record_seen = OrderedDict()
        self.reader.units.gallery.verified_store = store
        self._lock = threading.RLock()
        self._last_warning_at = None
        self._suppressed_warnings = 0
        self._error_count = 0
        self._seen_store_failures = store.failure_count

    def reset(self):
        """Clear the confirmation chain and all outstanding capture receipts."""
        with self._lock:
            self.previous = None
            self.source.reset()

    def fail(self, exc, operation='process'):
        """Fail closed for optional learning, without permanently disabling it."""
        with self._lock:
            self.reset()
            self._error_count += 1
            now = time.monotonic()
            if self._last_warning_at is None or now-self._last_warning_at >= 30:
                log.warning('Unit learning failed (%s, %s; suppressed=%d): %s', operation,
                            type(exc).__name__, self._suppressed_warnings, str(exc)[:300])
                self._last_warning_at, self._suppressed_warnings = now, 0
            else:
                self._suppressed_warnings += 1
            return dict(status='error', reason='optional_learning_failed', operation=operation,
                        error_type=type(exc).__name__, error=str(exc)[:500], error_count=self._error_count)

    def process(self, frame, transaction=None, *, check=None):
        with self._lock:
            try:
                # Matching fails closed on corrupt/unreadable assets. Surface the
                # diagnostic here too, before any confirmation/record can proceed.
                self.store.active_vectors()
                # A transient fault during main Gallery matching may already
                # have recovered by this call. It must still break the old
                # confirmation chain, rather than silently bridging the fault.
                failed = self.store.failure_count != self._seen_store_failures
                self._seen_store_failures = self.store.failure_count
                if self.store.last_error or failed:
                    raise StoreCorrupt(self.store.last_error or self.store.last_failure)
                if check:
                    check()
                return self._process(frame, transaction, check)
            except LearningCancelled as exc:
                self.reset()
                return dict(status='cancelled',reason=str(exc))
            except Exception as exc:
                return self.fail(exc)

    def _process(self, frame, transaction, check=None):
        if not self.source.validates(frame) or frame.sequence <= self._last_processed:
            self.previous = None
            return {'status': 'rejected', 'reason': 'stale_or_invalid_capture'}
        self._last_processed = frame.sequence
        if (str(self.reader.units.gallery.season) != self.store.season or not self.store.pool
                or (self.mode == 'panel' and (frame.phase != 'planning' or frame.viewing is not None))):
            self.previous = None
            return {'status': 'rejected', 'reason': 'unsupported_scene_or_season'}
        tracker = self.reader.units
        _, units = tracker.detect(frame.image)
        if self.mode == 'record':
            records = []
            deduplicated = 0
            for u in units:
                if check:
                    check()
                if u.crop is None:
                    continue
                digest = image_hash(u.crop)
                if digest in self._record_seen:
                    self._record_seen.move_to_end(digest)
                    deduplicated += 1
                    continue
                fp = feature_print(u.crop)
                ranked = sorted((fp_distance(fp, v), n) for n, v in tracker.gallery.vectors()
                                if fp is not None and fp.shape == v.shape)
                match = tracker.gallery.match(fp) if fp is not None else None
                if match is None or match[1] >= MATCH_THRESHOLD-MATCH_MARGIN:
                    if check:
                        check()
                    sid = self.store.record(dict(source='record', reason='unknown_or_low_confidence',
                        frame=frame.metadata(), x=u.x, y=u.y, zone=u.zone, slot=u.slot,
                        known_name=match[0] if match else None, distances=[dict(name=n, distance=d) for d,n in ranked[:6]],
                        inference_candidates=tracker.learning_candidates), {'unit':u.crop, 'panel':crop(frame.image, PANEL_REGION)})
                    records.append(sid)
                    self._record_seen[digest] = sid
                    while len(self._record_seen) > 512:
                        self._record_seen.popitem(last=False)
            self.previous = None
            return dict(status='recorded', records=records, deduplicated=deduplicated, promoted=0)
        panel = self.panel.read(frame.image)
        unit, binding, reason = bind_selection(frame.image, units, transaction.target if transaction else None)
        result = dict(status='pending', reason=reason, panel=panel.to_json(), binding=binding,
                      frame=frame.metadata())
        if not panel.valid:
            reason = panel.reason
        elif unit is None:
            pass
        elif transaction is not None:
            before = transaction.before
            if (not self.source.validates(before) or transaction.window_id != frame.window_id
                    or before.window_id != frame.window_id or before.size != frame.size
                    or before.phase != frame.phase or before.viewing != frame.viewing or before.stage != frame.stage
                    or not before.monotonic <= transaction.clicked_at < frame.monotonic
                    or frame.monotonic-before.monotonic > 10):
                reason = 'stale_or_cross_scene_transaction'
        valid = panel.valid and unit is not None and reason == 'unique_selection'
        if valid and not isolated_crop(unit,units,frame.image):
            valid,reason=False,'occluded_or_panel_crop'
        clean = unit.crop if unit else None
        before_crop = None
        before_after_distance = None
        if valid and transaction is not None:
            _, previous_units = tracker.detect(transaction.before.image)
            counterparts = [u for u in previous_units if u.zone == unit.zone and u.slot == unit.slot
                            and abs(u.x-unit.x)<.012 and abs(u.y-unit.y)<.02]
            if (len(counterparts) != 1 or not isolated_crop(counterparts[0],previous_units,transaction.before.image)):
                reason, valid = 'no_clean_before_unit', False
            else:
                before_crop = counterparts[0].crop
                a, b = feature_print(before_crop), feature_print(unit.crop)
                before_after_distance = fp_distance(a,b) if a is not None and b is not None else None
                if before_after_distance is None or before_after_distance >= TRACK_THRESHOLD:
                    reason, valid = 'before_after_unit_mismatch', False
                else:
                    clean = before_crop
        # A panel from one selected unit cannot confirm a later different target.
        key = None if unit is None else (unit.zone, unit.slot, round(unit.x,3), round(unit.y,3))
        prev = self.previous
        stable = (valid and prev is not None and prev['signature'] == panel.signature and prev['target'] == key
                  and prev['frame'].window_id == frame.window_id and prev['frame'].size == frame.size
                  and prev['frame'].phase == frame.phase and prev['frame'].viewing == frame.viewing
                  and prev['frame'].stage == frame.stage and frame.sequence == prev['frame'].sequence+1
                  and .10 <= frame.monotonic-prev['frame'].monotonic <= 5.)
        images = {'unit': clean, 'panel': crop(frame.image,PANEL_REGION), 'before_unit':before_crop}
        selected_rect = next((d['rect'] for d in binding if unit is not None and d['selected']
                              and d['x']==unit.x and d['y']==unit.y),None)
        if selected_rect:
            x0,y0,x1,y1=selected_rect
            images['selection'] = frame.image[y0:y1,x0:x1]
        if stable:
            images['previous_panel'] = crop(prev['frame'].image,PANEL_REGION)
            x0,y0,x1,y1=prev['selection_rect']
            images['previous_selection'] = prev['frame'].image[y0:y1,x0:x1]
        reason = 'two_fresh_consistent_panels' if stable else ('awaiting_second_capture' if valid else reason)
        if check:
            check()  # panel OCR/feature extraction must not bridge user takeover
        metadata = dict(source='panel', name=panel.name, cost=panel.cost, traits=panel.traits,
                        target=dict(x=unit.x,y=unit.y,zone=unit.zone,slot=unit.slot) if unit else None,
                        frame=frame.metadata(), before=transaction.before.metadata() if transaction else None,
                        previous=prev['frame'].metadata() if stable else None,
                        raw_ocr=panel.raw, previous_raw_ocr=prev['raw_ocr'] if stable else None,
                        before_after_distance=before_after_distance,
                        binding=binding, reason=reason, confirmation=bool(stable))
        sid = self.store.record(metadata, images, status='pending' if valid else 'rejected')
        self.previous = dict(signature=panel.signature, target=key, frame=frame,
                             selection_rect=selected_rect,raw_ocr=panel.raw) if valid else None
        result.update(status='pending' if valid else 'rejected', reason=reason, record=sid, target=metadata['target'])
        if stable and sid:
            fp = feature_print(clean) if clean is not None else None
            if fp is not None:
                result.update(self.store.promote(sid, fp, existing=(*tracker.gallery.items, *tracker.gallery.refs),check=check))
        return result
