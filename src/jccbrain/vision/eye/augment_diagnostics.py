"""Bounded passive evidence from an already captured/parsed game-window frame.

No OCR, capture, input, database, candidate matcher or ownership inference. Only
small ROI arrays are buffered (two frames); encoding/I/O happens inside events.
Unknown ownership is an observation reason, never permission to learn augments.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import stat
import time
import uuid

import cv2

from jccbrain.vision.eye.image import Region, crop, load_regions

log = logging.getLogger(__name__)
FORMAT = 'jccbrain.augment-diagnostics.v1'
MAX_EVENTS, MAX_FRAMES = 12, 5
EVENT_BYTES, SESSION_BYTES, TOTAL_BYTES = 512*1024, 8*1024*1024, 100*1024*1024
MAX_SESSIONS, META_BYTES = 10, 64*1024
COOLDOWN, SAMPLE_INTERVAL, SUSPECT_SECONDS = 10., 1., 2.
ROIS = (('stage',Region(.25,0,.50,.10)), ('permanent',Region(.24,.22,.18,.10)),
        ('right',Region(.80,.09,.20,.62)))
SESSION_RE = re.compile(r'session-[0-9a-f]{32}')
FILE_RE = re.compile(r'(roi-[0-9a-f]{64}\.jpg|event-[0-9]{2}\.json)')


def _json(value):
    return (json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n').encode()


def _text(value):
    return str(value)[:256] if value is not None else None


def _stage(value):
    match = re.fullmatch(r'([1-9][0-9]?)-([1-9][0-9]?)',str(value or ''))
    return tuple(map(int,match.groups())) if match else None


class EvidenceStore:
    """Event-only disk work. Flat, marked sessions; foreign/symlink = refuse.

    Root flock serializes quota accounting across providers/processes. Deletion
    requires the exact marker, a complete session and no unlisted children.
    """
    def __init__(self, root):
        self.root = Path(root)
        self.session = None
        self.files = {}
        self.complete = False
        self.fd = None

    @staticmethod
    def _regular(path):
        st = path.lstat()
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            raise OSError('unsafe diagnostic file')
        return st

    def _scan(self):
        sessions, total = [], 0
        for path in self.root.iterdir():
            if path.name in ('format.json','.lock'):
                total += self._regular(path).st_size
                continue
            if path.is_symlink() or not path.is_dir() or not SESSION_RE.fullmatch(path.name):
                raise OSError('unmanaged diagnostic entry; not removed')
            marker = path/'manifest.json'
            if self._regular(marker).st_size > META_BYTES:
                raise OSError('oversized diagnostic manifest')
            data = json.loads(marker.read_text())
            if (set(data)!={'format','session','complete','files'} or type(data.get('complete')) is not bool
                    or data.get('format') != FORMAT or data.get('session') != path.name):
                raise OSError('foreign diagnostic manifest')
            files = data.get('files')
            if not isinstance(files,dict) or len(files)>MAX_EVENTS*(MAX_FRAMES*3+1):
                raise OSError('invalid diagnostic file list')
            if {p.name for p in path.iterdir()} != set(files)|{'manifest.json'}:
                raise OSError('unlisted diagnostic entry; not removed')
            size = marker.stat().st_size
            for name, expected in files.items():
                if not FILE_RE.fullmatch(name):raise OSError('foreign diagnostic filename')
                st = self._regular(path/name)
                if st.st_size != expected:raise OSError('changed diagnostic file')
                size += st.st_size
            sessions.append((marker.stat().st_mtime,path,data.get('complete') is True,size))
            total += size
        return sorted(sessions),total

    @contextmanager
    def locked(self):
        if self.root.is_symlink():raise OSError('diagnostic root is a symlink')
        self.root.mkdir(parents=True,exist_ok=True)
        rootfd = os.open(self.root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        lockfd = None
        try:
            lockfd = os.open('.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600,dir_fd=rootfd)
            if os.fstat(lockfd).st_nlink != 1:raise OSError('unsafe diagnostic lock')
            fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            marker = self.root/'format.json'
            if not marker.exists():
                # Never claim ownership of an existing foreign directory.
                if {p.name for p in self.root.iterdir()} != {'.lock'}:
                    raise OSError('unmarked foreign diagnostics directory')
                fd=os.open('format.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=rootfd)
                with os.fdopen(fd,'wb') as stream:stream.write(_json({'format':FORMAT}))
            if self._regular(marker).st_size>1024 or json.loads(marker.read_text()).get('format')!=FORMAT:
                raise OSError('foreign diagnostic root marker')
            yield
        finally:
            try:
                if lockfd is not None:os.close(lockfd)
            finally:
                os.close(rootfd)

    def _manifest(self, files=None, complete=None):
        return _json(dict(format=FORMAT,session=self.session.name,
                          complete=self.complete if complete is None else complete,
                          files=self.files if files is None else files))

    def start(self):
        sessions,total = self._scan()
        while len(sessions)>=MAX_SESSIONS or total+EVENT_BYTES+META_BYTES>TOTAL_BYTES:
            victim=next((s for s in sessions if s[2]),None)
            if victim is None:raise OSError('diagnostic session/total quota')
            _,path,_,size=victim
            # _scan validated every flat child: no recursive deletion/traversal.
            for child in path.iterdir():child.unlink()
            path.rmdir();sessions.remove(victim);total-=size
        if shutil.disk_usage(self.root).free < EVENT_BYTES+META_BYTES:
            raise OSError('insufficient diagnostic disk space')
        self.session=self.root/('session-'+uuid.uuid4().hex)
        self.session.mkdir()
        self.fd=os.open(self.session,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        self.files={};self.complete=False
        try:self._put('manifest.json',self._manifest())
        except Exception:
            os.close(self.fd);self.fd=None
            raise

    def _put(self,name,data):
        if name!='manifest.json' and not FILE_RE.fullmatch(name):
            raise OSError('invalid diagnostic filename')
        fd=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_NOFOLLOW,0o600,dir_fd=self.fd)
        with os.fdopen(fd,'wb') as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode) or os.fstat(stream.fileno()).st_nlink!=1:
                raise OSError('unsafe diagnostic output')
            os.ftruncate(stream.fileno(),0)
            stream.write(data)

    def write(self,name,data):
        if not FILE_RE.fullmatch(name):raise OSError('invalid diagnostic filename')
        if len(data)>EVENT_BYTES:raise OSError('diagnostic object quota')
        _,total=self._scan()
        files=dict(self.files);old=files.get(name,0);files[name]=len(data)
        marker=self._manifest(files)
        if (len(marker)>META_BYTES or sum(files.values())+len(marker)>SESSION_BYTES
                or total-old+len(data)+len(marker)-len(self._manifest())>TOTAL_BYTES):
            raise OSError('diagnostic byte quota')
        self._put(name,data);self.files=files
        self._put('manifest.json',marker)

    def finish(self):
        if self.fd is not None:
            try:
                with self.locked():
                    self._scan()
                    self._put('manifest.json',self._manifest(complete=True))
                    self.complete=True
            finally:
                os.close(self.fd);self.fd=None


class AugmentDiagnostics:
    def __init__(self, root, *, clock=time.monotonic, utc=None):
        self.root=Path(root);self.clock=clock
        self.utc=utc or (lambda:datetime.now(timezone.utc).isoformat())
        regions=load_regions();a,b=regions['augment_names'],regions['augment_desc']
        self.choice=Region(a.x,a.y,a.w,b.y+b.h-a.y)
        self.counts=dict(events=0,frames=0,errors=0,skipped=0)
        self.latest=None;self.last_error=None;self._logged=set()
        self.closed=False
        self._clear()

    def _clear(self):
        self.ring=deque(maxlen=2);self.active=None;self.store=None
        self.window=None;self.previous_stage=None;self.previous_phase=None
        self.names=frozenset();self.last_sample=float('-inf');self.last_event=float('-inf')
        self.suspect_since=None;self.suspect_fired=False;self.events=0;self.ended=False;self.disabled=False

    def status(self):
        return dict(self.counts,last_path=self.latest,last_error=self.last_error)

    def close(self):
        """Terminal, idempotent close; no-event close creates no directory.

        A failed finish retains the incomplete evidence and releases descriptors.
        Provider restart creates a fresh observer rather than reviving this one.
        """
        if self.closed:return
        self.closed=True
        store=self.store
        try:
            if store is not None:store.finish()
        except Exception as exc:
            self.counts['errors']=min(9999,self.counts['errors']+1)
            self._error(exc)
        finally:
            try:
                if store is not None and store.fd is not None:
                    fd=store.fd;store.fd=None
                    os.close(fd)
            except OSError as exc:
                self._error(exc)
            finally:
                self._clear()

    def _error(self,reason):
        self.last_error=str(reason)[:160]
        self.counts['skipped']=min(9999,self.counts['skipped']+1)
        key=type(reason).__name__+str(reason)[:80]
        if len(self._logged)<8 and key not in self._logged:
            self._logged.add(key);log.warning('海克斯诊断已跳过：%s',self.last_error)

    def _snapshot(self,image,d,window,now):
        regions=list(ROIS)
        if d.get('phase')=='augment':regions[1]=('choice',self.choice)
        meta=dict(utc=self.utc(),monotonic=now,window_id=window.window_id,
                  size=[int(window.width),int(window.height)],stage=_text(d.get('stage')),
                  phase=_text(d.get('phase')),viewing=dict(name=_text(d['viewing'].get('name')),
                     slot=d['viewing'].get('slot')) if isinstance(d.get('viewing'),dict) else None,
                  players_count=len(d.get('players') or []),
                  augments=[dict(name=_text(a.get('name')),desc=_text(a.get('desc')))
                            for a in (d.get('augments') or [])[:3] if isinstance(a,dict)],
                  augments_taken=[_text(n) for n in (d.get('augments_taken') or [])[:12]],
                  ownership='unknown_owner' if len(d.get('players') or [])<5 or d.get('viewing') else 'unverified',
                  ownership_note='diagnostic only; absence of an eye/three icons is not identity evidence')
        rois=[]
        for name,region in regions:
            patch=crop(image,region)
            if patch is not None:
                rois.append((dict(name=name,region=[region.x,region.y,region.w,region.h],rect=region.px(image)),patch.copy()))
        return meta,rois

    def _save_frame(self,snapshot,role):
        meta,rois=snapshot;frame=dict(meta,role=role,rois=[])
        for region,patch in rois:
            digest=hashlib.sha256(str(patch.shape).encode()+patch.tobytes()).hexdigest()
            name='roi-'+digest+'.jpg'
            size=self.store.files.get(name)
            encoded=None
            if size is None:
                okay,data=cv2.imencode('.jpg',patch,[cv2.IMWRITE_JPEG_QUALITY,60])
                if not okay:raise OSError('diagnostic JPEG encoding failed')
                encoded=data.tobytes();size=len(encoded)
            used=self.active['images']
            if name not in used and sum(used.values())+size>EVENT_BYTES:
                self.active['limited']='event_image_quota';break
            if encoded is not None:self.store.write(name,encoded)
            used[name]=size
            frame['rois'].append(dict(region,pixel_sha=digest,file=name,encoded_bytes=size))
        if frame['rois']:
            self.active['frames'].append(frame);self.counts['frames']+=1
            self.active['frames'].sort(key=lambda f:(f['monotonic'],f['role']!='before'))
        document={k:v for k,v in self.active.items() if k not in ('images','post_remaining')}
        document['encoded_image_bytes']=sum(self.active['images'].values())
        payload=_json(document)
        if len(payload)>META_BYTES:raise OSError('diagnostic event metadata quota')
        self.store.write(f'event-{self.events-1:02d}.json',payload)
        self.latest=str(self.store.session/f'event-{self.events-1:02d}.json')

    def observe(self,image,d,window,*,now=None):
        """Best-effort observer. Never modify d/image and never raise to read()."""
        if self.closed:return
        try:self._observe(image,d,window,self.clock() if now is None else now)
        except Exception as exc:
            self.counts['errors']=min(9999,self.counts['errors']+1)
            self._error(exc);self.disabled=True
            if self.store and self.store.fd is not None:
                os.close(self.store.fd);self.store.fd=None

    def _observe(self,image,d,window,now):
        if (window is None or getattr(window,'window_id',None) is None
                or not hasattr(image,'shape') or len(image.shape)!=3
                or tuple(image.shape[:2][::-1])!=(int(window.width),int(window.height))):return
        # This task supports only the existing titled PlayCover window sizes.
        # Never retain a desktop/fullscreen or enlarged arbitrary image.
        h,w=image.shape[:2]
        if not (1576<=w<=1774 and 1013<=h<=1136 and abs(w/h-1774/1136)<=.02):return
        stage=_stage(d.get('stage'));phase=d.get('phase')
        boundary=(self.window is not None and self.window!=window.window_id) or (
            stage is not None and stage[0]==1 and self.previous_stage is not None and stage<self.previous_stage)
        if boundary:
            if self.store:self.store.finish()
            self._clear()
        if self.disabled:return
        if self.ended:
            if stage is None or stage[0]!=1:return
            self._clear()
        self.window=window.window_id
        names=frozenset(_text(n) for n in (d.get('augments_taken') or [])[:12])
        reasons=[]
        if phase=='augment' and self.previous_phase!='augment':reasons.append('enter_augment')
        if self.previous_phase=='augment' and phase!='augment':reasons.append('leave_augment')
        if names!=self.names:reasons.append('confirmed_changed')
        suspicious=bool(stage and stage>=(4,2) and len(names)<3 and len(d.get('players') or [])<5
                        and phase not in ('ended','dialog','carousel','spectate','unknown',None))
        if suspicious:
            if self.suspect_since is None:self.suspect_since=now
            if now-self.suspect_since>=SUSPECT_SECONDS and not self.suspect_fired:
                reasons.append('unknown_owner');self.suspect_fired=True
        else:self.suspect_since=None;self.suspect_fired=False
        self.previous_phase=phase;self.names=names
        if stage is not None:self.previous_stage=stage
        trigger=bool(reasons) and now-self.last_event>=COOLDOWN and self.events<MAX_EVENTS
        if reasons and self.events>=MAX_EVENTS and 'session_event_quota' not in self._logged:
            self._error('session_event_quota');self._logged.add('session_event_quota')
        # Reasons arriving during cooldown attach to the latest event rather than
        # disappearing because identical ROI bytes were already saved.
        merged=False
        if reasons and not trigger and self.active:
            for reason in reasons:
                if reason not in self.active['reasons']:self.active['reasons'].append(reason);merged=True
        closing=phase=='ended'
        if not trigger and now-self.last_sample<SAMPLE_INTERVAL and not merged and not closing:return
        snapshot=None
        if trigger or now-self.last_sample>=SAMPLE_INTERVAL or closing:
            snapshot=self._snapshot(image,d,window,now);self.last_sample=now
        if trigger or self.active and (merged or snapshot and self.active['post_remaining']>0):
            if self.store is None:self.store=EvidenceStore(self.root)
            with self.store.locked():
                if self.store.fd is None:self.store.start()
                if trigger:
                    self.events+=1;self.counts['events']+=1;self.last_event=now
                    self.active=dict(reasons=reasons,frames=[],images={},limited=None,post_remaining=2)
                    # Reserve the mandatory CURRENT evidence before spending the
                    # event byte budget on preframes; metadata stays chronological.
                    self._save_frame(snapshot,'current')
                    for prior in self.ring:self._save_frame(prior,'before')
                if not trigger and snapshot and self.active['post_remaining']>0:
                    self._save_frame(snapshot,'after');self.active['post_remaining']-=1
                elif merged:
                    document={k:v for k,v in self.active.items() if k not in ('images','post_remaining')}
                    document['encoded_image_bytes']=sum(self.active['images'].values())
                    self.store.write(f'event-{self.events-1:02d}.json',_json(document))
        if snapshot:self.ring.append(snapshot)
        if phase=='ended':
            if self.store:self.store.finish()
            self.ring.clear();self.active=None;self.ended=True
