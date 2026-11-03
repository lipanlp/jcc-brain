"""Small display ledger: no network, strategy, input sender or compute lock."""
from dataclasses import replace
import threading

from jccbrain.core.ai_source import AiSource, AiProjection, digest


class AiDisplay:
    def __init__(self, session, publish):
        self._lock = threading.RLock()
        self._publish = publish
        self._projection = AiProjection(AiSource(session))
        self._raw = None

    @property
    def projection(self):
        with self._lock:
            return self._projection

    def revoke(self, *, input_epoch=None):
        with self._lock:
            old = self._projection.source
            if input_epoch is not None and input_epoch < old.input_epoch:
                return old  # a raw observer must never undo a newer input fence
            source = replace(old, revision=old.revision + 1, sequence=old.sequence + 1,
                             ready=False, fingerprint='', input_epoch=old.input_epoch if input_epoch is None else input_epoch)
            self._projection = out = AiProjection(source)
        self._publish(out)
        return source

    def observe_raw(self, fingerprint, *, usable, input_epoch):
        with self._lock:
            if input_epoch != self._projection.source.input_epoch:
                return
            changed = fingerprint != self._raw
            self._raw = fingerprint
            revoke = changed or not usable and self._projection.source.ready
        if revoke:
            self.revoke(input_epoch=input_epoch)

    def confirm(self, fingerprint, context, buy_source, *, usable):
        identity = (buy_source.generation, buy_source.selection, buy_source.season,
                    buy_source.input_epoch, buy_source.stage, buy_source.phase)
        key = digest([fingerprint, context, identity])
        with self._lock:
            old = self._projection.source
            if buy_source.input_epoch != old.input_epoch or fingerprint != self._raw or not usable:
                return old
            if old.ready and old.fingerprint == key:
                return old
            source = AiSource(old.session, old.revision + 1, old.sequence + 1, *identity, key, True)
            self._projection = out = AiProjection(source)
        self._publish(out)
        return source

    def current(self, source):
        with self._lock:
            now = self._projection.source
            return bool(source and now.ready and source.ready and source.session == now.session
                        and source.revision == now.revision and source.fingerprint == now.fingerprint)

    def reply(self, source, text, augment, options, *, buy_frame=0):
        with self._lock:
            if not self.current(source):
                return False
            now = replace(self._projection.source, sequence=self._projection.source.sequence + 1)
            self._projection = out = AiProjection(now, True, text, augment, tuple(options), buy_frame)
        self._publish(out)
        return True
