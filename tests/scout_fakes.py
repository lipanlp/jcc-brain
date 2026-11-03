"""Explicit input injection for scout tests; never imports a native input sender."""
from dataclasses import replace
from jccbrain.vision.eye.input_monitor import EVENT_NAMES, EventLedger, InputSnapshot, counter_residual

class ControlledInput:
    def __init__(self, clock, window, mouse=lambda:(400.,300.)):
        self.clock, self.window, self.mouse = clock, window, mouse
        self.counts = dict.fromkeys(EVENT_NAMES, 0)
        self.ledger = EventLedger(clock=clock)
        self.generation = 1
        self.available = self.healthy = True
        self.held = False
        self.idle = 10.
        self.front = window.pid
        self.starts = self.stops = 0
        self.change = lambda s:s
        self.hook = lambda kind:None
    def start(self): self.starts += 1; return self.available
    def stop(self): self.stops += 1; self.ledger.clear_expected(); return True
    def clear_expected(self): self.ledger.clear_expected()
    def snapshot(self):
        row = self.ledger.state()
        return self.change(InputSnapshot(self.clock(), self.available, self.idle, tuple(self.counts.items()),
            self.held, self.mouse(), self.front, self.window, row['error'], self.generation,
            row['own_counters'], row['foreign_sequence'], self.healthy, row['pending'], row['events']))
    def expect_action(self, kind, point, tolerance):
        return self.ledger.expect(kind,point,tolerance,int(self.clock()*1e9)+1)
    def event(self, kind='MouseMoved', *, own=False, point=(1.,1.)):
        self.counts[kind] += 1
        self.ledger.observe(kind, self.ledger.pid if own else -1, point,
                            source_time_ns=int(self.clock()*1e9)+1)
        self.held = kind == 'LeftMouseDown'
        self.hook(kind)
    def finish_action(self, action, baseline, check):
        check()
        s=self.snapshot();p=s.pending
        if not p or p['id'] != action or p['consumed'] != len(p['order']) or any(counter_residual(baseline,s).values()):
            raise RuntimeError('expected_event_missing')
        self.clear_expected()
