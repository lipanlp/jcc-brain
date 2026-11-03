"""Raw semantic invalidation also runs unchanged against the frozen old source."""
from types import SimpleNamespace
import pytest
from test_assistant_buy_projection import raw, feed, model, isolated

@pytest.mark.parametrize('dependency', [
    {'gold':0}, {'level':7}, {'phase':'combat'}, {'stage':'4-1'}, {'shop':{'open':True,'slots':[{'name':'奥恩','cost':1}]}},
    {'shop':{'open':False}}, {'units_stale':True},
    {'board':[{'name':'绯红树怪','star':2}]},
    {'shop':{'open':True,'slots':[{'name':'绯红树怪','cost':2}]}},
    {'shop':{'open':True,'slots':[{'name':'绯红树怪','cost':1,'upgrade':2}]}}
])
def test_raw_dependency_revokes_before_slow_scout(db, rules, dependency):
    _, sink, mode = model(db, rules); feed(mode); assert sink.projections[-1].valid
    before = []
    mode.scout = SimpleNamespace(observe=lambda st:before.append(sink.projections[-1].valid), paused=False, mark_own=lambda:None, invalidate_notice=lambda *a:None)
    mode._process_frame(raw(**dependency))
    assert before == [False] and not sink.projections[-1].valid
