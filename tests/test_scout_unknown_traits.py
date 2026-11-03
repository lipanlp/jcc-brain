"""Missing trait scalars remain data; sorting must never discard a scout frame."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from itertools import permutations,product
import queue
import random
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest

from jccbrain.core.models import GameState,TraitStatus
from jccbrain.modes.scout import ScoutCapture,trait_signature
from test_scout_runner import setup


def raw(traits):return [(t.name,t.count,t.active,t.next) for t in traits]


def test_720_orders_preserve_missing_empty_duplicate_and_boolean_values():
    rows=[TraitStatus(None,None,None,None),TraitStatus('',0,False,0),
          TraitStatus('法师',None,True,2),TraitStatus('法师',None,None,None),
          TraitStatus('法师',0,None,None),TraitStatus(None,0,False,None)]
    signature=trait_signature(rows)
    for order in permutations(rows):assert trait_signature(order)==signature
    duplicated=rows+[deepcopy(rows[0])]
    assert Counter(trait_signature(duplicated))==Counter(raw(duplicated))
    assert trait_signature([TraitStatus(None)])!=trait_signature([TraitStatus('')])
    assert trait_signature([TraitStatus('法师',None)])!=trait_signature([TraitStatus('法师',0)])
    assert trait_signature([TraitStatus('法师',0,None)])!=trait_signature([TraitStatus('法师',0,False)])


def test_all_36_scalar_combinations_are_order_invariant_and_do_not_mutate_input():
    traits=[TraitStatus(*row) for row in product((None,'','法师'),(None,0),(None,False,True),(None,2))]
    before=deepcopy([asdict(t) for t in traits]);expected=trait_signature(traits);rng=random.Random(73)
    for _ in range(100):
        shuffled=traits[:];rng.shuffle(shuffled)
        assert trait_signature(shuffled)==expected
    assert len(expected)==36 and Counter(expected)==Counter(raw(traits))
    assert [asdict(t) for t in traits]==before


@pytest.mark.parametrize('place',['initial','result','both'])
def test_actual_background_finishes_job_and_preserves_unknowns(setup,place):
    ctx,runner,_,initial,_,_=setup
    unknown=[{'name':None,'count':None,'active':None,'next':None},
             {'name':'','count':0,'active':False,'next':None},
             {'name':'法师','count':3,'active':True,'next':5}]
    if place in ('initial','both'):initial.traits=[TraitStatus(**r) for r in unknown]
    traits=unknown if place in ('result','both') else [{'name':'法师','count':3}]
    result=dict(stage=initial.stage,viewing={'slot':1,'name':'甲'},traits=deepcopy(traits),
                board=[{'name':'?1','star':1}])
    runner._initial=initial;runner._status.update(captured=1,total=1)
    runner._reader_factory=lambda:NS(read_scout=lambda _:deepcopy(result))
    jobs=queue.Queue();jobs.put((initial.players[1],ScoutCapture(np.zeros((16,24,3),np.uint8),True,
          dict(cut_ms=None,settle_ms=None,total_ms=0)),1.))
    done=threading.Event();done.set()
    runner._recognize(jobs,done,ctx.scout.started,initial.stage)
    assert not runner.status['processing'] and runner.status['processed']==1 and jobs.unfinished_tasks==0
    rows=ctx.scout.latest();assert len(rows)==1 and not rows[0].units and not ctx.scout.taken()
    # Existing GameState parsing normalizes active=None to False; this task must
    # preserve that contract rather than change the core model.
    if place in ('result','both'):assert Counter(raw(rows[0].traits))==Counter(raw(GameState.from_dict(result).traits))
    stored=__import__('json').loads(ctx.scout.path.read_text().splitlines()[0])
    assert stored['board'][0]['name']=='?1'
    assert ctx.monitor.starts==0


@pytest.mark.parametrize('viewing',[{'slot':7,'name':'甲'},{'slot':1,'name':'别的玩家'}])
def test_unknown_traits_never_bypass_existing_identity_conflicts(setup,viewing):
    ctx,runner,_,initial,sink,_=setup;runner._initial=initial
    runner._reader_factory=lambda:NS(read_scout=lambda _:dict(viewing=viewing,
        traits=[{'name':None,'count':2,'active':None},{'name':'法师','count':3}],board=[{'name':'?1'}]))
    jobs=queue.Queue();jobs.put((initial.players[1],ScoutCapture(np.zeros((16,24,3),np.uint8),True,
        dict(cut_ms=0,settle_ms=0,total_ms=0)),1.))
    done=threading.Event();done.set();runner._recognize(jobs,done,ctx.scout.started,initial.stage)
    assert jobs.unfinished_tasks==0 and not runner.status['processing'] and not ctx.scout.latest()
    assert any('viewing_mismatch' in line for line in sink.logs) and ctx.monitor.starts==0
