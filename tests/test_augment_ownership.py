"""F2-0 offline probe boundaries: exact current text and safe output ownership."""
import json
from pathlib import Path

import pytest

from scripts.eval_augment_ownership import (
    CASES, POSITIVE, agreeing_name, anchor_candidate, private_ocr,
    single_text, strict_probe, validate_output,
)


def row(name='GameBoss', *, phase='planning', panel=True):
    return dict(reader=dict(phase=phase,viewing=None,players=[{'self':True}]+[{} for _ in range(7)]),
                probe=dict(plates=[dict(colour='green',exact_two_crop_name=name)],
                    raw_eye_score=.2,current_eye_slot=None,
                    panel=dict(lines=[dict(text='已拥有的强化符文')] if panel else [])))


def test_output_refuses_existing_input_nested_ancestor_and_symlink(tmp_path):
    source=tmp_path/'source';source.mkdir()
    with pytest.raises(ValueError,match='new directory'):validate_output(source,[source])
    with pytest.raises(ValueError,match='overlaps'):validate_output(source/'new',[source])
    link=tmp_path/'alias';link.symlink_to(source,target_is_directory=True)
    with pytest.raises(ValueError,match='overlaps'):validate_output(link/'new',[source])
    assert validate_output(tmp_path/'safe',[source])==tmp_path/'safe'
    with pytest.raises(ValueError,match='batch-F2'):
        validate_output(tmp_path/'outside',[],base=source)


def test_exact_two_raw_names_never_complete_a_partial_or_use_closest_spelling():
    assert agreeing_name([{'single_text':'GameBoss'},{'single_text':'GameBoss'}])=='GameBoss'
    assert agreeing_name([{'single_text':'ameBoss'},{'single_text':'GameBoss'}]) is None
    assert agreeing_name([{'single_text':'ameBoss'},{'single_text':'ameBoss'}])=='ameBoss'
    assert not strict_probe(row('ameBoss'), 'GameBoss')['trial_and']
    assert single_text([{'text':'GameBoss'},{'text':'other name'}]) is None
    assert single_text([{'text':'3'},{'text':'GameBoss'}])=='GameBoss'


@pytest.mark.parametrize('mutate',[
    lambda r:r['reader'].update(phase='combat'),
    lambda r:r['reader'].update(phase='unknown'),
    lambda r:r['reader'].update(viewing={'slot':1,'name':'GameBoss'}),
    lambda r:r['probe'].update(current_eye_slot=1),
    lambda r:r['probe'].update(raw_eye_score=.60),
    lambda r:r['probe']['plates'].append(dict(colour='green',exact_two_crop_name='GameBoss')),
    lambda r:r['probe']['plates'].append(dict(colour='orange',exact_two_crop_name='Other')),
    lambda r:r['probe']['panel'].update(lines=[]),
])
def test_green_name_title_and_old_anchor_cannot_override_current_conflicts(mutate):
    r=row();assert strict_probe(r,'GameBoss')['trial_and']
    mutate(r);assert not strict_probe(r,'GameBoss')['trial_and']


def test_anchor_requires_current_full_roster_unique_self_and_exact_single_green():
    r=row();assert anchor_candidate(r)=='GameBoss'
    r['reader']['players']=[{'self':True}]*5
    assert anchor_candidate(r) is None
    r=row();r['reader']['players']=[{'self':True}]
    assert anchor_candidate(r) is None
    assert anchor_candidate(row(None)) is None
    assert not strict_probe(row(),None)['trial_and']


def test_human_position_upper_bound_is_never_consumed_as_automatic_name_evidence():
    r=row(None)
    r['probe']['manual_upper_bound']=[dict(colour='manual-green',exact_two_crop_name='GameBoss')]
    assert anchor_candidate(r) is None
    assert not strict_probe(r,'GameBoss')['trial_and']


def test_probe_refuses_unexpected_backend_instead_of_silently_using_fallback():
    from types import SimpleNamespace
    with pytest.raises(RuntimeError,match='actual Rapid'):
        private_ocr(SimpleNamespace(name='vision'))


def test_private_probe_never_shares_writeable_caches_words_or_locks():
    from collections import OrderedDict,deque
    from threading import Lock
    class OCR:
        name='rapid'
        def __init__(self):
            self.cache=OrderedDict(a=('history',1));self.words=deque(['history'])
            self.lock=Lock();self._stats_lock=Lock();self._warm_lock=Lock()
        def reset_stats(self):self.stats={}
    main=OCR();private=private_ocr(main)
    assert private.cache is not main.cache and not private.cache
    assert private.lock is not main.lock and private._stats_lock is not main._stats_lock
    private.words.append('future word');assert not private.words and list(main.words)==['history']
    assert not private.fallback


def test_frozen_review_is_source_bound_and_separate_from_model_inputs():
    path=Path(__file__).parent/'fixtures/augment_ownership_truth.json'
    review=json.loads(path.read_text());rows={r['id']:r for r in review['rows']}
    assert set(rows)==set(CASES) and len(POSITIVE)==4
    assert sum(r['real'] and not r['eligible_context'] for r in rows.values())>=8
    assert all(len(r['sha256'])==64 for r in rows.values())
    assert rows['augment-taken/0006.png']['green_legend_name_fully_visible'] is False
    assert rows['augment-taken/0006.png']['panel_owner_label_visible'] is False
    assert rows['augment-taken/0001.png']['panel_visible'] and rows['augment-taken/0001.png']['board_owner']=='other'
    assert review['probe_plan']['max_geometry_combinations']==2
