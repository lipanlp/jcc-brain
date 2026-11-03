"""Offline diagnostics reject unsafe entry points before any production context starts."""
import importlib.util
import json
from pathlib import Path
import socket

import pytest

spec = importlib.util.spec_from_file_location('profile_assistant', Path(__file__).parents[1]/'scripts/profile_assistant.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def fixture(tmp_path):
    home=tmp_path/'home'; home.mkdir(); source=tmp_path/'inputs'/'raw.json'; source.parent.mkdir(); source.write_text('[]')
    (home/'jcc.sqlite3').touch()
    (home/'.assistant-profile-offline.json').write_text(json.dumps({'purpose':'offline-assistant-profile','home':str(home)}))
    cfg={'llm':{'enabled':False},'scout':{'enabled':False},'data':{'auto_sync':False},'actuator':{'dry_run':True},'vision':{'unit_learning_mode':'off'}}
    (home/'settings.json').write_text(json.dumps(cfg))
    return home,source,cfg


@pytest.mark.parametrize('section,key,value',[('llm','enabled',True),('scout','enabled',True),('data','auto_sync',True),('actuator','dry_run',False),('vision','unit_learning_mode','auto')])
def test_reject_unsafe_settings_before_output_or_home_switch(tmp_path,monkeypatch,section,key,value):
    h,inp,cfg=fixture(tmp_path);cfg[section][key]=value;(h/'settings.json').write_text(json.dumps(cfg));monkeypatch.setenv('JCC_HOME','unchanged')
    out=tmp_path/'out'
    with pytest.raises(ValueError):p.validate(h,out,inp)
    import os
    assert os.environ['JCC_HOME']=='unchanged' and not out.exists()


def test_explicit_marker_required_not_just_safe_flags(tmp_path):
    h,inp,_=fixture(tmp_path);(h/'.assistant-profile-offline.json').unlink()
    with pytest.raises(FileNotFoundError):p.validate(h,tmp_path/'out',inp)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('kind',['missing_input','existing_output','input_home_overlap','output_home_overlap'])
def test_reject_path_reuse(tmp_path,kind):
    h,inp,_=fixture(tmp_path);out=tmp_path/'out'
    if kind=='missing_input':inp=tmp_path/'absent'
    if kind=='existing_output':out.mkdir()
    if kind=='input_home_overlap':inp=h/'raw.json';inp.write_text('[]')
    if kind=='output_home_overlap':out=h/'output'
    with pytest.raises(ValueError):p.validate(h,out,inp)


def test_validated_home_and_new_output_are_explicit(tmp_path,monkeypatch):
    h,inp,_=fixture(tmp_path);monkeypatch.setenv('JCC_HOME','previous')
    out=tmp_path/'out';assert p.validate(h,out,inp)==(h,out,inp);assert out.is_dir()


def test_statistics_empty_single_nearest_rank_even_and_generator():
    assert p.summary([])=={'n':0,'min':None,'median':None,'p95':None,'max':None}
    assert p.summary([2])=={'n':1,'min':2,'median':2,'p95':2,'max':2}
    assert p.summary(x for x in range(1,21))=={'n':20,'min':1,'median':10.5,'p95':19,'max':20}


def test_network_guard_restores_after_error():
    original=socket.socket.connect
    with pytest.raises(RuntimeError):
        with p.no_network():socket.socket.connect(None,('127.0.0.1',1))
    assert socket.socket.connect is original


def test_cli_missing_inputs_cannot_enter_profiling(monkeypatch):
    monkeypatch.setattr(p,'sequence',lambda *a:pytest.fail('should not enter mode'))
    with pytest.raises(SystemExit) as e:p.main(['sequence'])
    assert e.value.code==2
