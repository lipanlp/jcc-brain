from importlib.util import module_from_spec,spec_from_file_location
from pathlib import Path
import pytest

SPEC=spec_from_file_location('eval_learning',Path(__file__).resolve().parents[1]/'scripts/eval_unit_learning.py')
evaluation=module_from_spec(SPEC);SPEC.loader.exec_module(evaluation)


def row(name='?1',**extra):
    return {'_file':'0001.jpg','prediction':{'phase':'planning','board':[dict(name=name,x=.4,y=.3)],**extra}}


def test_identity_comparison_only_exempts_exact_identity_fields():
    old=[row()];new=[row('甲')]
    assert len(evaluation.compare_identity_outputs(old,new)['allowed'])==1
    assert not evaluation.compare_identity_outputs(old,new)['rejected']
    for field,value in [('phase','combat'),('opponent',{'board':[]} ),('stage','2-1')]:
        altered=row('甲');altered['prediction'][field]=value
        assert evaluation.compare_identity_outputs(old,[altered])['rejected']
    altered=row();altered['prediction']['board'][0]['x']=.5
    assert evaluation.compare_identity_outputs(old,[altered])['rejected']
    altered=row();altered['prediction']['board']=[]
    assert evaluation.compare_identity_outputs(old,[altered])['rejected']
    with pytest.raises(ValueError):evaluation.compare_identity_outputs(old,[])
    old_with_inference=row();old_with_inference['prediction']['auto_labeled']={'name':'甲','slot':0}
    assert not evaluation.compare_identity_outputs([old_with_inference],old)['rejected']
    assert evaluation.compare_identity_outputs(old,[old_with_inference])['rejected']


def test_evaluation_refuses_input_and_existing_output(tmp_path):
    source=tmp_path/'input';source.mkdir()
    with pytest.raises(ValueError):evaluation.require_new_output(source/'results',[source])
    output=tmp_path/'out';evaluation.require_new_output(output,[source])
    with pytest.raises(FileExistsError):evaluation.require_new_output(output,[source])
