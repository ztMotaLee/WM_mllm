"""Navigation keeps each semantic/geometry criterion."""
import pytest

from wmmllm.agent import navigation


def verdict():return {'criteria':{k:'PASS' for k in navigation.CRITERIA},'semantic':{**{k:'pass' for k in navigation.SEMANTIC_FIELDS},'reason':'Visible evidence.'},'reason':'Coherent destination.','retry_instruction':''}


def test_all_navigation_criteria_gate_the_result():
    v=verdict();v['criteria']['spatial']='UNCERTAIN';d=navigation.navigation_decision(v)
    assert d['status']=='UNCERTAIN' and d['continuation_allowed'] and not d['semantic_accepted']
    v['criteria']['movement']='FAIL';d=navigation.navigation_decision(v)
    assert d['status']=='FAIL' and not d['continuation_allowed']


def test_navigation_templates_escape_json_and_normalize_context():
    specs=navigation.app_navigation_specs(subjects=[{'name':'A "quoted" cup'}],context='A room.\n  Clear light.')
    assert set(specs)==set(navigation.CRITERIA); assert 'A \\"quoted\\" cup' in specs['entity']['prompt']
    assert 'A room. Clear light.' in specs['entity']['prompt']


def test_navigation_semantic_error_uses_public_terminology():
    invalid={'criteria':{key:'PASS' for key in navigation.CRITERIA},'semantic':{},'reason':'','retry_instruction':''}
    with pytest.raises(ValueError,match=r'^Invalid endpoint semantic fields\.$'):
        navigation.navigation_decision(invalid)

