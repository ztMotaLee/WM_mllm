"""Navigation keeps depth provenance and each semantic/geometry criterion."""
import json
import numpy as np
import pytest
from wmmllm.agent import navigation,planning,contracts
from wmmllm.state.panorama import create_panorama_state,panorama_state_sha256,pixel_sha256,erp_array
from wmmllm.session.io import file_ref


def verdict():return {'criteria':{k:'PASS' for k in navigation.CRITERIA},'semantic':{**{k:'pass' for k in navigation.SEMANTIC_FIELDS},'reason':'Visible evidence.'},'reason':'Coherent destination.','retry_instruction':''}


def test_all_navigation_criteria_gate_the_result():
    v=verdict();v['criteria']['spatial']='UNCERTAIN';d=navigation.navigation_decision(v)
    assert d['status']=='UNCERTAIN' and d['continuation_allowed'] and not d['semantic_accepted']
    v['criteria']['movement']='FAIL';d=navigation.navigation_decision(v)
    assert d['status']=='FAIL' and not d['continuation_allowed']
    with pytest.raises(ValueError):navigation.navigation_decision(v,allow_uncertain_continuation=1)


def test_navigation_templates_escape_json_and_normalize_context():
    specs=navigation.app_navigation_specs(anchors=[{'text':'A "quoted" anchor'}],subjects=[{'name':'blue cup'}],context='A room.\n  Clear light.')
    assert set(specs)==set(navigation.CRITERIA)
    assert 'A \\"quoted\\" anchor' in specs['spatial']['prompt']
    assert 'A room. Clear light.' in specs['entity']['prompt']


def test_depth_is_exact_shape_bound_and_never_invented(tmp_path):
    state=create_panorama_state(np.zeros((32,64,3),np.uint8),width=64,height=32)
    assert navigation.validate_depth(state,None,run_root=tmp_path)['generative_navigation_required']
    path=tmp_path/'depth.npy';np.save(path,np.full((32,64),2,np.float32),allow_pickle=False)
    ref={**file_ref(path,run_root=tmp_path),'panorama_state_sha256':panorama_state_sha256(state),'erp_pixel_sha256':pixel_sha256(erp_array(state)),'units':'radial_eye_heights'}
    assert navigation.validate_depth(state,ref,run_root=tmp_path)['depth_ready']
    assert not navigation.validate_depth(state,{**ref,'panorama_state_sha256':'0'*64},run_root=tmp_path)['depth_ready']
    np.save(path,np.zeros((16,32),np.float32),allow_pickle=False);ref.update(file_ref(path,run_root=tmp_path))
    assert 'no resize' in navigation.validate_depth(state,ref,run_root=tmp_path)['reason']
