"""Packaged prompts retain exact bytes and validate the frozen endpoint protocol."""
import hashlib
import pytest
from wmmllm.prompts import manifest,read_text,read_json_data
from wmmllm.prompts import policy


def test_all_prompt_resources_have_checked_bytes():
    for name,entry in manifest().items():
        value=read_text(name)
        assert hashlib.sha256(value.encode()).hexdigest()==entry['sha256']
        if name.endswith('.json'):assert read_json_data(name) is not None

@pytest.mark.parametrize('caption,multi',[
    ('The person cuts an apple.',False),('Pan right without cuts to another scene.',False),
    ('A close-up of the cup, then the camera cuts to the doorway.',True),
    ('Two shots show the room.',True),('A single shot shows the room.',False),
    ('先展示房间，然后切换到下一个镜头。',True)])
def test_explicit_shots_only(caption,multi):
    assert policy.explicit_multiple_shots(caption) is multi


def test_policy_preserves_current_caption_and_action():
    caption='The camera pans to the right.'
    plan={'current_state':'A room contains a red cup.','remaining_action':'Keep the cup still.','endpoint_instruction':'A red cup stays on the same table.'}
    assert policy.verification_input(policy.planner_input([],caption),plan)['actual_h3_action']==caption
    assert policy.validate_plan(plan)['remaining_action']=='Keep the cup still.'
    with pytest.raises(ValueError):policy.validate_plan({**plan,'endpoint_instruction':'Move the object.'})


def test_semantic_uncertain_is_labeled_and_fail_wins():
    v={k:'pass' for k in policy.SEMANTIC_FIELDS};v['reason']='Visible evidence.'
    v['state_matches_source']='uncertain';d=policy.semantic_decision(v)
    assert d['semantic_status']=='UNCERTAIN' and d['uncertain_criteria']==['state_matches_source']
    v['endpoint_matches_action']='fail';d=policy.semantic_decision(v)
    assert d['semantic_status']=='FAIL' and d['failed_criteria']==['endpoint_matches_action']
    with pytest.raises(ValueError):policy.validate_semantic_verification({**v,'reason':1})
