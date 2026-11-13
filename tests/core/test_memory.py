"""Accepted observation, source continuity, retrieval and hold regressions (no network)."""
from __future__ import annotations
import json
from pathlib import Path
import pytest
from PIL import Image
from wmmllm.memory import facts
from wmmllm.memory.store import MemoryStore, MemoryConflict, World
from wmmllm.memory.retrieval import retrieve
from wmmllm.memory.observer import ObserverDeps, observer_request, observe_state
from wmmllm.providers.types import ChatResponse, CallReceipt, Usage, ImageInput
from wmmllm.providers.errors import TransportError
from wmmllm.session.io import file_ref, write_once, read_json


def image_ref(root, name, color):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (32, 16), color).save(path)
    return file_ref(path, run_root=root, image=True)


def observation(value='closed', indices=(0,1)):
    return {'summary':'The box is visible.', 'facts':[{'subject_id':'box', 'label':'box',
        'kind':'state', 'key':'lid', 'value':value, 'evidence':'Visible lid position.',
        'frame_indices':list(indices)}]}


def memory(root):
    source = image_ref(root, 'source.png', 'blue')
    store = MemoryStore(root / 'memory', task_id='world', setting='accepted', scene_id='scene',run_root=root)
    store.initialize(source)
    return store,source


def commit(store, root, n, caption, source, endpoint, value):
    observed = observation(value)
    ref = write_once(root / ('observation%d.json'%n), {'observation':observed},run_root=root)
    return store.commit(segment=n,caption=caption,start=source,endpoint=endpoint,
                        observation=observed,observation_record=ref)


def test_fact_validation_sanitize_caps(monkeypatch):
    monkeypatch.setattr(facts.time, 'time', lambda:123.0)
    assert facts.sanitize('a\n"b') == 'a \\"b'
    world=World()
    for n in range(251):
        facts.append_fact(world,kind='state',fact='property %d'%n,scene_id='scene')
    assert len(world.memory['facts']) == 127
    assert any(f['kind']=='summary' for f in world.memory['facts'])
    with pytest.raises(ValueError):
        facts.validate_observation(observation(indices=(2,)))
    with pytest.raises(ValueError):
        facts.validate_observation({**observation(), 'extra':True})


def test_commit_idempotence_and_continuity(tmp_path):
    store,source=memory(tmp_path)
    endpoint=image_ref(tmp_path,'end.png','red')
    first=commit(store,tmp_path,1,'Open the box.',source,endpoint,'open')
    second=commit(store,tmp_path,1,'Open the box.',source,endpoint,'open')
    assert first==second
    assert store.snapshot().completed_segments==1
    with pytest.raises(MemoryConflict):
        commit(store,tmp_path,2,'Close it.',source,endpoint,'closed')
    with pytest.raises(MemoryConflict):
        retrieve(store,segment=2,caption='Look.',source=source,preceding_captions=['Open the box.'],text_purpose='navigate')


def test_changed_state_excludes_old_reference_and_preserves_slots(tmp_path):
    store,source=memory(tmp_path)
    first=image_ref(tmp_path,'one.png','red'); second=image_ref(tmp_path,'two.png','green')
    commit(store,tmp_path,1,'Look at the box.',source,first,'closed')
    context=retrieve(store,segment=2,caption='Show box lid.',source=first,
        preceding_captions=['Look at the box.'],text_purpose='navigate')
    assert context.historical_reference.segment==1
    commit(store,tmp_path,2,'Open it.',first,second,'open')
    snapshot=store.snapshot()
    assert snapshot.state_change_segment=={'box':2}
    assert snapshot.latest_state['box|state|lid']['value']=='open'
    assert len(snapshot.world_memory['facts'])==1
    context=retrieve(store,segment=3,caption='Return to the original view.',source=second,
        preceding_captions=['Look at the box.','Open it.'],text_purpose='navigate')
    assert context.historical_reference is None
    assert context.selection['initial_view_requested']


def test_semantic_restore_keeps_sequential_journal(tmp_path):
    store,source=memory(tmp_path)
    first=image_ref(tmp_path,'one.png','red');second=image_ref(tmp_path,'two.png','green')
    commit(store,tmp_path,1,'Look.',source,first,'closed')
    saved=store.snapshot_to(tmp_path/'snapshots/one.json')
    commit(store,tmp_path,2,'Open.',first,second,'open')
    store.restore_semantics(saved,tmp_path/'restore.json')
    snap=store.snapshot()
    assert snap.completed_segments==2
    assert snap.commits[-1]['endpoint']==second
    assert snap.latest_state['box|state|lid']['value']=='closed'
    assert store.restore_semantics(saved,tmp_path/'restore.json')


def test_hold_advances_caption_only_and_is_idempotent(tmp_path):
    store,source=memory(tmp_path);before=store.snapshot()
    ref=write_once(tmp_path/'hold.json',{'status':'FAILURE_HOLD','hold_eligible':True,
        'rejected_candidate_committed':False,'segment':1,'caption':'Open.',
        'source':source,'endpoint':source},run_root=tmp_path)
    result=store.commit_hold(segment=1,caption='Open.',source=source,failure_record=ref)
    assert result==store.commit_hold(segment=1,caption='Open.',source=source,failure_record=ref)
    after=store.snapshot()
    assert after.completed_segments==1 and after.commits[0]['facts']==0
    assert before.world_memory==after.world_memory and before.states==after.states
    context=retrieve(store,segment=2,caption='Look.',source=source,preceding_captions=['Open.'],text_purpose='navigate')
    assert context.source==source


class TestChat:
    """Only a test transport; this class never contacts a provider."""
    __test__=False
    route='test'
    def __init__(self, answers):self.answers=list(answers);self.requests=[]
    def complete(self, request):
        self.requests.append(request)
        answer=self.answers.pop(0)
        if isinstance(answer,Exception):raise answer
        receipt=CallReceipt('chat','observer','test','test','known',Usage(1,2),'known',
            'complete','test',0.0,'0'*64,'1'*64,(),())
        return ChatResponse(json.dumps(answer),'stop','test',receipt)


def deps(root,chat):
    return ObserverDeps(chat,{'chat':2,'image':0},1,(0,1),('source','endpoint'),
                        'observe',{'type':'json_object'},lambda:0.0,root)


def test_observer_request_images_and_recovery(tmp_path):
    source=image_ref(tmp_path,'source.png','blue');end=image_ref(tmp_path,'end.png','red')
    chat=TestChat([observation()]);d=deps(tmp_path,chat)
    request=observer_request('Open.',source,end,{},d)
    assert [i.label for i in request.images]==['source','endpoint']
    result=observe_state(task_id='world',segment=1,caption='Open.',start=source,endpoint=end,
        known_subjects={},deps=d,out_dir=tmp_path/'observer')
    assert result.status=='OBSERVED'
    assert observe_state(task_id='world',segment=1,caption='Open.',start=source,endpoint=end,
        known_subjects={},deps=d,out_dir=tmp_path/'observer')==result
    assert len(chat.requests)==1
    with pytest.raises(ValueError):
        observe_state(task_id='world',segment=1,caption='Different.',start=source,endpoint=end,
            known_subjects={},deps=d,out_dir=tmp_path/'observer')


def test_observer_invalid_output_is_error_not_retry(tmp_path):
    source=image_ref(tmp_path,'source.png','blue');end=image_ref(tmp_path,'end.png','red')
    chat=TestChat([{'summary':'Planned','facts':[]}])
    result=observe_state(task_id='world',segment=1,caption='Open.',start=source,endpoint=end,
        known_subjects={},deps=deps(tmp_path,chat),out_dir=tmp_path/'observer')
    assert result.status=='ERROR' and result.observation is None
    assert len(chat.requests)==1
    assert 'Planned' not in json.dumps(read_json(result.record,run_root=tmp_path))


def test_observer_technical_retry_and_observation_only_resume(tmp_path):
    source=image_ref(tmp_path,'source.png','blue');end=image_ref(tmp_path,'end.png','red')
    chat=TestChat([TransportError('Timeout'),TransportError('Timeout'),observation()])
    d=deps(tmp_path,chat)
    first=observe_state(task_id='world',segment=1,caption='Open.',start=source,endpoint=end,
        known_subjects={},deps=d,out_dir=tmp_path/'observer')
    assert first.status=='ERROR' and first.technical_retries==1
    second=observe_state(task_id='world',segment=1,caption='Open.',start=source,endpoint=end,
        known_subjects={},deps=d,out_dir=tmp_path/'observer')
    assert second.status=='OBSERVED' and len(chat.requests)==3
    assert read_json(first.record,run_root=tmp_path)['status']=='ERROR'


def test_error_observation_cannot_be_committed(tmp_path):
    store,source=memory(tmp_path);endpoint=image_ref(tmp_path,'end.png','red')
    observed=observation('open')
    ref=write_once(tmp_path/'error.json',{'status':'ERROR','observation':observed},run_root=tmp_path)
    with pytest.raises(MemoryConflict,match='validated accepted-state record'):
        store.commit(segment=1,caption='Open.',start=source,endpoint=endpoint,
                     observation=observed,observation_record=ref)
    assert store.snapshot().completed_segments==0
