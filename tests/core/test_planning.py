"""Ordered images, endpoint schema correction and source-bound retry contracts."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from wmmllm.agent import planning,contracts,router
from wmmllm.providers.types import ChatResponse,CallReceipt
from wmmllm.session.io import save_png,write_once


def receipt(index=1):
    return CallReceipt('chat','planner','test', 'test','known',None,'unknown','complete','test',0.,str(index)*64,'f'*64,(),())

def decision():
    return {'operations':[{'operation':'local_edit','targets':['red cup'],'scope':'local','instruction':'Paint the red cup blue.'}],'ending_camera':{'yaw':0.,'pitch':0.},'editing_camera':{'yaw':0.,'pitch':0.},'editing_hfov':90.,'restore_node':'','physical_state_changed':True,'endpoint_instruction':'A blue cup is on the table.','decision_reason':'The cup changes color.'}


def refs(tmp_path):
    image=save_png(tmp_path/'source.png',np.zeros((32,48,3),np.uint8),run_root=tmp_path)
    pano=save_png(tmp_path/'erp.png',np.zeros((32,64,3),np.uint8),run_root=tmp_path)
    memory=write_once(tmp_path/'memory.json',{'memory':'source'},run_root=tmp_path)
    return image,pano,memory


def request(tmp_path):
    source,pano,memory=refs(tmp_path)
    return planning.SegmentRequest('interactive','case','case',0,1,0,'Paint the cup blue.',(),source,'official_initial_letterboxed',{},memory,None,None,{},5.,tmp_path),pano


def test_request_images_and_policy_text(tmp_path):
    req,pano=request(tmp_path);endpoint=save_png(tmp_path/'end.png',np.ones((32,48,3),np.uint8),run_root=tmp_path)
    req.verify_inputs(run_root=tmp_path);p=planning.Plan('A red cup is on a table.','Paint the red cup blue.','A blue cup is on the table.')
    v=planning.verifier_request(req,p,endpoint)
    assert [i.path.name for i in v.images]==['source.png','end.png']
    assert v.payload['actual_h3_action']==req.current_caption
    built=planning.endpoint_request(req,decision=decision(),payload={'operation_decision':decision()},editing_crop=req.source,physical_pano_ref=pano)
    assert [i.path.name for i in built.images]==['source.png','source.png','erp.png']
    assert built.response_format['json_schema']['schema']['required'][-3:]==['scope_bbox','support_polygons','final_view_entity_bboxes']
    assert 'EDITING_VIEW' in built.system


def test_one_schema_reask_preserves_request(tmp_path):
    req,pano=request(tmp_path);built=planning.endpoint_request(req,decision=decision(),payload={},editing_crop=req.source,physical_pano_ref=pano)
    plan={'current_state':'A red cup is on a table.','remaining_action':'Paint the red cup blue.','endpoint_instruction':'A blue cup is on the table.','scope_bbox':[.2,.2,.8,.8],'support_polygons':[[[.2,.2],[.8,.2],[.8,.8],[.2,.8]]],'final_view_entity_bboxes':[]}
    class Chat:
        def __init__(self):self.requests=[]
        def complete(self,r):
            self.requests.append(r);return ChatResponse('{}' if len(self.requests)==1 else json.dumps(plan),'stop','test',receipt(len(self.requests)))
    chat=Chat();out=planning.endpoint_plan(chat=chat,request=built,decision=decision(),caption=req.current_caption,out=tmp_path/'attempt',deps=SimpleNamespace(run_root=tmp_path))
    assert len(chat.requests)==2 and out['scope_bbox']==plan['scope_bbox']
    assert chat.requests[0].images==chat.requests[1].images
    assert chat.requests[0].response_format==chat.requests[1].response_format
    assert chat.requests[1].payload['technical_schema_correction']['does_not_consume_semantic_retry']


def test_invalid_schema_second_response_raises(tmp_path):
    req,pano=request(tmp_path)
    class Chat:
        count=0
        def complete(self,r):self.count+=1;return ChatResponse('{}','stop','test',receipt())
    chat=Chat()
    with pytest.raises(KeyError):planning.endpoint_plan(chat=chat,request=planning.planner_request(req),decision=decision(),caption=req.current_caption,out=tmp_path/'attempt',deps=SimpleNamespace(run_root=tmp_path))
    assert chat.count==2


def test_bound_generic_entity_schema_repair():
    d=decision();d['operations']=[{'operation':'subject_action','targets':['the maroon-robed figure'],'scope':'local','instruction':'Raise a hand.'}]
    p={'current_state':'A maroon-robed figure stands.','remaining_action':'The maroon-robed figure raises a hand.','endpoint_instruction':'The maroon-robed figure stands with a raised hand. The subject smiles.'}
    metadata=planning.bound_entity_metadata(d,p);result,record=planning.repair_plan_schema(p,entity_metadata=metadata,validate_plan=planning.policy.validate_plan)
    assert 'The maroon-robed figure smiles.' in result['endpoint_instruction']
    assert record['remaining_action_unchanged']


def test_router_schema_rejects_stale_restore_and_camera_physical_change():
    d=decision();d['restore_node']='node-old'
    with pytest.raises(ValueError,match='not in saved'):router.validate_operation_response(json.dumps(d),{})
    d=decision();d['operations'][0]['operation']='rotate'
    with pytest.raises(ValueError,match='camera-only|Camera-only'):router.validate_operation_response(json.dumps(d),{})
    d=decision();d['ending_camera']['yaw']=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):router.validate_operation_response(json.dumps(d),{})


def test_retry_feedback_binds_actual_source_and_failure(tmp_path):
    req,pano=request(tmp_path);d=decision();d.update(source_sha256=req.source['sha256'],panorama_sha256=pano['sha256'])
    c=contracts.build_endpoint_constraints({'key':'case','task_id':'case','segment':1,'seed':0,'current_caption':req.current_caption},d,source=req.source,panorama=pano)
    sem={'status':'FAIL','verification':{**{k:'pass' for k in planning.policy.SEMANTIC_FIELDS},'endpoint_matches_action':'fail','reason':'Cup remains red.'},'failed':['endpoint_matches_action'],'uncertain':[],'reason':'Cup remains red.'}
    v=write_once(tmp_path/'VERIFIER_1.json',sem,run_root=tmp_path)
    prior={'attempt':1,'image':req.source,'endpoint':req.source,'semantic':sem,'verification_ref':v,'verifier_source':req.source,'editor_inputs':[req.source],'editor_receipt':receipt().to_record(),'verifier_receipt':receipt().to_record()}
    feedback=contracts.build_feedback(current_attempt=2,attempts=[prior],stage=tmp_path,constraints=c,actual_editor_source=req.source,navigation=False,run_root=tmp_path)
    base=planning.planner_request(req);new=contracts.planner_feedback(base,feedback,c,run_root=tmp_path)
    assert len(new.images)==3 and [i.label for i in new.images[-2:]]==['actual rejected attempt1 benchmark endpoint','actual rejected attempt1 raw image proposal']
    with pytest.raises(ValueError,match='different accepted physical'):contracts.build_feedback(current_attempt=2,attempts=[prior],stage=tmp_path,constraints=c,actual_editor_source=pano,navigation=False,run_root=tmp_path)


def test_schema_reask_keeps_second_semantic_attempt_number(tmp_path):
    from dataclasses import replace
    req,pano=request(tmp_path)
    built=planning.endpoint_request(req,decision=decision(),payload={},editing_crop=req.source,physical_pano_ref=pano)
    built=replace(built,payload={**built.payload,'previous_failed_proposal_provenance':{'current_attempt':2}})
    answer={'current_state':'A red cup is on a table.','remaining_action':'Paint the red cup blue.','endpoint_instruction':'A blue cup is on the table.','scope_bbox':[0,0,1,1],'support_polygons':[[[0,0],[1,0],[1,1],[0,1]]],'final_view_entity_bboxes':[]}
    class Chat:
        def __init__(self):self.requests=[]
        def complete(self,r):
            self.requests.append(r)
            return ChatResponse('{}' if len(self.requests)==1 else json.dumps(answer),'stop','test',receipt(len(self.requests)))
    chat=Chat()
    planning.endpoint_plan(chat=chat,request=built,decision=decision(),caption=req.current_caption,out=tmp_path/'attempt',deps=SimpleNamespace(run_root=tmp_path))
    assert chat.requests[1].payload['technical_schema_correction']['semantic_attempt']==2
