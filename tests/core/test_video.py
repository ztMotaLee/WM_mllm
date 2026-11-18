"""Real CPU camera rendering and explicit unavailable video paths."""
from __future__ import annotations
from pathlib import Path
import shutil
import numpy as np
import pytest
from PIL import Image
from wmmllm.state.panorama import create_panorama_state,render_view,panorama_state_sha256
from wmmllm.video import cpu
from wmmllm.video.presentation import present_transition
from wmmllm.providers import NoneVideoBackend, PresentationResult, VideoTransition
from wmmllm.session.io import file_ref, read_json


def setup(root):
    yy,xx=np.indices((32,64));pano=np.stack((xx*3,yy*7,(xx+yy)*2),axis=-1).astype(np.uint8)
    state=create_panorama_state(pano,source=None,width=64,height=32)
    profile={'width':32,'height':16,'num_frames':3,'fps':12}
    viewport={'width':32,'height':16,'content_box':[0,0,32,16],'padding_ltrb':[0,0,0,0]}
    nodes=[]
    for index,yaw in enumerate((0,20)):
        path=root/('view%d.png'%index)
        Image.fromarray(render_view(state,yaw=yaw,width=32,height=16)).save(path)
        nodes.append({'origin_id':'origin','physical_revision':0,'view_identity':'viewer',
            'pose':{'yaw':yaw,'pitch':0,'hfov':90},'viewport_profile':viewport,
            'view':file_ref(path,run_root=root,image=True)})
    transition=VideoTransition('turn0001','rotate',root/'view0.png',root/'view1.png',.25,root/'video.mp4')
    return state,profile,nodes,transition


def test_rotation_endpoint_inclusive_source_path():
    assert cpu.rotation_poses(170,-170,num_frames=3)==[
        {'yaw':170.0,'pitch':0.0},{'yaw':0.0,'pitch':0.0},{'yaw':-170.0,'pitch':0.0}]


def test_none_backend_never_claims_rendered_or_generated_video(tmp_path):
    transition=VideoTransition('transition-1','local_edit',tmp_path/'first.png',tmp_path/'last.png',
                               duration_seconds=5,output_path=tmp_path/'video.mp4')
    backend=NoneVideoBackend()
    result=backend.present(transition)
    assert result.status=='none' and result.video_path is None and result.backend=='none'
    assert not list(tmp_path.iterdir())
    assert PresentationResult('rendered',Path('a.mp4'),'cpu').status!=PresentationResult('generated',Path('b.mp4'),'x').status
    state,profile,nodes,transition=setup(tmp_path)
    transition=VideoTransition('edit','local_edit',transition.first_frame,transition.last_frame,
                               output_path=tmp_path/'edit.mp4')
    before=panorama_state_sha256(state)
    result=present_transition(transition,panorama=state,source_node=nodes[0],target_node=nodes[1],
                             backend=NoneVideoBackend(),profile=profile,run_root=tmp_path)
    assert result.status=='none' and result.video_path is None
    receipt=read_json(transition.output_path.with_suffix('.presentation.json'),run_root=tmp_path)
    assert receipt['presentation_before_observation_and_commit'] is True
    assert receipt['panorama_state_sha256_after']==before; assert not receipt['video_writes_state']


def test_missing_ffmpeg_writes_real_frames(tmp_path,monkeypatch):
    state,profile,nodes,transition=setup(tmp_path)
    monkeypatch.setattr(cpu.shutil,'which',lambda _:None)
    result=present_transition(transition,panorama=state,source_node=nodes[0],target_node=nodes[1],
                             backend=NoneVideoBackend(),profile=profile,run_root=tmp_path)
    assert result.status=='none' and not transition.output_path.exists()
    frames=sorted((tmp_path/'video.frames').glob('*.png'))
    assert len(frames)==3
    assert np.array_equal(np.array(Image.open(frames[0])),np.array(Image.open(transition.first_frame)))
    assert np.array_equal(np.array(Image.open(frames[-1])),np.array(Image.open(transition.last_frame)))


def test_changed_endpoint_rejected_before_render(tmp_path):
    state,profile,nodes,transition=setup(tmp_path)
    Image.new('RGB',(32,16),'black').save(transition.first_frame)
    nodes[0]['view']=file_ref(transition.first_frame,run_root=tmp_path,image=True)
    with pytest.raises(ValueError,match='authoritative panorama render'):
        present_transition(transition,panorama=state,source_node=nodes[0],target_node=nodes[1],
                             backend=NoneVideoBackend(),profile=profile,run_root=tmp_path)


def test_actual_cpu_encoder_exact_endpoints_and_receipt_reuse(tmp_path):
    if not shutil.which('ffmpeg'):pytest.skip('CPU ffmpeg unavailable')
    state,profile,nodes,transition=setup(tmp_path)
    result=present_transition(transition,panorama=state,source_node=nodes[0],target_node=nodes[1],
                             backend=NoneVideoBackend(),profile=profile,run_root=tmp_path)
    assert result.status=='rendered' and result.video_path.stat().st_size>0
    assert present_transition(transition,panorama=state,source_node=nodes[0],target_node=nodes[1],
                             backend=NoneVideoBackend(),profile=profile,run_root=tmp_path)==result
    cpu.verify_encoded_endpoints(result.video_path,np.array(Image.open(transition.first_frame)),
                                 np.array(Image.open(transition.last_frame)),3)
