"""Real CPU camera rendering and explicit unavailable video paths."""
from __future__ import annotations
from pathlib import Path
import shutil
import numpy as np
import pytest
from PIL import Image
from wmmllm.state.panorama import create_panorama_state,render_view,panorama_state_sha256
from wmmllm.video import cpu
from wmmllm.video.presentation import present_transition,read_presentation_receipt
from wmmllm.providers.video import NoneVideoBackend
from wmmllm.providers.types import VideoTransition
from wmmllm.session.io import file_ref


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
    assert cpu.rotation_poses(170,-170,num_frames=3,shortest_yaw=True)[1]['yaw']==180
    assert cpu.rotation_poses(0,40,num_frames=3,easing='smoothstep')[1]['yaw']==20


def test_none_backend_retains_state_and_receipt(tmp_path):
    state,profile,nodes,transition=setup(tmp_path)
    transition=VideoTransition('edit','local_edit',transition.first_frame,transition.last_frame,
                               output_path=tmp_path/'edit.mp4')
    before=panorama_state_sha256(state)
    result=present_transition(transition,panorama=state,source_node=nodes[0],target_node=nodes[1],
                             backend=NoneVideoBackend(),profile=profile,run_root=tmp_path)
    assert result.status=='none' and result.video_path is None
    receipt=read_presentation_receipt(transition,run_root=tmp_path)
    assert receipt['presentation_before_observation_and_commit'] is True
    assert receipt['panorama_state_sha256_after']==before
    assert not receipt['video_writes_state']


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


def test_stale_revision_uses_none_backend(tmp_path,monkeypatch):
    state,profile,nodes,transition=setup(tmp_path)
    nodes[0]['physical_revision']=1
    monkeypatch.setattr(cpu,'render_rotation_video',lambda *a,**k:pytest.fail('stale revision rendered'))
    result=present_transition(transition,panorama=state,source_node=nodes[0],target_node=nodes[1],
                             backend=NoneVideoBackend(),profile=profile,run_root=tmp_path)
    assert result.status=='none'


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


def test_real_cpu_hold_exact_pixels(tmp_path):
    if not shutil.which('ffmpeg'):pytest.skip('CPU ffmpeg unavailable')
    state,profile,nodes,transition=setup(tmp_path)
    result=cpu.encode_hold_video(nodes[0]['view'],profile,tmp_path/'hold.mp4',run_root=tmp_path)
    assert len(set(result['raw_frame_pixel_sha256']))==1
    assert result['accepted_state_unchanged']


def test_same_origin_revisit_renders_current_state(tmp_path,monkeypatch):
    state,profile,nodes,transition=setup(tmp_path)
    monkeypatch.setattr(cpu.shutil,'which',lambda _:None)
    transition=VideoTransition('return','revisit',transition.first_frame,transition.last_frame,
                               output_path=tmp_path/'return.mp4')
    result=present_transition(transition,panorama=state,source_node=nodes[0],target_node=nodes[1],
                             backend=NoneVideoBackend(),profile=profile,run_root=tmp_path)
    assert result.status=='none'
    assert len(list((tmp_path/'return.frames').glob('*.png')))==3


def test_missing_two_view_geometry_is_explicit_unsupported(tmp_path):
    from wmmllm.session.io import read_json
    state,profile,nodes,transition=setup(tmp_path)
    # Saved-node integrity runs before any geometry product or mesh use.
    with pytest.raises((KeyError,ValueError)):
        cpu.render_known_return_video({'status':'NOT_READY'},nodes[0]['view'],nodes[1],
            tmp_path/'geometry.mp4',width=32,height=16,num_frames=3,fps=12,run_root=tmp_path)
    assert not (tmp_path/'geometry.mp4').exists()
