"""Bootstrap preserves native source rays and bounds its reachable seam gate."""
import numpy as np
from wmmllm.agent.bootstrap import reachable_poses,corrected_single_erp_state,correct_colour_collar,prior_erp_mask,preserve_prior_overlap
from wmmllm.state.panorama import render_view,source_profile,erp_array


def test_reachable_schedule_contains_intermediate_camera_and_margin():
    poses=reachable_poses({'camera_path':[{'segment':1,'yaw':45,'pitch':0}]})
    assert (0.,0.) in poses and (45.,0.) in poses and (65.,0.) in poses
    assert (15.,20.) in poses and (30.,-20.) in poses
    assert (-180.,80.) not in poses


def test_coherent_collar_keeps_the_exact_source_patch():
    rng=np.random.default_rng(0);source=rng.integers(0,256,(768,1344,3),np.uint8);erp=rng.integers(0,256,(64,128,3),np.uint8)
    state,record=corrected_single_erp_state(erp,source,None,width=128)
    assert np.array_equal(render_view(state,width=1344,height=768),source)
    row={'initial_image':source,'source_profile':source_profile(source)}
    # Pure collar uses the source viewport for its residual estimate.
    state2,detail=correct_colour_collar(row,state)
    assert np.array_equal(render_view(state2,width=1344,height=768),source)
    assert record['known_ERP_changed_pixels']==0 and detail['outside_support_changed']==0
    assert len(state2['patches'])==1


def test_bootstrap_rejects_unsupported_live_viewport_before_input_or_provider(tmp_path):
    from types import SimpleNamespace
    import pytest
    from wmmllm.agent.bootstrap import initialize
    from wmmllm.config import ConfigError
    deps=SimpleNamespace(config=SimpleNamespace(video=SimpleNamespace(size=(64,32)),image=SimpleNamespace(view_output_size=(1344,768))),run_root=tmp_path)
    with pytest.raises(ConfigError,match='Core sessions require a 1344x768 viewport and image output; no request was sent.'):
        initialize({},tmp_path,deps=deps)
