"""The offline example runs the real session engine with drawn, credential-free providers."""
import importlib
import json
from pathlib import Path
import pytest
from wmmllm.cli import main
from wmmllm.session.io import validate_session, verify_ref


@pytest.fixture(scope="module")
def offline_session(tmp_path_factory):
    example = Path(__file__).resolve().parents[2] / "examples/offline_session"
    patch = pytest.MonkeyPatch()
    patch.syspath_prepend(str(example))
    try:
        yield importlib.import_module("run").run(tmp_path_factory.mktemp("offline-session") / "run")
    finally:
        patch.undo()


def test_full_session_and_memory(offline_session):
    runner = offline_session
    record = validate_session(runner.session_record(), run_root=runner.run_root)
    assert record["schema"] == "wmmllm.session.v1" and record["status"] == "ready"
    assert record["video_writes_state"] is False
    assert record["state_authority"] == "accepted_panorama_and_endpoint"
    assert [t["operations"] for t in record["transitions"]] == [["rotate"], ["local_edit"], ["revisit"]]
    assert all(t["status"] == "accepted" for t in record["transitions"])
    assert runner.store.snapshot().completed_segments == 3
    fixture = json.loads((runner.run_root / "OFFLINE_FIXTURE.json").read_text())
    assert fixture["model_calls"] == fixture["network_requests"] == 0
    for node in record["nodes"]:
        assert set(node["panorama_state"]) == {"path", "sha256", "panorama_state_sha256"}
        assert set(node["viewport_profile"]) == {"width", "height", "content_box", "padding_ltrb"}
        verify_ref(node["panorama_state"], run_root=runner.run_root)
        verify_ref(node["view"], run_root=runner.run_root)


def test_current_revision_survives_revisit(offline_session):
    root, rotation, edit, revisit = offline_session.session_record()["nodes"]
    assert [n["physical_revision"] for n in (root, rotation, edit, revisit)] == [0, 0, 1, 1]
    assert root["panorama_state"] == rotation["panorama_state"]
    assert edit["panorama_state"] == revisit["panorama_state"]
    assert root["panorama_state"]["panorama_state_sha256"] != edit["panorama_state"]["panorama_state_sha256"]
    assert revisit["pose"]["yaw"] == root["pose"]["yaw"] == 0


def test_cpu_view_uses_no_provider_config(offline_session, tmp_path, monkeypatch):
    from wmmllm.providers import factory
    monkeypatch.setattr(factory, "create_providers", lambda *a, **k: pytest.fail("CPU command constructed providers"))
    for key in ("WMM_LLM_API_KEY", "WMM_LLM_MODEL", "WMM_IMAGE_API_KEY", "WMM_IMAGE_MODEL"):
        monkeypatch.delenv(key, raising=False)
    out = tmp_path / "view.png"
    assert main(["view", "--run", str(offline_session.run_root), "--yaw", "0", "--pitch", "0", "--out", str(out)]) == 0
    assert out.stat().st_size > 0
