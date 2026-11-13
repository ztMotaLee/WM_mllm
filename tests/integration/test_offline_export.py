"""Real offline session -> packaged exporter -> schema-validated static viewer."""
import importlib.util
import json
from pathlib import Path
import socket
import pytest
import jsonschema
from wmmllm.cli import main
from wmmllm.export_viewer import export_bundle, ExportError
from wmmllm.session.io import validate_session, verify_ref
from wmmllm.viewer import asset_root


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Integration tests forbid network requests")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)


@pytest.fixture(scope="module")
def offline_session(tmp_path_factory):
    example = Path(__file__).resolve().parents[2] / "examples/offline_session"
    patch = pytest.MonkeyPatch()
    patch.syspath_prepend(str(example))
    def blocked(*args, **kwargs):
        raise AssertionError("Offline example forbids network requests")
    patch.setattr(socket.socket, "connect", blocked)
    patch.setattr(socket, "create_connection", blocked)
    patch.setattr(socket, "getaddrinfo", blocked)
    spec = importlib.util.spec_from_file_location("integration_offline_example", example / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        yield module.run(tmp_path_factory.mktemp("offline-session") / "run")
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


def test_closed_session_projection(offline_session):
    for node in offline_session.session_record()["nodes"]:
        assert set(node["panorama_state"]) == {"path", "sha256", "panorama_state_sha256"}
        assert set(node["viewport_profile"]) == {"width", "height", "content_box", "padding_ltrb"}
        verify_ref(node["panorama_state"], run_root=offline_session.run_root)
        verify_ref(node["view"], run_root=offline_session.run_root)


def test_current_revision_survives_revisit(offline_session):
    root, rotation, edit, revisit = offline_session.session_record()["nodes"]
    assert [n["physical_revision"] for n in (root, rotation, edit, revisit)] == [0, 0, 1, 1]
    assert root["panorama_state"] == rotation["panorama_state"]
    assert edit["panorama_state"] == revisit["panorama_state"]
    assert root["panorama_state"]["panorama_state_sha256"] != edit["panorama_state"]["panorama_state_sha256"]
    assert revisit["pose"]["yaw"] == root["pose"]["yaw"] == 0


@pytest.fixture
def exported(offline_session, tmp_path):
    out = tmp_path / "viewer"
    assert main(["export-viewer", "--run", str(offline_session.run_root), "--out", str(out)]) == 0
    return out, json.loads((out / "worlds/prepared/world.json").read_text())


def test_export_schema_and_procedural_label(exported):
    out, world = exported
    schema = json.loads((asset_root() / "schema/world.v1.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(world)
    assert world["procedural_fixture"] is True
    assert "OFFLINE PROCEDURAL FIXTURE" in world["title"]
    assert len(world["nodes"]) == 4 and len(world["actions"]) == 3
    assert world["current_node_id"] == world["nodes"][-1]["id"]
    assert 'name="wmmllm-default-world" content="prepared"' in (out / "index.html").read_text()
    assert (out / "assets/app.js").is_file()


def test_export_contains_real_media_and_excludes_private_state(exported):
    out, world = exported
    for node in world["nodes"]:
        assert not {"memory_snapshot", "panorama_state", "view_identity"} & node.keys()
        for key in ("panorama", "endpoint"):
            assert (out / "worlds/prepared" / node[key]).is_file()
    assert [a["presentation"]["kind"] for a in world["actions"]] == ["rendered", "none", "rendered"]
    for action in world["actions"]:
        present = action["presentation"]
        if present["kind"] == "none":
            assert present["video"] is None and present["reason"]
        else:
            assert (out / "worlds/prepared" / present["video"]).stat().st_size > 0
    names = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    assert not any(part in name for name in names for part in ("MEMORY", "receipt", "runtime_config", "decision"))


def test_existing_export_is_not_modified(offline_session, tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    sentinel = out / "sentinel.txt"
    sentinel.write_text("retain")
    with pytest.raises(ExportError, match="output_exists"):
        export_bundle(offline_session.run_root, out)
    assert sentinel.read_text() == "retain" and list(out.iterdir()) == [sentinel]


def test_failed_export_cleans_only_partial_output(tmp_path):
    empty_run = tmp_path / "empty-run"
    empty_run.mkdir()
    out = tmp_path / "failed-viewer"
    with pytest.raises(ExportError):
        export_bundle(empty_run, out)
    assert not out.exists() and empty_run.is_dir()


def test_export_cli_rejects_source_symlink(tmp_path, capsys):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "linked-run"
    link.symlink_to(target, target_is_directory=True)
    out = tmp_path / "viewer"
    assert main(["export-viewer", "--run", str(link), "--out", str(out)]) == 1
    assert "symlink_rejected" in capsys.readouterr().err
    assert not out.exists()


def test_cpu_cli_uses_no_provider_config(offline_session, tmp_path, monkeypatch):
    from wmmllm.providers import factory
    monkeypatch.setattr(factory, "create_providers", lambda *a, **k: pytest.fail("CPU command constructed providers"))
    for key in ("WMM_LLM_API_KEY", "WMM_LLM_MODEL", "WMM_IMAGE_API_KEY", "WMM_IMAGE_MODEL"):
        monkeypatch.delenv(key, raising=False)
    assert main(["view", "--run", str(offline_session.run_root), "--yaw", "0", "--pitch", "0", "--out", str(tmp_path / "view.png")]) == 0
    assert main(["export-viewer", "--run", str(offline_session.run_root), "--out", str(tmp_path / "viewer")]) == 0
