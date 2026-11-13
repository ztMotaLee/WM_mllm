# SPDX-License-Identifier: Apache-2.0
"""Exporter privacy, integrity, and containment checks with real small images."""

from __future__ import annotations

import builtins
import copy
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps, PngImagePlugin
import pytest

from .conftest import (
    RELEASE_ROOT,
    file_ref,
    image_ref,
    load_script,
    procedural_image,
    write_session,
)


def export(exporter, run_root, worlds_dir, **kwargs):
    world_dir = exporter.export_world(run_root, worlds_dir, **kwargs)
    assert isinstance(world_dir, Path)
    manifest_path = world_dir / "world.json"
    return manifest_path, json.loads(manifest_path.read_text(encoding="utf-8"))


def test_exporter_import_has_no_engine_dependency(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith(("wmmllm.agent", "wmmllm.providers", "wmmllm.config", "wmmllm.session")):
            raise AssertionError("Exporter must not import engine or provider code")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    from wmmllm import export_viewer
    module = load_script("viewer_export_no_engine", Path(export_viewer.__file__))
    assert callable(module.export_world)
    assert issubclass(module.ExportError, ValueError)


def test_export_remaps_graph_and_keeps_media_relative(exporter, synthetic_run, tmp_path):
    run_root, session = synthetic_run
    path, world = export(exporter, run_root, tmp_path / "worlds", world_id="test-world")
    assert path == tmp_path / "worlds/test-world/world.json"
    assert world["id"] == "test-world"
    assert world["schema"] == "wmmllm.world.v1"
    assert world["procedural_fixture"] is False
    assert world["title"] == session["title"]
    first, second = world["nodes"]
    action = world["actions"][0]
    assert first["id"] != session["nodes"][0]["node_id"]
    assert second["id"] != session["nodes"][1]["node_id"]
    assert action["id"] != session["transitions"][0]["transition_id"]
    assert first["place_id"] != session["nodes"][0]["origin_id"]
    assert first["place_id"] == second["place_id"]
    assert first["parent_id"] is None
    assert second["parent_id"] == first["id"]
    assert world["initial_node_id"] == first["id"]
    assert world["current_node_id"] == second["id"]
    assert (action["from"], action["to"]) == (first["id"], second["id"])
    assert action["label"] == "Move the blue cube."
    assert action["semantic"] == "PASS"
    assert action["presentation"]["kind"] == "none"
    assert action["presentation"]["video"] is None
    assert action["presentation"]["reason"]
    for node in world["nodes"]:
        for key in ("panorama", "endpoint"):
            relative = node[key]
            assert relative.startswith("media/") and relative.endswith(".png")
            assert not Path(relative).is_absolute()
            assert ".." not in Path(relative).parts
            assert (path.parent / relative).is_file()


def test_export_uses_only_public_field_allowlist(exporter, synthetic_run, tmp_path):
    run_root, _ = synthetic_run
    path, world = export(exporter, run_root, tmp_path / "worlds")
    assert set(world) == {
        "schema", "id", "title", "procedural_fixture", "attribution",
        "initial_node_id", "current_node_id", "nodes", "actions",
    }
    for node in world["nodes"]:
        assert set(node) == {
            "id", "parent_id", "place_id", "revision", "pose", "viewport",
            "panorama", "endpoint", "seam_status",
        }
        assert set(node["pose"]) == {"yaw", "pitch", "hfov"}
        assert set(node["viewport"]) == {"width", "height", "content_box", "padding_ltrb"}
    for action in world["actions"]:
        assert set(action) == {
            "id", "from", "to", "label", "operations", "status", "semantic", "presentation",
        }
        assert set(action["presentation"]) == {
            "kind", "video", "reason", "num_frames", "fps", "duration_seconds",
        }
    serialized = path.read_text(encoding="utf-8")
    for value in (
        "sourceNodeA", "sourceNodeB", "sourceOriginA", "sourceTurnA", "sourceSessionA",
        "view_identity", "sensitiveViewIdentityCanary", "UNEXPORTED_LOGICAL_PAYLOAD",
        "panorama_state", "memory_snapshot", "verification", "decision", "extensions",
    ):
        assert value not in serialized
    files = [entry for entry in path.parent.rglob("*") if entry.is_file()]
    assert all(entry == path or entry.suffix in {".png", ".mp4"} for entry in files)
    assert all(entry == path or entry.relative_to(path.parent).parts[0] == "media" for entry in files)


def test_new_origins_remap_to_distinct_places(exporter, synthetic_run, tmp_path):
    run_root, session = synthetic_run
    session["nodes"][1]["origin_id"] = "sourceOriginB"
    write_session(run_root, session)
    _, world = export(exporter, run_root, tmp_path / "worlds")
    assert world["nodes"][0]["place_id"] != world["nodes"][1]["place_id"]
    assert "sourceOriginB" not in json.dumps(world)


def test_unreachable_saved_node_is_rejected(exporter, synthetic_run, tmp_path):
    run_root, session = synthetic_run
    orphan = copy.deepcopy(session["nodes"][1])
    orphan["node_id"] = "unreachableSavedState"
    session["nodes"].append(orphan)
    write_session(run_root, session)
    with pytest.raises(exporter.ExportError):
        exporter.export_world(run_root, tmp_path / "worlds")


def test_png_metadata_is_erased_without_changing_pixels(exporter, synthetic_run, tmp_path):
    run_root, session = synthetic_run
    source = run_root / "images/first.png"
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Comment", "PRIVATE_IMAGE_METADATA_CANARY")
    metadata.add_itxt("Description", "PRIVATE_IMAGE_METADATA_CANARY")
    exif = Image.Exif()
    exif[0x010E] = "PRIVATE_IMAGE_METADATA_CANARY"
    procedural_image((8, 6)).save(source, pnginfo=metadata, exif=exif)
    session["nodes"][0]["view"] = image_ref(run_root, "images/first.png")
    write_session(run_root, session)
    path, world = export(exporter, run_root, tmp_path / "worlds")
    public = path.parent / world["nodes"][0]["endpoint"]
    with Image.open(source) as before, Image.open(public) as after:
        assert after.format == "PNG"
        assert after.convert("RGB").tobytes() == before.convert("RGB").tobytes()
        assert not after.getexif()
        assert not after.text
        assert "exif" not in after.info
        assert "icc_profile" not in after.info
    assert b"PRIVATE_IMAGE_METADATA_CANARY" not in public.read_bytes()


def test_jpeg_orientation_is_applied_after_raw_pixel_verification(exporter, synthetic_run, tmp_path):
    run_root, session = synthetic_run
    source = run_root / "images/oriented.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6
    exif[0x010E] = "PRIVATE_JPEG_METADATA_CANARY"
    procedural_image((8, 6)).save(source, quality=98, exif=exif)
    session["nodes"][0]["view"] = image_ref(run_root, "images/oriented.jpg")
    write_session(run_root, session)
    with Image.open(source) as image:
        expected = ImageOps.exif_transpose(image).convert("RGB")
        expected_pixels, expected_size = expected.tobytes(), expected.size
    path, world = export(exporter, run_root, tmp_path / "worlds")
    public = path.parent / world["nodes"][0]["endpoint"]
    with Image.open(public) as image:
        assert image.format == "PNG"
        assert image.size == expected_size
        assert image.convert("RGB").tobytes() == expected_pixels
        assert not image.getexif()
    assert b"PRIVATE_JPEG_METADATA_CANARY" not in public.read_bytes()


@pytest.mark.parametrize("field", ["sha256", "pixel_sha256", "size"])
def test_image_integrity_fields_are_verified(exporter, synthetic_run, tmp_path, field):
    run_root, session = synthetic_run
    session["nodes"][0]["view"][field] = [7, 6] if field == "size" else "0" * 64
    write_session(run_root, session)
    with pytest.raises(exporter.ExportError):
        exporter.export_world(run_root, tmp_path / "worlds")
    assert not (tmp_path / "worlds/procedural/world.json").exists()


def test_pixel_hash_requires_the_source_header_convention(exporter, synthetic_run, tmp_path):
    run_root, session = synthetic_run
    with Image.open(run_root / "images/first.png") as source:
        raw_only = hashlib.sha256(source.convert("RGB").tobytes()).hexdigest()
    session["nodes"][0]["view"]["pixel_sha256"] = raw_only
    write_session(run_root, session)
    with pytest.raises(exporter.ExportError):
        exporter.export_world(run_root, tmp_path / "worlds")


@pytest.mark.parametrize("media_key", ["panorama", "view"])
def test_missing_required_image_is_an_error(exporter, synthetic_run, tmp_path, media_key):
    run_root, session = synthetic_run
    (run_root / session["nodes"][0][media_key]["path"]).unlink()
    with pytest.raises(exporter.ExportError):
        exporter.export_world(run_root, tmp_path / "worlds")


@pytest.mark.parametrize("kind", ["rendered", "generated"])
def test_missing_required_video_is_an_error(exporter, synthetic_run, tmp_path, kind):
    run_root, session = synthetic_run
    session["transitions"][0]["presentation"].update(
        status=kind,
        backend="cpu_rotation" if kind == "rendered" else "external_video",
        video={"path": "videos/missing.mp4", "sha256": "0" * 64},
        reason=None,
        num_frames=2,
        fps=2,
        duration_seconds=1,
    )
    write_session(run_root, session)
    with pytest.raises(exporter.ExportError):
        exporter.export_world(run_root, tmp_path / "worlds")


@pytest.mark.parametrize("contents", [b"", b"invalid encoded media"])
def test_empty_or_invalid_video_is_an_error(exporter, synthetic_run, tmp_path, contents):
    run_root, session = synthetic_run
    video = run_root / "videos/invalid.mp4"
    video.parent.mkdir()
    video.write_bytes(contents)
    session["transitions"][0]["presentation"].update(
        status="rendered", backend="cpu_rotation", video=file_ref(run_root, "videos/invalid.mp4"),
        reason=None, num_frames=2, fps=2, duration_seconds=1,
    )
    write_session(run_root, session)
    with pytest.raises(exporter.ExportError):
        exporter.export_world(run_root, tmp_path / "worlds")


INVALID_PATHS = [
    "", ".", "..", "/endpoint.png", "../endpoint.png", "images/../endpoint.png",
    "images//endpoint.png", "./images/first.png", "images/./first.png",
    "images\\first.png", "C:/endpoint.png", "//example.invalid/share.png",
    "https" + "://example.invalid/endpoint.png", "images/first.png?x=1",
    "images/first.png#caption", "images/%2e%2e/endpoint.png", "images/a%20b.png",
    "images/a:b.png", "images/user@host.png", "images/first.png\x00extra",
]


@pytest.mark.parametrize("relative", INVALID_PATHS)
def test_unsafe_input_media_paths_are_rejected(exporter, synthetic_run, tmp_path, relative):
    run_root, session = synthetic_run
    session["nodes"][0]["view"]["path"] = relative
    write_session(run_root, session)
    with pytest.raises(exporter.ExportError):
        exporter.export_world(run_root, tmp_path / "worlds")


@pytest.mark.parametrize("where", ["session", "media", "media_parent", "run_root", "run_ancestor"])
@pytest.mark.parametrize("target_kind", ["inside", "outside", "dangling"])
def test_input_symlinks_are_rejected(exporter, synthetic_run, tmp_path, where, target_kind):
    run_root, session = synthetic_run
    outside = tmp_path / "outside"
    outside.mkdir()
    if where in {"session", "media"}:
        original = run_root / ("session.json" if where == "session" else "images/first.png")
        contents = original.read_bytes()
        original.unlink()
        target = (run_root / "target.bin") if target_kind == "inside" else outside / "target.bin"
        if target_kind != "dangling":
            target.write_bytes(contents)
        original.symlink_to(target)
    elif where == "media_parent":
        original = run_root / "images"
        target = (run_root / "saved_images") if target_kind == "inside" else outside / "images"
        if target_kind == "dangling":
            for child in original.iterdir():
                child.unlink()
            original.rmdir()
        else:
            original.rename(target)
        original.symlink_to(target, target_is_directory=True)
    elif where == "run_root":
        target = run_root
        run_root = tmp_path / "run_alias"
        if target_kind == "dangling":
            target = tmp_path / "absent_run"
        run_root.symlink_to(target, target_is_directory=True)
    else:
        target = tmp_path / "alias_parent"
        target.symlink_to(tmp_path / ("absent_parent" if target_kind == "dangling" else "."), target_is_directory=True)
        run_root = target / "run"
    with pytest.raises(exporter.ExportError):
        exporter.export_world(run_root, tmp_path / "worlds")


@pytest.mark.parametrize("where", ["worlds", "worlds_ancestor", "world_leaf"])
@pytest.mark.parametrize("target_kind", ["inside", "outside", "dangling"])
def test_output_symlinks_are_rejected(exporter, synthetic_run, tmp_path, where, target_kind):
    run_root, _ = synthetic_run
    destination = tmp_path / "worlds"
    target = tmp_path / ("missing_target" if target_kind == "dangling" else "output_target")
    if target_kind != "dangling":
        target.mkdir()
        (target / "sentinel.txt").write_text("preserve", encoding="utf-8")
    if where == "worlds":
        destination.symlink_to(target, target_is_directory=True)
    elif where == "worlds_ancestor":
        alias = tmp_path / "alias"
        alias.symlink_to(target, target_is_directory=True)
        destination = alias / "worlds"
    else:
        destination.mkdir()
        (destination / "procedural").symlink_to(target, target_is_directory=True)
    with pytest.raises(exporter.ExportError):
        exporter.export_world(run_root, destination)
    if target_kind != "dangling":
        assert (target / "sentinel.txt").read_text(encoding="utf-8") == "preserve"
        assert not (target / "world.json").exists()


@pytest.mark.parametrize("world_id", ["", ".", "..", "../other", "a/b", "a\\b", "/world", "C:world", "a%20b", "a?b", "a#b"])
def test_unsafe_world_ids_are_rejected(exporter, synthetic_run, tmp_path, world_id):
    run_root, _ = synthetic_run
    with pytest.raises(exporter.ExportError):
        exporter.export_world(run_root, tmp_path / "worlds", world_id=world_id)


def test_existing_output_is_rejected_without_clobber(exporter, synthetic_run, tmp_path):
    run_root, _ = synthetic_run
    path, _ = export(exporter, run_root, tmp_path / "worlds")
    before = {file.relative_to(path.parent).as_posix(): file.read_bytes() for file in path.parent.rglob("*") if file.is_file()}
    with pytest.raises(exporter.ExportError):
        exporter.export_world(run_root, tmp_path / "worlds")
    after = {file.relative_to(path.parent).as_posix(): file.read_bytes() for file in path.parent.rglob("*") if file.is_file()}
    assert after == before


def test_existing_nonempty_world_directory_is_not_reused(exporter, synthetic_run, tmp_path):
    run_root, _ = synthetic_run
    destination = tmp_path / "worlds/procedural"
    destination.mkdir(parents=True)
    sentinel = destination / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(exporter.ExportError):
        exporter.export_world(run_root, tmp_path / "worlds")
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert list(destination.iterdir()) == [sentinel]


def redaction_canaries():
    # Construct synthetic sensitive forms so static scanners see no real identity.
    return [
        "https" + "://example.invalid/private?token=synthetic",
        "reader" + chr(64) + "example.invalid",
        "/" + "/".join(("srv", "private", "scene")),
        "open" + "ai/" + "open" + "ai/synthetic-model",
        "Bear" + "er " + "SyntheticSecretValue123456",
        "sk" + "-" + "A" * 40,
        "api" + "_key=" + "SyntheticSecretValue123456",
    ]


@pytest.mark.parametrize("field", ["title", "caption", "reason"])
@pytest.mark.parametrize("canary", redaction_canaries())
def test_display_strings_redact_sensitive_forms(exporter, synthetic_run, tmp_path, field, canary):
    run_root, session = synthetic_run
    value = "Procedural scene " + canary
    if field == "title":
        session["title"] = value
    elif field == "caption":
        session["transitions"][0]["caption"] = value
    else:
        session["transitions"][0]["presentation"]["reason"] = value
    write_session(run_root, session)
    path, _ = export(exporter, run_root, tmp_path / "worlds")
    assert canary not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("status", ["held", "failed"])
def test_held_and_failed_actions_preserve_the_graph(exporter, synthetic_run, tmp_path, status):
    run_root, session = synthetic_run
    transition = session["transitions"][0]
    transition["status"] = status
    transition["to_node_id"] = "sourceNodeA" if status == "held" else None
    transition["semantic"].update(status="FAIL", semantic_accepted=False)
    transition["memory_after"] = session["memory"] if status == "held" else None
    session["nodes"] = session["nodes"][:1]
    session["current_node_id"] = "sourceNodeA"
    write_session(run_root, session)
    _, world = export(exporter, run_root, tmp_path / "worlds")
    assert world["current_node_id"] == world["initial_node_id"]
    assert world["actions"][0]["status"] == status
    expected_to = world["initial_node_id"] if status == "held" else None
    assert world["actions"][0]["to"] == expected_to


def test_uncertain_semantics_remain_labeled(exporter, synthetic_run, tmp_path):
    run_root, session = synthetic_run
    session["transitions"][0]["semantic"].update(
        status="UNCERTAIN", semantic_accepted=False, uncertain_accepted_and_labeled=True
    )
    write_session(run_root, session)
    _, world = export(exporter, run_root, tmp_path / "worlds")
    assert world["actions"][0]["semantic"] == "UNCERTAIN"


def test_uninitialized_session_does_not_create_a_world(exporter, synthetic_run, tmp_path):
    run_root, session = synthetic_run
    session.update(status="incomplete", initial_node_id=None, current_node_id=None, nodes=[], transitions=[], memory=None)
    write_session(run_root, session)
    with pytest.raises(exporter.ExportError):
        exporter.export_world(run_root, tmp_path / "worlds")
    assert not (tmp_path / "worlds/procedural/world.json").exists()


MALFORMED_SESSION_VALUES = [
    (("status",), []),
    (("initial_node_id",), []),
    (("current_node_id",), {}),
    (("nodes",), {}),
    (("transitions",), {}),
    (("extensions",), []),
    (("title",), {}),
    (("nodes", 0, "node_id"), []),
    (("nodes", 0, "parent_node_id"), []),
    (("nodes", 0, "origin_id"), {}),
    (("nodes", 0, "seam_status"), {}),
    (("nodes", 0, "pose", "yaw"), "invalid"),
    (("nodes", 0, "pose", "hfov"), True),
    (("nodes", 0, "view", "path"), []),
    (("nodes", 0, "view", "pixel_sha256"), {}),
    (("nodes", 0, "view", "size"), [8, True]),
    (("transitions", 0, "caption"), {}),
    (("transitions", 0, "status"), {}),
    (("transitions", 0, "operations"), [[]]),
    (("transitions", 0, "semantic", "status"), []),
    (("transitions", 0, "semantic", "attempt_count"), True),
    (("transitions", 0, "presentation", "status"), []),
    (("transitions", 0, "presentation", "reason"), {}),
]


@pytest.mark.parametrize("location,value", MALFORMED_SESSION_VALUES)
def test_malformed_session_values_raise_safe_export_errors(exporter, synthetic_run, tmp_path, location, value):
    run_root, session = synthetic_run
    target = session
    for component in location[:-1]:
        target = target[component]
    target[location[-1]] = value
    write_session(run_root, session)
    with pytest.raises(exporter.ExportError) as caught:
        exporter.export_world(run_root, tmp_path / "worlds")
    assert str(tmp_path) not in str(caught.value)
    assert "UNEXPORTED_LOGICAL_PAYLOAD" not in str(caught.value)


@pytest.mark.parametrize("kind", ["duplicate_node", "duplicate_transition", "cycle", "unknown_parent", "unknown_current", "nonconsecutive", "unknown_target", "discontinuous", "semantic_flags", "wrong_authority", "video_state_write"])
def test_inconsistent_session_graph_is_rejected(exporter, synthetic_run, tmp_path, kind):
    run_root, session = synthetic_run
    if kind == "duplicate_node":
        session["nodes"].append(copy.deepcopy(session["nodes"][0]))
    elif kind == "duplicate_transition":
        duplicate = copy.deepcopy(session["transitions"][0])
        duplicate.update(index=2, from_node_id="sourceNodeB")
        session["transitions"].append(duplicate)
    elif kind == "cycle":
        session["nodes"][1]["parent_node_id"] = "sourceNodeB"
    elif kind == "unknown_parent":
        session["nodes"][1]["parent_node_id"] = "missingNode"
    elif kind == "unknown_current":
        session["current_node_id"] = "missingNode"
    elif kind == "nonconsecutive":
        session["transitions"][0]["index"] = 2
    elif kind == "unknown_target":
        session["transitions"][0]["to_node_id"] = "missingNode"
    elif kind == "discontinuous":
        session["transitions"][0]["from_node_id"] = "sourceNodeB"
    elif kind == "semantic_flags":
        session["transitions"][0]["semantic"]["semantic_accepted"] = False
    elif kind == "wrong_authority":
        session["state_authority"] = "video"
    else:
        session["video_writes_state"] = True
    write_session(run_root, session)
    with pytest.raises(exporter.ExportError):
        exporter.export_world(run_root, tmp_path / "worlds")


def test_duplicate_json_keys_are_rejected(exporter, synthetic_run, tmp_path):
    run_root, _ = synthetic_run
    source = (run_root / "session.json").read_text(encoding="utf-8")
    (run_root / "session.json").write_text(
        source.replace('"title":', '"title": "Duplicate title", "title":', 1), encoding="utf-8"
    )
    with pytest.raises(exporter.ExportError):
        exporter.export_world(run_root, tmp_path / "worlds")


def test_external_symlink_target_is_not_exposed_in_error(exporter, synthetic_run, tmp_path):
    run_root, _ = synthetic_run
    source = run_root / "images/first.png"
    target = tmp_path / "external-canary.png"
    source.rename(target)
    source.symlink_to(target)
    with pytest.raises(exporter.ExportError) as caught:
        exporter.export_world(run_root, tmp_path / "worlds")
    assert str(target) not in str(caught.value)
    assert "external-canary" not in str(caught.value)


@pytest.mark.parametrize("source_kind", ["extension", "logical", "extra_file"])
def test_internal_files_are_never_recursively_copied(exporter, synthetic_run, tmp_path, source_kind):
    run_root, session = synthetic_run
    if source_kind == "extension":
        session["extensions"]["test:private"]["asset"]["path"] = "../do-not-open.json"
    elif source_kind == "logical":
        (run_root / "logical/private_receipt.txt").write_text("PRIVATE_RECEIPT_CANARY", encoding="utf-8")
    else:
        (run_root / "unused.bin").write_bytes(b"PRIVATE_UNUSED_CANARY")
    write_session(run_root, session)
    path, _ = export(exporter, run_root, tmp_path / "worlds")
    for file in path.parent.rglob("*"):
        if file.is_file():
            assert b"PRIVATE_RECEIPT_CANARY" not in file.read_bytes()
            assert b"PRIVATE_UNUSED_CANARY" not in file.read_bytes()
            assert b"UNEXPORTED_LOGICAL_PAYLOAD" not in file.read_bytes()
