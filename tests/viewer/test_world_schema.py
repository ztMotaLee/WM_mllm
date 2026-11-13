# SPDX-License-Identifier: Apache-2.0
"""Published procedural playback must satisfy the closed public schema."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from .conftest import RELEASE_ROOT, VIEWER_ROOT, load_script


@pytest.fixture(scope="session")
def committed_world():
    path = VIEWER_ROOT / "worlds/procedural/world.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def validate(schema, value):
    jsonschema.Draft202012Validator(schema).validate(value)


def test_public_schema_is_valid(world_schema):
    jsonschema.Draft202012Validator.check_schema(world_schema)


def test_committed_procedural_fixture_matches_schema(world_schema, committed_world):
    path, world = committed_world
    validate(world_schema, world)
    assert world["procedural_fixture"] is True
    assert "procedural" in world["title"].casefold()
    assert world["attribution"]
    assert world["nodes"] and world["actions"]
    for node in world["nodes"]:
        for key in ("panorama", "endpoint"):
            relative = node[key]
            assert relative.startswith("media/")
            assert (path.parent / relative).is_file()
    videos = [action["presentation"]["video"] for action in world["actions"] if action["presentation"]["kind"] == "rendered"]
    assert videos, "The fixture must exercise actual CPU-rendered playback"
    assert all((path.parent / relative).is_file() for relative in videos)
    assert any(action["presentation"]["kind"] == "none" for action in world["actions"])


def test_generated_fixture_matches_schema(exporter, world_schema, tmp_path):
    fixture = load_script("viewer_procedural_fixture", RELEASE_ROOT / "examples/viewer_fixture/make_fixture.py")
    run_root = tmp_path / "generated_run"
    result = fixture.make_run(run_root)
    assert isinstance(result, Path)
    assert (run_root / "session.json").is_file()
    world_dir = exporter.export_world(run_root, tmp_path / "worlds", procedural_fixture=True)
    path = world_dir / "world.json"
    world = json.loads(path.read_text(encoding="utf-8"))
    validate(world_schema, world)
    assert world["procedural_fixture"] is True
    assert any(action["presentation"]["kind"] == "rendered" for action in world["actions"])
    assert any(action["presentation"]["kind"] == "none" for action in world["actions"])


SCHEMA_MUTATIONS = [
    ((), "schema", "other.schema"),
    ((), "private_payload", "UNEXPORTED_SCHEMA_CANARY"),
    (("nodes", 0), "view_identity", "UNEXPORTED_SCHEMA_CANARY"),
    (("nodes", 0), "panorama_state", {"path": "logical/state.json"}),
    (("nodes", 0), "endpoint", "/endpoint.png"),
    (("nodes", 0), "endpoint", "media/../endpoint.png"),
    (("nodes", 0), "endpoint", "media//endpoint.png"),
    (("nodes", 0), "endpoint", "media/./endpoint.png"),
    (("nodes", 0), "endpoint", "media\\endpoint.png"),
    (("nodes", 0), "endpoint", "https" + "://example.invalid/endpoint.png"),
    (("nodes", 0), "endpoint", "media/endpoint.png?x=1"),
    (("nodes", 0), "endpoint", "media/endpoint.png#label"),
    (("nodes", 0), "endpoint", "media/a%20b.png"),
    (("nodes", 0), "endpoint", "media/endpoint.jpg"),
    (("nodes", 0), "revision", -1),
    (("nodes", 0), "seam_status", "hidden"),
    (("nodes", 0, "pose"), "private_payload", "UNEXPORTED_SCHEMA_CANARY"),
    (("nodes", 0, "viewport"), "width", 0),
    (("nodes", 0, "viewport"), "padding_ltrb", [0, 0, 0]),
    (("actions", 0), "status", "unknown"),
    (("actions", 0), "semantic", "ACCEPTED"),
    (("actions", 0), "operations", ["unknown"]),
    (("actions", 0), "decision", {"path": "logical/decision.json"}),
    (("actions", 0, "presentation"), "backend", "unexported_backend"),
    (("actions", 0, "presentation"), "kind", "crossfade"),
    (("actions", 0, "presentation"), "num_frames", 0),
    (("actions", 0, "presentation"), "fps", -1),
    (("actions", 0, "presentation"), "duration_seconds", -1),
]


@pytest.mark.parametrize("location,key,value", SCHEMA_MUTATIONS)
def test_schema_rejects_adversarial_public_manifest(world_schema, committed_world, location, key, value):
    _, fixture = committed_world
    world = copy.deepcopy(fixture)
    target = world
    for component in location:
        target = target[component]
    target[key] = value
    with pytest.raises(jsonschema.ValidationError):
        validate(world_schema, world)


@pytest.mark.parametrize("kind", ["rendered", "generated"])
def test_schema_requires_video_for_visible_presentation(world_schema, committed_world, kind):
    _, fixture = committed_world
    world = copy.deepcopy(fixture)
    world["actions"][0]["presentation"].update(
        kind=kind, video=None, reason=None, num_frames=2, fps=2, duration_seconds=1
    )
    with pytest.raises(jsonschema.ValidationError):
        validate(world_schema, world)


@pytest.mark.parametrize("reason", [None, ""])
def test_schema_requires_reason_for_absent_video(world_schema, committed_world, reason):
    _, fixture = committed_world
    world = copy.deepcopy(fixture)
    world["actions"][0]["presentation"].update(kind="none", video=None, reason=reason)
    with pytest.raises(jsonschema.ValidationError):
        validate(world_schema, world)


def test_schema_forbids_video_when_presentation_is_none(world_schema, committed_world):
    _, fixture = committed_world
    world = copy.deepcopy(fixture)
    world["actions"][0]["presentation"].update(
        kind="none", video="media/video.mp4", reason="Generated video is unavailable."
    )
    with pytest.raises(jsonschema.ValidationError):
        validate(world_schema, world)
