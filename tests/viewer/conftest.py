# SPDX-License-Identifier: Apache-2.0
"""Small procedural inputs for static-export tests; never dispatch providers."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import sys

from PIL import Image
import pytest


from wmmllm.viewer import asset_root

RELEASE_ROOT = Path(__file__).resolve().parents[2]
VIEWER_ROOT = asset_root()


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_ref(run_root: Path, relative: str) -> dict:
    return {"path": relative, "sha256": sha256_file(run_root / relative)}


def image_ref(run_root: Path, relative: str) -> dict:
    ref = file_ref(run_root, relative)
    with Image.open(run_root / relative) as source:
        rgb = source.convert("RGB")
        header = f"RGB uint8 {rgb.width} {rgb.height}\n".encode("ascii")
        ref.update(
            pixel_sha256=hashlib.sha256(header + rgb.tobytes()).hexdigest(),
            size=[rgb.width, rgb.height],
        )
    return ref


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_session(run_root: Path, value: dict) -> None:
    write_json(run_root / "session.json", value)


def procedural_image(size: tuple[int, int]) -> Image.Image:
    image = Image.new("RGB", size)
    image.putdata(
        [
            ((x * 31) % 256, (y * 47) % 256, ((x + y) * 19) % 256)
            for y in range(size[1])
            for x in range(size[0])
        ]
    )
    return image


def make_synthetic_run(run_root: Path) -> dict:
    """Produce real images and explicit test-only logical artifacts."""
    (run_root / "images").mkdir(parents=True)
    procedural_image((16, 8)).save(run_root / "images/panorama.png")
    procedural_image((8, 6)).save(run_root / "images/first.png")
    procedural_image((8, 6)).transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(
        run_root / "images/second.png"
    )
    logical = {"schema": "test.logical.v1", "text": "UNEXPORTED_LOGICAL_PAYLOAD"}
    for relative in (
        "logical/state.json",
        "logical/memory.json",
        "logical/decision.json",
        "logical/verification.json",
        "logical/extension.json",
    ):
        write_json(run_root / relative, logical)
    state_ref = file_ref(run_root, "logical/state.json")
    state_ref["panorama_state_sha256"] = hashlib.sha256(b"test state").hexdigest()
    memory_ref = file_ref(run_root, "logical/memory.json")
    profile = {
        "width": 8,
        "height": 6,
        "content_box": [0, 0, 8, 6],
        "padding_ltrb": [0, 0, 0, 0],
    }
    nodes = []
    for index, node_id in enumerate(("sourceNodeA", "sourceNodeB")):
        nodes.append(
            {
                "node_id": node_id,
                "parent_node_id": None if index == 0 else "sourceNodeA",
                "origin_id": "sourceOriginA",
                "physical_revision": index,
                "view_identity": {"identity": "sensitiveViewIdentityCanary"},
                "pose": {"yaw": 45 * index, "pitch": 0, "hfov": 90},
                "viewport_profile": copy.deepcopy(profile),
                "panorama_state": copy.deepcopy(state_ref),
                "panorama": image_ref(run_root, "images/panorama.png"),
                "view": image_ref(run_root, f"images/{'first' if index == 0 else 'second'}.png"),
                "memory_snapshot": copy.deepcopy(memory_ref),
                "seam_status": "resolved",
            }
        )
    session = {
        "schema": "wmmllm.session.v1",
        "session_id": "sourceSessionA",
        "title": "Procedural test room",
        "seed": 0,
        "status": "ready",
        "state_authority": "accepted_panorama_and_endpoint",
        "video_writes_state": False,
        "initial_node_id": "sourceNodeA",
        "current_node_id": "sourceNodeB",
        "nodes": nodes,
        "transitions": [
            {
                "transition_id": "sourceTurnA",
                "index": 1,
                "from_node_id": "sourceNodeA",
                "to_node_id": "sourceNodeB",
                "caption": "Move the blue cube.",
                "operations": ["local_edit"],
                "status": "accepted",
                "semantic": {
                    "status": "PASS",
                    "attempt_count": 1,
                    "semantic_accepted": True,
                    "uncertain_accepted_and_labeled": False,
                    "verification": file_ref(run_root, "logical/verification.json"),
                },
                "presentation": {
                    "status": "none",
                    "backend": "none",
                    "video": None,
                    "reason": "Generated video is unavailable.",
                    "num_frames": None,
                    "fps": None,
                    "duration_seconds": None,
                },
                "decision": file_ref(run_root, "logical/decision.json"),
                "memory_after": copy.deepcopy(memory_ref),
            }
        ],
        "memory": copy.deepcopy(memory_ref),
        "extensions": {
            "test:private": {
                "asset": file_ref(run_root, "logical/extension.json"),
                "observation": "UNEXPORTED_LOGICAL_PAYLOAD",
            }
        },
    }
    write_session(run_root, session)
    return session


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Viewer tests must not use the network")

    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)


@pytest.fixture(scope="session")
def exporter():
    from wmmllm import export_viewer
    return export_viewer


@pytest.fixture
def synthetic_run(tmp_path):
    run_root = tmp_path / "run"
    session = make_synthetic_run(run_root)
    return run_root, session


@pytest.fixture(scope="session")
def world_schema():
    return json.loads((VIEWER_ROOT / "schema/world.v1.json").read_text(encoding="utf-8"))

