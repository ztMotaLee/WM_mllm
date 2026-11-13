#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export an explicit session.v1 graph into a portable static viewer world.

Only panorama previews, accepted endpoint pixels and transition video are
exported. Logical state, memory, decisions, receipts and extension data stay in
the run directory. No provider configuration or SDK is imported.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from fractions import Fraction
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any

from PIL import Image, ImageOps


class ExportError(ValueError):
    """A safe field name and failure class, without source data or paths."""


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
_SEGMENT = re.compile(r"[A-Za-z0-9_.-]+\Z")
_HASH = re.compile(r"[a-f0-9]{64}\Z")
_OPS = {"rotate", "revisit", "navigate", "local_edit", "global_change", "subject_action"}
_NONE_REASON = "No generated video; this action uses an instant cut to its saved endpoint."
_TOP = {"schema", "session_id", "title", "seed", "status", "state_authority",
        "video_writes_state", "initial_node_id", "current_node_id", "nodes",
        "transitions", "memory"}
_NODE = {"node_id", "parent_node_id", "origin_id", "physical_revision", "view_identity",
         "pose", "viewport_profile", "panorama_state", "panorama", "view",
         "memory_snapshot", "seam_status"}
_TURN = {"transition_id", "index", "from_node_id", "to_node_id", "caption", "operations",
         "status", "semantic", "presentation", "decision", "memory_after"}


def _fail(field: str, problem: str) -> None:
    raise ExportError(f"{field}: {problem}") from None


def _object(value: Any, required: set[str], field: str, optional: set[str] = frozenset()) -> dict:
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - required - optional:
        _fail(field, "invalid_fields")
    return value


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value) or value in {".", ".."}:
        _fail(field, "invalid_identifier")
    return value


def _number(value: Any, field: str, *, minimum: float | None = None,
            maximum: float | None = None, integer: bool = False) -> float | int:
    if type(value) not in (int, float):
        _fail(field, "invalid_number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        _fail(field, "invalid_number")
    if integer and type(value) is not int:
        _fail(field, "expected_integer")
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        _fail(field, "out_of_range")
    return value


def _relative(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) > 1024:
        _fail(field, "invalid_relative_path")
    parts = value.split("/")
    if not value or any(p in {"", ".", ".."} or not _SEGMENT.fullmatch(p) for p in parts):
        _fail(field, "invalid_relative_path")
    return value


def _anchor(path: Path, field: str, *, create: bool = False) -> Path:
    """Reject symlinks in the explicit anchor and each of its ancestors."""
    path = Path(os.path.abspath(path))
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            if not create:
                _fail(field, "missing_directory")
            try:
                current.mkdir()
            except OSError:
                _fail(field, "directory_creation_failed")
            info = current.lstat()
        if stat.S_ISLNK(info.st_mode):
            _fail(field, "symlink_rejected")
        if not stat.S_ISDIR(info.st_mode):
            _fail(field, "not_directory")
    return path


@contextmanager
def _input_file(root: Path, relative: str, field: str):
    """Walk with openat/O_NOFOLLOW so checked components cannot be swapped."""
    parts = _relative(relative, field).split("/")
    handles: list[int] = []
    try:
        handles.append(os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW))
        for part in parts[:-1]:
            handles.append(os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                   dir_fd=handles[-1]))
        leaf = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                       dir_fd=handles[-1])
        handles.append(leaf)
        info = os.fstat(leaf)
        if not stat.S_ISREG(info.st_mode):
            _fail(field, "not_regular_file")
        if info.st_size < 1 or info.st_size > 256 * 1024 * 1024:
            _fail(field, "invalid_file_size")
        yield leaf, info.st_size
    except OSError:
        _fail(field, "missing_or_unsafe_file")
    finally:
        for handle in reversed(handles):
            os.close(handle)


def _bytes(root: Path, ref: dict, field: str, *, limit: int = 64 * 1024 * 1024) -> bytes:
    with _input_file(root, ref["path"], field) as (descriptor, size):
        if size > limit:
            _fail(field, "file_too_large")
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            data = stream.read(limit + 1)
    if len(data) > limit or hashlib.sha256(data).hexdigest() != ref["sha256"]:
        _fail(field, "encoded_hash_mismatch")
    return data


def _ref(value: Any, field: str, *, image: bool = False, state: bool = False) -> dict:
    keys = {"path", "sha256"} | ({"pixel_sha256", "size"} if image else set())
    keys |= {"panorama_state_sha256"} if state else set()
    obj = _object(value, keys, field)
    _relative(obj["path"], field)
    for key in keys & {"sha256", "pixel_sha256", "panorama_state_sha256"}:
        if not isinstance(obj[key], str) or not _HASH.fullmatch(obj[key]):
            _fail(field, "invalid_hash")
    if image:
        if not isinstance(obj["size"], list) or len(obj["size"]) != 2:
            _fail(field, "invalid_image_size")
        for number in obj["size"]:
            _number(number, field, minimum=1, maximum=32768, integer=True)
    return obj


def _pixel_hash(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    prefix = f"RGB uint8 {rgb.width} {rgb.height}\n".encode("ascii")
    return hashlib.sha256(prefix + rgb.tobytes()).hexdigest()


def _clean_image(root: Path, ref: dict, target: Path, field: str, *, panorama: bool) -> None:
    data = _bytes(root, ref, field)
    try:
        with Image.open(BytesIO(data)) as source:
            if source.format not in {"PNG", "JPEG", "WEBP"} or getattr(source, "n_frames", 1) != 1:
                _fail(field, "unsupported_image")
            if source.width * source.height > 64 * 1024 * 1024:
                _fail(field, "image_too_large")
            source.load()
            if list(source.size) != ref["size"] or _pixel_hash(source) != ref["pixel_sha256"]:
                _fail(field, "decoded_hash_mismatch")
            oriented = ImageOps.exif_transpose(source).convert("RGB")
            if panorama and oriented.width != 2 * oriented.height:
                _fail(field, "panorama_requires_two_to_one")
            # Reconstruct pixels so info/EXIF/XMP/comments/profiles are not inherited.
            clean = Image.frombytes("RGB", oriented.size, oriented.tobytes())
            clean.save(target, format="PNG", compress_level=9)
        with Image.open(target) as check:
            if check.info or check.getexif() or _pixel_hash(check) != _pixel_hash(clean):
                _fail(field, "metadata_or_pixel_mismatch")
    except ExportError:
        raise
    except Exception:
        _fail(field, "image_decode_failed")


def _process(command: list[str], field: str, *, pass_fds: tuple[int, ...] = ()) -> bytes:
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                timeout=120, check=False, pass_fds=pass_fds)
        if result.returncode or len(result.stdout) > 4 * 1024 * 1024:
            _fail(field, "media_tool_failed")
        return result.stdout
    except (OSError, subprocess.TimeoutExpired):
        _fail(field, "media_tool_unavailable_or_timeout")


def _probe(path: str, field: str, *, pass_fds: tuple[int, ...] = ()) -> dict:
    raw = _process(["ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe",
                    "-count_frames", "-show_streams", "-show_format", "-of", "json", path],
                   field, pass_fds=pass_fds)
    try:
        result = json.loads(raw)
        streams = result["streams"]
        video = [s for s in streams if s.get("codec_type") == "video"]
        if len(video) != 1 or video[0].get("codec_name") != "h264":
            _fail(field, "requires_single_h264_video")
        if "mp4" not in result["format"]["format_name"].split(","):
            _fail(field, "requires_mp4_container")
        return result
    except (ValueError, KeyError, TypeError):
        _fail(field, "invalid_media_profile")


def _video_pixel_hash(path: str, field: str, *, pass_fds: tuple[int, ...] = ()) -> bytes:
    return _process(["ffmpeg", "-v", "error", "-nostdin", "-threads", "1",
                     "-protocol_whitelist", "file,pipe", "-i", path, "-map", "0:v:0",
                     "-an", "-sn", "-dn", "-threads", "1", "-pix_fmt", "rgb24",
                     "-f", "hash", "-hash", "sha256", "-"], field, pass_fds=pass_fds)


def _clean_video(root: Path, ref: dict, target: Path, presentation: dict, field: str) -> None:
    if not ref["path"].lower().endswith(".mp4"):
        _fail(field, "requires_mp4_extension")
    with _input_file(root, ref["path"], field) as (descriptor, _):
        digest = hashlib.sha256()
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != ref["sha256"]:
            _fail(field, "encoded_hash_mismatch")
        os.lseek(descriptor, 0, os.SEEK_SET)
        source = f"/proc/self/fd/{descriptor}"
        before = _probe(source, field, pass_fds=(descriptor,))
        stream = next(s for s in before["streams"] if s["codec_type"] == "video")
        try:
            if (int(stream["nb_read_frames"]) != presentation["num_frames"] or
                    abs(float(Fraction(stream["avg_frame_rate"])) - presentation["fps"]) > 1e-4):
                _fail(field, "video_profile_mismatch")
        except (KeyError, ValueError, ZeroDivisionError):
            _fail(field, "invalid_media_profile")
        _process(["ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-n",
                  "-protocol_whitelist", "file,pipe", "-i", source, "-map", "0:v:0",
                  "-an", "-sn", "-dn", "-map_metadata", "-1", "-map_metadata:s:v", "-1",
                  "-map_chapters", "-1", "-c:v", "copy", "-bsf:v", "filter_units=remove_types=6",
                  "-fflags", "+bitexact", "-metadata", "encoder=", "-metadata:s:v:0", "encoder=",
                  "-metadata:s:v:0", "language=und", "-metadata:s:v:0", "handler_name=VideoHandler",
                  "-metadata:s:v:0", "vendor_id=", "-write_tmcd", "0", "-movflags", "+faststart",
                  "-brand", "mp42", str(target)], field, pass_fds=(descriptor,))
        after = _probe(str(target), field)
        # Only MP4 structural tags with known values may survive the rewrite.
        allowed_format = {"major_brand": {"mp42"}, "minor_version": {"512"},
                          "compatible_brands": {"mp42iso2avc1mp41"}}
        allowed_stream = {"language": {"und"}, "handler_name": {"VideoHandler"},
                          "vendor_id": {"[0][0][0][0]"}}
        for obj, allowed in [(after["format"], allowed_format), *[(s, allowed_stream) for s in after["streams"]]]:
            if any(k not in allowed or v not in allowed[k] for k, v in obj.get("tags", {}).items()):
                _fail(field, "unexpected_video_metadata")
        if _video_pixel_hash(source, field, pass_fds=(descriptor,)) != _video_pixel_hash(str(target), field):
            _fail(field, "video_pixel_mismatch")


def _public_text(value: Any, field: str, terms: tuple[str, ...]) -> str:
    if not isinstance(value, str) or len(value) > 16384:
        _fail(field, "invalid_text")
    for term in sorted(set(terms), key=len, reverse=True):
        if term:
            value = value.replace(term, "[redacted]")
    patterns = [r"\b(?:https?|wss?|ftp)://[^\s<>\"']+", r"\b[^\s/@]+@[^\s/@]+\.[^\s/@]+",
                r"(?i)\b(?:Bearer|Authorization)\s*[:=]?\s*[^\s,;]+",
                r"(?i)\b(?:[\w-]*api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*[^\s,;]+",
                r"\b(?:sk-|gh[pousr]_|github_pat_|xox[baprs]-)[\w-]+", r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
                r"(?:[A-Za-z]:[\\/]|/|\\\\)[^\s<>\"']+", r"\b[\w.-]+/[\w./-]+",
                r"\b[\w-]+(?:\.[\w-]+)*\.(?:com|org|net|edu|io|ai)\b", r"\b[A-Za-z0-9_-]{24,}\b"]
    for pattern in patterns:
        value = re.sub(pattern, "[redacted]", value)
    return " ".join("".join(c for c in value if c >= " " and c != "\x7f").split())[:500]


def _validate_session(session: Any) -> tuple[dict[str, dict], list[dict]]:
    session = _object(session, _TOP, "session", {"extensions"})
    if (session["schema"] != "wmmllm.session.v1" or
            session["state_authority"] != "accepted_panorama_and_endpoint" or
            session["video_writes_state"] is not False):
        _fail("session", "unsupported_schema_or_authority")
    _identifier(session["session_id"], "session_id")
    _number(session["seed"], "seed", integer=True)
    if session["status"] not in {"ready", "incomplete", "failed"}:
        _fail("status", "invalid_status")
    if not isinstance(session["nodes"], list) or not isinstance(session["transitions"], list):
        _fail("session", "expected_arrays")
    if "extensions" in session and not isinstance(session["extensions"], dict):
        _fail("extensions", "expected_object")
    if len(session["nodes"]) > 10000 or len(session["transitions"]) > 10000:
        _fail("session", "graph_too_large")
    if not session["nodes"]:
        _fail("nodes", "unavailable_session")
    nodes: dict[str, dict] = {}
    for node in session["nodes"]:
        _object(node, _NODE, "node")
        key = _identifier(node["node_id"], "node_id")
        if key in nodes:
            _fail("node_id", "duplicate_identifier")
        nodes[key] = node
        _identifier(node["origin_id"], "origin_id")
        _number(node["physical_revision"], "physical_revision", minimum=0, integer=True)
        identity = node["view_identity"]
        if not (isinstance(identity, str) or (isinstance(identity, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in identity.items()))):
            _fail("view_identity", "invalid_identity")
        pose = _object(node["pose"], {"yaw", "pitch", "hfov"}, "pose")
        _number(pose["yaw"], "pose.yaw")
        _number(pose["pitch"], "pose.pitch", minimum=-90, maximum=90)
        _number(pose["hfov"], "pose.hfov", minimum=1, maximum=179)
        viewport = _object(node["viewport_profile"], {"width", "height", "content_box", "padding_ltrb"}, "viewport_profile")
        for name in ("width", "height"):
            _number(viewport[name], "viewport_profile", minimum=1, maximum=32768, integer=True)
        for name in ("content_box", "padding_ltrb"):
            if not isinstance(viewport[name], list) or len(viewport[name]) != 4:
                _fail("viewport_profile", "invalid_rectangle")
            for n in viewport[name]:
                _number(n, "viewport_profile", minimum=0, maximum=32768, integer=True)
        x0, y0, x1, y1 = viewport["content_box"]
        if not (x0 < x1 <= viewport["width"] and y0 < y1 <= viewport["height"]):
            _fail("viewport_profile", "invalid_content_box")
        if viewport["padding_ltrb"] != [x0, y0, viewport["width"] - x1, viewport["height"] - y1]:
            _fail("viewport_profile", "inconsistent_padding")
        _ref(node["panorama"], "panorama", image=True)
        _ref(node["view"], "view", image=True)
        _ref(node["panorama_state"], "panorama_state", state=True)
        _ref(node["memory_snapshot"], "memory_snapshot")
        if node["seam_status"] not in {"resolved", "seam_unresolved", "not_checked"}:
            _fail("seam_status", "invalid_status")
    if session["initial_node_id"] not in nodes or session["current_node_id"] not in nodes:
        _fail("session", "unknown_initial_or_current_node")
    if nodes[session["initial_node_id"]]["parent_node_id"] is not None:
        _fail("parent_node_id", "root_has_parent")
    for node in nodes.values():
        seen = {node["node_id"]}
        parent = node["parent_node_id"]
        while parent is not None:
            if parent not in nodes or parent in seen:
                _fail("parent_node_id", "unknown_or_cyclic_parent")
            seen.add(parent)
            parent = nodes[parent]["parent_node_id"]
        if session["initial_node_id"] not in seen:
            _fail("parent_node_id", "disconnected_root")
    turns = session["transitions"]
    seen_turns: set[str] = set()
    current = session["initial_node_id"]
    reached = {current}
    for index, turn in enumerate(turns, 1):
        _object(turn, _TURN, "transition")
        key = _identifier(turn["transition_id"], "transition_id")
        if key in seen_turns:
            _fail("transition_id", "duplicate_identifier")
        seen_turns.add(key)
        if type(turn["index"]) is not int or turn["index"] != index:
            _fail("index", "nonconsecutive_transition")
        if turn["from_node_id"] != current:
            _fail("from_node_id", "discontinuous_transition")
        status = turn["status"]
        if status not in {"accepted", "held", "failed"}:
            _fail("transition.status", "invalid_status")
        if not isinstance(turn["operations"], list) or any(op not in _OPS for op in turn["operations"]):
            _fail("operations", "invalid_operation")
        if ((status == "accepted" and turn["to_node_id"] not in nodes) or
                (status == "held" and turn["to_node_id"] != current) or
                (status == "failed" and turn["to_node_id"] is not None)):
            _fail("to_node_id", "invalid_transition_target")
        semantic = _object(turn["semantic"], {"status", "attempt_count", "semantic_accepted", "uncertain_accepted_and_labeled", "verification"}, "semantic")
        label = semantic["status"]
        if label not in {"PASS", "FAIL", "UNCERTAIN", "NOT_CHECKED"}:
            _fail("semantic", "invalid_status")
        _number(semantic["attempt_count"], "attempt_count", minimum=0, maximum=2, integer=True)
        if (semantic["semantic_accepted"] is not (label == "PASS") or
                semantic["uncertain_accepted_and_labeled"] is not (status == "accepted" and label == "UNCERTAIN") or
                (status == "accepted" and label not in {"PASS", "UNCERTAIN"}) or
                (status != "accepted" and label not in {"FAIL", "NOT_CHECKED"})):
            _fail("semantic", "inconsistent_acceptance")
        for value, field in [(semantic["verification"], "verification"), (turn["decision"], "decision")]:
            if value is not None:
                _ref(value, field)
        if status != "failed":
            _ref(turn["memory_after"], "memory_after")
        elif turn["memory_after"] is not None:
            _fail("memory_after", "failed_transition_has_memory")
        presentation = _object(turn["presentation"], {"status", "backend", "video", "reason", "num_frames", "fps", "duration_seconds"}, "presentation")
        _identifier(presentation["backend"], "presentation.backend")
        if presentation["status"] == "none":
            if presentation["video"] is not None or not isinstance(presentation["reason"], str) or not presentation["reason"].strip():
                _fail("presentation", "invalid_absent_video")
        elif presentation["status"] in {"rendered", "generated"}:
            _ref(presentation["video"], "presentation.video")
            _number(presentation["num_frames"], "num_frames", minimum=1, integer=True)
            _number(presentation["fps"], "fps", minimum=0.001, maximum=240)
            _number(presentation["duration_seconds"], "duration_seconds", minimum=0)
            if abs(presentation["duration_seconds"] - presentation["num_frames"] / presentation["fps"]) > 1e-6:
                _fail("presentation", "inconsistent_duration")
        else:
            _fail("presentation", "invalid_status")
        if status != "failed":
            current = turn["to_node_id"]
            reached.add(current)
    if current != session["current_node_id"]:
        _fail("current_node_id", "not_last_settled_state")
    if reached != set(nodes):
        _fail("nodes", "unreachable_state")
    _ref(session["memory"], "memory")
    return nodes, turns


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("session", "duplicate_json_key")
        result[key] = value
    return result


def export_world(run_root: Path, worlds_dir: Path, *, world_id: str = "procedural",
                 procedural_fixture: bool = False, redact_terms: tuple[str, ...] = ()) -> Path:
    """Write a fresh world folder, rejecting existing outputs and all symlinks.

    ``redact_terms`` can supply additional private display terms in memory. No
    credential source is opened. Only allowlisted media refs are dereferenced.
    ffmpeg/ffprobe are needed only when the session contains actual video.
    """
    _identifier(world_id, "world_id")
    root = _anchor(Path(run_root), "run_root")
    with _input_file(root, "session.json", "session") as (descriptor, size):
        if size > 16 * 1024 * 1024:
            _fail("session", "file_too_large")
        try:
            with os.fdopen(os.dup(descriptor), "r", encoding="utf-8") as stream:
                session = json.load(stream, parse_constant=lambda _: _fail("session", "nonfinite_json"),
                                    object_pairs_hook=_json_pairs)
        except (ValueError, UnicodeError):
            _fail("session", "invalid_json")
    try:
        nodes, turns = _validate_session(session)
    except ExportError:
        raise
    except (TypeError, KeyError, RecursionError):
        _fail("session", "invalid_value_type")
    public_nodes = {key: f"n{i:04d}" for i, key in enumerate(nodes)}
    public_places: dict[str, str] = {}
    for node in nodes.values():
        if node["origin_id"] not in public_places:
            public_places[node["origin_id"]] = f"p{len(public_places):04d}"
    terms = tuple(redact_terms) + tuple(nodes) + tuple(public_places) + tuple(t["transition_id"] for t in turns) + (session["session_id"],)
    world = {"schema": "wmmllm.world.v1", "id": world_id,
             "title": _public_text(session["title"], "title", terms) or "Prepared world",
             "procedural_fixture": bool(procedural_fixture),
             "attribution": ("Procedural artwork and CPU panorama clips, drawn locally; Apache-2.0." if procedural_fixture
                             else "Prepared session playback; media rights remain with their owner."),
             "initial_node_id": public_nodes[session["initial_node_id"]],
             "current_node_id": public_nodes[session["current_node_id"]],
             "nodes": [], "actions": []}
    destination_root = _anchor(Path(worlds_dir), "worlds_dir", create=True)
    destination = destination_root / world_id
    if os.path.lexists(destination):
        _fail("world_id", "output_exists")
    parent_stat = destination_root.stat()
    stage = Path(tempfile.mkdtemp(prefix=".export-", dir=destination_root))
    created = False
    published = False
    output_stat = None
    try:
        (stage / "media").mkdir()
        image_cache: dict[tuple, str] = {}
        def image(ref: dict, field: str, panorama: bool = False) -> str:
            # Validate every source ref even when a previous ref has identical bytes.
            cache_key = (ref["path"], ref["sha256"], ref["pixel_sha256"], tuple(ref["size"]), panorama)
            if cache_key not in image_cache:
                name = f"media/image{len(image_cache):04d}.png"
                _clean_image(root, ref, stage / name, field, panorama=panorama)
                image_cache[cache_key] = name
            return image_cache[cache_key]
        for node in nodes.values():
            world["nodes"].append({"id": public_nodes[node["node_id"]],
                                   "parent_id": public_nodes.get(node["parent_node_id"]),
                                   "place_id": public_places[node["origin_id"]],
                                   "revision": node["physical_revision"], "pose": dict(node["pose"]),
                                   "viewport": dict(node["viewport_profile"]),
                                   "panorama": image(node["panorama"], "panorama", True),
                                   "endpoint": image(node["view"], "view"),
                                   "seam_status": node["seam_status"]})
        for i, turn in enumerate(turns, 1):
            source = turn["presentation"]
            present = {"kind": source["status"], "video": None,
                       "reason": _NONE_REASON if source["status"] == "none" else None,
                       "num_frames": None, "fps": None, "duration_seconds": None}
            if source["status"] != "none":
                path = f"media/clip{i:04d}.mp4"
                _clean_video(root, source["video"], stage / path, source, "presentation.video")
                present.update(video=path, **{k: source[k] for k in ("num_frames", "fps", "duration_seconds")})
            world["actions"].append({"id": f"a{i:04d}", "from": public_nodes[turn["from_node_id"]],
                                      "to": public_nodes.get(turn["to_node_id"]),
                                      "label": _public_text(turn["caption"], "caption", terms) or f"Action {i}",
                                      "operations": list(turn["operations"]), "status": turn["status"],
                                      "semantic": turn["semantic"]["status"], "presentation": present})
        (stage / "world.json").write_text(json.dumps(world, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        # Publish media first and the manifest last. Exclusive directory creation
        # rejects even an empty existing world; hard-link publication cannot replace it.
        _anchor(destination_root, "worlds_dir")
        current_parent = destination_root.stat()
        if (current_parent.st_dev, current_parent.st_ino) != (parent_stat.st_dev, parent_stat.st_ino):
            _fail("worlds_dir", "directory_changed")
        try:
            destination.mkdir()
        except FileExistsError:
            _fail("world_id", "output_exists")
        created = True
        output_stat = destination.lstat()
        with _output_directory(destination, output_stat) as output_fd:
            os.rename(stage / "media", "media", dst_dir_fd=output_fd)
            _anchor(destination, "world_id")
            os.link(stage / "world.json", "world.json", dst_dir_fd=output_fd, follow_symlinks=False)
        published = True
        return destination
    except ExportError:
        raise
    except Exception:
        _fail("export", "local_export_failed")
    finally:
        shutil.rmtree(stage)
        if created and not published and output_stat is not None:
            try:
                current = destination.lstat()
                if (current.st_dev, current.st_ino) == (output_stat.st_dev, output_stat.st_ino):
                    shutil.rmtree(destination)
            except FileNotFoundError:
                pass


@contextmanager
def _output_directory(path: Path, expected):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        current = os.fstat(descriptor)
        if (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino):
            _fail("world_id", "directory_changed")
        yield descriptor
    finally:
        os.close(descriptor)


def export_bundle(run_dir: Path, out: Path, *, procedural_fixture: bool = False) -> Path:
    """Write a standalone viewer containing only explicitly exported media."""
    from .viewer import asset_root

    # Validate the source before creating the destination. The world exporter
    # repeats these checks and applies its closed field/asset allowlist.
    root = _anchor(Path(run_dir), "run_dir")
    out = Path(os.path.abspath(out))
    _anchor(out.parent, "out_parent", create=True)
    if os.path.lexists(out):
        _fail("out", "output_exists")
    try:
        out.mkdir()
    except OSError:
        _fail("out", "directory_creation_failed")
    identity = out.stat()
    try:
        assets = asset_root()
        for name in ("index.html", "assets/viewer.css", "assets/world.js",
                     "assets/app.js", "assets/renderer.js", "assets/favicon.svg",
                     "schema/world.v1.json"):
            target = out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write((assets / name).read_bytes())
        # The offline example explicitly identifies its deterministic providers.
        # Preserve that label when a user exports it without an extra flag.
        procedural_fixture = procedural_fixture or (root / "OFFLINE_FIXTURE.json").is_file()
        export_world(root, out / "worlds", world_id="prepared",
                     procedural_fixture=procedural_fixture)
        # The static viewer's built-in default is the distributed artwork.
        # An exported bundle instead opens this run's actual graph.
        index = out / "index.html"
        index.write_text(index.read_text(encoding="utf-8").replace(
            "<head>", '<head>\n  <meta name="wmmllm-default-world" content="prepared">'),
            encoding="utf-8")
        return out
    except BaseException:
        current = out.lstat()
        if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
            shutil.rmtree(out)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--worlds-dir", type=Path, default=Path("viewer/worlds"))
    parser.add_argument("--world-id", default="prepared")
    parser.add_argument("--procedural-fixture", action="store_true",
                        help="Label locally authored procedural playback explicitly.")
    args = parser.parse_args(argv)
    try:
        export_world(args.run_dir, args.worlds_dir, world_id=args.world_id,
                     procedural_fixture=args.procedural_fixture)
    except ExportError as error:
        print(f"Export rejected: {error}", file=sys.stderr)
        return 1
    print("Prepared world exported with relative media paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
