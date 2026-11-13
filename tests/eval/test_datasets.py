"""CPU-only tests using isolated synthetic dataset files; no downloads."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def dataset_tools(monkeypatch):
    return importlib.import_module("wmmllm_datasets.verify_dataset"), importlib.import_module("wmmllm_datasets.prepare_compbench")


@pytest.mark.parametrize("value", ["/absolute", "../escape", "a/../b", "./a", "a//b", "a\\b", "C:/a", "https://example.org/a", "", "a\x00b", "s3:asset", "a?b", "a#b", "a\nb"])
def test_dataset_relative_paths_reject_invalid_values(dataset_tools, value):
    verifier, _ = dataset_tools
    with pytest.raises(ValueError):
        verifier.relative_path(value)


def test_dataset_manifest_rejects_duplicate_and_invalid_hash(dataset_tools, tmp_path):
    verifier, _ = dataset_tools
    manifest = tmp_path / "checksums"
    line = hashlib.sha256(b"sample").hexdigest() + "  nested/input.png\n"
    manifest.write_text(line)
    assert verifier.read_manifest(manifest) == [("nested/input.png", hashlib.sha256(b"sample").hexdigest())]
    manifest.write_text(line + line)
    with pytest.raises(ValueError, match="duplicate"):
        verifier.read_manifest(manifest)
    manifest.write_text("z" * 64 + "  nested/input.png\n")
    with pytest.raises(ValueError, match="invalid format"):
        verifier.read_manifest(manifest)


def test_dataset_verification_detects_corruption_and_missing_file(dataset_tools, tmp_path):
    verifier, _ = dataset_tools
    source = tmp_path / "source.png"
    source.write_bytes(b"synthetic test bytes")
    entries = [("source.png", verifier.file_sha256(source))]
    assert verifier.verify_files(tmp_path, entries)["status"] == "PASS"
    source.write_bytes(b"corrupted")
    assert verifier.verify_files(tmp_path, entries)["failures"] == [{"path": "source.png", "reason": "sha256 mismatch"}]
    source.unlink()
    assert verifier.verify_files(tmp_path, entries)["status"] == "FAIL"


def test_dataset_verification_rejects_child_symlinks(dataset_tools, tmp_path):
    verifier, _ = dataset_tools
    target = tmp_path / "actual.png"
    target.write_bytes(b"synthetic test bytes")
    (tmp_path / "linked.png").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        verifier.safe_path(tmp_path, "linked.png")
    directory = tmp_path / "directory"
    directory.mkdir()
    (directory / "actual.png").write_bytes(b"synthetic test bytes")
    (tmp_path / "linked_directory").symlink_to(directory, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        verifier.safe_path(tmp_path, "linked_directory/actual.png")


@pytest.fixture
def synthetic_compbench(dataset_tools, tmp_path):
    _, prepare = dataset_tools
    for subtask, metadata in prepare.METADATA.items():
        metadata_path = tmp_path / metadata
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        if subtask in {"add", "remove", "replace"}:
            key = "scene/0/test.png"
            record = {"image_path": key, "instruction": "synthetic test instruction"}
            metadata_path.write_text(json.dumps([record]))
            dirs = [f"tasks/{subtask}/{kind}" for kind in ["input_image", "edited_image", "mask"]]
        else:
            key = "test.png"
            field = "instruction" if subtask == "implicit" else "add_ins" if subtask == "multi_object_add" else "remove_ins"
            record = {"image_name": key, field: "synthetic test instruction"}
            metadata_path.write_text(json.dumps(record) + "\n")
            prefix = "tasks/implicit_reasoning" if subtask == "implicit" else f"tasks/multi_turn_editing/{subtask}"
            dirs = [f"{prefix}/{kind}" for kind in ["ori_images", "edited_images", "masks"]]
        for directory in dirs:
            path = tmp_path / directory / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic test bytes")
    return tmp_path


def test_dataset_index_maps_all_six_task_layouts(dataset_tools, synthetic_compbench):
    verifier, prepare = dataset_tools
    rows = prepare.build_index(synthetic_compbench, expected_counts={key: 1 for key in prepare.METADATA})
    assert len(rows) == 6
    assert len({row["row_id"] for row in rows}) == 6
    for row in rows:
        assert "direct" not in row
        assert row["instruction"] == "synthetic test instruction"
        assert row["item_id"] == f"{row['subtask']}/{row['eval_key']}"
        assert row["row_id"] == hashlib.sha256(row["item_id"].encode()).hexdigest()[:24]
        for key in ["source", "target", "mask"]:
            assert verifier.safe_path(synthetic_compbench, row[key]).is_file()


def test_dataset_index_rejects_missing_reference_and_wrong_population(dataset_tools, synthetic_compbench):
    _, prepare = dataset_tools
    with pytest.raises(ValueError, match="population"):
        prepare.build_index(synthetic_compbench)
    (synthetic_compbench / "tasks/add/mask/scene/0/test.png").unlink()
    with pytest.raises(ValueError, match="missing regular file"):
        prepare.build_index(synthetic_compbench, expected_counts={key: 1 for key in prepare.METADATA})


def test_dataset_index_rejects_duplicate_metadata_item(dataset_tools, synthetic_compbench):
    _, prepare = dataset_tools
    path = synthetic_compbench / "tasks/add/metadata.json"
    record = json.loads(path.read_text())[0]
    path.write_text(json.dumps([record, record]))
    with pytest.raises(ValueError, match="duplicate item"):
        prepare.build_index(synthetic_compbench, expected_counts={key: 1 for key in prepare.METADATA})


def test_dataset_release_manifest_identities(dataset_tools):
    verifier, _ = dataset_tools
    import wmmllm_datasets
    directory = Path(wmmllm_datasets.__file__).parent
    descriptor = json.loads((directory / "DATASET_MANIFESTS.json").read_text())
    for item in descriptor["datasets"]:
        path = directory / item["checksum_manifest"]
        entries = verifier.read_manifest(path)
        assert len(entries) == item["files"]
        assert verifier.file_sha256(path) == item["checksum_manifest_sha256"]
        assert len({name for name, _ in entries}) == len(entries)
