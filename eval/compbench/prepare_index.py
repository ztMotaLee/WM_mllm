"""Create an explicit relative-path index from an independently obtained CompBench copy."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .export import contained_file, relative_reference

EXPECTED_COUNTS = {"add": 982, "remove": 1331, "replace": 152,
                   "multi_object_add": 72, "multi_object_remove": 72, "implicit": 100}
METADATA = {
    "add": "tasks/add/metadata.json",
    "remove": "tasks/remove/metadata.json",
    "replace": "tasks/replace/metadata.json",
    "multi_object_add": "tasks/multi_turn_editing/multi_object_add.jsonl",
    "multi_object_remove": "tasks/multi_turn_editing/multi_object_remove.jsonl",
    "implicit": "tasks/implicit_reasoning/implicit_info.jsonl",
}
# SHA-256 of the seven metadata files of the paper's 2,709-item CompBench population.
METADATA_SHA256 = {
    "tasks/add/metadata.json": "f94fa9e433a25ec06f6bc15407a21f6d743f3ca6686850bea231d7639be6bb2a",
    "tasks/implicit_reasoning/implicit_data.jsonl": "f69387c5977869e8c3357e2f95e4ad9103cada473d1351fe6f1be154a722fcf8",
    "tasks/implicit_reasoning/implicit_info.jsonl": "cdeaec1b6a33089b6bca479dc55ee49c418a7364a8ec605767b21715e631a270",
    "tasks/multi_turn_editing/multi_object_add.jsonl": "242e34af8be726a9a25ca78a35a10915f2fd54447cc757265358633535983a0f",
    "tasks/multi_turn_editing/multi_object_remove.jsonl": "c291e349584cefba2f5ab9996232475eb914597954b86964189fd6960615bfa1",
    "tasks/remove/metadata.json": "c1420dcb663ee33ac387161a3c4c2863d579f39bdb35bac8ad7c63382e52656b",
    "tasks/replace/metadata.json": "b421e2f71d1df1c480cf1f3f7750ada98a897acf970a93b34b828ce7fd357e45",
}


def read_records(root: Path, name: str) -> list[dict]:
    path = contained_file(root, name)
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError(f"{name}: expected metadata records")
    return records


def build_index(root: Path, *, expected_counts: dict[str, int] | None = None) -> list[dict]:
    """Read source metadata. Targets and masks are evaluation references only."""
    rows: list[dict] = []
    seen: set[str] = set()
    for subtask, metadata in METADATA.items():
        for record in read_records(root, metadata):
            if subtask in {"add", "remove", "replace"}:
                key = relative_reference(record["image_path"]).as_posix()
                prefix = f"tasks/{subtask}"
                paths = {"source": f"{prefix}/input_image/{key}", "target": f"{prefix}/edited_image/{key}", "mask": f"{prefix}/mask/{key}"}
                instruction = record["instruction"]
            else:
                key = relative_reference(record["image_name"]).as_posix()
                if subtask == "implicit":
                    prefix = "tasks/implicit_reasoning"
                    instruction = record["instruction"]
                else:
                    prefix = f"tasks/multi_turn_editing/{subtask}"
                    instruction = record["add_ins" if subtask == "multi_object_add" else "remove_ins"]
                paths = {"source": f"{prefix}/ori_images/{key}", "target": f"{prefix}/edited_images/{key}", "mask": f"{prefix}/masks/{key}"}
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"{metadata}: missing instruction")
            item_id = f"{subtask}/{key}"
            if item_id in seen:
                raise ValueError(f"{metadata}: duplicate item")
            seen.add(item_id)
            for value in paths.values():
                contained_file(root, value)
            rows.append({
                "row_id": hashlib.sha256(item_id.encode("utf-8")).hexdigest()[:24],
                "item_id": item_id,
                "subtask": subtask,
                "eval_key": key,
                **paths,
                "instruction": instruction,
            })
    expected = EXPECTED_COUNTS if expected_counts is None else expected_counts
    if dict(Counter(row["subtask"] for row in rows)) != expected:
        raise ValueError("metadata counts do not match the selected benchmark population")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Downloaded CompBench root containing tasks/")
    parser.add_argument("--output", required=True, type=Path,
                        help="New JSONL index, outside the dataset root; accepted directly by eval.compbench.compose")
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve(strict=True)
        if args.output.resolve().is_relative_to(root):
            raise ValueError("output must be outside the dataset root")
        failures = [{"path": name, "reason": "sha256 mismatch"} for name, expected in METADATA_SHA256.items()
                    if hashlib.sha256(contained_file(root, name).read_bytes()).hexdigest() != expected]
        if failures:
            print(json.dumps({"status": "FAIL", "failures": failures}, indent=2))
            return 1
        rows = build_index(root)
        with args.output.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    except (OSError, ValueError, KeyError, TypeError):
        print(json.dumps({"status": "FAIL", "reason": "invalid dataset, mismatched population, or unavailable output"}))
        return 1
    print(json.dumps({"status": "PASS", "items": len(rows), "subtasks": dict(Counter(row["subtask"] for row in rows))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
