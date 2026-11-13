#!/usr/bin/env python3
"""Create an explicit relative-path index from an independently obtained CompBench copy."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

if __package__:
    from .verify_dataset import read_manifest, relative_path, safe_path, verify_files
else:
    from verify_dataset import read_manifest, relative_path, safe_path, verify_files


EXPECTED_COUNTS = {
    "add": 982,
    "remove": 1331,
    "replace": 152,
    "multi_object_add": 72,
    "multi_object_remove": 72,
    "implicit": 100,
}

METADATA = {
    "add": "tasks/add/metadata.json",
    "remove": "tasks/remove/metadata.json",
    "replace": "tasks/replace/metadata.json",
    "multi_object_add": "tasks/multi_turn_editing/multi_object_add.jsonl",
    "multi_object_remove": "tasks/multi_turn_editing/multi_object_remove.jsonl",
    "implicit": "tasks/implicit_reasoning/implicit_info.jsonl",
}


def read_records(root: Path, name: str) -> list[dict]:
    path = safe_path(root, name)
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
                key = relative_path(record["image_path"])
                prefix = f"tasks/{subtask}"
                paths = {"source": f"{prefix}/input_image/{key}", "target": f"{prefix}/edited_image/{key}", "mask": f"{prefix}/mask/{key}"}
                instruction = record["instruction"]
            else:
                key = relative_path(record["image_name"])
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
                safe_path(root, value)
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
                        help="New JSONL index, outside the dataset root; accepted directly by reproduce.compbench")
    parser.add_argument("--verify", action="store_true", help="Verify every image as well as metadata before indexing")
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve(strict=True)
        if args.output.resolve().is_relative_to(root):
            raise ValueError("output must be outside the dataset root")
        entries = read_manifest(Path(__file__).parent / "compbench_paper_2709.sha256")
        selected = entries if args.verify else [entry for entry in entries if entry[0].endswith((".json", ".jsonl"))]
        verified = verify_files(root, selected)
        if verified["status"] != "PASS":
            print(json.dumps(verified, indent=2))
            return 1
        rows = build_index(root)
        with args.output.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    except (OSError, ValueError, KeyError, TypeError):
        print(json.dumps({"status": "FAIL", "reason": "invalid dataset, mismatched population, or unavailable output"}))
        return 1
    print(json.dumps({"status": "PASS", "items": len(rows), "subtasks": dict(Counter(row["subtask"] for row in rows)), "image_hashes_verified": args.verify}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
