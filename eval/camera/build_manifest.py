"""Select the common item population and its videos with explicit portable inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any


def video_path(video_root: Path, value: str) -> Path:
    """Resolve a normalized video reference under the caller's explicit root."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("video_path must be a nonempty relative POSIX path")
    parts = value.split("/")
    ref = PurePosixPath(value)
    if ref.is_absolute() or any(p in ("", ".", "..") for p in parts) or ":" in value:
        raise ValueError("video_path must be a normalized relative POSIX path")
    root = Path(video_root).resolve(strict=True)
    path = root.joinpath(*parts).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("video_path must resolve to a file under video_root")
    return path


def build_manifest(
    table: dict[str, Any], generated_rows: list[dict[str, Any]],
    public_rows: list[dict[str, Any]], *, video_root: Path,
    generated_model_id: str = "aiwm", expected_items: int = 224,
) -> list[dict[str, Any]]:
    """Preserve common-item order, sorted model order, and last-row lookup."""
    population = table["populations"]["common"]
    common = [x if isinstance(x, str) else x["item_id"] for x in population["items"]]
    if len(common) != expected_items:
        raise ValueError(f"Expected {expected_items} common items; found {len(common)}")
    models = sorted({cell["model_id"] for cell in population["cells"]})
    videos = {}
    for row in generated_rows:
        videos[(generated_model_id, row["item_id"])] = (row["video_path"], row["condition_id"])
    for row in public_rows:
        mid = row.get("model_id")
        if not mid:
            parts = PurePosixPath(row["final_path"]).parts
            mid = parts[parts.index("models") + 1]
        videos[(mid, row["item_id"])] = (row["final_path"], row["condition_id"])
    result = []
    for model in models:
        for item in common:
            path, condition = videos[(model, item)]
            video_path(video_root, path)
            result.append({"model_id": model, "item_id": item,
                           "condition_id": condition, "video_path": path})
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--generated-inputs", type=Path, required=True)
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--generated-model-id", default="aiwm")
    parser.add_argument("--expected-items", type=int, default=224)
    args = parser.parse_args(argv)
    rows = build_manifest(json.loads(args.table.read_text(encoding="utf-8")),
                          read_jsonl(args.generated_inputs),
                          json.loads(args.public_manifest.read_text(encoding="utf-8"))["rows"],
                          video_root=args.video_root, generated_model_id=args.generated_model_id,
                          expected_items=args.expected_items)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} video references")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
