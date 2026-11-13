"""Requested camera round-trip completion from DA3 camera-to-world poses (32 uniform frames per video).
Conditions: left_then_right_25s / right_then_left_25s.
Camera convention: OpenCV (x right, y down, z forward); yaw_i = signed angle of frame-i forward axis in frame-0 coordinates about the
frame-0 vertical, positive = right.  Prespecified rule (thresholds fixed before inspecting outcomes):
  first_dir_ok : the largest-|yaw| frame within the first half of the clip turns in the requested first direction
  excursion    : max yaw in the requested first direction over the whole clip, degrees
  return_err   : geodesic rotation angle between the first and last frame, degrees
  completed    : first_dir_ok and excursion >= 15 and return_err <= 10
Sensitivity grid: excursion in {10, 15, 20, 30} x return in {5, 10, 15}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CONDITIONS = ("left_then_right_25s", "right_then_left_25s")


def metrics(c2w, cond):
    """Compute the original wrapped-yaw and geodesic-return expressions."""
    c2w = np.asarray(c2w)
    if c2w.ndim != 3 or c2w.shape[1:] not in ((3, 4), (4, 4)) or len(c2w) < 2:
        raise ValueError("c2w must contain at least two 3x4 or 4x4 camera poses")
    if not np.isfinite(c2w).all():
        raise ValueError("c2w poses must be finite")
    if cond not in CONDITIONS:
        raise ValueError("Unsupported camera round-trip condition")
    R = c2w[:, :3, :3]
    rel = np.einsum('ji,njk->nik', R[0], R)
    fwd = rel[:, :, 2]
    yaw = np.degrees(np.arctan2(fwd[:, 0], fwd[:, 2]))
    first = -1.0 if cond.startswith('left') else 1.0
    half = len(yaw) // 2
    k = int(np.argmax(np.abs(yaw[:half + 1])))
    first_dir_ok = bool(np.sign(yaw[k]) == first and abs(yaw[k]) > 1.0)
    excursion = float(np.max(first * yaw))
    Rl = rel[-1]
    ret = float(np.degrees(np.arccos(np.clip((np.trace(Rl) - 1) / 2, -1, 1))))
    return dict(first_dir_ok=first_dir_ok, excursion_deg=excursion, return_err_deg=ret,
                max_abs_yaw_deg=float(np.max(np.abs(yaw))))


def comp(m, ex=15, rt=10):
    return m['first_dir_ok'] and m['excursion_deg'] >= ex and m['return_err_deg'] <= rt


def load_pose_shards(poses_dir: Path):
    """Read explicitly selected compact pose shards; never load pickled arrays."""
    files = sorted(Path(poses_dir).glob("poses_shard*.jsonl"))
    if not files:
        raise ValueError("No poses_shard*.jsonl files found")
    rows = []
    for path in files:
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as archive:
            poses = archive["c2w"]
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row["status"] not in ("MEASURED", "FAILED"):
                        raise ValueError("Unsupported pose status")
                    if row["status"] == "MEASURED":
                        k = row["k"]
                        if type(k) is not int or k < 0 or k >= len(poses):
                            raise ValueError("Measured pose index is outside the shard")
                        row["c2w"] = poses[k].copy()
                    else:
                        row["c2w"] = None
                    rows.append(row)
    return rows


def analyze(rows, *, bootstrap_draws=20000, seed=55):
    """Analyze rows in shard order and share one seeded RNG across sorted models."""
    if type(bootstrap_draws) is not int or bootstrap_draws < 1:
        raise ValueError("bootstrap_draws must be a positive integer")
    out = []
    for row in rows:
        m = metrics(row['c2w'], row['condition_id']) if row['c2w'] is not None else None
        out.append({k: row[k] for k in ('model_id', 'item_id', 'condition_id', 'status')} | (m or {}))
    models = sorted({o['model_id'] for o in out})
    rng = np.random.default_rng(seed)
    summary = {}
    for mid in models:
        ms = [o for o in out if o['model_id'] == mid]
        meas = [o for o in ms if o['status'] == 'MEASURED']
        c = np.array([comp(o, 15, 10) for o in meas], float)
        boot = [c[rng.integers(0, len(c), len(c))].mean() for _ in range(bootstrap_draws)] if len(c) else []
        summary[mid] = {
            'items': len(ms), 'measured': len(meas), 'failed': len(ms) - len(meas),
            'completed_rate': float(c.mean()) if len(c) else None,
            'completed_ci95': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if boot else [None, None],
            'first_dir_ok_rate': float(np.mean([o['first_dir_ok'] for o in meas])) if meas else None,
            'median_excursion_deg': float(np.median([o['excursion_deg'] for o in meas])) if meas else None,
            'median_return_err_deg': float(np.median([o['return_err_deg'] for o in meas])) if meas else None,
            'static_rate_max_abs_yaw_lt_5': float(np.mean([o['max_abs_yaw_deg'] < 5 for o in meas])) if meas else None,
            'sensitivity': {f'ex{ex}_ret{rt}': float(np.mean([comp(o, ex, rt) for o in meas])) if meas else None
                            for ex in (10, 15, 20, 30) for rt in (5, 10, 15)}}
    return {'rule': __doc__, 'per_item': out, 'summary': summary}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Analyze saved camera poses without model weights")
    parser.add_argument("--poses-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(load_pose_shards(args.poses_dir))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    for model, summary in result["summary"].items():
        print(f"{model}: {summary['measured']}/{summary['items']} measured; completion={summary['completed_rate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
