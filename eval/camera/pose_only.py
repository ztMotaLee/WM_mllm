"""Extract compact DA3 poses with user-downloaded, verified local model files.

Importing this module requires neither Torch nor DA3. Run this command in the
isolated evaluation environment; downloaded model files are never bundled.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from .build_manifest import read_jsonl, video_path


def extract(args):
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    rows = read_jsonl(args.manifest)[args.shard_index::args.num_shards]
    # Resolve every input before loading models. Stored manifest paths remain portable.
    rows = [dict(row, video_path=str(video_path(args.video_root, row["video_path"]))) for row in rows]
    from ..mbench.mem6_scorer.runtime import verify_weights, configure_offline, Runtime

    lock, _ = verify_weights(args.weights_dir, family="camera")
    args.out.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    exts, indices, statuses = [], [], []
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="camera_", dir=args.work_dir) as temp:
        work = Path(temp)
        configure_offline(lock, work)
        runtime = Runtime.for_pose(args.device, work, lock)
        for index, row in enumerate(rows):
            base = {key: row[key] for key in ("model_id", "item_id", "condition_id")}
            try:
                with tempfile.TemporaryDirectory(prefix="video_", dir=work) as video_temp:
                    path, product = runtime.prepare_da3(row, Path(video_temp) / "da3")
                    with np.load(path, allow_pickle=False) as archive:
                        exts.append(archive['extrinsics'].astype(np.float32))
                    indices.append(np.asarray(product['sampled_frame_indices'], dtype=np.int32))
                    statuses.append(dict(base, k=len(exts) - 1, status="MEASURED",
                                         c2w_sha256=product['camera_to_world_array_sha256'],
                                         producer_seconds=product['producer_seconds']))
            except Exception as error:
                # Raw exception text can contain private paths or provider state.
                statuses.append(dict(base, k=None, status="FAILED", error=type(error).__name__))
            if index % 25 == 0:
                progress = {'done': index + 1, 'of': len(rows), 'elapsed_s': time.monotonic() - started}
                (args.out / f'PROGRESS_{args.shard_index}.json').write_text(json.dumps(progress), encoding="utf-8")
    np.savez_compressed(args.out / f'poses_shard{args.shard_index}.npz',
                        c2w=np.stack(exts) if exts else np.zeros((0, 32, 3, 4), np.float32),
                        frame_idx=np.stack(indices) if indices else np.zeros((0, 32), np.int32))
    with (args.out / f'poses_shard{args.shard_index}.jsonl').open('w', encoding="utf-8") as stream:
        for status in statuses:
            stream.write(json.dumps(status) + "\n")
    progress = {'done': len(rows), 'of': len(rows), 'elapsed_s': time.monotonic() - started,
                'complete': True, 'measured': sum(s['status'] == 'MEASURED' for s in statuses)}
    (args.out / f'PROGRESS_{args.shard_index}.json').write_text(json.dumps(progress), encoding="utf-8")
    return progress


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--video-root', type=Path, required=True)
    parser.add_argument('--weights-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    progress = extract(args)
    print(f"Measured {progress['measured']} of {progress['done']} videos")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
