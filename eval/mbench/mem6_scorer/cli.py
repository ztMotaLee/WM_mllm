"""Offline environment scorer and CPU-only saved-score aggregation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

from .common import (PACKAGE, FAMILIES, atomic, clean, digest, file_ref, now,
                     read_manifest, record_id, select_rows, verify_code)


def _measured(item):
    value = item.get("scores", {}).get("score")
    return bool(item.get("valid") and isinstance(value, (float, int))
                and not isinstance(value, bool) and math.isfinite(value))


def _invalid_payload(metric, item, error):
    from eval.mbench._upstream.core.result import UnitResult
    from eval.mbench._upstream.core.pipeline import _project_unit_scores, _project_item_scores
    from eval.mbench._upstream.aggregators.mean import MeanAggregator
    unit = UnitResult(item.item_id, metric.name, metric.unit_type,
                      item.item_id + "__scorer_failure", valid=False, error=error)
    result = metric.aggregate_item(item, [unit])
    result.valid = False
    result.error = error
    projected = _project_item_scores(metric.name, result)
    summary = MeanAggregator().aggregate(metric.name, [projected])
    return {"units": [asdict(_project_unit_scores(metric.name, unit))],
            "item": asdict(projected), "summary": asdict(summary), "raw_item_scores": {}}


def _aggregate(records, output):
    from eval.mbench._upstream.aggregators.mean import MeanAggregator
    from eval.mbench._upstream.core.result import UnitResult, ItemResult, ResultWriter
    writer = ResultWriter(output)
    writer.write_items([{"item_id": row["item_id"], "model_id": row["model_id"]} for row in records])
    summary = {}
    for name in sorted({metric for row in records for metric in row["metrics"]}):
        units, items = [], []
        for row in records:
            payload = row["metrics"].get(name)
            if payload is not None:
                units.extend(UnitResult(**unit) for unit in payload["units"])
                items.append(ItemResult(**payload["item"]))
        metric_summary = MeanAggregator().aggregate(name, items)
        writer.write_metric_results(name, units, items, metric_summary)
        summary[name] = asdict(metric_summary)
    atomic(output / "SUMMARY.json", summary)
    return summary


def _run(args):
    if args.device != "cpu" and not args.device.startswith("cuda:"):
        raise ValueError("device must be cpu or cuda:N")
    code = verify_code()
    rows = select_rows(read_manifest(args.manifest), family=args.family, model=args.model,
                       index=args.index, limit=args.limit,
                       shard_index=args.shard_index, shard_count=args.shard_count)
    if not rows:
        raise ValueError("No manifest videos match this partition")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records_dir = output / "records"
    records_dir.mkdir(exist_ok=True)
    # Each concurrently running partition must have its own output directory.
    with (output / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another scorer is using output directory " + str(output)) from exc
        return _run_locked(args, rows, output, records_dir, code)


def _run_locked(args, rows, output, records_dir, code):
    from .runtime import verify_weights, configure_offline, Runtime, video_metadata, environment_versions
    started = time.monotonic()
    run_id = digest({"created": now(), "pid": os.getpid(), "device": args.device,
                     "family": args.family, "rows": [(r["model_id"], r["item_id"]) for r in rows]})
    preflight_started = time.monotonic()
    lock, weight_provenance = verify_weights(args.weights_dir, family=args.family)
    work_parent = Path(args.work_dir).resolve()
    work_parent.mkdir(parents=True, exist_ok=True)
    records = []
    with tempfile.TemporaryDirectory(prefix="mem6_" + run_id[:12] + "_", dir=work_parent) as work:
        work_dir = Path(work)
        bindings = configure_offline(lock, work_dir)
        import torch
        if args.device.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise RuntimeError("Requested CUDA device is unavailable")
            torch.cuda.set_device(args.device)
        runtime = Runtime(args.family, args.device, work_dir, lock)
        runtime.synchronize()
        startup_seconds = time.monotonic() - started
        provenance = {"code": code, "weights": weight_provenance,
                      "runtime_versions": environment_versions(), "load_path_bindings": bindings,
                      "protocol": file_ref(PACKAGE / "PROTOCOL.json"),
                      "upstream_commit": code["upstream_commit"]}
        atomic(output / "PREFLIGHT.json", {"status": "READY", "created_utc": now(),
               "run_id": run_id, "gpu": runtime.gpu, "startup_seconds": startup_seconds,
               "provenance": provenance, "selected_videos": len(rows),
               "owned_temporary_path": str(work_dir), "cleanup": "TemporaryDirectory finally"})
        from eval.mbench._upstream.core.item import EvaluationItem, Media, VideoRef, ArtifactRef
        from eval.mbench._upstream.metrics.video_io import sample_bgr_frames
        for position, row in enumerate(rows):
            key = record_id(row)
            destination = records_dir / (key + ".json")
            # Hash every selected video and compare the HF LFS/provenance pin.
            # This also prevents using a changed video with a completed record.
            video = file_ref(row["video_path"])
            if row.get("expected_sha256") and video["sha256"] != row["expected_sha256"]:
                raise RuntimeError("Video SHA256 differs from manifest: " + row["video_path"])
            binding = {"video": video, "model_id": row["model_id"], "item_id": row["item_id"],
                       "subset": row["subset"], "condition_id": row.get("condition_id"),
                       "sample_id": row.get("sample_id"), "code": code,
                       "weights_lock_sha256": weight_provenance["lock"]["sha256"],
                       "protocol_sha256": provenance["protocol"]["sha256"]}
            input_sha = digest(binding)
            if destination.is_file():
                prior = json.loads(destination.read_text())
                if prior["input_sha256"] != input_sha:
                    raise RuntimeError("Existing score binding differs: " + str(destination))
                records.append(prior)
                print(json.dumps({"event": "REUSED", "model_id": row["model_id"],
                                  "item_id": row["item_id"], "record": str(destination)}), flush=True)
                continue
            family_started = time.monotonic()
            runtime.synchronize()
            if args.device.startswith("cuda:"):
                torch.cuda.reset_peak_memory_stats(args.device)
            item = EvaluationItem(row["item_id"], media=Media(videos=[VideoRef(video["path"])]),
                                  model_id=row["model_id"],
                                  metadata={"subset": args.family, "sample_id": row.get("sample_id"),
                                            "condition_id": row.get("condition_id")})
            metric_results, metric_seconds, errors = {}, {}, []
            producer = None
            producer_seconds = 0.0
            metadata = None
            with tempfile.TemporaryDirectory(prefix="video_", dir=work_dir) as video_work:
                try:
                    metadata = video_metadata(video["path"])
                    if args.family == "environment":
                        producer_started = time.monotonic()
                        try:
                            artifact, producer = runtime.prepare_da3(row, Path(video_work) / "da3")
                            item.artifacts["da3"] = ArtifactRef(name="da3", path=str(artifact), format="npz")
                        finally:
                            producer_seconds = time.monotonic() - producer_started
                    else:
                        sampled, indices = sample_bgr_frames(video["path"], 32)
                        metadata["sampled_frame_indices"] = indices
                        metadata["sampled_decoded_frames"] = len(sampled)
                        del sampled
                except Exception as exc:
                    errors.append({"stage": "producer", "error": str(exc),
                                   "traceback": traceback.format_exc()})
                for metric in runtime.metrics:
                    runtime.synchronize()
                    metric_started = time.monotonic()
                    try:
                        metric_results[metric.name] = runtime.evaluate_metric(metric, item)
                    except Exception as exc:
                        error = str(exc)
                        errors.append({"stage": metric.name, "error": error,
                                       "traceback": traceback.format_exc()})
                        metric_results[metric.name] = _invalid_payload(metric, item, error)
                    runtime.synchronize()
                    metric_seconds[metric.name] = time.monotonic() - metric_started
                # Pair ordinals are official; real indices are recorded
                # alongside them, without changing upstream pair metadata.
                if producer is not None:
                    real_indices = producer["sampled_frame_indices"]
                    producer["pair_source_indices"] = {}
                    for name, payload in metric_results.items():
                        producer["pair_source_indices"][name] = [
                            {"unit_id": unit["unit_id"],
                             "frame_out": unit["metadata"].get("frame_out"),
                             "frame_in": unit["metadata"].get("frame_in"),
                             "source_frame_out": real_indices[unit["metadata"]["frame_out"]],
                             "source_frame_in": real_indices[unit["metadata"]["frame_in"]]}
                            for unit in payload["units"]
                            if "frame_out" in unit["metadata"] and "frame_in" in unit["metadata"]]
            runtime.synchronize()
            elapsed = time.monotonic() - family_started
            valid_metrics = sum(_measured(payload["item"]) for payload in metric_results.values())
            no_departure = {name: sum(unit.get("metadata", {}).get("has_departure_return") is False
                                      for unit in payload["units"])
                            for name, payload in metric_results.items()}
            unit_errors = [{"metric": name, "unit_id": unit["unit_id"], "error": unit["error"]}
                           for name, payload in metric_results.items() for unit in payload["units"]
                           if unit.get("error")]
            processing_errors = errors + unit_errors
            timing = {"run_id": run_id, "record_id": key, "model_id": row["model_id"],
                      "item_id": row["item_id"], "family": args.family, "stage": "evaluation",
                      "elapsed_seconds": elapsed, "family_seconds": elapsed,
                      "producer_seconds": producer_seconds, "metric_seconds": metric_seconds,
                      "startup_seconds": startup_seconds if position == 0 else 0.0,
                      "allocated_gpu_count": int(args.device.startswith("cuda:")),
                      "gpu_seconds": elapsed if args.device.startswith("cuda:") else 0.0,
                      "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(args.device) if args.device.startswith("cuda:") else 0,
                      "gpu": runtime.gpu}
            record = {"schema": "mem6.score.v1", "created_utc": now(), "run_id": run_id,
                      "record_id": key, "input_sha256": input_sha, "family": args.family,
                      "item_id": row["item_id"], "model_id": row["model_id"],
                      "condition_id": row.get("condition_id"), "sample_id": row.get("sample_id"),
                      "status": "PARTIAL_PROCESSING_FAILURE" if processing_errors and valid_metrics else
                                "PROCESSING_FAILED" if processing_errors else
                                "MEASURED" if valid_metrics == len(metric_results) else
                                "PARTIAL_INVALID" if valid_metrics else "INVALID",
                      "metrics": metric_results, "video": video, "video_metadata": metadata,
                      "source_row": row, "provenance": provenance, "da3": producer,
                      "invalid_diagnostics": {"no_departure_return_units": no_departure},
                      "timings": timing, "errors": errors, "processing_errors": processing_errors,
                      "cleanup": {"video_temporary_workspace_removed": True}}
            atomic(destination, record)
            records.append(record)
            atomic(output / "PROGRESS.json", {"run_id": run_id, "completed": len(records),
                   "selected": len(rows), "last_record": str(destination), "updated_utc": now()})
            print(json.dumps({"event": "SCORED", "model_id": row["model_id"],
                              "item_id": row["item_id"], "status": record["status"],
                              "elapsed_seconds": elapsed, "record": str(destination)}), flush=True)
        _aggregate(records, output)
        atomic(output / "TIMING.json", {"schema": "mem6.timing.v1", "run_id": run_id,
               "family": args.family, "startup_seconds": startup_seconds,
               "wall_seconds": time.monotonic() - started,
               "records": [record["timings"] for record in records],
               "note": "One timing per video/family. Shared DA3 is counted once; GPU-seconds are elapsed wall seconds only for a CUDA device."})
    atomic(output / "DONE.json", {"schema": "mem6.partition.v1", "run_id": run_id,
           "status": "COMPLETE", "selected_videos": len(rows), "records": len(records),
           "measured": sum(record["status"] == "MEASURED" for record in records),
           "invalid_or_partial": sum(record["status"] != "MEASURED" for record in records),
           "created_utc": now(), "temporary_workspace_removed": not Path(work).exists(),
           "owned_children_remaining": []})
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Score a partition using locally prepared weights")
    run.add_argument("--manifest", required=True)
    run.add_argument("--weights-dir", required=True)
    run.add_argument("--family", choices=tuple(FAMILIES), required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--work-dir", required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--model")
    run.add_argument("--index", type=int)
    run.add_argument("--limit", type=int)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--shard-count", type=int, default=1)
    table = commands.add_parser("aggregate", help="Aggregate saved canonical receipts without loading models")
    table.add_argument("--manifest", action="append", required=True)
    table.add_argument("--records", action="append", required=True)
    table.add_argument("--output", required=True)
    table.add_argument("--iterations", type=int, default=20000)
    table.add_argument("--seed", type=int, default=45)
    commands.add_parser("describe", help="Print frozen protocol and verify code pins without GPU imports")
    args = parser.parse_args(argv)
    try:
        if args.command == "aggregate":
            from ..aggregation import aggregate_files
            aggregate_files(args.manifest, args.records, Path(args.output), args.seed, args.iterations)
            return 0
        if args.command == "describe":
            print(json.dumps({"code": verify_code(), "protocol": json.loads(
                (PACKAGE / "PROTOCOL.json").read_text())}, indent=2))
            return 0
        return _run(args)
    except Exception as exc:
        failure = {"status": "FAILED", "error": str(exc), "traceback": traceback.format_exc(),
                   "created_utc": now()}
        if getattr(args, "output", None):
            atomic(Path(args.output).resolve() / "FAILURE.json", failure)
        print(json.dumps(failure), file=sys.stderr, flush=True)
        return 1
