"""CPU-only camera tests on procedural poses and empty local input files."""
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from eval.camera.analyze_camera import analyze, comp, load_pose_shards, metrics
from eval.camera.build_manifest import build_manifest, video_path

TEMP_ROOT = Path(os.environ.get("WMM_TEST_TEMP_DIR", tempfile.gettempdir()))


def poses(yaws, *, initial_rotation=None):
    transforms = np.tile(np.eye(4), (len(yaws), 1, 1))
    for i, angle in enumerate(np.radians(yaws)):
        c, s = np.cos(angle), np.sin(angle)
        transforms[i, :3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    if initial_rotation is not None:
        transforms[:, :3, :3] = np.einsum("ij,njk->nik", initial_rotation, transforms[:, :3, :3])
    return transforms[:, :3, :4]


def row(item, yaws=None, *, model="model", condition="right_then_left_25s"):
    return dict(model_id=model, item_id=item, condition_id=condition,
                status="MEASURED" if yaws is not None else "FAILED",
                c2w=poses(yaws) if yaws is not None else None)


class CameraMetricsTest(unittest.TestCase):
    def test_right_roundtrip(self):
        result = metrics(poses([0, 20, 40, 20, 0]), "right_then_left_25s")
        self.assertTrue(comp(result))
        self.assertAlmostEqual(result["excursion_deg"], 40)
        self.assertAlmostEqual(result["return_err_deg"], 0)

    def test_left_roundtrip(self):
        result = metrics(poses([0, -20, -40, -20, 0]), "left_then_right_25s")
        self.assertTrue(comp(result))
        self.assertAlmostEqual(result["excursion_deg"], 40)

    def test_wrong_first_turn_fails(self):
        result = metrics(poses([0, -40, -30, 20, 0]), "right_then_left_25s")
        self.assertFalse(result["first_dir_ok"])
        self.assertFalse(comp(result))

    def test_static_fails(self):
        result = metrics(poses([0, 0, 0, 0]), "right_then_left_25s")
        self.assertFalse(result["first_dir_ok"])
        self.assertFalse(comp(result))

    def test_final_rotation_error_fails(self):
        result = metrics(poses([0, 40, 35, 25, 20]), "right_then_left_25s")
        self.assertAlmostEqual(result["return_err_deg"], 20)
        self.assertFalse(comp(result))

    def test_geodesic_return_includes_pitch(self):
        trajectory = poses([0, 40, 0])
        c, s = np.cos(np.radians(25)), np.sin(np.radians(25))
        trajectory[-1, :3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
        result = metrics(trajectory, "right_then_left_25s")
        self.assertAlmostEqual(result["return_err_deg"], 25)
        self.assertFalse(comp(result))

    def test_initial_world_orientation_cancels(self):
        rotation = poses([70])[0, :3, :3]
        first = metrics(poses([0, 40, 0]), "right_then_left_25s")
        second = metrics(poses([0, 40, 0], initial_rotation=rotation), "right_then_left_25s")
        for key in first:
            self.assertAlmostEqual(first[key], second[key])

    def test_excursion_uses_whole_clip(self):
        result = metrics(poses([0, 2, 4, 3, 40, 0]), "right_then_left_25s")
        self.assertTrue(result["first_dir_ok"])
        self.assertAlmostEqual(result["excursion_deg"], 40)
        self.assertTrue(comp(result))

    def test_wrapped_yaw_is_not_unwrapped(self):
        result = metrics(poses([0, 179, 181, 0]), "right_then_left_25s")
        self.assertAlmostEqual(result["excursion_deg"], 179)
        self.assertAlmostEqual(result["max_abs_yaw_deg"], 179)

    def test_inclusive_completion_thresholds(self):
        self.assertTrue(comp(dict(first_dir_ok=True, excursion_deg=15, return_err_deg=10)))
        self.assertFalse(comp(dict(first_dir_ok=True, excursion_deg=14.999, return_err_deg=10)))
        self.assertFalse(comp(dict(first_dir_ok=True, excursion_deg=15, return_err_deg=10.001)))

    def test_invalid_pose_or_condition_rejected(self):
        for malformed in (np.zeros((1, 3, 4)), np.zeros((4, 3, 3)), np.full((3, 3, 4), np.nan)):
            with self.assertRaises(ValueError):
                metrics(malformed, "right_then_left_25s")
        with self.assertRaises(ValueError):
            metrics(poses([0, 40, 0]), "unknown")

    def test_failed_scores_excluded_and_deterministic_bootstrap(self):
        rows = [row("a", [0, 40, 0]), row("b", [0, 0, 0]), row("c")]
        result = analyze(rows, bootstrap_draws=200)
        summary = result["summary"]["model"]
        self.assertEqual((summary["items"], summary["measured"], summary["failed"]), (3, 2, 1))
        self.assertEqual(summary["completed_rate"], 0.5)
        self.assertEqual(result, analyze(rows, bootstrap_draws=200))

    def test_no_measured_scores_emit_json_null(self):
        result = analyze([row("failed")], bootstrap_draws=10)
        self.assertIsNone(result["summary"]["model"]["completed_rate"])
        self.assertEqual(result["summary"]["model"]["completed_ci95"], [None, None])
        json.dumps(result, allow_nan=False)

    def test_model_dependencies_are_lazy(self):
        original_import = __import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "transformers", "depth_anything_3"):
                raise AssertionError("Heavy model dependency imported")
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guarded):
            importlib.reload(importlib.import_module("eval.camera.pose_only"))
            importlib.reload(importlib.import_module("eval.camera.analyze_camera"))


class CameraManifestTest(unittest.TestCase):
    def setUp(self):
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=TEMP_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_manifest_order_and_public_model_inference(self):
        for path in ("generated/b.mp4", "generated/a.mp4", "models/public/b.mp4", "models/public/a.mp4"):
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()
        table = {"populations": {"common": {"items": ["b", {"item_id": "a"}],
                  "cells": [{"model_id": "public"}, {"model_id": "aiwm"}]}}}
        gen = [dict(item_id=i, condition_id="right_then_left_25s", video_path=f"generated/{i}.mp4") for i in ("a", "b")]
        pub = [dict(item_id=i, condition_id="right_then_left_25s", final_path=f"models/public/{i}.mp4") for i in ("a", "b")]
        result = build_manifest(table, gen, pub, video_root=self.root, expected_items=2)
        self.assertEqual([(x["model_id"], x["item_id"]) for x in result],
                         [("aiwm", "b"), ("aiwm", "a"), ("public", "b"), ("public", "a")])
        with self.assertRaises(ValueError):
            build_manifest(table, gen, pub, video_root=self.root)

    def test_video_path_rejects_traversal_absolute_and_symlink_escape(self):
        for ref in ("../x", "/x", "a//b", "./x", "a/../b", "C:/x", "a\\b"):
            with self.assertRaises(ValueError):
                video_path(self.root, ref)
        outside = self.root.parent / "outside-camera-input"
        outside.touch()
        self.addCleanup(outside.unlink)
        (self.root / "escape").symlink_to(outside)
        with self.assertRaises(ValueError):
            video_path(self.root, "escape")

    def test_saved_pose_shard_loader(self):
        np.savez_compressed(self.root / "poses_shard0.npz", c2w=np.stack([poses([0, 40, 0])]))
        measured = dict(k=0, model_id="model", item_id="a", condition_id="right_then_left_25s", status="MEASURED")
        failed = dict(measured, k=None, item_id="b", status="FAILED")
        (self.root / "poses_shard0.jsonl").write_text(json.dumps(measured) + "\n" + json.dumps(failed) + "\n")
        rows = load_pose_shards(self.root)
        self.assertTrue(comp(metrics(rows[0]["c2w"], rows[0]["condition_id"])))
        self.assertIsNone(rows[1]["c2w"])
        (self.root / "poses_shard0.jsonl").write_text(json.dumps(dict(measured, k=-1)) + "\n")
        with self.assertRaises(ValueError):
            load_pose_shards(self.root)

    def test_extractor_array_alignment_failure_privacy_and_cleanup(self):
        from argparse import Namespace
        from eval.camera.pose_only import extract
        from eval.mbench.mem6_scorer import runtime as runtime_module
        entries = [dict(model_id="model", item_id=item, condition_id="right_then_left_25s",
                        video_path=f"{item}.mp4") for item in ("first", "failed", "third")]
        for entry in entries:
            (self.root / entry["video_path"]).touch()
        manifest = self.root / "inputs.jsonl"
        manifest.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
        class TestProducer:
            def prepare_da3(self, entry, dest):
                dest.mkdir()
                if entry["item_id"] == "failed":
                    (dest / "partial.txt").write_text("test-only partial artifact")
                    raise ValueError("private diagnostic must be excluded")
                value = 1 if entry["item_id"] == "first" else 3
                np.savez_compressed(dest / "results.npz", extrinsics=np.full((32, 3, 4), value, np.float32))
                return dest / "results.npz", dict(sampled_frame_indices=list(range(32)),
                        camera_to_world_array_sha256="0" * 64, producer_seconds=0)
        args = Namespace(num_shards=1, shard_index=0, manifest=manifest, video_root=self.root,
                         weights_dir=self.root / "unused", out=self.root / "out",
                         work_dir=self.root / "work", device="cpu")
        with patch.object(runtime_module, "verify_weights", return_value=({}, {})) as verified, \
             patch.object(runtime_module, "configure_offline"), \
             patch.object(runtime_module.Runtime, "for_pose", return_value=TestProducer()):
            progress = extract(args)
        self.assertEqual(verified.call_args.kwargs["family"], "camera")
        self.assertEqual(progress["measured"], 2)
        saved = [json.loads(line) for line in (args.out / "poses_shard0.jsonl").read_text().splitlines()]
        self.assertEqual([entry["k"] for entry in saved], [0, None, 1])
        self.assertEqual(saved[1]["error"], "ValueError")
        self.assertNotIn("private diagnostic", json.dumps(saved))
        with np.load(args.out / "poses_shard0.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["c2w"][:, 0, 0, 0], [1, 3])
        self.assertEqual(list(args.work_dir.iterdir()), [])

    def test_shard_range_checked_before_model_import(self):
        from argparse import Namespace
        from eval.camera.pose_only import extract
        with self.assertRaises(ValueError):
            extract(Namespace(num_shards=2, shard_index=2))


if __name__ == "__main__":
    unittest.main()
