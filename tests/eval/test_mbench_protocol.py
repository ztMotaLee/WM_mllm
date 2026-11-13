"""CPU-only synthetic MBench protocol tests; no weights, videos, or network."""
from copy import deepcopy
import importlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eval.mbench import aggregation as table
from eval.mbench._upstream.core.item import EvaluationItem
from eval.mbench._upstream.core.result import UnitResult, ItemResult
from eval.mbench._upstream.core.pipeline import _project_item_scores, _project_unit_scores
from eval.mbench._upstream.aggregators.mean import MeanAggregator
from eval.mbench._upstream.metrics.environment_consistency import SpatialEpipolarMetric, SpatialReprojectionMetric
from eval.mbench._upstream.metrics.canonical import canonicalize_scores
from eval.mbench.mem6_scorer.common import read_manifest, select_rows
from eval.mbench.mem6_scorer.cli import _measured


class MBenchProtocolTests(unittest.TestCase):
    def test_raw_units_aggregate_before_exponential_projection(self):
        for metric, key, tau in ((SpatialEpipolarMetric(), "median_epi_dist", 5.),
                                 (SpatialReprojectionMetric(), "median_reproj", 12.)):
            item = EvaluationItem("synthetic", model_id="procedural_test")
            units = [UnitResult(item.item_id, metric.name, "frame_pair", str(index),
                                scores={key: value}) for index, value in enumerate((0., tau * 2))]
            units.append(UnitResult(item.item_id, metric.name, "frame_pair", "invalid",
                                    scores={key: 999.}, valid=False))
            result = _project_item_scores(metric.name, metric.aggregate_item(item, units))
            self.assertEqual(result.scores["score_raw"], tau)
            self.assertEqual(result.scores["score"], math.exp(-1))
            self.assertNotAlmostEqual(result.scores["score"], (1 + math.exp(-2)) / 2)
            self.assertEqual(result.num_units, 3)
            self.assertTrue(result.valid)

    def test_style_lighting_projection_and_invalid_flags(self):
        self.assertEqual(canonicalize_scores("mbencha.environment.rendering_style", {"gram_dist": .3, "dino_sim": .99}),
                         {"score": .7, "score_raw": .3})
        self.assertEqual(canonicalize_scores("mbencha.environment.rendering_lighting", {"lighting_score": .7, "lighting_distance": .3}),
                         {"score": .7, "score_raw": .3})
        metric = SpatialEpipolarMetric()
        unit = UnitResult("synthetic", metric.name, "frame_pair", "u", valid=False)
        self.assertFalse(_project_unit_scores(metric.name, unit).valid)
        self.assertFalse(_project_item_scores(metric.name, metric.aggregate_item(EvaluationItem("synthetic"), [unit])).valid)

    def test_official_summary_and_report_validity_are_distinct(self):
        name = table.METRICS.keys().__iter__().__next__()
        valid = ItemResult("one", "aiwm", name, {"score": .8}, valid=True)
        invalid = ItemResult("two", "aiwm", name, {"score": .2}, valid=False)
        summary = MeanAggregator().aggregate(name, [valid, invalid])
        self.assertEqual(summary.results["aiwm"]["scores"]["score"], .5)
        self.assertTrue(_measured({"valid": True, "scores": {"score": .8}}))
        for value in (True, float("nan"), float("inf"), None):
            self.assertFalse(_measured({"valid": True, "scores": {"score": value}}))
        self.assertFalse(_measured({"valid": False, "scores": {"score": .2}}))

    def test_manifest_order_aliases_shards_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory(prefix="mbench_synthetic_") as directory:
            path = Path(directory) / "manifest.json"
            rows = [{"model": "procedural", "item_id": f"environment:synthetic{index}:condition",
                     "final_path": f"synthetic{index}.mp4", "lfs_sha256": "a" * 64}
                    for index in range(5)]
            path.write_text(json.dumps({"rows": rows}))
            parsed = read_manifest(path)
            self.assertEqual(parsed[0]["expected_sha256"], "a" * 64)
            self.assertEqual(select_rows(parsed, family="environment", index=3)[0]["item_id"], rows[3]["item_id"])
            shards = [select_rows(parsed, family="environment", shard_index=i, shard_count=2) for i in range(2)]
            self.assertEqual([[row["manifest_index"] for row in shard] for shard in shards], [[0, 2, 4], [1, 3]])
            path.write_text(json.dumps(rows + [rows[0]]))
            with self.assertRaises(ValueError):
                read_manifest(path)

    def test_runtime_module_import_is_lazy(self):
        runtime = importlib.import_module("eval.mbench.mem6_scorer.runtime")
        self.assertTrue(callable(runtime.Runtime.for_pose))
        self.assertTrue(callable(runtime.verify_weights))


    def test_weight_verification_uses_explicit_root_and_camera_subset(self):
        import hashlib
        from eval.mbench.mem6_scorer import runtime
        with tempfile.TemporaryDirectory(prefix="mbench_synthetic_") as directory:
            root = Path(directory)
            weights = root / "weights"
            weights.mkdir()
            (weights / "config.json").write_text("{}")
            reference = {"path": "config.json", "bytes": 2,
                         "sha256": hashlib.sha256(b"{}").hexdigest(),
                         "families": ["camera", "environment"],
                         "repo": "depth-anything/DA3NESTED-GIANT-LARGE"}
            lock = {"files": [reference, {**reference, "path": "not-needed-for-camera", "families": ["environment"]}],
                    "snapshots": {"depth-anything/DA3NESTED-GIANT-LARGE": {"path": ".", "revision": "test-only"}},
                    "da3_source": {}}
            (root / "MODEL_LOCK.json").write_text(json.dumps(lock))
            with patch.object(runtime, "PACKAGE", root):
                resolved, checked = runtime.verify_weights(weights, family="camera")
                self.assertEqual(len(checked["verified_files"]), 1)
                self.assertTrue(Path(resolved["files"][0]["path"]).is_absolute())
                (weights / "config.json").write_text("[]")
                with self.assertRaisesRegex(RuntimeError, "missing or changed"):
                    runtime.verify_weights(weights, family="camera")


class MBenchTableTests(unittest.TestCase):
    def fixture(self):
        expected = {}
        records = {}
        metric = next(iter(table.METRICS))
        for model in table.MODELS:
            for index in range(3):
                key = (model, f"environment:synthetic{index}:condition", "condition")
                expected[key] = {"model_id": model, "item_id": key[1], "condition_id": key[2],
                                 "video_path": "procedural.mp4"}
                records[key] = {"model_id": model, "item_id": key[1], "condition_id": key[2],
                                "status": "SYNTHETIC_TEST_ONLY", "metrics": {metric: {
                                    "item": {"valid": index != 1, "scores": {"score": [.25, .9, .75][index]}},
                                    "units": []}}}
        return metric, expected, records

    def test_common_population_tracks_video_presence_and_retains_invalid_scores(self):
        metric, expected, records = self.fixture()
        expected[("yume", "environment:synthetic2:condition", "condition")]["status"] = "MISSING_TEST_ONLY"
        common, present = table.populations(expected)
        self.assertEqual(len(common), 2)
        self.assertEqual(len(present["aiwm"]), 3)
        result = table.cell("aiwm", metric, present["aiwm"], expected, records, "own_full", iterations=17)
        self.assertEqual((result["valid_count"], result["invalid_count"], result["missing_count"]), (2, 1, 0))
        self.assertEqual(result["mean"], .5)
        self.assertEqual(result["mean_display100"], 50.)
        self.assertEqual(result["bootstrap_95_ci"], [.25, .75])

    def test_missing_receipt_does_not_shrink_population(self):
        metric, expected, records = self.fixture()
        del records[("aiwm", "environment:synthetic2:condition", "condition")]
        common, _ = table.populations(expected)
        result = table.cell("aiwm", metric, common, expected, records, "common", iterations=5)
        self.assertEqual(result["population_count"], 3)
        self.assertEqual((result["valid_count"], result["invalid_count"], result["missing_count"]), (1, 1, 1))
        self.assertEqual(result["bootstrap_95_ci"], [.25, .25])

    def test_nonfinite_boolean_scores_are_unmeasured_and_bootstrap_is_deterministic(self):
        metric, expected, records = self.fixture()
        for value in (True, float("nan"), float("inf"), None):
            changed = deepcopy(records)
            changed[("aiwm", "environment:synthetic0:condition", "condition")]["metrics"][metric]["item"]["scores"]["score"] = value
            result = table.cell("aiwm", metric, table.populations(expected)[0], expected, changed, "common", iterations=5)
            self.assertEqual(result["valid_count"], 1)
        self.assertEqual(table.bootstrap([.1, .2, .8], 101, 19), table.bootstrap([.1, .2, .8], 101, 19))
        self.assertIsNone(table.bootstrap([], 7, 3))
        self.assertEqual(table.percentile([0., 2., 4.], .25), 1.)


    def test_cli_aggregate_end_to_end_without_models(self):
        from eval.mbench.mem6_scorer.cli import main
        metric, expected, records = self.fixture()
        with tempfile.TemporaryDirectory(prefix="mbench_synthetic_") as directory:
            root = Path(directory)
            manifests = root / "manifest.json"
            manifests.write_text(json.dumps(list(expected.values())))
            saved = root / "records"
            saved.mkdir()
            for index, (key, row) in enumerate(records.items()):
                row["schema"], row["family"] = "mem6.score.v1", "environment"
                row["metrics"][metric]["item"].update({"metric": metric, "item_id": key[1], "model_id": key[0]})
                row["provenance"] = {"code": {"tree_sha256": "a" * 64},
                                     "weights": {"lock": {"sha256": "b" * 64}},
                                     "protocol": {"sha256": "c" * 64}}
                (saved / f"synthetic{index}.json").write_text(json.dumps(row))
            output = root / "table.json"
            self.assertEqual(main(["aggregate", "--manifest", str(manifests), "--records", str(saved),
                                   "--output", str(output), "--iterations", "17"]), 0)
            report = json.loads(output.read_text())
            self.assertEqual(report["common_population_count"], 3)
            self.assertEqual(len(report["populations"]["common"]["cells"]), 28)
            first = report["populations"]["common"]["cells"][0]
            self.assertEqual(first["mean"], .5)
            with self.assertRaises(FileExistsError):
                table.aggregate_files([manifests], [saved], output, iterations=17)
            bad = json.loads((saved / "synthetic0.json").read_text())
            bad["provenance"]["protocol"]["sha256"] = "d" * 64
            (saved / "synthetic0.json").write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "different code/weights/protocol"):
                table.load_inputs([manifests], [saved])


if __name__ == "__main__":
    unittest.main()
