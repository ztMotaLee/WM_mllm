# Modified: A-environment closure only; metric calculations are unchanged.
# Copyright 2026 MBench community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Zhang Zhang / Tsinghua University] in [2026].

"""MBench-A / MBench-T environment consistency metric plugins.

- MBench-A: DA3 fold-pair spatial/rendering metrics with pose-return trigger scores.
- MBench-T: caption-segment pair selection with VLM trigger and shared reliability keys.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

from eval.mbench._upstream.core.contract import InputContract
from eval.mbench._upstream.core.item import EvaluationItem
from eval.mbench._upstream.core.registry import Metric
from eval.mbench._upstream.core.result import ItemResult, UnitResult
from eval.mbench._upstream.metrics.environment_calculations import (
    clamp01,
    compute_epipolar_error,
    compute_reprojection_error,
    distance_to_score,
    find_fold_pairs,
    load_da3_data,
    pose_return_pair_score,
    read_image_at_depth_res,
)
from eval.mbench._upstream.metrics.metric_math import finite_mean
from eval.mbench._upstream.metrics.video_io import first_video_path, sample_bgr_frames

ENV_TRIGGER_SCORE_KEYS = (
    "triggered",
    "trigger_score",
    "pair_selection_score",
    "return_similarity",
    "middle_similarity",
)
ENV_RELIABILITY_SCORE_KEYS = ("reliability_score",)

A_ENV_SPATIAL_EPIPOLAR_METRIC = "mbencha.environment.spatial_epipolar"
A_ENV_SPATIAL_REPROJECTION_METRIC = "mbencha.environment.spatial_reprojection"
A_ENV_RENDERING_STYLE_METRIC = "mbencha.environment.rendering_style"
A_ENV_RENDERING_LIGHTING_METRIC = "mbencha.environment.rendering_lighting"

T_ENV_SPATIAL_EPIPOLAR_METRIC = "mbencht.environment.spatial_epipolar"
T_ENV_SPATIAL_REPROJECTION_METRIC = "mbencht.environment.spatial_reprojection"
T_ENV_RENDERING_STYLE_METRIC = "mbencht.environment.rendering_style"
T_ENV_RENDERING_LIGHTING_METRIC = "mbencht.environment.rendering_lighting"


def _apply_trigger_conditioned_item_scores(
    scores: dict[str, float],
    *,
    trigger_score: float,
    reliability_score: float | None,
) -> None:
    scores["trigger_score"] = round(float(trigger_score), 6)
    scores["triggered"] = 1.0 if trigger_score > 0.5 else 0.0
    if reliability_score is not None:
        scores["reliability_score"] = round(float(reliability_score), 6)


# =============================================================================
# MBench-A Environment
# =============================================================================


class _DA3PairMetric(Metric):
    artifact_name = "da3"
    unit_type = "frame_pair"
    num_pairs = 20

    @property
    def contract(self) -> InputContract:  # type: ignore[override]
        return InputContract.from_field_paths(
            required=["media.videos", f"artifacts.{self.artifact_name}"],
        )

    def evaluate(self, item: EvaluationItem, **kwargs: Any) -> list[UnitResult]:
        artifact = item.artifacts.get(self.artifact_name)
        if artifact is None or artifact.path is None:
            return [self._invalid(item, "missing DA3 artifact")]
        try:
            data = load_da3_data(artifact.path)
            pairs = find_fold_pairs(data["extrinsics"], num_pairs=self.num_pairs)
            return [self._evaluate_pair(item, data, pair) for pair in pairs]
        except Exception as e:
            return [self._invalid(item, str(e))]

    def _evaluate_pair(
        self,
        item: EvaluationItem,
        data: dict[str, Any],
        pair: dict[str, Any],
    ) -> UnitResult:
        raise NotImplementedError


    def aggregate_item(self, item: EvaluationItem, units: list[UnitResult]) -> ItemResult:
        if not units:
            return ItemResult(
                item_id=item.item_id,
                model_id=item.model_id,
                metric=self.name,
                scores={},
                num_units=0,
                valid=False,
                error="no frame-pair unit results",
            )

        valid_units = [unit for unit in units if unit.valid]
        invalid_units = [unit for unit in units if not unit.valid]
        scores: dict[str, float] = {
            "num_valid_units": float(len(valid_units)),
            "num_invalid_units": float(len(invalid_units)),
            "valid_unit_rate": float(len(valid_units) / len(units)) if units else 0.0,
        }

        primary_key = self.score_keys[0] if self.score_keys else None
        if primary_key:
            primary_values = [
                float(unit.scores[primary_key])
                for unit in valid_units
                if primary_key in unit.scores and math.isfinite(float(unit.scores[primary_key]))
            ]
            if primary_values:
                scores[primary_key] = float(sum(primary_values) / len(primary_values))

        for key in self.score_keys[1:]:
            values = [
                float(unit.scores[key])
                for unit in valid_units
                if key in unit.scores and math.isfinite(float(unit.scores[key]))
            ]
            if values:
                scores[key] = float(sum(values) / len(values))

        match_values = [
            float(unit.scores["num_matches"])
            for unit in units
            if "num_matches" in unit.scores and math.isfinite(float(unit.scores["num_matches"]))
        ]
        if match_values:
            scores["num_matches"] = float(sum(match_values) / len(match_values))

        translation_values = []
        rotation_values = []
        pose_return_values = []
        for unit in units:
            metadata = unit.metadata or {}
            translation = metadata.get("translation_dist")
            rotation = metadata.get("rotation_dist_deg")
            if translation is not None:
                translation_values.append(float(translation))
            if rotation is not None:
                rotation_values.append(float(rotation))
            pose_score = self._pose_return_pair_score(unit)
            if pose_score is not None:
                pose_return_values.append(pose_score)

        if translation_values:
            scores["mean_translation_dist"] = float(sum(translation_values) / len(translation_values))
        if rotation_values:
            scores["mean_rotation_dist_deg"] = float(sum(rotation_values) / len(rotation_values))
        if pose_return_values:
            pose_return_coverage = float(sum(pose_return_values) / len(pose_return_values))
            scores["pose_return_score"] = pose_return_coverage
            scores["pose_return_coverage"] = pose_return_coverage

        trigger_score = float(scores.get("pose_return_coverage", 0.0))
        reliability_score = self._item_reliability_score(scores)
        _apply_trigger_conditioned_item_scores(
            scores,
            trigger_score=trigger_score,
            reliability_score=reliability_score,
        )

        primary_key = self.score_keys[0] if self.score_keys else None
        has_primary_score = bool(primary_key and primary_key in scores)
        return ItemResult(
            item_id=item.item_id,
            model_id=item.model_id,
            metric=self.name,
            scores=scores,
            num_units=len(units),
            valid=has_primary_score,
            error=None if has_primary_score else "no valid frame-pair scores",
        )

    def _item_reliability_score(self, scores: dict[str, float]) -> float | None:
        raise NotImplementedError

    def _pose_return_pair_score(self, unit: UnitResult) -> float | None:
        metadata = unit.metadata or {}
        translation = metadata.get("translation_dist")
        rotation = metadata.get("rotation_dist_deg")
        if translation is None or rotation is None:
            return None
        return pose_return_pair_score(float(translation), float(rotation))

    def _invalid(self, item: EvaluationItem, error: str) -> UnitResult:
        return UnitResult(
            item_id=item.item_id,
            metric=self.name,
            unit_type=self.unit_type,
            unit_id=f"{item.item_id}__da3",
            scores={},
            valid=False,
            error=error,
        )


class SpatialEpipolarMetric(_DA3PairMetric):
    name = A_ENV_SPATIAL_EPIPOLAR_METRIC
    description = "Viewpoint consistency via SIFT matches and DA3-pose epipolar geometry."
    score_keys = (
        "median_epi_dist",
        "mean_epi_dist",
        "std_epi_dist",
        "num_matches",
    )

    def _evaluate_pair(
        self,
        item: EvaluationItem,
        data: dict[str, Any],
        pair: dict[str, Any],
    ) -> UnitResult:
        i, j = int(pair["frame_out"]), int(pair["frame_in"])
        img_i = read_image_at_depth_res(data["image_paths"][i], data["depth_shape"])
        img_j = read_image_at_depth_res(data["image_paths"][j], data["depth_shape"])
        scores = compute_epipolar_error(
            img_i,
            img_j,
            data["intrinsics"][i],
            data["extrinsics"][i],
            data["intrinsics"][j],
            data["extrinsics"][j],
        )
        return UnitResult(
            item_id=item.item_id,
            metric=self.name,
            unit_type=self.unit_type,
            unit_id=f"{item.item_id}__f{i}_f{j}",
            scores=scores,
            valid=bool(scores.get("median_epi_dist") is not None),
            metadata=pair,
        )

    def _item_reliability_score(self, scores: dict[str, float]) -> float | None:
        if "median_epi_dist" not in scores:
            return None
        return distance_to_score(scores["median_epi_dist"], tau=5.0)


class SpatialReprojectionMetric(_DA3PairMetric):
    name = A_ENV_SPATIAL_REPROJECTION_METRIC
    description = "Viewpoint consistency via DA3-depth reprojection."
    score_keys = (
        "median_reproj",
        "mean_reproj",
        "std_reproj",
        "num_matches",
        "num_valid_fwd",
        "num_valid_bwd",
    )

    def _evaluate_pair(
        self,
        item: EvaluationItem,
        data: dict[str, Any],
        pair: dict[str, Any],
    ) -> UnitResult:
        i, j = int(pair["frame_out"]), int(pair["frame_in"])
        img_i = read_image_at_depth_res(data["image_paths"][i], data["depth_shape"])
        img_j = read_image_at_depth_res(data["image_paths"][j], data["depth_shape"])
        scores = compute_reprojection_error(
            img_i,
            img_j,
            data["depth"][i],
            data["depth"][j],
            data["intrinsics"][i],
            data["extrinsics"][i],
            data["intrinsics"][j],
            data["extrinsics"][j],
        )
        return UnitResult(
            item_id=item.item_id,
            metric=self.name,
            unit_type=self.unit_type,
            unit_id=f"{item.item_id}__f{i}_f{j}",
            scores=scores,
            valid=bool(scores.get("median_reproj") is not None),
            metadata=pair,
        )

    def _item_reliability_score(self, scores: dict[str, float]) -> float | None:
        if "median_reproj" not in scores:
            return None
        return distance_to_score(scores["median_reproj"], tau=12.0)


class RenderingStyleConsistencyMetric(_DA3PairMetric):
    name = A_ENV_RENDERING_STYLE_METRIC
    description = "Rendering style consistency from DINOv2, LPIPS, SSIM, and VGG Gram metrics."
    score_keys = (
        "dino_sim",
        "lpips",
        "ssim",
        "gram_dist",
    )

    def _evaluate_pair(
        self,
        item: EvaluationItem,
        data: dict[str, Any],
        pair: dict[str, Any],
    ) -> UnitResult:
        i, j = int(pair["frame_out"]), int(pair["frame_in"])
        try:
            scores = _StyleMetricBundle().compute_all(
                str(data["image_paths"][i]),
                str(data["image_paths"][j]),
            )
            valid = True
            error = None
        except Exception as e:
            scores = {}
            valid = False
            error = str(e)
        return UnitResult(
            item_id=item.item_id,
            metric=self.name,
            unit_type=self.unit_type,
            unit_id=f"{item.item_id}__f{i}_f{j}",
            scores=scores,
            valid=valid,
            error=error,
            metadata=pair,
        )

    def _item_reliability_score(self, scores: dict[str, float]) -> float | None:
        if "gram_dist" not in scores:
            return None
        return clamp01(1.0 - float(scores["gram_dist"]))


class _StyleMetricBundle:
    """Migrated memory_consistency style bundle."""

    def __init__(self, device: str = "cuda"):
        import torch

        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self._lpips = None
        self._dino = None
        self._gram = None

    def compute_all(self, path1: str, path2: str) -> dict[str, float]:
        img1 = self._load_image_tensor(path1)
        img2 = self._load_image_tensor(path2)
        scores = {"ssim": round(self._compute_ssim(img1, img2), 5)}
        scores["lpips"] = round(self._compute_lpips(img1, img2), 5)
        scores["dino_sim"] = round(self._compute_dino_sim(img1, img2), 5)
        scores["gram_dist"] = round(self._compute_gram_distance(img1, img2), 5)
        return scores

    def _load_image_tensor(self, path: str, size: int = 512):
        import numpy as np
        from PIL import Image

        img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
        tensor = self.torch.tensor(np.array(img), dtype=self.torch.float32).permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0).to(self.device)

    def _compute_ssim(self, img1_t, img2_t) -> float:
        from skimage.metrics import structural_similarity

        a = img1_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        b = img2_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        return float(structural_similarity(a, b, channel_axis=2, data_range=1.0))

    def _compute_lpips(self, img1_t, img2_t) -> float:
        if self._lpips is None:
            import lpips

            self._lpips = lpips.LPIPS(net="alex").to(self.device).eval()
        with self.torch.no_grad():
            distance = self._lpips(img1_t * 2 - 1, img2_t * 2 - 1)
        return float(distance.item())

    def _compute_dino_sim(self, img1_t, img2_t) -> float:
        import torch.nn.functional as F
        from PIL import Image

        if self._dino is None:
            from transformers import AutoImageProcessor, AutoModel

            processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
            model = AutoModel.from_pretrained("facebook/dinov2-large").to(self.device).eval()
            self._dino = (processor, model)

        processor, model = self._dino

        def feat(img_t):
            arr = (img_t.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255.0).astype("uint8")
            inputs = processor(images=Image.fromarray(arr), return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                out = model(**inputs)
            return F.normalize(out.last_hidden_state[:, 0, :], dim=-1)

        return float(F.cosine_similarity(feat(img1_t), feat(img2_t), dim=-1).item())

    def _compute_gram_distance(self, img1_t, img2_t) -> float:
        import torch.nn.functional as F

        if self._gram is None:
            from torchvision.models import VGG19_Weights, vgg19

            vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(self.device).eval()
            mean = self.torch.tensor([0.485, 0.456, 0.406]).to(self.device).view(-1, 1, 1)
            std = self.torch.tensor([0.229, 0.224, 0.225]).to(self.device).view(-1, 1, 1)
            self._gram = (vgg, mean, std)

        vgg, mean, std = self._gram

        def grams(img_t):
            x = (img_t - mean) / std
            out = []
            for idx, layer in enumerate(vgg):
                x = layer(x)
                if str(idx) in {"3", "8", "17", "26"}:
                    _b, c, h, w = x.shape
                    feat = x.view(c, h * w)
                    out.append(self.torch.mm(feat, feat.t()))
            return out

        with self.torch.no_grad():
            g1 = grams(img1_t)
            g2 = grams(img2_t)

        dists = []
        for a, b in zip(g1, g2):
            dists.append(1.0 - F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item())
        return float(sum(dists) / len(dists))


class RenderingLightingConsistencyMetric(Metric):
    name = A_ENV_RENDERING_LIGHTING_METRIC
    description = "Rendering lighting consistency via mirrored temporal segment LAB statistics."
    unit_type = "video"
    score_keys = (
        "lighting_score",
        "lighting_distance",
        "luminance_distance",
        "color_shift_distance",
    )

    @property
    def contract(self) -> InputContract:  # type: ignore[override]
        return InputContract.from_field_paths(required=["media.videos"])

    def evaluate(self, item: EvaluationItem, **kwargs: Any) -> list[UnitResult]:
        try:
            scores = compute_lighting_consistency(
                first_video_path(item),
                num_segments=int(kwargs.get("num_segments", 5)),
            )
            return [
                UnitResult(
                    item_id=item.item_id,
                    metric=self.name,
                    unit_type=self.unit_type,
                    unit_id=f"{item.item_id}__lighting",
                    scores=scores,
                    valid=True,
                )
            ]
        except Exception as e:
            return [
                UnitResult(
                    item_id=item.item_id,
                    metric=self.name,
                    unit_type=self.unit_type,
                    unit_id=f"{item.item_id}__lighting",
                    scores={},
                    valid=False,
                    error=str(e),
                )
            ]

    def aggregate_item(self, item: EvaluationItem, units: list[UnitResult]) -> ItemResult:
        if not units:
            return ItemResult(
                item_id=item.item_id,
                model_id=item.model_id,
                metric=self.name,
                scores={},
                num_units=0,
                valid=False,
                error="no lighting unit result",
            )
        unit = units[0]
        scores = dict(unit.scores)
        lighting_score = float(scores.get("lighting_score", 0.0))
        _apply_trigger_conditioned_item_scores(
            scores,
            trigger_score=1.0,
            reliability_score=lighting_score,
        )
        return ItemResult(
            item_id=item.item_id,
            model_id=item.model_id,
            metric=self.name,
            scores=scores,
            num_units=len(units),
            valid=unit.valid,
            error=unit.error,
        )


def compute_lighting_consistency(
    video_path: str,
    *,
    num_segments: int = 5,
) -> dict[str, float]:
    """Compare mirrored early/late temporal segments in LAB space."""
    import cv2
    import numpy as np

    frames, _ = sample_bgr_frames(video_path, n_frames=max(num_segments * 8, 16))
    if len(frames) < 2:
        raise ValueError(f"not enough frames for lighting consistency: {video_path}")

    segment_size = max(1, len(frames) // num_segments)
    segment_stats: list[tuple[float, float, float]] = []
    for seg_idx in range(num_segments):
        start = seg_idx * segment_size
        end = min(len(frames), (seg_idx + 1) * segment_size)
        chunk = frames[start:end]
        if not chunk:
            continue
        lab_values = [cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32) for frame in chunk]
        stacked = np.stack(lab_values, axis=0)
        l_mean = float(stacked[:, :, :, 0].mean())
        a_mean = float(stacked[:, :, :, 1].mean())
        b_mean = float(stacked[:, :, :, 2].mean())
        segment_stats.append((l_mean, a_mean, b_mean))

    if len(segment_stats) < 2:
        raise ValueError(f"not enough segments for lighting consistency: {video_path}")

    half = len(segment_stats) // 2
    early = segment_stats[:half]
    late = list(reversed(segment_stats[-half:]))
    pair_count = min(len(early), len(late))
    lum_dists = []
    color_dists = []
    for left, right in zip(early[:pair_count], late[:pair_count]):
        lum_dists.append(abs(left[0] - right[0]) / 255.0)
        color_dists.append(
            float(np.linalg.norm(np.array(left[1:]) - np.array(right[1:])) / 255.0)
        )
    luminance_distance = float(sum(lum_dists) / len(lum_dists))
    color_shift_distance = float(sum(color_dists) / len(color_dists))
    lighting_distance = float(luminance_distance + color_shift_distance)
    lighting_score = clamp01(float(np.exp(-lighting_distance)))
    return {
        "lighting_score": round(lighting_score, 6),
        "lighting_distance": round(lighting_distance, 6),
        "luminance_distance": round(luminance_distance, 6),
        "color_shift_distance": round(color_shift_distance, 6),
    }


@lru_cache(maxsize=1)
def _cached_style_metric_bundle() -> "_StyleMetricBundle":
    return _StyleMetricBundle()


A_ENVIRONMENT_METRICS = (
    SpatialEpipolarMetric,
    SpatialReprojectionMetric,
    RenderingStyleConsistencyMetric,
    RenderingLightingConsistencyMetric,
)


