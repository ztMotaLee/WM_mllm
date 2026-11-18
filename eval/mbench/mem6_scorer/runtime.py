"""Lazy offline metric runtime. Weights are supplied by the user."""
from __future__ import annotations
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import time
from .common import PACKAGE, FAMILIES, file_ref, sha256

def video_metadata(path):
    import cv2
    capture = cv2.VideoCapture(str(path))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if total <= 0 or fps <= 0:
        raise ValueError("Cannot read actual video frame count/FPS: " + str(path))
    return {"frames": total, "fps": fps, "duration_seconds": total / fps,
            "width": width, "height": height,
            "sampling": "upstream sample_bgr_frames, numpy.linspace dtype=int, seek/decode"}



class Runtime:
    def __init__(self, family, device, work_dir, lock):
        import torch
        from mbench.metrics import environment_consistency as env

        self.family = family
        self.device = device
        self.work_dir = work_dir
        self.lock = lock
        self.torch = torch
        self.da3 = None
        self.style = None
        classes = {cls.name: cls for cls in (
            env.SpatialEpipolarMetric, env.SpatialReprojectionMetric,
            env.RenderingLightingConsistencyMetric, env.RenderingStyleConsistencyMetric)}
        self.metrics = [classes[name]() for name in FAMILIES[family]]
        if family == "environment":
            # Upstream A style constructs this bundle per pair. Holding the
            # same instance changes model lifetime only; every method is the
            # unchanged upstream implementation.
            self.style = env._StyleMetricBundle(device=device)
            env._StyleMetricBundle = lambda *args, **kwargs: self.style
        self.gpu = {"device": device}

    @classmethod
    def for_pose(cls, device, work_dir, lock):
        """Construct the shared DA3 producer without constructing style metrics."""
        import torch
        obj = cls.__new__(cls)
        obj.family, obj.device, obj.work_dir, obj.lock = "camera", device, work_dir, lock
        obj.torch, obj.da3 = torch, None
        return obj

    def synchronize(self):
        if self.device.startswith("cuda:"):
            self.torch.cuda.synchronize(self.device)

    def prepare_da3(self, row, dest):
        import cv2
        import hashlib
        import numpy as np
        from depth_anything_3.api import DepthAnything3
        from mbench.metrics.video_io import sample_bgr_frames

        verify_da3_source(self.lock)

        started = time.monotonic()
        frames, indices = sample_bgr_frames(row["video_path"], 32)
        if len(frames) != 32 or len(set(indices)) != 32:
            raise ValueError("Uniform32 DA3 requires 32 distinct decoded frames")
        decode_seconds = time.monotonic() - started
        if self.da3 is None:
            self.da3 = DepthAnything3.from_pretrained(
                self.lock["snapshots"]["depth-anything/DA3NESTED-GIANT-LARGE"]["path"]
            ).to(self.device).eval()
        self.synchronize()
        inference_started = time.monotonic()
        prediction = self.da3.inference(
            [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames],
            process_res=504, process_res_method="upper_bound_resize",
            infer_gs=False, export_dir=None)
        self.synchronize()
        inference_seconds = time.monotonic() - inference_started
        arrays = {name: np.asarray(getattr(prediction, name), dtype=np.float32)
                  for name in ("depth", "intrinsics", "extrinsics")}
        raw_w2c = arrays["extrinsics"].copy()
        # Preserve the frozen camera-transform numerical expression.
        source = np.asarray(raw_w2c, dtype=np.float32)
        if source.ndim != 3 or source.shape[1:] not in ((3, 4), (4, 4)):
            raise ValueError("Unexpected DA3 camera shape")
        transforms = np.broadcast_to(np.eye(4, dtype=np.float32), (len(source), 4, 4)).copy()
        transforms[:, :3, :4] = source[:, :3, :4]
        arrays["extrinsics"] = np.linalg.inv(transforms).astype(np.float32)[:, :3, :4]
        if any(len(value) != 32 or not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("DA3 arrays invalid")
        dest.mkdir()
        images = dest / "input_images"
        images.mkdir()
        # The producer saves original BGR samples. The unmodified official
        # reader later resizes these to depth_shape with INTER_LINEAR.
        for ordinal, frame in enumerate(frames):
            if not cv2.imwrite(str(images / f"{ordinal:05d}.png"), frame):
                raise RuntimeError("Failed to save original DA3 input pixels")
        np.savez_compressed(dest / "results.npz", **arrays)
        product = {"artifact_sha256": sha256(dest / "results.npz"),
                   "sampled_frame_indices": indices,
                   "sampled_bgr_pixel_sha256": [hashlib.sha256(f.tobytes()).hexdigest() for f in frames],
                   "sampled_bgr_shapes": [list(f.shape) for f in frames],
                   "depth_shape": list(arrays["depth"].shape),
                   "intrinsics_shape": list(arrays["intrinsics"].shape),
                   "extrinsics_shape": list(arrays["extrinsics"].shape),
                   "world_to_camera_array_sha256": hashlib.sha256(raw_w2c.tobytes()).hexdigest(),
                   "camera_to_world_array_sha256": hashlib.sha256(arrays["extrinsics"].tobytes()).hexdigest(),
                   "pose_convention": "camera-to-world, inverse raw DA3 world-to-camera; original OpenCV reference world",
                   "image_policy": "Original sampled BGR PNG; upstream read_image_at_depth_res INTER_LINEAR resize",
                   "process_res": 504, "process_res_method": "upper_bound_resize",
                   "infer_gs": False, "export_dir": None,
                   "frame_policy": "sample_bgr_frames(video,32); 32 distinct original indices; compact arrays",
                   "decode_seconds": decode_seconds, "inference_seconds": inference_seconds,
                   "producer_seconds": time.monotonic() - started}
        del prediction, arrays, raw_w2c, frames
        return dest / "results.npz", product

    def evaluate_metric(self, metric, item):
        from mbench.core.pipeline import _project_unit_scores, _project_item_scores
        from mbench.aggregators.mean import MeanAggregator
        kwargs = {"device": self.device, "ignore_trigger": True}
        units = metric.evaluate(item, **kwargs)
        # Official order: aggregate native raw unit signals before projecting.
        raw_item = metric.aggregate_item(item, units)
        raw_item_scores = dict(raw_item.scores)
        units = [_project_unit_scores(metric.name, unit) for unit in units]
        canonical_item = _project_item_scores(metric.name, raw_item)
        summary = MeanAggregator().aggregate(metric.name, [canonical_item])
        return {"units": [asdict(unit) for unit in units], "item": asdict(canonical_item),
                "summary": asdict(summary), "raw_item_scores": raw_item_scores}



def environment_versions():
    names = ("torch", "torchvision", "transformers", "depth-anything-3",
             "lpips", "numpy", "opencv-python", "scikit-image", "Pillow")
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result



def verify_weights(weights_dir, family="environment"):
    """Verify user-supplied files and resolve the portable model lock in memory."""
    if family not in {"environment", "camera"}:
        raise ValueError("Supported families are environment and camera")
    root = Path(weights_dir).resolve(strict=True)
    lock_path = PACKAGE / "MODEL_LOCK.json"
    lock = json.loads(lock_path.read_text())
    checked = []
    for ref in lock["files"]:
        if family not in ref["families"]:
            continue
        path = (root / ref["path"]).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("Weight path escapes weights directory")
        if not path.is_file() or path.stat().st_size != ref["bytes"] or sha256(path) != ref["sha256"]:
            raise RuntimeError("Pinned weight is missing or changed: " + ref["path"])
        checked.append({**ref, "path": str(path)})
    lock["files"] = checked
    lock["snapshots"] = {
        repo: {**value, "path": str(root / value["path"])}
        for repo, value in lock["snapshots"].items()
        if family == "environment" or repo.startswith("depth-anything/")
    }
    lock["weights_dir"] = str(root)
    lock["torch_home"] = str(root / "torch")
    return lock, {"lock": file_ref(lock_path), "verified_files": checked}


def verify_da3_source(lock):
    """Require the pinned producer implementation; never search a checkout."""
    import depth_anything_3.api as api
    package_root = Path(api.__file__).resolve().parent
    for ref in lock["da3_source"]["files"]:
        path = (package_root / ref["path"]).resolve(strict=True)
        if not path.is_relative_to(package_root) or path.stat().st_size != ref["bytes"] or sha256(path) != ref["sha256"]:
            raise RuntimeError("Installed DA3 source differs from the pinned implementation: " + ref["path"])



def configure_offline(lock, work_dir):
    """Bind local model files. Call only in an explicit evaluation process."""
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
                 "HF_HUB_DISABLE_TELEMETRY", "HF_HUB_DISABLE_PROGRESS_BARS"):
        os.environ[name] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TORCH_HOME"] = lock["torch_home"]
    for name, subdir in (("HF_HOME", "hf"), ("XDG_CACHE_HOME", "cache"),
                         ("MPLCONFIGDIR", "matplotlib"), ("TMPDIR", "tmp")):
        path = Path(work_dir) / subdir
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    import torch

    def download_disabled(*args, **kwargs):
        raise RuntimeError("Model downloads are disabled; prepare the pinned weights first")

    torch.hub.download_url_to_file = download_disabled
    import transformers
    snapshots = lock["snapshots"]
    for name in ("AutoModel", "AutoImageProcessor"):
        cls = getattr(transformers, name)
        original = cls.from_pretrained

        def pinned_loader(identifier, *args, _original=original, **kwargs):
            identifier = str(identifier)
            if identifier in snapshots:
                identifier = snapshots[identifier]["path"]
            elif identifier not in {value["path"] for value in snapshots.values()}:
                raise RuntimeError("Unpinned transformer model requested")
            kwargs["local_files_only"] = True
            return _original(identifier, *args, **kwargs)

        cls.from_pretrained = staticmethod(pinned_loader)
    # LPIPS uses the same v0.1 calibration bytes, supplied explicitly instead
    # of relying on the location of the installed package's copy.
    if any(ref["repo"] == "lpips-v0.1-alex-calibration" for ref in lock["files"]):
        import lpips
        original_lpips = lpips.LPIPS
        calibration = next(ref["path"] for ref in lock["files"]
                           if ref["repo"] == "lpips-v0.1-alex-calibration")

        def pinned_lpips(*args, **kwargs):
            kwargs["model_path"] = calibration
            return original_lpips(*args, **kwargs)

        lpips.LPIPS = pinned_lpips
    return {"transformers": "explicit pinned local snapshots; local_files_only=True",
            "torchvision": "verified checkpoint cache; downloads disabled",
            "lpips": "verified v0.1 AlexNet calibration bytes"}
