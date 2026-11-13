# MBench-A recovery evaluator

This port contains the four environment metrics used for the recovery comparison:
Epipolar, Reprojection, Lighting, and Style. It uses MBench commit
`c44dbd9dfa0278be728bda2101b0fb96f8547f11` from
[study-overflow/MBench](https://github.com/study-overflow/MBench).
The derived code retains its MIT notices and [complete license](LICENSE.upstream).
The action-conditioned metrics require no API judge. Human, object, causal, and
text-conditioned metrics are outside this port.

The evaluator uses 32 distinct frames sampled by OpenCV with
`numpy.linspace(0, total-1, 32, dtype=int)`, joint DA3 inference at resolution
504 and `upper_bound_resize`, and float32 inversion of world-to-camera poses.
It stores original sampled BGR frames; the metric reader resizes them to the
depth shape using `INTER_LINEAR`. Fold pairing uses the unchanged upstream
20-pair selector. Native unit values aggregate before canonical projection:
`exp(-mean_distance/5)` for Epipolar, `exp(-mean_distance/12)` for Reprojection,
`clip(1-mean_gram_distance,0,1)` for Style, and `lighting_score` for Lighting.
[PROTOCOL.json](PROTOCOL.json) records the full numerical protocol.

These choices reproduce the paper's common rescoring protocol. The original
leaderboard producer's complete pose settings and normalization are unavailable;
absolute equality to the original leaderboard scores is unverified. Aggregation
on the saved paper receipts is checked separately from model inference. No
weights, benchmark images, or existing videos are included.

## Prepare a separate evaluation environment

Core package startup never imports metric models. Use a separate Python 3.10+
environment for inference. The recorded evaluator used Python 3.10, PyTorch
2.4.0+cu124, torchvision 0.19.0+cu124, Transformers 4.57.6, LPIPS 0.1.4,
NumPy 1.26.4, OpenCV 4.11.0.86, scikit-image 0.25.2, and Pillow 12.3.0.
[requirements-mbench.txt](requirements-mbench.txt) records these Python package
versions. Select the appropriate PyTorch wheel source for your hardware.
A complete transitive environment lock and fresh inference installation have
not been validated. CPU unit tests and saved-score aggregation need no model
packages or weights. DA3 requires **NumPy <2**.

Install Depth Anything 3 from its public repository at commit
`3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`:
[ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3).
Its dependencies include packages beyond the short requirements file. Runtime
checks all 93 recorded DA3 source/resource files from [MODEL_LOCK.json](MODEL_LOCK.json),
including `depth_anything_3.api` with SHA256
`042ae8f6bfd1e5610100a585ebb00f8cbc3320f9aa9e03a07530fd4deda8d2d8`.
If that source differs, stop and resolve the version mismatch explicitly.

## Download the exact weights yourself

Use one explicit weights directory with this layout:

```text
weights/
  da3nested-giant-large/config.json, model.safetensors
  dinov2-large/config.json, model.safetensors, preprocessor_config.json
  torch/hub/checkpoints/alexnet-owt-7be5be79.pth
  torch/hub/checkpoints/vgg19-dcbb9e9d.pth
  lpips/weights/v0.1/alex.pth
```

The model snapshots are:

| Model | Revision |
|---|---|
| `depth-anything/DA3NESTED-GIANT-LARGE` | `8615eefb62f2db4f8d6ebaa59160086981672829` |
| `facebook/dinov2-large` | `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c` |

For example, after installing the Hugging Face CLI, download each snapshot with
`hf download MODEL_ID --revision REVISION --local-dir LOCAL_DIRECTORY` using
the table and layout above. No download occurs inside the scorer.
Torchvision uses its published
[AlexNet](https://download.pytorch.org/models/alexnet-owt-7be5be79.pth) and
[VGG19](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth) files.
Copy `weights/v0.1/alex.pth` from the installed `lpips==0.1.4` package into the
corresponding weights directory location. [MODEL_LOCK.json](MODEL_LOCK.json)
contains every required size and SHA256; startup verifies all selected files.
The camera-only runtime selects the two DA3 files. Models are read from these
local files and downloads are disabled. Model-weight licenses are separate
from the MIT evaluator code; verify each provider's current terms yourself.
Download URLs and external availability were not checked during this offline port.

## Score your videos

Run from the release root with a prepared evaluation environment:

```bash
python -B -m eval.mbench.mem6_scorer describe
python -B -m eval.mbench.mem6_scorer run \
  --manifest scoring_inputs.jsonl --family environment \
  --weights-dir weights --output scores/method_a --work-dir work \
  --device cuda:0
```

`--device cpu` is supported by the wrapper and is unvalidated for full model
inference. Each row needs `model_id`, `item_id`, `subset: "environment"`, and
`video_path`; optional `condition_id`, `sample_id`, and `sha256` bind the video.
JSON arrays and JSONL work. Relative video paths resolve against the manifest.
Duplicate model/item rows fail. Optional `--model`, `--index`, `--limit`, and
`--shard-index/--shard-count` preserve partition order. Concurrent invocations
need separate output directories. Input/code/weights/protocol hashes bind reused
records; changed bindings fail. Temporary per-video artifacts are removed on exit.

Output `records/*.json` preserves official units, raw item diagnostics, projected
items, summary, validity, input binding, and timing. Official summary arithmetic
can retain diagnostic scores from invalid items. For a paper table, use the
valid-only aggregation below. A finite score with `valid: false`, or `valid:
true` without a finite non-boolean score, is unmeasured.

## Aggregate saved scores without models

```bash
python -B -m eval.mbench.mem6_scorer aggregate \
  --manifest all_models_manifest.jsonl \
  --records scores/method_a/records --records scores/method_b/records \
  --output environment_table.json
```

The paper table adapter expects the seven model IDs listed in `aggregation.py`.
Supply every model/item/condition row, including absent-video rows with
`status: "MISSING..."`. Supply each exact record file or a `records` directory
explicitly; the adapter never searches the original checkout or opens media.
It rejects duplicate identities, video hash conflicts, and mixed scorer cohorts.
Common population means exact video-presence intersection over all seven models.
Own-full uses each model's declared-present videos. Score validity never changes
these populations. Report means and confidence intervals use only valid finite
canonical item scores; invalid/missing outcomes remain counted without imputation.
The percentile bootstrap uses 20,000 draws, base seed 45, sorted item identities,
`random.Random`, and `math.fsum`; model/metric/population seeds use the original
SHA256 derivation. Display values multiply the canonical result by 100 once.
The output omits private campaign provenance and historical route metadata.
