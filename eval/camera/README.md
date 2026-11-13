# Camera round-trip evaluation

This evaluation measures camera return from saved camera-to-world poses.
It uses the recorded recovery protocol. It does not run the state engine or
require a model API. No benchmark videos, model files, or saved project poses
are included.

Run these commands from the repository root. All paths below are explicit
user inputs. Manifest video paths must be normalized POSIX paths relative to
`--video-root`; absolute paths and traversal are rejected.

```sh
python -m eval.camera.build_manifest --table inputs/final_table.json \
  --generated-inputs inputs/generated_videos.jsonl \
  --public-manifest inputs/public_videos.json --video-root videos \
  --out inputs/pose_inputs.jsonl
python -m eval.camera.pose_only --manifest inputs/pose_inputs.jsonl \
  --video-root videos --weights-dir weights --out poses --work-dir work/camera
python -m eval.camera.analyze_camera --poses-dir poses --out camera_completion.json
```

The first command reads `populations.common.items` and `.cells` from the table.
Items are IDs or objects containing `item_id`; cells contain `model_id`.
Generated-video JSONL rows contain `item_id`, `condition_id`, and `video_path`.
Their default model ID is `aiwm` (`--generated-model-id` changes it).
Public JSON contains `rows` with `model_id`, `item_id`, `condition_id`, and
`final_path`; a missing model ID is derived from the segment after `models/`
in `final_path`. It selects 224 common items by default, in table item order
and sorted model order. `--expected-items` supports a different supplied
population. Duplicate model/item lookups retain the last supplied row, as in
the recorded selection script. Missing videos or mappings fail.

Pose extraction shares the MBench DA3 producer in `eval/mbench`. It uniformly
samples 32 distinct decoded frames with `numpy.linspace(..., dtype=int)`,
runs them jointly at `process_res=504`, `upper_bound_resize`, `infer_gs=False`,
and inverts float32 world-to-camera transforms as 4x4 matrices before retaining
camera-to-world 3x4 poses. Input video pixels are unchanged. Temporary decoded
images and depth arrays are removed after each video. Saved shards contain
`c2w` and `frame_idx` arrays plus JSONL status rows. A measured row's `k` indexes
its shard; failed rows have `k=null`. Extraction supports explicit sharding
with `--shard-index` and `--num-shards`. The recorded extraction used CUDA;
CPU extraction has not been validated. Saved-pose analysis is CPU-only.

For each video, the first-half frame with largest absolute yaw must turn in
the requested direction by more than one degree. Excursion is the largest
requested-direction yaw over the **whole** clip. Return error is the geodesic
rotation between first and last poses. Completion requires correct first
turn, excursion at least 15 degrees, and return error at most 10 degrees.
Yaw is wrapped to [-180,180]; it is not unwrapped. The supported conditions
are `left_then_right_25s` and `right_then_left_25s`. Failed measurements stay
in item/failed counts and are excluded from valid-score means. Intervals use
20,000 item resamples, seed 55, with one RNG across sorted model IDs. The
sensitivity grid uses excursion thresholds 10/15/20/30 and return thresholds
5/10/15 degrees. With no measured items, unavailable fields are JSON null.

## Model and environment

Users must obtain the model separately and verify its terms:

- Model: `depth-anything/DA3NESTED-GIANT-LARGE`
- Model revision: `8615eefb62f2db4f8d6ebaa59160086981672829`
- DA3 implementation commit: `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`

Use the DA3-only `camera` selection in `eval/mbench`'s local weight verifier.
See that directory's model lock and setup instructions for the required
relative paths and SHA-256 values. DA3 code is Apache-2.0; checkpoint terms
remain separate and require user verification. No DA3 source or weights are
vendored here. The extractor requires an independently installed DA3 checkout
at the stated commit. It does not download models or install packages.

Create an **isolated evaluation environment** for DA3, which requires
`numpy<2`. Install that checkout's declared dependencies and the MBench
evaluation requirements; do not combine it with a NumPy 2 core environment.
`requirements-analysis.txt` is sufficient to analyze saved poses and run the
synthetic numerical tests. Analysis also passed in the release development
environment with NumPy 2; that check does not validate DA3 inference there.

The metric expressions and measured-population bootstrap preserve the
recorded analysis. Added checks reject malformed/nonfinite poses, unknown
conditions, invalid shard indices, and out-of-root video references. Empty
measured populations use null instead of nonstandard JSON NaN. Failure
records retain exception classes rather than raw exception text; progress
records omit machine and scheduler identifiers.
