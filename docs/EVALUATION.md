# Evaluation and reproduction

Fresh generation is stochastic. Paper numbers are reproduced by scoring the original outputs, rather than by bit-identical regeneration. This release does not distribute those outputs, datasets, benchmark code or the video generation model/runtime. Evaluation runs from the source checkout; core CPU tests need no weights, datasets, credentials or metric inference packages.

Use an isolated environment for inference. DA3 requires NumPy below 2; do not combine it with a NumPy 2 generation environment. Select hardware-appropriate PyTorch wheels and independently install the pinned DA3 source. See [MBench](../eval/mbench/README.md), [camera](../eval/camera/README.md), and [CompBench](../eval/compbench/README.md).

## Benchmark code

The adapters load the upstream benchmark code at the commits that scored the paper:

```bash
git clone https://github.com/study-overflow/MBench && git -C MBench checkout c44dbd9dfa0278be728bda2101b0fb96f8547f11 && export PYTHONPATH=$PWD/MBench
git clone https://github.com/comp-bench/CompBench && git -C CompBench checkout cb9165b01d14a321db14611afd9922c43a04665d
```

MBench is imported from the checkout on `PYTHONPATH`; its 18 scorer files are hash-checked against `eval/mbench/CODE_LOCK.json`. CompBench's `eval_all.py` is passed with `--evaluator CompBench/eval_all.py` and must match its pinned SHA-256. Both codebases are MIT licensed.

## MBench environment recovery

Commands: [eval/mbench/README.md](../eval/mbench/README.md). Manifest rows identify model, item, environment subset and video. The scorer validates code, protocol, input and local weight hashes. It samples 32 frames, runs joint DA3 and evaluates Epipolar, Reprojection, Lighting and Style. Raw values aggregate before canonical projection. The paper adapter defines common video-presence intersection separately from own-full populations and averages only valid finite scores. Invalid or absent scores stay counted. Bootstrap intervals use the frozen 20,000-draw procedure. This port excludes human, object, causal and text/judge metrics.

## Camera analysis

Commands: [eval/camera/README.md](../eval/camera/README.md). Pose extraction needs the separately installed model. Saved-pose analysis is CPU-only. Missing measurements remain failures or unavailable values.

The executable camera formula requires the largest absolute first-half yaw to have the requested sign and exceed one degree. Requested-direction excursion is measured over the whole clip and must reach 15 degrees. Final geodesic return error must be at most 10 degrees. The paper appendix's shorter first-half description differs; this release preserves the actual scored formula. See `eval/camera/README.md` for exact arithmetic and threshold sensitivity.

## CompBench

Commands: [eval/compbench/README.md](../eval/compbench/README.md). Supply one real saved proposal at `saved_proposals/<row_id>.png` for every prepared row. The composer accepts the prepared JSONL directly, retains each `subtask` and `eval_key`, and exports final PNG bytes under the evaluator's original image names. The result-model directory must be new. `--check-inputs` verifies the handoff without metric inference; the final command runs the metrics. Target images and benchmark masks do not select committed pixels. The final aggregator (`eval.compbench.aggregate`) consumes supplied tables for the three historical installation arms.

Historical proposals used GPT-Image 2. Composition derives support from mean RGB changes strictly above 25, reconstructs thin structures, retains components of at least 64 pixels, dilates by 7 and uses Gaussian feather radius 3. The session engine instead selects polygons after generation, dilates by 16 and feathers 12 pixels inward. Keep these configurations separate. The composer provenance is recorded in [the CompBench README](../eval/compbench/README.md).

## Data

Obtain the benchmark data from the maintainers after checking their current instructions and terms; no download is part of this release. Do not treat either code repository's MIT license as a license for its dataset, baseline videos or model weights. No redistribution permission is claimed.

- **MBench-A** data (recorded as `huggingface.co/datasets/studyOverflow/MBench-Data`, revision `412581ebc498a752567e47bc1fb2a434692c47ad`; verify the current location and terms). The recovery inputs use 227 environment first frames under `MBench-A/samples/environment/<sample_id>/first_frame.png`. Scoring also needs sample metadata, conditions, videos and per-method availability; the paper's common comparison contains 224 item–condition pairs.
- **CompBench** data (recorded as `huggingface.co/datasets/BohanJia/CompBench`; the dataset revision is unrecorded, so verify the current location and terms). The paper population contains 2,709 items: `add` 982, `remove` 1,331, `replace` 152, `multi_object_add` 72, `multi_object_remove` 72 and `implicit` 100. Preserve the original `tasks/` layout: `metadata.json` with `input_image/`, `edited_image/` and `mask/` for add/remove/replace; `multi_turn_editing/<subtask>.jsonl` with `ori_images/`, `edited_images/` and `masks/`; and `implicit_reasoning/implicit_info.jsonl`, `implicit_data.jsonl` with the same three image directories.

`prepare_index` checks the SHA-256 of the seven metadata files, the population counts and every referenced file, rejects child symlinks, and writes a new index outside the dataset root. The index contains `item_id`, `row_id`, `subtask`, `eval_key`, `source`, `target`, `mask` and `instruction`. `row_id` is the first 24 hexadecimal characters of SHA-256 of `subtask + '/' + eval_key`.

## Paper results and code

| Paper result or study | Release code | What must be supplied or remains excluded |
| --- | --- | --- |
| MBench environment recovery | `eval/mbench/`, four environment metrics and valid-only aggregation | Original input/output manifest, videos, MBench checkout and verified weights; historical recovery generator is excluded |
| Camera round-trip completion and thresholds | `eval/camera/`, shared MBench pose producer | Original videos or saved pose shards; fresh DA3 inference requires separate setup |
| CompBench installation comparison | `eval/compbench/` | Original 2,709 GPT-Image 2 proposals, CompBench checkout, benchmark references and saved scores |
| Same-pose content preservation, exact revisits and outside-support invariants | `state`, `agent.executor` and core tests | Pooled historical outputs, study scripts and judgments are excluded |
| Revision lookup, observed memory and multi-turn components | `state.graph`, `memory`, full `agent`/`session` engine | Historical replay harness, study arms, generated videos and judges are excluded |
| Runtime and costs | Provider receipts and user-priced budget accounting | Historical hardware, service prices and video runtime are excluded |

## Caveats

The historical recovery profile used high chat effort and accelerated video presentation. Its 454 segments comprised 215 rotations, 186 restores, 10 generated-video segments and 43 holds. This package does not implement that historical generator profile.

The 817-segment diagnostic population pooled trajectories and reruns across source revisions and reasoning efforts: 606 segments recorded an effort setting above `high` that is not offered by public endpoints, 154 high, 23 medium and one max; 33 bootstrap holds had no recorded chat effort. These diagnostics must not be described as uniformly using high effort. Their lookup path used direct visit-key lookup; the explicit saved-node, origin, viewport-profile and revision checks cannot be attributed retroactively to that entire population.

All chat roles in the paper's multi-turn sessions used GPT-6.1-sol with high effort, and images used GPT-Image 2.5 via image editing. Other models or endpoints change the configuration. Removing generated video preserves the presentation boundary but does not reproduce video quality or latency.

Port verification reproduced all 56 saved MBench common/own-full numeric cells, including intervals and validity counts; camera analysis matched all 1,568 saved rows; final CompBench three-arm aggregation matched every value and count in the saved table. These checks concern saved results. Clean CPU installation and synthetic tests do not validate fresh model inference. Inference requirement lists and weight locks are not complete transitive installation locks, and no weights download implicitly. No portable generation cost or time guarantee is provided.
