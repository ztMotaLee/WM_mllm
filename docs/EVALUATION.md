# Evaluation

The wheel installs `eval`, `reproduce` and `wmmllm_datasets`. Their module commands work outside the source checkout. The source archive contains their detailed READMEs, direct requirements and locks. Core CPU tests need no weights, datasets, credentials or metric inference packages.

Use an isolated environment for inference. DA3 requires NumPy below 2; do not combine it with a NumPy 2 generation environment. Select hardware-appropriate PyTorch wheels and independently install the pinned DA3 source. The `eval` optional extra is a convenience dependency set, not a complete historical environment. See [Mbench](../eval/mbench/README.md), [camera](../eval/camera/README.md), and [CompBench](../eval/compbench/README.md).

## MBench environment recovery

```bash
python -B -m eval.mbench.mem6_scorer describe
python -B -m eval.mbench.mem6_scorer run --manifest scoring_inputs.jsonl --family environment --weights-dir weights --output scores/method_a --work-dir work --device cuda:0
python -B -m eval.mbench.mem6_scorer aggregate --manifest all_models_manifest.jsonl --records scores/method_a/records --records scores/method_b/records --output environment_table.json
```

Manifest rows identify model, item, environment subset and video. The scorer validates code, protocol, input and local weight hashes. It samples 32 frames, runs joint DA3 and evaluates Epipolar, Reprojection, Lighting and Style. Raw values aggregate before canonical projection. The paper adapter defines common video-presence intersection separately from own-full populations and averages only valid finite scores. Invalid or absent scores stay counted. Bootstrap intervals use the frozen 20,000-draw procedure. This port excludes human, object, causal and text/judge metrics.

## Camera analysis

```bash
python -m eval.camera.build_manifest --table inputs/final_table.json --generated-inputs inputs/generated_videos.jsonl --public-manifest inputs/public_videos.json --video-root videos --out inputs/pose_inputs.jsonl
python -m eval.camera.pose_only --manifest inputs/pose_inputs.jsonl --video-root videos --weights-dir weights --out poses --work-dir work/camera
python -m eval.camera.analyze_camera --poses-dir poses --out camera_completion.json
```

Pose extraction needs the separately installed model. Saved-pose analysis is CPU-only. See the camera README and reproduction notes for exact direction/excursion/return formulas and the paper-description difference. Missing measurements remain failures/unavailable values.

## CompBench

```bash
python -m wmmllm_datasets.prepare_compbench --root data/CompBench --output compbench_inputs.jsonl --verify
python -m reproduce.compbench --inputs compbench_inputs.jsonl --input-root data/CompBench --proposal-root saved_proposals --out composed
python -m eval.compbench.prepare_results --composed composed --results-root editing_results --model my_model
python -m eval.compbench.run_eval my_model --check-inputs --index compbench_inputs.jsonl --data-root data/CompBench/tasks --results-root editing_results --output-dir scores --tasks local multi_object implicit
python -m eval.compbench.run_eval my_model --clip-path weights/clip-vit-large-patch14 --data-root data/CompBench/tasks --results-root editing_results --output-dir scores --device cpu --tasks local multi_object implicit
```

Supply one real saved proposal at `saved_proposals/<row_id>.png` for every prepared row. The composer accepts the prepared JSONL directly, retains each `subtask` and `eval_key`, and exports final PNG bytes under the evaluator's original image names. The result-model directory must be new. `--check-inputs` verifies the handoff without metric inference; the final command runs the metrics. See [composition inputs](../reproduce/compbench/README.md) for explicit proposal paths. Target images and benchmark masks do not select committed pixels. Use the recovered pinned evaluator and its local CLIP/LPIPS setup. The final aggregator consumes supplied tables for the three historical installation arms. The session engine's current support selector is a separate configuration.

## Data and CPU checks

```bash
python -m wmmllm_datasets.verify_dataset --help
python -m wmmllm_datasets.prepare_compbench --help
python -m pytest -q tests/providers tests/core tests/eval tests/viewer tests/integration
```

[Dataset instructions](../scripts/datasets/README.md) describe the separately obtained files and checksums. URLs/weight terms were recorded from local sources and require independent verification; no download is part of the scorer. All distributed examples and test media are procedural. See [reproduction](REPRODUCTION.md) for result coverage and limitations.
