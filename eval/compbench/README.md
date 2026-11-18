# CompBench evaluation

Run these tools from the release source root. Index preparation, composition, export and saved-score aggregation need only NumPy, Pillow, SciPy and the standard library. Metric inference uses a separate environment (`requirements.txt`). No weights, dataset images, generated outputs or evaluator code are distributed.

## Evaluator

Clone `https://github.com/comp-bench/CompBench` (MIT) and check out commit `cb9165b01d14a321db14611afd9922c43a04665d`. `run_eval.py` requires `--evaluator CompBench/eval_all.py` and refuses any file whose SHA-256 differs from `EVALUATOR_SHA256` (`04f622a3…dea0e5e`) before importing it. It replaces the CLIP path in memory with `--clip-path` and uses local-only loading for the torchmetrics CLIP adapter.

Download `openai/clip-vit-large-patch14` yourself, including the processor and tokenizer files. Archived evaluation logs identify these two model files:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| config.json | 4519 | 8a09b467700c58138c29d53c605b34ebc69beaadd13274a8a2af8ad2c2f4032a |
| model.safetensors | 1710540580 | a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb |

An upstream model revision and processor/tokenizer content hashes were not recorded in those logs. They remain unverified. LPIPS uses the upstream `net_type='squeeze'` path, including its SqueezeNet backbone and torchmetrics calibration weights; it does not use the MBench AlexNet LPIPS variant. Obtain those weights under their own terms. The evaluator version is torchmetrics 1.4.3. Changing it can change SSIM; the final paper numbers use the saved scores from that environment.

The evaluator uses 512-square bicubic RGB preprocessing and nearest mask resizing, CLIP on the mask bounding rectangle, and zero-masked background metrics. It preserves the original skip/resume behavior; check detailed CSV counts before using a mean. Do not interpret missing outputs as completed evaluation.

## Historical composition

`compose.py` is the historical RGB-difference composer used with the paper's 2,709 saved GPT-Image 2 proposals. It computes mean RGB absolute differences in float32, thresholds strictly above 25, reconstructs thin-safe connected components, keeps components with at least 64 pixels, dilates by 7 and uses Pillow Gaussian feathering with radius 3. Pillow performs the final RGB composition. Pixels where the stored alpha equals zero remain source-exact. The session engine has a separate support mechanism (post-generation MLLM polygons, dilation 16 and inward feather 12); keep their results separate.

The recorded production composer and selector hashes (`c82b75c2…80b1` and `39b2d465…901d`) differ from the available source files (`aab589da…7cae` and `d2c8c838…c2f7`); no matching copies were found in a bounded historical search. This port uses the available files. On 24 saved production items, four per subtask across all six task types, support, alpha and final RGB match exactly. This establishes sample equivalence; it does not establish whole-population pixel equivalence for every historical proposal.

```bash
python -m eval.compbench.prepare_index --root data/CompBench --output compbench_inputs.jsonl
python -m eval.compbench.compose --inputs compbench_inputs.jsonl --input-root data/CompBench --proposal-root saved_proposals --out composed
python -m eval.compbench.export --composed composed --results-root editing_results --model my_model
python -m eval.compbench.run_eval my_model --evaluator CompBench/eval_all.py --check-inputs --index compbench_inputs.jsonl \
  --data-root data/CompBench/tasks --results-root editing_results --output-dir scores --tasks local multi_object implicit
python -m eval.compbench.run_eval my_model --evaluator CompBench/eval_all.py --clip-path weights/clip-vit-large-patch14 \
  --data-root data/CompBench/tasks --results-root editing_results --output-dir scores --device cpu --tasks local multi_object implicit
```

Supply a real saved image proposal at `saved_proposals/<row_id>.png` for every row. With `--proposal-root`, `source` resolves within `--input-root` and `direct` resolves within `--proposal-root`; a missing `direct` field means `<row_id>.png`. Without `--proposal-root`, both paths resolve within `--input-root`, and a nonempty JSON array of objects with `row_id`, `source` and `direct` is also accepted. Target images, benchmark masks and metric files never select committed pixels. The loader keeps the original RGB conversion and uniform Lanczos resize of a proposal to the source dimensions. There is no model call.

The composition output directory must be new. It contains `support/`, `alpha/` and `final/` PNGs and `RUN_MANIFEST.json`, which records each `row_id`, `subtask`, `eval_key`, final path and final-file hash. The composer reads its written PNGs back and checks the zero-alpha guarantee. The support-area ceiling 0.56 lists global edits in `out_of_scope_rows` for manual review; it does not prune pixels.

The exporter requires a new model directory, preserves every original `eval_key` (including nested paths), verifies the composition hashes and copies final PNG bytes without re-encoding. `RESULTS_MANIFEST.json` records the full `row_id` to evaluator-path mapping. `--check-inputs` verifies the pinned evaluator and selected source/target/mask/result files, then exits before loading metric models. The evaluator expects result images at:

```text
editing_results/my_model/my_model_local/{add,remove,replace}/<eval_key>
editing_results/my_model/my_model_multi/{multi_object_add,multi_object_remove}/<eval_key>
editing_results/my_model/my_model_implicit/<eval_key>
```

## Paper aggregation

The final three-arm paper aggregation uses the detailed per-image CSVs:

```bash
python -m eval.compbench.aggregate --scores saved_scores.tar.gz --output ablation.json
```

The default archive model names match the saved paper artifacts; override them with `--direct-model`, `--paste-model` and `--ours-model` for your own archives. Each CSV path ends in `<model>/<task-tag>/<model>/<subtask>_detailed.csv`. Task tags are add, remove, replace, multi_object and implicit. Members are read in memory without extraction.

Non-PSNR fields use their own available rows, skipping NaN. PSNR uses image IDs with finite values in all three arms, leaving 2,462 local items. Round each subtask mean to three decimals using NumPy scalar rounding, then average subtasks equally and round again. The final aggregation reproduces all 45 group/arm/metric values and counts of the saved paper table exactly. Only the descriptive `rules` text differs from the saved file. An empty valid population or a nonfinite aggregate raises an error.
