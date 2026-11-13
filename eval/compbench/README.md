# CompBench evaluation

Run these tools from the release source root. Metric inference uses a separate
environment; saved-score aggregation needs only NumPy and the standard library.
No weights, dataset images or generated outputs are distributed.

The upstream evaluator was recovered byte-exactly from git commit
`cb9165b01d14a321db14611afd9922c43a04665d` of
`https://github.com/BhJia/CompBench` (verify upstream URL before downloading).
`EVALUATOR_LOCK.json` records its SHA-256 and the portable copy's SHA-256.
The bundled `upstream/eval_all.py` has the same numerical implementation;
three author-specific path defaults are replaced with explicit paths and a
relative CLIP default. The adapter supplies the actual local CLIP directory.
The MIT text is retained in `LICENSE.upstream`.

Download `openai/clip-vit-large-patch14` yourself, including the processor and
tokenizer files. Archived evaluation logs identify these two model files:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| config.json | 4519 | 8a09b467700c58138c29d53c605b34ebc69beaadd13274a8a2af8ad2c2f4032a |
| model.safetensors | 1710540580 | a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb |

An upstream model revision and processor/tokenizer content hashes were not
recorded in those logs. They remain unverified. LPIPS uses the upstream
`net_type='squeeze'` path, including its SqueezeNet backbone and torchmetrics
calibration weights; it does not use the MBench AlexNet LPIPS variant. Obtain
those weights under their own terms. Their complete hashes and weight terms
remain unverified. The evaluator version is torchmetrics 1.4.3. Changing it can
change SSIM; the final paper numbers use the saved scores from that environment.

Prepare and compose the dataset index as described in
`reproduce/compbench/README.md`, then export and verify the identity mapping:

```bash
python -m eval.compbench.prepare_results --composed composed --results-root editing_results --model my_model
python -m eval.compbench.run_eval my_model --check-inputs --index compbench_inputs.jsonl \
  --data-root data/CompBench/tasks --results-root editing_results --output-dir scores \
  --tasks local multi_object implicit
```

The exporter requires a new model directory, preserves every original
`eval_key` (including nested paths), verifies the composition hashes and copies
final PNG bytes without re-encoding. `RESULTS_MANIFEST.json` records the full
`row_id` to evaluator-path mapping. `--check-inputs` verifies the pinned evaluator
and selected source/target/mask/result files, then exits before loading metric
models. It does not measure a score. Run actual metric inference separately:

```bash
python -m eval.compbench.run_eval my_model \
  --clip-path weights/clip-vit-large-patch14 --data-root data/CompBench/tasks \
  --results-root editing_results --output-dir scores \
  --device cpu --tasks local multi_object implicit
```

The evaluator expects result images at:

```text
editing_results/my_model/my_model_local/{add,remove,replace}/<eval_key>
editing_results/my_model/my_model_multi/{multi_object_add,multi_object_remove}/<eval_key>
editing_results/my_model/my_model_implicit/<eval_key>
```

`scripts/datasets/prepare_compbench.py` provides the relative dataset index accepted by composition and the evaluation preflight. Composition preserves `row_id`, `subtask` and `eval_key` in its run manifest.
The recovered evaluator uses 512-square bicubic RGB preprocessing and nearest
mask resizing, CLIP on the mask bounding rectangle, and zero-masked background
metrics. It preserves the original skip/resume behavior; check detailed CSV
counts before using a mean. The expected six subtask populations are 982, 1331,
152, 72, 72 and 100 respectively. Do not interpret missing outputs as completed
evaluation. No fresh metric inference was run during release preparation.

The adapter checks the evaluator hash before import. `--evaluator` can select
the exact upstream file instead of the bundled portable copy. It replaces the
CLIP path in memory and uses local-only loading for the torchmetrics CLIP
adapter. Install/download metric dependencies explicitly before inference.

The final three-arm paper aggregation uses the detailed per-image CSVs:

```bash
python -m eval.compbench.aggregate --scores saved_scores.tar.gz --output ablation.json
```

The default archive model names match the saved paper artifacts; override them
with `--direct-model`, `--paste-model` and `--ours-model` for your own archives.
Each CSV path ends in `<model>/<task-tag>/<model>/<subtask>_detailed.csv`.
Task tags are add, remove, replace, multi_object and implicit. Members are read
in memory without extraction.

Non-PSNR fields use their own available rows, skipping NaN. PSNR uses image IDs
with finite values in all three arms, leaving 2,462 local items. Round each
subtask mean to three decimals using NumPy scalar rounding, then average
subtasks equally and round again. The final aggregation reproduces all 45 group/arm/metric values and counts of the saved paper table exactly. Only the descriptive `rules` text differs from the saved file.
An empty valid population or a nonfinite aggregate raises an error; it does
not emit an unmeasured number as JSON NaN.
