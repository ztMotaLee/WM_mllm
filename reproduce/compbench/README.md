# Historical CompBench composition

This is the historical RGB-difference composer used with the paper's 2,709
saved GPT-Image 2 proposals. It computes mean RGB absolute differences in
float32, thresholds strictly above 25, reconstructs thin-safe connected
components, keeps components with at least 64 pixels, dilates by 7 and uses
Pillow Gaussian feathering with radius 3. Pillow performs the final RGB
composition. Pixels where the stored alpha equals zero remain source-exact.

The session engine has a separate support mechanism: post-generation MLLM
polygons, dilation 16 and inward feather 12. These implementations measure
different configurations. Keep their results separate.

Run from the release source root with NumPy, Pillow and SciPy installed:

```bash
python -m wmmllm_datasets.prepare_compbench --root data/CompBench --output compbench_inputs.jsonl --verify
python -m reproduce.compbench --inputs compbench_inputs.jsonl --input-root data/CompBench --proposal-root saved_proposals --out composed
python -m eval.compbench.prepare_results --composed composed --results-root editing_results --model my_model
```

The prepared JSONL can be passed directly. Supply a real saved image proposal at
`saved_proposals/<row_id>.png` for every row. With `--proposal-root`, `source`
resolves within `--input-root` and `direct` resolves within `--proposal-root`; a
missing `direct` field means `<row_id>.png`. To use other proposal filenames,
add an explicit relative `direct` field to each row.

A nonempty JSON array of objects with `row_id`, `source` and `direct` is also
accepted. Without `--proposal-root`, both paths resolve within `--input-root`,
as in `python -m reproduce.compbench --inputs inputs.json --input-root saved_inputs --out composed`. Optional fields are `subtask`,
`instruction`, `split`, and diagnostic image paths `protected`, `base_target`
and `declared_box`, or a `grounding` JSON file with `edit_regions_norm` boxes.
Target images, benchmark masks and metric files never select committed pixels.
Grounding and diagnostic masks only populate measurements. Absent diagnostics
are labeled as not measured. Source and proposal images are required.

The loader preserves the original RGB conversion and uniform Lanczos resize
of a proposal to the source dimensions. There is no model call. The output
directory must be new. It contains support, alpha, final RGB, per-row receipts
and code/input-manifest hashes. `RUN_MANIFEST.json` also preserves each
`row_id`, `subtask`, `eval_key`, final path and final-file hash. The export
command verifies those hashes and copies the unchanged PNG bytes into a new
`editing_results/my_model/` directory using the exact benchmark `eval_key`,
including nested paths and original filename extensions. No re-encoding occurs.
Evaluation exports require `subtask` and `eval_key` in every composition row. The CLI reads its written PNGs back and checks
the zero-alpha guarantee. The support-area ceiling 0.56 classifies global edits
for manual review; it does not prune pixels. A global edit can retain a real
nonempty exact region while being excluded from the in-scope headline.

The recorded production composer/selector hashes differ from the available
source files. No matching copies were found in a bounded historical search.
This port uses the available files. On 24 saved production items, four per
subtask across all six task types, support, alpha and final RGB match exactly.
This establishes sample equivalence; it does not establish whole-population
pixel equivalence for every historical proposal.

Dataset inputs and historical proposals are not distributed. See
`scripts/datasets/README.md` for dataset preparation and `eval/compbench/README.md`
for evaluation and the paper's final aggregation.
