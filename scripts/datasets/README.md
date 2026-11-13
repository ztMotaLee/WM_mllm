# Benchmark data preparation

Obtain the benchmark data separately. This release contains checksums and preparation code; it contains no benchmark images, videos, instructions, model weights, or generated proposals. These scripts use only the Python standard library and never download data.

## Download status and terms

| Benchmark | Source recorded locally | Status |
|---|---|---|
| MBench code | `https://github.com/study-overflow/MBench` | **verify upstream URL**; availability was not checked online. The inspected code license is MIT. |
| MBench data | `https://huggingface.co/datasets/studyOverflow/MBench-Data` | **verify upstream URL**. Local acquisition records identify revision `412581ebc498a752567e47bc1fb2a434692c47ad`. Dataset and baseline-video terms are **unverified**. |
| CompBench code | `https://github.com/BhJia/CompBench` | **verify upstream URL**; availability was not checked online. The inspected code license is MIT. |
| CompBench data | `https://huggingface.co/datasets/BohanJia/CompBench` | **verify upstream URL**. The dataset revision and dataset terms are **unverified**. |

Download from the benchmark maintainers after checking their current instructions and terms. Preserve the paths below. The inspected CompBench README names `download_from_hf.py` as its dataset downloader; verify that recipe and its dependencies in the upstream checkout before use. Do not treat either code repository's MIT license as a license for its dataset, baseline videos, or model weights. No dataset-specific license file was found in the inspected local dataset copies. No redistribution permission is claimed.

## MBench-A

Place first frames under a user-selected root, for example:

```text
data/MBench/
  MBench-A/samples/environment/<sample_id>/first_frame.png
  MBench-A/samples/object/<sample_id>/first_frame.png
  MBench-A/samples/human/<sample_id>/first_frame.png
  MBench-A/samples/causal/<sample_id>/first_frame.png
```

The recovery-input manifest covers **227 environment first frames**. The broader manifest covers **547 first frames**: 227 environment, 100 object, 120 human, and 100 causal. This first-frame collection is **not the complete official MBench-A release**. The paper's common comparison contains 224 item–condition pairs; selecting that shared population requires the evaluation manifests, sample metadata, conditions and per-method availability.

```bash
python scripts/datasets/verify_dataset.py --dataset mbench-a-environment-227 --root data/MBench
python scripts/datasets/verify_dataset.py --dataset mbench-a-first-frames-547 --root data/MBench
```

These checksums cover first frames only. They do not cover captions, sample metadata, conditions, baseline videos, method outputs, poses, or metric weights. Obtain those separately for scoring and use the evaluator's required input manifest. MBench-T is outside these checksum manifests.

## CompBench

The paper population contains **2,709 items**. The metadata defines these subtasks:

| Subtask | Items | Path under `data/CompBench/` |
|---|---:|---|
| `add` | 982 | `tasks/add/` |
| `remove` | 1,331 | `tasks/remove/` |
| `replace` | 152 | `tasks/replace/` |
| `multi_object_add` | 72 | `tasks/multi_turn_editing/multi_object_add/` |
| `multi_object_remove` | 72 | `tasks/multi_turn_editing/multi_object_remove/` |
| `implicit` | 100 | `tasks/implicit_reasoning/` |

For `add`, `remove` and `replace`, preserve `metadata.json` and the matching `input_image/`, `edited_image/`, and `mask/` trees. Each metadata `image_path` is relative to those trees. For the two multi-object subtasks, preserve `multi_turn_editing/<subtask>.jsonl` and the matching `ori_images/`, `edited_images/`, and `masks/` trees. For implicit reasoning, preserve `implicit_info.jsonl`, `implicit_data.jsonl`, and the same three image-directory names. JSONL `image_name` values identify their image files.

The CompBench checksum manifest covers **8,134 files**: source, target and mask for each item, plus seven metadata files. It excludes the action/location/view tasks, sequential turn tasks, proposals and method outputs.

```bash
python scripts/datasets/verify_dataset.py --dataset compbench-paper-2709 --root data/CompBench
python scripts/datasets/prepare_compbench.py --root data/CompBench --output compbench_inputs.jsonl --verify
```

Preparation verifies the locked metadata, checks the population and every referenced file, and writes a new index outside the dataset root. `--verify` also hashes every image. Without that option it hashes the seven metadata files and checks image presence. It fails if the output already exists. The scripts leave the dataset untouched and reject child symlinks.

The generated index contains `item_id`, `row_id`, `subtask`, `eval_key`, `source`, `target`, `mask` and `instruction`. All file references are relative to the explicit CompBench root. `row_id` is the first 24 hexadecimal characters of SHA-256 of `subtask + '/' + eval_key`; it is a portable identifier, independent of historical proposal filenames.

To use the historical compositor, supply one real proposal at `saved_proposals/<row_id>.png` per indexed row, then run:

```bash
python -m reproduce.compbench --inputs compbench_inputs.jsonl --input-root data/CompBench --proposal-root saved_proposals --out composed
python -m eval.compbench.prepare_results --composed composed --results-root editing_results --model my_model
python -m eval.compbench.run_eval my_model --check-inputs --index compbench_inputs.jsonl --data-root data/CompBench/tasks --results-root editing_results --output-dir scores --tasks local multi_object implicit
```

The prepared JSONL needs no format conversion. An explicit relative `direct` field can override each proposal filename; it resolves within `--proposal-root`. Without that option, explicit source/proposal paths share `--input-root`. Composition records the original `subtask` and `eval_key`. Export preserves that mapping in the evaluator's required task directories, using a new model directory. The preflight verifies selected paths without metric inference; see `eval/compbench/README.md` for the scoring command. **Only the source and proposal are composition inputs. Targets and benchmark masks are evaluation references.** The prepared dataset index alone cannot reproduce the historical proposals or the paper's scores. The CompBench evaluator uses its own upstream dataset layout; retain the original task directories for that adapter.

## Open issues

- **verify upstream URL** for both benchmarks before advertising a working download recipe; CompBench's exact dataset revision remains unresolved.
- Verify dataset and baseline-video licenses and terms separately from code and weight terms.
- Acquire and pin MBench sample metadata, baseline videos, conditions and comparison populations. First-frame hashes alone do not establish exact score reproduction.
- Historical image proposals and method outputs are not distributed. Their acquisition and attribution remain separate from benchmark preparation.

`DATASET_MANIFESTS.json` records the checksum-file identities. The checksums identify the inspected local input bytes and do not claim that an upstream download is complete or still available.
