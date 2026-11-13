# Evaluation tools

These source-tree commands run from the release root. They are independent
of the live session engine and require no provider credentials. Analysis and
aggregation import no model weights. Model inference is an explicit separate
step using user-downloaded checkpoints.

- `mbench/`: the four MBench-A environment-recovery metrics, portable locks,
  model instructions, scoring and valid-only bootstrap aggregation.
- `camera/`: round-trip manifest preparation, DA3 pose extraction and CPU
  analysis of already saved poses.
- `compbench/`: the recovered evaluator adapter and final saved-score
  aggregation for the three installation rules.
- `../reproduce/compbench/`: historical RGB-difference composition on saved
  proposals, separate from the session engine's polygon compositor.

Create a dedicated Python 3.10+ environment. `eval/requirements-cpu.txt` covers analysis,
composition and tests. Add `eval/mbench/requirements-mbench.txt` for MBench/DA3
inference, or `eval/compbench/requirements.txt` for CompBench metric inference.
DA3 requires NumPy below 2. Core generation and evaluation environments are
separate. These direct requirement lists do not certify every transitive
dependency or a clean installation; installation and model inference were not
run during this CPU-only preparation stage.

```bash
python -m pytest -q tests/eval
```

Tests use synthetic arrays, poses, per-item scores and procedural images. They
need no network, benchmark payload or model checkpoint. Dataset tooling and
checksum manifests are in `scripts/datasets/`.
