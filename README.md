# WM^MLLM: Turning MLLMs into World Models

[Project page](https://ztmotalee.github.io/wmmllm/) · [Code](https://github.com/ztMotaLee/WM_mllm)

WM^MLLM uses multimodal language models to plan and check changes to a persistent panorama world. Saved panorama state and endpoint images determine subsequent turns. Video presents a transition and never updates world state or memory.

This release contains the full session engine used for the paper's multi-turn sessions, API providers, CPU panorama geometry, observed semantic memory, revision-aware saved views, evaluation tools, historical CompBench composition, and a static viewer. Procedural examples use explicitly labeled deterministic providers. They test software behavior and do not measure model quality.

The video generation model and its runtime are not included. With `video.backend: none`, eligible rotations and revisits are rendered on CPU. Other transitions have no generated video. Model weights, datasets, paper outputs, experimental ablation arms, and historical campaign drivers are not included.

## Install

Use Linux or a compatible POSIX system with Python 3.10 or later. Clean installation is checked on Python 3.12. CPU clips require `ffmpeg` and `ffprobe` on PATH. The static viewer requires a WebGL2 browser. GPU and model packages are unnecessary for core installation.

```bash
git clone https://github.com/ztMotaLee/WM_mllm.git
cd WM_mllm
python -m venv .venv
. .venv/bin/activate
python -m pip install .
```

Install `'.[dev]'` to run tests. Evaluation inference uses a separate environment; see [evaluation](docs/EVALUATION.md). The wheel includes prompt resources, evaluator code, dataset checksum tools and viewer assets. The source archive also includes examples, documentation and tests.

## Use your own API

```bash
cp .env.example .env
# Fill the WMM_* values in your own .env file.
wmmllm check-config --env-file .env
```

Keys come only from the process environment or `--env-file`. The program never searches for an env file. Do not put keys in YAML or command-line arguments. Model names are required and have no built-in default. Set both `WMM_LLM_MODEL` and `WMM_IMAGE_MODEL` for distinct chat and image models. An empty image field inherits the corresponding chat field.

The paper's multi-turn sessions and MBench-A recovery runs used GPT-6.1-sol with reasoning effort **high** for all chat roles and GPT-Image 2.5 through the image edit endpoint. Some diagnostic segments used other reasoning efforts; see [reproduction notes](docs/REPRODUCTION.md). The historical CompBench proposals used GPT-Image 2. Other models may behave differently. Your endpoint must support the selected models, ordered multimodal inputs, strict JSON schemas, reasoning controls and multi-image editing. Endpoint compatibility and live model quality require separate verification.

`chat_completions` is the paper transport. `responses_stream` is an option for endpoints that cut long non-streamed requests. See every key, precedence rule, spend cap and privacy behavior in [configuration](docs/CONFIGURATION.md).

Without configuration, `wmmllm check-config` exits with code 2 before constructing a client:

```text
Missing WMM_LLM_API_KEY for chat provider; set the process environment or an explicitly selected env file; no request was sent.
```

## Quickstart

First run the offline procedural example from this source checkout or source archive:

```bash
python -B examples/offline_session/run.py --run offline-run
wmmllm export-viewer --run offline-run --out offline-viewer
python -m http.server 8000 --bind 127.0.0.1 --directory offline-viewer
```

Open `http://127.0.0.1:8000`. The viewer identifies the exported world as a procedural fixture. Stop the server when finished. Use new output directories. The example performs initialization, rotation, a drawn local edit and a revisit with real engine geometry, memory and transactions. It makes no model calls. Its providers do not implement navigation; separate core tests exercise navigation with queued test responses.

For a real session, supply your own initial image and configured API:

```bash
wmmllm init --image room.png --run my-run --title "My room" --env-file .env
wmmllm step --run my-run --env-file .env 'Look right by 45 degrees.'
wmmllm step --run my-run --env-file .env 'Put a blue bowl on the table.'
wmmllm export-viewer --run my-run --out my-viewer
```

Each real command may incur provider charges. Supply an explicit env file again on later commands. Configure prices and a spend cap before running if needed. `wmmllm session --run my-run --script turns.json --env-file .env` resumes a session and executes an array of instruction strings or an object with a `turns` array. Each array entry may also be an object with `instruction` and optional `metadata`. `wmmllm view --run my-run --yaw 0 --pitch 0 --out view.png` renders current state without credentials.

If initialization fails, retry its saved checkpoint with `wmmllm resume --run my-run --env-file .env`. Use the same configuration. A labeled `BOOTSTRAP_HOLD` keeps the original image and leaves the world uninitialized, with no panorama or memory facts; `resume` can retry it. Repeating `init` also requires the same image pixels, captions, seed, panorama width, source profile and runtime settings. Changed initialization inputs are rejected. Each retry keeps prior attempt files intact.

## Viewer and evaluation

The [viewer guide](docs/VIEWER.md) covers export, local playback, unavailable clips and procedural artwork. Export copies only allowed graph summaries and verified media. Keep raw runs outside the directory served by your web server.

The [evaluation guide](docs/EVALUATION.md) covers MBench environment scoring, saved-pose camera analysis and CompBench. Obtain datasets, saved outputs and weights separately. CPU tests require no downloads:

```bash
python -m pytest -q tests/providers tests/core tests/eval tests/viewer tests/integration
```

## Reproduction and limitations

[Reproduction notes](docs/REPRODUCTION.md) map each paper result to included code and missing components. Historical recovery, pooled diagnostics, multi-turn sessions and CompBench used distinct configurations. Fresh generation is stochastic. Paper numbers are reproduced by scoring the original outputs, rather than by bit-identical regeneration. Original outputs and video generation are not distributed here.

The current session engine uses one full session configuration. Historical recovery profiles, study arms and independent judge pipelines are absent. Supplied depth must be bound to the current state; no depth estimator is included. Unsupported revisit geometry has no clip. External API access, model availability and new model behavior are user-specific. The dependency ranges are not a full transitive reproduction lock. See [architecture](docs/ARCHITECTURE.md) for state and transaction behavior.

## License

Original code and procedural assets use [Apache-2.0](LICENSE). MIT-derived evaluator files retain their licenses. See [NOTICE](NOTICE) and [third-party notices](THIRD_PARTY_NOTICES.md). Model weights, datasets and user outputs are governed by their own terms.

## Citation

Technical report, 2026:

```bibtex
@techreport{li2026wmmllm,
  title = {{WM^MLLM}: Turning MLLMs into World Models},
  author = {Li, Baiang and Xie, Enze and Yu, Jincheng and Wang, Jin and Liu, Haozhe and Li, Yitong and Tian, Juanxi and Heide, Felix},
  year = {2026},
  type = {Technical report},
  url = {https://ztmotalee.github.io/wmmllm/}
}
```
