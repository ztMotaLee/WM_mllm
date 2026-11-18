# Architecture

The public package implements the full session engine used for the paper's multi-turn sessions. Its module groups correspond to the paper's panorama state, state update, revision lookup, observed memory and presentation components. Historical result regimes are described in [evaluation](EVALUATION.md).

| Paper component | Public implementation | Role |
| --- | --- | --- |
| Persistent panorama state and rendering | `wmmllm.state.panorama`, `projection`, `registration` | Immutable render base, ordered native patches, camera projection, registration and exact protected pixels |
| State-update planning and checking | `wmmllm.agent.router`, `planning`, `contracts`, `executor` | Operation routing, endpoint proposals, source-bound checks and one regeneration after semantic FAIL |
| Initialization and navigation | `wmmllm.agent.bootstrap`, `navigation` | Outpainting, reachable-view seam checks, generative navigation with a combined destination check |
| Revision-aware state lookup | `wmmllm.state.graph` | Saved node, origin, profile and physical-revision validation |
| Observed semantic memory | `wmmllm.memory.facts`, `store`, `retrieval`, `observer` | Facts from saved endpoint images and eligible historical-image references |
| Presentation separated from authority | `wmmllm.video.cpu`, `presentation`; `providers.video` | CPU camera clips and explicit absent generated video |
| Local session execution | `wmmllm.session.runner`, `io`, `receipts` | Single-writer transactions, contained references, checkpoints and sanitized receipts |
| Model request boundary | `wmmllm.providers`, `config`, `budget`, `logging_utils` | Injected clients, transport, accounting and privacy |
| Method instructions | `wmmllm.prompts` | Packaged text and JSON schemas |

The current endpoint image and panorama are the authority. The renderer combines the immutable base and ordered native patches; the ERP preview alone is not the complete numerical state. Local support is selected from source and actual generated view after generation. Polygon support is dilated by 16 pixels and blended 12 pixels inward. Reprojection preserves every pixel outside its alpha. Whole-panorama replacement retires native patches.

A turn reads current state and memory, routes the action, creates/checks an endpoint, persists its assets, completes presentation, observes the saved endpoint, and commits memory and graph state. PASS commits. UNCERTAIN commits with an explicit label and false semantic-success flag. Generated FAIL permits one regeneration. A second FAIL cannot enter ordinary presentation or memory commit. Pure geometry and exact returns get a single semantic check. Classified failure holds retain the preceding state and add no requested-action facts.

A pending packet is persisted only after successful checking, presentation and observation. Recovery validates its full bindings and completes the same memory commit and graph publication exactly once. It does not issue replacement model or video requests. Budget recovery occurs before provider use. Sessions serialize normalized run-relative references with encoded, pixel, array and canonical state hashes. POSIX locks serialize writers.

Initialization has a separate immutable checkpoint. `wmmllm resume --run RUN` retries failed or held initialization from that checkpoint with the same configuration. Repeated `init` must match the bound image pixels, captions, seed, panorama width and runtime settings. Attempts use separate directories. A `BOOTSTRAP_HOLD` preserves the original image and records an incomplete, uninitialized world; it creates no panorama, graph nodes, memory facts or generated video. Transport and fatal provider failures leave a resumable failed session and raise a sanitized error.

Rotations and same-origin revisits use real CPU panorama rendering. Other transitions go to the injected `VideoBackend`; the shipped `none` backend emits no generated video. Presentation never updates panorama state, endpoint images or semantic memory.

Evaluation lives in the source checkout's `eval` directory and has separate inference dependencies. Core imports do not load evaluation models, weights or credentials. Prompt resources work from an installed wheel. Offline providers are confined to tests and the explicitly named offline example.
