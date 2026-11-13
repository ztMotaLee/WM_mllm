# Configuration

`load_config(config_path=None, env_file=None, overrides=None, environ=None)` resolves configuration without constructing providers. Importing the package loads no credentials. The CLI accepts `--config FILE` and `--env-file FILE` on `check-config`, `init`, `resume`, `step`, and `session`. The `view` and `export-viewer` commands need neither.

## Resolution

Nonsecret settings resolve as package defaults, explicit YAML, documented WMM endpoint environment values, then explicit nonsecret `load_config(overrides=...)` values. The CLI exposes config/env-file selection rather than arbitrary key/value overrides. Unknown fields and inline secrets fail locally. YAML permits `env_file` plus the sections below. Each endpoint permits `base_url`, `model`, `organization`, and `project` in YAML. Keys are never allowed there.

For each WMM variable, a present process value wins over the explicitly selected env file, including an empty process value. Only recognized WMM names are read. An env file permits `NAME=value`, optional `export `, blank lines, comments and a single pair of enclosing quotes. Values are not executed or shell-expanded. After resolving each field, an empty IMAGE field independently inherits its LLM counterpart. An empty final chat base URL becomes `https://api.openai.com/v1`. Models have no built-in default. An omitted image model inherits the explicitly chosen chat model; choose a separate image model when the endpoints differ. A rejected inherited model fails explicitly.

Env-file selection is direct `env_file`/`--env-file`, then overrides `env_file`, then YAML `env_file`. Relative YAML file references use the YAML directory. A direct env-file argument uses the current directory. No parent directory or credential-store search occurs. Unrelated `OPENAI_*` variables do not configure the SDK. API keys, URLs, organization and project settings are passed explicitly.

| Variable | Meaning |
| --- | --- |
| `WMM_LLM_API_KEY` | Required chat credential |
| `WMM_LLM_BASE_URL` | Chat-compatible API base URL |
| `WMM_LLM_MODEL` | Required model name supported by that endpoint |
| `WMM_LLM_ORGANIZATION` | Optional organization |
| `WMM_LLM_PROJECT` | Optional project |
| `WMM_IMAGE_API_KEY` | Image credential; empty inherits chat |
| `WMM_IMAGE_BASE_URL` | Image base URL; empty inherits chat |
| `WMM_IMAGE_MODEL` | Image model; empty inherits chat |
| `WMM_IMAGE_ORGANIZATION` | Image organization; empty inherits chat |
| `WMM_IMAGE_PROJECT` | Image project; empty inherits chat |

There are no WMM variables for retry, video or budget fields. Configure those through YAML or the Python API. `env_file` is optional and has no default. `.env.example` is a template for a user-authored file.

## All settings

| Key | Default and allowed values |
| --- | --- |
| `chat.base_url` | Standard API base URL when empty |
| `chat.model` | No default; explicit supported string |
| `chat.organization`, `chat.project` | null; optional strings |
| `chat.transport` | `chat_completions`; or `responses_stream` |
| `chat.reasoning_effort` | `high`; `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` |
| `chat.max_output_tokens` | 32000; positive integer |
| `chat.timeout_seconds` | 600; positive finite number |
| `image.base_url`, `image.model`, `image.organization`, `image.project` | Each inherits corresponding chat setting when empty |
| `image.quality` | `high`; `low`, `medium`, `high`, `auto` |
| `image.view_request_size` | `1792x1024`; positive WIDTHxHEIGHT string |
| `image.panorama_request_size` | `2048x1024`; positive WIDTHxHEIGHT string |
| `image.view_output_size` | [1344, 768]; pair of positive integers |
| `image.panorama_output_size` | [2048, 1024]; pair of positive integers |
| `image.timeout_seconds` | 420; positive finite number |
| `transport.max_attempts` | 5; positive integer, including initial dispatch |
| `transport.base_delay_seconds` | 20; positive number |
| `transport.max_delay_seconds` | 160; positive base cap |
| `transport.jitter` | 0.2; number in [0, 1] |
| `budget.max_cost_usd` | null; or a nonnegative finite cap |
| `budget.prices` | {}; role map described below |
| `budget.price_table_file` | null; explicit YAML file holding the role map |
| `video.backend` | `none`; no other backend ships |
| `video.num_frames` | 124; integer at least 2 |
| `video.fps` | 24; positive finite number |
| `video.size` | [1344, 768]; positive width/height |

Live core sessions require the 1344x768 viewport and normal image output. Geometry functions and isolated fixtures can use other sizes. Providers are fixed to the supplied OpenAI-compatible chat/image implementations; `provider`, `operation`, `weights_dir`, `adapter_path`, `sdk_max_retries` and arbitrary headers are not config keys. Image operations always use the edit endpoint. SDK retries are always disabled. The one transport loop uses exponential delay with its base capped before jitter; Retry-After may increase the delay. Schema repair, observer technical retries and semantic regeneration are separate algorithmic counters.

`chat_completions` preserves the paper request contract. `responses_stream` maps the same model, effort, strict schema, image order and output cap to a foreground Responses stream with storage disabled. It requires terminal completion and complete usable output. Partial, refused and length-truncated answers fail. Streaming can help endpoints that terminate long non-streamed requests; long-duration reliability for an arbitrary endpoint is unverified. Unsupported API fields cause explicit failure without model or effort substitution.

## Spend cap and user prices

No provider prices ship. `budget.prices` maps `chat` and `image` to entries with required positive `upper_bound_usd` and optional nonnegative token rates: `input_usd_per_million`, `cached_input_usd_per_million` (chat), `output_usd_per_million`, `text_input_usd_per_million`, `image_input_usd_per_million`, `image_output_usd_per_million`. `budget.price_table_file` holds the same role map and cannot be combined with nonempty inline prices. Enter prices and conservative per-attempt bounds from your own provider before enabling a cap.

When the chat cached rate and cached token count are both known, cost is `((input_tokens - cached_input_tokens) * input_rate + cached_input_tokens * cached_rate + output_tokens * output_rate) / 1_000_000`. Cached counts must be nonnegative integers no greater than input counts. Invalid counts retain the per-attempt bound. Without a cached rate, all input is charged at the full input rate. With a cached rate but a missing cached count, the per-attempt bound is retained. Missing counts never imply a cache discount.

A cap requires both roles and both bounds at startup. Each physical dispatch, including retries, reserves its bound before sending. Complete supported measured usage settles at configured token prices. Missing usage, interrupted outcomes and possibly billed errors retain the full bound. Unpriced cost is unknown. It is never reported as zero. Actual cost above a bound is recorded in full and stops further dispatch. The bound is a user assumption, not a provider invoice guarantee.

`budget/checkpoint.json` persists complete exact accounting before dispatch and after settlement. Resume restores historical charges and settles unresolved reservations conservatively. Unknown historical costs prevent enabling a cap. A capped run cannot resume uncapped. CLI resume retains saved nonsecret settings and prices unless a new config is explicitly supplied. Changing credentials or models requires their environment settings again. A cap cannot be reset by restarting a session process.

## Logging and local privacy

Config summaries show presence flags and nonsecret settings. Receipts keep sanitized requested/returned models or explicit unknown attribution, usage or unknown, attempt status, HTTP status, timing and hashes. They omit prompts, raw response objects, credentials, URLs, image data URLs and raw image payload sidecars. Streaming receipts record first-byte and maximum-silence timing. Exceptions retain a class and numeric HTTP status.

Local run state includes user instructions, decisions, support polygons, checked observations and memory. These files may contain user scene content. The viewer exporter uses a closed field and asset allowlist. It copies graph summaries, redacted labels, endpoint pixels, panorama previews and verified clips. It excludes runtime config, state manifests, memory, decisions and receipts. It rejects symlinks and escaping references. Export is not automatic review of sensitive visual content; inspect your own media before sharing it.

Image ingestion decodes pixels and applies a non-sRGB embedded ICC profile once to produce sRGB. Standard sRGB transform data is recognized independently of profile descriptions and leaves decoded RGB pixels unchanged. EXIF orientation is not applied, preserving pixel coordinates. Invalid or incompatible ICC profiles fail locally. Uploads and saved run images are reconstructed from pixels and encoded without ICC, EXIF, comments or other ancillary metadata; every RGB PNG encoding verifies decoded pixel equality. Run input records and provider receipts retain a safe conversion status and pixel hash, never embedded profile names or EXIF values. A clean saved image has no profile to convert again. Grayscale and alpha coverage masks retain their channel layout and exact coverage values.

Provider log filtering runs before handler dispatch. It covers SDK child loggers and handlers added after provider construction. Payload diagnostics are suppressed; warning details, exception bodies and credential-bearing fields are omitted or redacted. Terminal machine codes such as `insufficient_quota` stop transport and observer retries. HTTP 401, 403 and 404 and budget failures also stop optional processing; other optional provider failures keep their labelled local result.
