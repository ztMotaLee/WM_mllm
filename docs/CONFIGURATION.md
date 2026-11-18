# Configuration

`load_config(config_path=None, env_file=None, overrides=None, environ=None)` resolves configuration without constructing providers. Importing the package loads no credentials. The CLI accepts `--config FILE` and `--env-file FILE` on `check-config`, `init`, `resume`, `step`, and `session`. The `view` command needs neither.

## Resolution

The order and env-file rules are in the `load_config` docstring: defaults, YAML, `WMM_*` endpoint variables (a process value, even an empty one, wins over the selected env file), then nested nonsecret `overrides`. YAML permits `env_file` plus the sections below; each endpoint permits `base_url`, `model`, `organization` and `project`, never a key. Unknown fields and inline secrets fail locally. `--env-file` wins over an overrides `env_file`, which wins over the YAML `env_file`. An env file accepts `NAME=value`, optional `export `, comments and one pair of enclosing quotes, without shell expansion. An empty IMAGE field inherits its LLM counterpart, the chat base URL and the models are required and have no default; choose a separate image model when the endpoints differ. No other file or `OPENAI_*` variable is read.

| Variable | Meaning |
| --- | --- |
| `WMM_LLM_API_KEY` | Required chat credential |
| `WMM_LLM_BASE_URL` | Base URL of your own chat-compatible endpoint; required, no default |
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
| `chat.base_url` | Required; no default endpoint is provided |
| `chat.model` | No default; explicit supported string |
| `chat.organization`, `chat.project` | null; optional strings |
| `chat.transport` | `chat_completions`; the only supported value |
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

Live core sessions require the 1344x768 viewport and normal image output. Geometry functions and isolated fixtures can use other sizes. Providers are the supplied OpenAI-compatible chat/image implementations without custom headers. Image operations always use the edit endpoint. SDK retries are always disabled. The one transport loop uses exponential delay with its base capped before jitter; Retry-After may increase the delay. Schema repair, observer technical retries and semantic regeneration are separate algorithmic counters.

## Spend cap and user prices

No provider prices ship. `budget.prices` maps `chat` and `image` to entries with required positive `upper_bound_usd` and optional nonnegative token rates: `input_usd_per_million`, `cached_input_usd_per_million` (chat), `output_usd_per_million`, `text_input_usd_per_million`, `image_input_usd_per_million`, `image_output_usd_per_million`. `budget.price_table_file` holds the same role map and cannot be combined with nonempty inline prices. Enter prices and conservative per-attempt bounds from your own provider before enabling a cap.

When the chat cached rate and cached token count are both known, cost is `((input_tokens - cached_input_tokens) * input_rate + cached_input_tokens * cached_rate + output_tokens * output_rate) / 1_000_000`. Cached counts must be nonnegative integers no greater than input counts. Invalid counts retain the per-attempt bound. Without a cached rate, all input is charged at the full input rate. With a cached rate but a missing cached count, the per-attempt bound is retained. Missing counts never imply a cache discount.

A cap requires both roles and both bounds. Each dispatch, including retries, reserves its bound first; missing usage and possibly billed errors keep the full bound, unpriced cost stays unknown, and a cost above a bound stops further dispatch (see the `budget` module docstring). `budget/checkpoint.json` holds the exact accounting, so restarting a session cannot reset a cap. CLI resume keeps the saved nonsecret settings and prices unless a new config is supplied; credentials and models come from the environment again.
