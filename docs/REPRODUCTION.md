# Reproduction

Fresh generation is stochastic. Paper numbers are reproduced by scoring the original outputs, rather than by bit-identical regeneration. This release does not distribute those outputs, datasets or the video generation model/runtime. It implements the full engine used for the paper's multi-turn sessions and selected deterministic evaluation/composition tools. Matching a mechanism does not reproduce a historical campaign.

| Paper result or study | Corresponding release code | What must be supplied or remains excluded |
| --- | --- | --- |
| MBench environment recovery | `eval/mbench/`, four environment metrics and valid-only aggregation | Original input/output manifest, videos and verified weights; historical recovery generator is excluded |
| Camera round-trip completion and thresholds | `eval/camera/`, shared MBench pose producer | Original videos or saved pose shards; fresh DA3 inference requires separate setup |
| Blind camera motion/usability and joint judgments | Core panorama rendering is related | Study samples, independent judge pipeline and labels are excluded |
| Same-pose content preservation and panorama coverage | `state.panorama` and native-patch rendering | Dedicated content/coverage study scripts and judgments are excluded |
| CompBench installation comparison | `reproduce/compbench/`, `eval/compbench/` | Original 2,709 GPT-Image 2 proposals, benchmark references and saved scores |
| Blind composition and support-timing comparison | `agent.executor`, local support/composition | Study arms, 150-proposal and 112-edit populations, judgments and outputs are excluded |
| Revision lookup ablation | `state.graph` and core tests | Historical replay harness and diagnostic population are excluded |
| Observed-memory comparison | `memory` full implementation | Cache-only arm, paired-session scripts and independent judging are excluded |
| Multi-turn component study | Full `agent`/`memory`/`session` engine | Other router, video-tail and integrity-only arms, generated videos and judges are excluded |
| Exact revisits, outside-support invariants and endpoint diagnostics | Core state, lookup, composition and deterministic tests | Pooled historical outputs and judgments are excluded |
| WBench navigation and full diagnostic populations; causal diagnostics | Related state mechanisms only | These benchmarks, scorers and outputs are excluded; the MBench environment port does not score them |
| Runtime, costs and video latency | Provider receipts and user-priced budget accounting | Historical hardware, service prices and video runtime are excluded |
| Qualitative worlds | Static viewer and procedural artwork | Project-page worlds and paper media are excluded |

## Distinct generation regimes

The historical recovery profile used high chat effort and accelerated video presentation. Its 454 segments comprised 215 rotations, 186 restores, 10 generated-video segments and 43 holds. The panorama module agrees with the multi-turn session source in the inspected copies, but the router and campaign failure/source/export behavior differ. This package does not implement that historical generator profile.

The 817-segment diagnostic population pooled trajectories and reruns across source revisions and reasoning efforts: 606 segments recorded ultra, 154 high, 23 medium and one max; 33 bootstrap holds had no recorded chat effort. Twelve benchmark items included medium-effort segments. These diagnostics must not be described as uniformly using high effort. Its measured lookup path used direct visit-key lookup. The public core's explicit saved-node, origin, viewport-profile and revision checks cannot be attributed retroactively to that entire population.

The full session engine uses saved panorama/endpoint authority and observed semantic memory. All chat roles in the paper's multi-turn sessions used GPT-6.1-sol with high effort. Images used GPT-Image 2.5 via image editing. This release permits user-specified endpoints and models. Other choices change the configuration. Removing generated video preserves the presentation boundary but does not reproduce video quality or the historical request-to-playable latency.

The public source profile is identified by an implementation seal in `release_manifest.json`. Individual extracted source functions were checked during porting. The seal identifies the source selection; it does not assert that every file in the historical tree is distributed or that every historical result is reproduced.

## Historical CompBench composition

Historical proposals used GPT-Image 2. Composition derives support from mean RGB changes strictly above 25, reconstructs thin structures, retains components of at least 64 pixels, dilates by 7 and uses Gaussian feather radius 3. The session engine instead selects polygons after generation, dilates by 16 and feathers 12 pixels inward. Keep these configurations separate.

The available composer and selector have different hashes from the recorded production sources. A bounded source search did not recover the production-hash copies. The public port matched support, alpha and final pixels on 24 saved production items, four per each of six subtasks. It has not recomposed every historical proposal. Final three-arm aggregation matched every value and count in the saved table. Only the descriptive `rules` text in the output was reworded for this release, so the raw JSON bytes differ in that field. Source/evaluator identifiers remain in the manifest and evaluator locks.

## Evaluation identities and verification scope

The MBench port retains its pinned upstream metrics and scoring protocol. Port verification reproduced all 56 saved common/own-full numeric cells, including intervals and validity counts. Camera analysis matched all 1,568 saved rows. These checks concern saved results. Clean CPU installation and synthetic tests do not validate fresh model inference or reproduce the missing paper media.

The executable camera formula requires the largest absolute first-half yaw to have the requested sign and exceed one degree. Requested-direction excursion is measured over the whole clip and must reach 15 degrees. Final geodesic return error must be at most 10 degrees. The paper appendix's shorter first-half description differs; this release preserves the actual scored formula. See `eval/camera/README.md` for exact arithmetic and threshold sensitivity.

Mbench/DA3 and CompBench inference have separate requirement lists and weight locks. They are not complete verified transitive installation locks. No weights download implicitly. Dataset checksum manifests bind locally recorded subsets; 547 first frames are not the complete official MBench-A release. Obtain original outputs, confirm matching identities/populations, score them with the locked evaluator, and aggregate saved receipts using the documented common/own-full rules.

No portable generation cost or time guarantee is provided. Provider charges, hardware, model access and generation durations vary. The shipped budget table contains no prices. Historical timing and accounting figures belong to the technical report's runtime/cost appendix and are not new measurements of this package.
