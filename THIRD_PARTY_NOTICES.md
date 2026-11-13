# Third-party notices

Original WM^MLLM code and procedural artwork are Apache-2.0. The following derived evaluator files retain MIT terms. The top-level license does not replace their licenses.

## MBench

Upstream: https://github.com/study-overflow/MBench
Commit: `c44dbd9dfa0278be728bda2101b0fb96f8547f11`.
Copyright (c) 2026 MBench Contributors. Source headers also name MBench community.
The port retains the environment metric closure, excludes unrelated metrics, and uses portable package imports and explicit data inputs. Every bundled file in this group is listed here:

- `eval/mbench/_upstream/__init__.py`
- `eval/mbench/_upstream/aggregators/__init__.py`
- `eval/mbench/_upstream/aggregators/mean.py`
- `eval/mbench/_upstream/core/__init__.py`
- `eval/mbench/_upstream/core/artifact.py`
- `eval/mbench/_upstream/core/contract.py`
- `eval/mbench/_upstream/core/exceptions.py`
- `eval/mbench/_upstream/core/item.py`
- `eval/mbench/_upstream/core/pipeline.py`
- `eval/mbench/_upstream/core/registry.py`
- `eval/mbench/_upstream/core/result.py`
- `eval/mbench/_upstream/metrics/__init__.py`
- `eval/mbench/_upstream/metrics/canonical.py`
- `eval/mbench/_upstream/metrics/environment_calculations.py`
- `eval/mbench/_upstream/metrics/environment_consistency.py`
- `eval/mbench/_upstream/metrics/metric_math.py`
- `eval/mbench/_upstream/metrics/video_io.py`

License: `eval/mbench/LICENSE.upstream` (MIT), reproduced below.

```text
MIT License

Copyright (c) 2026 MBench Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## CompBench

Upstream: https://github.com/BhJia/CompBench
Commit: `cb9165b01d14a321db14611afd9922c43a04665d`.
Derived file: `eval/compbench/upstream/eval_all.py`. Portable path defaults and a modification notice replace author-specific paths. Metric and population expressions remain unchanged.
License: `eval/compbench/LICENSE.upstream` (MIT), reproduced below.

```text
MIT License

Copyright (c) 2026 comp-bench

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Dependencies and excluded assets

NumPy, Pillow, OpenCV, SciPy, OpenAI Python, HTTPX, PyYAML and optional evaluation packages are installed separately and retain their own package licenses. No copies of their implementation are vendored. The top-level Apache license text is the unmodified standard text from a locally installed Apache-2.0 package.

DA3 code, CLIP, LPIPS, DINOv2 and other model implementations/weights are not distributed. Datasets, benchmark images, baseline videos and original model outputs are not distributed. Model weights and datasets are governed by their own terms. A code license does not grant permission to redistribute those assets. No video generation code, runtime, adapter, model weight or generated video from that model is included.
