# Viewer

```bash
wmmllm export-viewer --run my-run --out my-viewer
python -m http.server 8000 --bind 127.0.0.1 --directory my-viewer
```

Open `http://127.0.0.1:8000`. The output directory must be new. The exported viewer opens `worlds/prepared/world.json` by default. `?world=prepared` selects it explicitly. Stop the local server when finished. The wheel contains all browser assets; export works without a checkout, credentials, model calls or JavaScript build step.

Exported offline-example sessions retain their procedural-fixture label automatically. `--procedural-fixture` explicitly labels other locally drawn fixtures. Distributed artwork is procedural prepared playback. The viewer is not a live generation interface.

Drag or use the keyboard to look around. Choose an available graph action to play its actual transition clip or visit its endpoint when no clip exists. Undo restores the prior browser state and camera. Saved-state controls return to visited browser states. The stored endpoint panel shows the saved endpoint image. Seam and uncertain-verification labels remain visible. Browser actions do not mutate the source session or memory.

The canvas renders the ERP preview with its viewport padding. That preview does not encode the complete numerical panorama state or native patch history. Eligible CPU clips are labeled rendered. The absent backend produces no generated clip. Local edits, global changes, generative navigation and unsupported revisit geometry show an explicit video-unavailable result. There is no crossfade substitute. A broken provided clip leaves the previous state visible and reports an error.

The exporter validates the closed session schema, graph, normalized relative references, encoded/pixel hashes and media profiles. It rejects symlinks, missing required files, traversal, remote references and existing output worlds. Image pixels are rewritten without private metadata. Actual H.264/MP4 clips are remuxed with metadata removal and decoded-frame comparison; this requires `ffmpeg`, `ffprobe` and Linux/POSIX descriptor support.

Only public graph summaries, redacted display labels, panorama previews, endpoints and clips are copied. Numerical state, native patches, memory, decisions, runtime configuration and receipts remain in the raw run. Keep raw runs outside the served directory. Inspect your own media before sharing it.

`tools/export_viewer.py` is a thin compatibility entry for exporting a world into an existing viewer: `python tools/export_viewer.py my-run --worlds-dir viewer/worlds --world-id prepared`. The `wmmllm export-viewer` command also copies the standalone browser shell and selects its exported world by default.
