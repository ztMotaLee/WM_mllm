<!-- SPDX-License-Identifier: Apache-2.0
     Copyright 2026 The WM^MLLM Authors -->
# Prepared session viewer

This static viewer plays exported session graphs. The bundled world is a visibly labeled **procedural fixture**. It uses locally generated test images and prepared CPU clips. The viewer performs no model calls and does not modify engine state or memory.

From the release root, serve the viewer locally:

```sh
cd viewer && python -m http.server 8000 --bind 127.0.0.1
```

Open `http://127.0.0.1:8000/` in a browser with WebGL2. Opening `index.html` through `file:` is unsupported because browsers restrict local module and manifest requests. No build, package installation, external font, CDN, analytics, or credential is needed.

Export a run from the release root:

```sh
python tools/export_viewer.py RUN_DIR --worlds-dir viewer/worlds --world-id my-world
```

Then open `http://127.0.0.1:8000/?world=my-world`. World IDs are single ASCII path segments, starting with a letter or digit and containing only letters, digits, `_`, `.`, or `-`. The default is `worlds/procedural/world.json`. The public manifest contract is [schema/world.v1.json](schema/world.v1.json).

The exporter requires Pillow. Sessions with video also require local `ffmpeg` and `ffprobe`, with H.264 support. The exporter verifies encoded and decoded image hashes, writes clean RGB PNGs, and removes video metadata without changing decoded frames. It rejects missing media, unsafe paths, symlinks in input/output paths, and an existing world folder. Choose a fresh world ID when exporting again. It reads only the session's explicit media references; memory, numerical panorama state, decisions, and receipts remain private.

With `pytest` and `jsonschema` available, run the offline checks from the release root with `python -m pytest -q tests/viewer`. Rebuild the fixture with `python examples/viewer_fixture/make_fixture.py --world-id procedural-copy`; the existing bundled world is preserved.

Drag or swipe the panorama to look around. While it is focused, arrow keys or W/A/S/D move the camera, `+`/`-` zoom, `R` resets the view, and `Z` undoes the last action. Scrolling and pinch gestures also zoom. Actions and visited states are ordinary keyboard-accessible buttons. Undo restores the exact preceding displayed state and camera, including zoom. A visited-state button restores that state's latest viewed camera. These controls stay in browser memory; reloading starts at `initial_node_id`.

Transitions retain their recorded presentation labels:

- **CPU panorama rendering** plays an actual prepared CPU clip.
- **Generated video** plays an actual supplied generated clip. The shipped engine has no generated-video backend.
- **No generated video** makes an immediate cut to the prepared endpoint state, with its recorded reason.

There are no crossfades. **Skip prepared clips** and **Skip to result** explicitly jump to the recorded result. Reduced-motion preferences initially select Skip prepared clips; uncheck it to play a clip. A missing or undecodable clip produces a visible error and retains the displayed state. Select Skip prepared clips and retry the action to inspect its prepared result. Failed and held actions retain their preceding state.

The canvas renders the exported equirectangular panorama preview and preserves its viewport padding. **Stored endpoint** displays the separately exported endpoint image. The preview is a presentation asset; it does not contain the engine's complete numerical panorama state or native patch history. Seam status and uncertain semantic checks remain visible.

The viewer validates the complete manifest, parent graph, action targets, presentation timing, and local media paths before loading media. Only normalized paths under the selected world's `media/` directory are accepted. URLs, traversal, encoded paths, backslashes, and unknown manifest fields are rejected. A local-only Content Security Policy restricts requests to this viewer's origin. Serve the viewer folder as the static root; keep private run files outside it.
