<!-- SPDX-License-Identifier: Apache-2.0 -->

# Procedural playback fixture

This example authors a small playback session using local NumPy geometry,
Pillow PNG encoding, and CPU FFmpeg. It uses no model, API key, network,
benchmark image, or generated project media. The viewer labels it
**Procedural playback fixture**.

From the project root, with the package dependencies and FFmpeg available:

```bash
python -B examples/viewer_fixture/make_fixture.py --world-id procedural-new
```

The command creates an isolated temporary run, exports the public assets to
`viewer/worlds/procedural-new/`, then removes the run. The checkout already includes `viewer/worlds/procedural/`. It refuses to replace an
existing world, including a previously generated `procedural-new` world. Select a fresh identifier when generating another copy:

```bash
python -B examples/viewer_fixture/make_fixture.py \
  --worlds-dir viewer/worlds --world-id procedural-two
```

`--work-dir DIRECTORY` selects the parent for temporary files. The generator
removes its own temporary directory on success and failure. FFmpeg must provide
the `libx264` encoder. Missing FFmpeg or an encoding failure produces a local
error and no substitute clip.

The five-node graph has two places and three panorama states. Look right in
the room, change its blue vase to coral, look back toward the window, visit the
courtyard, then restore the exact saved room node with the coral vase. The
revisit reuses that node's saved endpoint, panorama, revision, and authored
memory snapshot. The two rotation clips contain 24 perspective renders each,
including both endpoint poses, at 12 fps. They are H.264 baseline / YUV420p
MP4s at 480 × 270 pixels. The panoramas are 1024 × 512 pixels.

Local editing, navigation, and the cross-place revisit have explicit
`presentation.status: "none"` records. The release supplies no generated
transition video for them. Their saved endpoint images remain viewable.

This is prepared playback. It does not exercise the agent or demonstrate a
model's performance. Every semantic result is labeled `UNCERTAIN`, with zero
verification attempts and no verifier output. State manifests and memory are
small authored fixture records. They are internal to the temporary run and
are excluded from the public viewer export. All images and videos originate
from the included procedural source, under Apache-2.0.
