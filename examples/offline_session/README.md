This is an **offline procedural fixture, not a model**. The explicit demo providers draw deterministic pixels. They are confined to this example and are unavailable from the live CLI. The example runs the real bootstrap, operation parser, projection, support selection, compositor, verifier parser, observation, memory and session commit code. No credentials or network access are needed.

From the installed source checkout:

```bash
python -B examples/offline_session/run.py --run ./offline-run
```

Use a new output directory. The seed is generated locally; no photographs, benchmark inputs or generated-model media are bundled. Three frames per CPU transition keep this fixture small; normal configuration uses 124 frames at 24 fps. Without ffmpeg, real frame sequences are saved and MP4 availability is reported explicitly.
