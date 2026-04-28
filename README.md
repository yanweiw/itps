---
title: ITPS Maze2D (browser-side, ACT)
emoji: 🌀
colorFrom: red
colorTo: yellow
sdk: static
pinned: false
license: mit
models:
  - felixw/itps-act
short_description: Real-time multimodal ACT predictions on YOUR device. No server compute.
---

# ITPS Maze2D — browser-side ACT

Real-time, multimodal motion predictions from the
[*Inference-Time Policy Steering through Human Interactions*](https://yanweiw.github.io/itps/)
paper, running entirely in your browser. Move your mouse over the maze and
watch 32 sampled trajectories follow your cursor.

## How to use

Open the page, wait a few seconds for `act.onnx` (~44 MB, FP16) to download,
then move the mouse over the maze. Trajectories that pass through walls are
tinted toward white. The first visit downloads the model; subsequent visits
load it from the browser cache instantly.

This Space hosts the unconditional ACT path of the original CLI:

```
python interact_maze2d.py -p act -u
```

The Diffusion-Policy variant and sketch-based guidance from the paper are
deferred to follow-on Spaces — see "What's NOT here yet" below.

## What's powering this

| Layer | Tech |
| --- | --- |
| ML runtime | [ONNX Runtime Web 1.20](https://onnxruntime.ai/docs/tutorials/web/) via jsDelivr CDN |
| Acceleration | WebGPU when available (any modern browser on Mac, Windows, recent Android, iPhone 13+); WASM fallback otherwise |
| Model | [ACT](https://huggingface.co/felixw/itps-act) — Action Chunking Transformer, 153 MB → 44 MB FP16 ONNX |
| UI | Vanilla `<canvas>` + `<script type="module">`. No build step, no framework. |
| Server compute | **Zero**. The Hugging Face Static Space serves a handful of files; inference happens on the visitor's device. |

The whole client side is three files: `index.html`, `app.js`, `style.css`,
plus the model weights in `act.onnx`. The `scripts/export_onnx.py` script
runs once locally to produce `act.onnx` from the PyTorch checkpoint.

## Performance

Per-frame inference latency on the visitor's device:

| Device | Backend | ms / frame | FPS |
| --- | --- | --- | --- |
| M-series Mac | WebGPU | ~3-10 | 30+ |
| Modern Windows / Linux laptop | WebGPU | ~5-15 | 30+ |
| iPhone 13+ / iPad Pro | WebGPU | ~10-30 | 20-30 |
| Older laptop, no WebGPU | WASM SIMD | ~50-100 | 8-15 |

The status pill at the top of the page shows the actual measured ms/frame
and FPS once the model is loaded.

## Run locally

You don't need Hugging Face Spaces — any static file server works. From the
repo root:

```bash
python scripts/serve.py            # preferred: COOP/COEP -> WASM threading
# or: python -m http.server 8000   # works, but WASM falls back to single-thread

# then open http://localhost:8000 in Chrome / Edge / Safari 18+
```

`scripts/serve.py` is a tiny wrapper around `http.server` that adds the
COOP/COEP headers needed to enable `SharedArrayBuffer` — without those,
ORT Web's WASM backend silently falls back to single-threaded SIMD, which
is ~4x slower. (If WebGPU works on your device, the headers don't matter;
they only affect the WASM fallback path.)

The first `act.onnx` request downloads from the local server; the model
runs in your browser via WebGPU / WASM exactly as it does on the deployed
Space.

## Re-exporting the model

If you retrain ACT or change the input/output shape, regenerate `act.onnx`
locally:

```bash
source .venv/bin/activate
pip install onnx onnxruntime onnxconverter-common onnxscript
python scripts/export_onnx.py
```

The script wraps the trained `ACTPolicy` so the ONNX graph takes
`(state, env_state, latent)` as positional inputs. The latent is supplied
from JS (a seedable PRNG matching `seeded_context(0)` from the original
CLI), which keeps the ONNX graph deterministic and avoids ORT's spotty
WebGPU support for `RandomNormal`.

## What's NOT here yet

| Feature | Where it would live |
| --- | --- |
| Unconditional DP (mouse-follow with diffusion) | A future variant: export the DP UNet to ONNX, add a ~30-line DDIM loop in JS. |
| Sketch input + post-hoc / biased-init strategies | Pure browser, no autograd needed. ~70 LOC additive. |
| Sketch input + guided-diffusion / stochastic-sampling | Either a hybrid call to a Gradio Space, or finite-difference gradient approximation in JS. |

This Space is intentionally minimal — the goal is to show that the realtime
unconditional case lives perfectly well as a static page on free hosting,
and to make adding the rest of the paper's strategies a sequence of small
incremental commits rather than a flag day.

## Source

Code is on the `browser-demo` branch of
[github.com/yanweiw/itps](https://github.com/yanweiw/itps). The Python CLI,
training code, and original `pygame` interface live on `main`.

## Acknowledgement

Pre-trained ACT weights and the original `interact_maze2d.py` are by
Yanwei Wang. ACT builds on [LeRobot](https://github.com/huggingface/lerobot).
