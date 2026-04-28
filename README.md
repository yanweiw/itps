---
title: ITPS Maze2D (browser-side, ACT + DP)
emoji: 🌀
colorFrom: red
colorTo: yellow
sdk: static
pinned: false
license: mit
models:
  - felixw/itps-act
  - felixw/itps-dp
short_description: Real-time ACT + Diffusion Policy predictions on YOUR device. No server compute.
---

# ITPS Maze2D — browser-side ACT &amp; DP

Real-time motion-policy predictions from the
[*Inference-Time Policy Steering through Human Interactions*](https://yanweiw.github.io/itps/)
paper, running entirely in your browser. Pick an engine (Diffusion Policy or
Action Chunking Transformer), move your mouse over the maze, and watch the
sampled trajectories follow your cursor.

## How to use

The page loads with **DP** (Diffusion Policy) selected by default. The first
DP visit downloads `dp_unet.onnx` (~31 MB FP16); switching to ACT then
downloads `act.onnx` (~44 MB) on first use. Both are cached after the first
fetch, so subsequent visits load instantly.

Three controls live beneath the maze:

- **Engine** — DP vs ACT. Switching is instant if a session for the current
  batch size is already cached, otherwise triggers a brief recompile.
- **Batch size** (1-32) — number of trajectories sampled per frame. Bigger
  is richer-looking but slower; the slider triggers a kernel recompile for
  the active engine on release.
- **DDIM steps** (1-25, DP only) — denoising iterations per frame. The paper
  uses 10. Lower = faster but rougher samples; this slider is instant
  because step count controls only the JS loop length, not the ONNX shape.

Trajectories that pass through walls are tinted toward white.

This Space hosts the unconditional paths of the original CLI:

```
python interact_maze2d.py -p [act, dp] -u
```

Sketch-based guidance from the paper is deferred to follow-on phases — see
"What's NOT here yet" below.

## What's powering this

| Layer | Tech |
| --- | --- |
| ML runtime | [ONNX Runtime Web 1.24](https://onnxruntime.ai/docs/tutorials/web/) via jsDelivr CDN |
| Acceleration | WebGPU when available (modern browser on Mac, Windows, recent Android, iPhone 13+); WASM SIMD fallback otherwise |
| Models | [ACT](https://huggingface.co/felixw/itps-act) (~44 MB FP16 ONNX) + [DP](https://huggingface.co/felixw/itps-dp) UNet (~31 MB FP16 ONNX) |
| DP scheduler | DDIM, ported to ~30 LOC of plain JS (verified bit-exact against the diffusers PyTorch path) |
| UI | Vanilla `<canvas>` + `<script type="module">`. No build step, no framework. |
| Server compute | **Zero**. The Hugging Face Static Space serves a handful of files; all inference happens on the visitor's device. |

The whole client side is three files (`index.html`, `app.js`, `style.css`)
plus the model weights (`act.onnx`, `dp_unet.onnx`). The `scripts/export_onnx.py`
script runs once locally to produce the ONNX files from the PyTorch checkpoints.

## Performance

Per-frame inference latency on the visitor's device. ACT is one forward pass
per frame; DP runs the UNet `num_steps` times per frame, so its per-frame
cost scales roughly linearly with the DDIM steps slider.

| Device | Backend | ACT (1 pass) | DP @ 10 steps | DP @ 4 steps |
| --- | --- | --- | --- | --- |
| M-series Mac | WebGPU | ~5-10 ms | ~80-150 ms | ~30-60 ms |
| Modern Windows / Linux laptop | WebGPU | ~10-20 ms | ~150-300 ms | ~60-120 ms |
| iPhone 13+ / iPad Pro | WebGPU | ~15-30 ms | ~250-500 ms | ~100-200 ms |
| Older laptop, no WebGPU | WASM SIMD | ~80-150 ms | ~1-2 s | ~400-800 ms |

The status pill at the top of the page reports the live measured ms/frame
and FPS for whatever engine + batch + steps you've picked.

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

The first ONNX request downloads from the local server; both models then
run in your browser via WebGPU / WASM exactly as they do on the deployed
Space.

## Re-exporting the models

If you retrain ACT or DP, regenerate the ONNX files locally:

```bash
source .venv/bin/activate
pip install onnx onnxruntime onnxconverter-common onnxscript
python scripts/export_onnx.py --engine act        # produces act.onnx
python scripts/export_onnx.py --engine dp         # produces dp_unet.onnx
```

Both wrappers take fully positional inputs (ONNX hates dict feeds):

- ACT: `(state, env_state, latent)` -> `actions`. The VAE latent is supplied
  from JS via a seedable PRNG matching `seeded_context(0)` from the CLI;
  this keeps the ONNX graph deterministic and avoids ORT's spotty WebGPU
  support for `RandomNormal`.
- DP UNet: `(sample, timestep, global_cond)` -> `noise_pred`. JS owns the
  outer DDIM loop (sample initialization, timestep schedule, scheduler
  step); only the UNet forward pass goes through ORT. The DDIM scheduler
  port is verified bit-exact against `diffusers.DDIMScheduler` via
  `scripts/verify_dp_js_math.py`.

The DP UNet is converted to FP16 with `onnxruntime.transformers.float16`
(rather than `onnxconverter_common.float16`) so the int64 timestep encoder's
`Cast` nodes stay consistent.

## What's NOT here yet

| Feature | Where it would live |
| --- | --- |
| Sketch input + post-hoc / biased-init strategies | Phase 3. Pure browser, no autograd needed. ~70 LOC additive: stroke capture on canvas, skeletonization, JS guide tensor. |
| Sketch input + guided-diffusion / stochastic-sampling | Phase 4. Either a hybrid call to a Gradio Space (the `hf-space-demo` branch), or finite-difference approximation of `guide_gradient` in JS. |
| Visualizing intermediate DDIM samples (`-v` flag) | Phase 4. Easy: render mid-loop instead of just the final sample. ~10 LOC. |
| INT8 quantization | Optional optimization if first-visit downloads ever feel slow. The current FP16 sizes (~75 MB total) are already small enough that this is unlikely to matter. |

This Space focuses on the realtime *unconditional* case — sketch input plus
gradient-based alignment strategies are a clean follow-on rather than a
flag day.

## Source

Code is on the `browser-demo` branch of
[github.com/yanweiw/itps](https://github.com/yanweiw/itps). The Python CLI,
training code, and original `pygame` interface live on `main`.

## Acknowledgement

Pre-trained ACT and DP weights and the original `interact_maze2d.py` are by
Yanwei Wang. Both policies build on [LeRobot](https://github.com/huggingface/lerobot).
