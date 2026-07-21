"""Export the fused DP "DDIM step" ONNX graph used by demo/app.js.

The original demo shipped the bare UNet (dp_unet.onnx) and ran the DDIM
scheduler in JavaScript. That had two performance problems on ONNX Runtime
Web's WebGPU backends:

  1. On the JSEP backend (the `webgpu` EP in the plain ort.min.js bundle up
     to at least 1.24.x), `Softplus` has no WebGPU kernel. The UNet uses
     Mish = x * tanh(softplus(x)) 27 times, so every forward pass bounced
     27 tensors GPU -> CPU -> GPU. That capped DP at ~6 FPS (batch=8,
     4 DDIM steps) on an M-series Mac.
  2. Every DDIM step downloaded noise_pred to JS and re-uploaded the updated
     sample, adding a sync point per step.

This export fixes both by fusing one whole denoising step into the graph:

    eps  = UNet(sample, timestep, global_cond)
    eps += guide_ratio * (sample - guide) / (max(dist, 1e-8) * horizon)
    x0   = clip(inv_sqrt_at * sample - som_over_sa * eps, -1, 1)
    out  = c0 * x0 + c1 * eps + c2 * noise

The per-step scalars come in as runtime inputs, so a single graph serves
every (DDIM steps, alignment mode) combination the demo exposes:

    normal DDIM step to t_prev: c0=sqrt(a_prev), c1=sqrt(1-a_prev), c2=0
    MCMC re-noise at t:         c0=sqrt(a_t),    c1=0,              c2=sqrt(1-a_t)
    guidance off:               guide_ratio=0 (guide/noise can be zeros)

demo/app.js chains steps with the sample staying on the GPU (the previous
step's output tensor is fed straight back as the next step's input) and only
downloads the final actions. Run on the deployed ORT >= 1.26 native WebGPU EP
bundle (ort.webgpu.min.js), which has Softplus/Tanh kernels, this takes DP
from ~6 to ~34 FPS at batch=8 / 4 steps, and the same graph also runs on the
WASM fallback for browsers without WebGPU.

Usage (needs the policy sources + weights; run from a checkout that has
them, e.g. the browser-demo branch, with the repo venv):

    .venv/bin/python tools/export_dp_step.py \
        --sources /path/to/itps-package-dir \
        --weights itps/weights_dp/pretrained_model \
        --out demo/dp_step.onnx

The exported graph is verified against PyTorch (fp32 exact, fp16 <= ~1e-3)
including full multi-step loops for every alignment mode.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch import nn


class FusedDDIMStep(nn.Module):
    """One denoising step with optional sketch guidance; see module docstring."""

    def __init__(self, unet, horizon: int):
        super().__init__()
        self.unet = unet
        self.inv_horizon = 1.0 / horizon

    def forward(
        self,
        sample,        # (B, H, 2) f32
        timestep,      # (B,)      i64
        global_cond,   # (B, 8)    f32
        guide,         # (H, 2)    f32, zeros when unused
        guide_ratio,   # (1,)      f32, 0 disables guidance
        inv_sqrt_at,   # (1,)      f32  = 1 / sqrt(alpha_t)
        som_over_sa,   # (1,)      f32  = sqrt(1 - alpha_t) / sqrt(alpha_t)
        c0,            # (1,)      f32
        c1,            # (1,)      f32
        c2,            # (1,)      f32
        noise,         # broadcastable to (B, H, 2); (1, 1, 1) zeros when unused
    ):
        eps = self.unet(sample, timestep, global_cond=global_cond)

        # Sketch-guidance gradient: d/d_sample of the mean L2 distance to the
        # guide trajectory -- identical math to interact_maze2d.py's
        # guided-diffusion / stochastic-sampling strategies.
        diff = sample - guide
        dist = torch.sqrt(torch.sum(diff * diff, dim=-1, keepdim=True))
        dist = torch.clamp(dist, min=1e-8)
        eps = eps + guide_ratio * (diff / dist) * self.inv_horizon

        # DDIM update (prediction_type="epsilon", clip_sample=True, eta=0),
        # refactored so all schedule-dependent factors are runtime scalars.
        x0 = inv_sqrt_at * sample - som_over_sa * eps
        x0 = torch.clamp(x0, -1.0, 1.0)
        return c0 * x0 + c1 * eps + c2 * noise


def ddim_timesteps(num_train_timesteps: int, n: int) -> list[int]:
    step = num_train_timesteps // n
    return [(n - 1 - i) * step for i in range(n)]


def verify_loops(policy, sess, horizon: int, alphas: np.ndarray) -> None:
    """Full-loop parity: fused ONNX graph vs PyTorch reference for all modes."""
    B = 8
    cond = torch.randn(B, 8)
    init = np.random.RandomState(7).randn(B, horizon, 2).astype(np.float32)
    guide_np = (np.random.RandomState(8).randn(horizon, 2) * 0.4).astype(np.float32)

    def torch_loop(steps, guide=None, ratio=0.0, mcmc=1, start_influence=None, seed=1234):
        rng = np.random.RandomState(seed)
        s = init.copy()
        ts = ddim_timesteps(len(alphas), steps)
        for i, t in enumerate(ts):
            if start_influence is not None and t > start_influence:
                continue
            t_prev = ts[i + 1] if i + 1 < len(ts) else -1
            a_t, a_p = alphas[t], (1.0 if t_prev < 0 else alphas[t_prev])
            for m in range(mcmc):
                with torch.no_grad():
                    eps = policy.diffusion.unet(
                        torch.from_numpy(s), torch.full((B,), t, dtype=torch.long), global_cond=cond
                    ).numpy().copy()
                if ratio and t > 0:
                    d = s - guide[None]
                    dist = np.maximum(np.sqrt((d ** 2).sum(-1, keepdims=True)), 1e-8)
                    eps = eps + ratio * (d / dist) / horizon
                x0 = np.clip((s - np.sqrt(1 - a_t) * eps) / np.sqrt(a_t), -1, 1)
                if m < mcmc - 1:
                    noise = rng.randn(*s.shape).astype(np.float32)
                    s = (np.sqrt(a_t) * x0 + np.sqrt(1 - a_t) * noise).astype(np.float32)
                else:
                    s = (np.sqrt(a_p) * x0 + np.sqrt(1 - a_p) * eps).astype(np.float32)
        return s

    def onnx_loop(steps, guide=None, ratio=0.0, mcmc=1, start_influence=None, seed=1234):
        rng = np.random.RandomState(seed)
        s = init.copy()
        ts = ddim_timesteps(len(alphas), steps)
        g = guide if guide is not None else np.zeros((horizon, 2), np.float32)
        for i, t in enumerate(ts):
            if start_influence is not None and t > start_influence:
                continue
            t_prev = ts[i + 1] if i + 1 < len(ts) else -1
            a_t, a_p = alphas[t], (1.0 if t_prev < 0 else alphas[t_prev])
            for m in range(mcmc):
                renoise = m < mcmc - 1
                s = sess.run(None, {
                    "sample": s,
                    "timestep": np.full((B,), t, np.int64),
                    "global_cond": cond.numpy(),
                    "guide": g,
                    "guide_ratio": np.array([ratio if t > 0 else 0.0], np.float32),
                    "inv_sqrt_at": np.array([1 / np.sqrt(a_t)], np.float32),
                    "som_over_sa": np.array([np.sqrt(1 - a_t) / np.sqrt(a_t)], np.float32),
                    "c0": np.array([np.sqrt(a_t) if renoise else np.sqrt(a_p)], np.float32),
                    "c1": np.array([0.0 if renoise else np.sqrt(1 - a_p)], np.float32),
                    "c2": np.array([np.sqrt(1 - a_t) if renoise else 0.0], np.float32),
                    "noise": rng.randn(*s.shape).astype(np.float32) if renoise
                             else np.zeros((1, 1, 1), np.float32),
                })[0]
        return s

    cases = [
        ("unconditional steps=4", dict(steps=4)),
        ("unconditional steps=10", dict(steps=10)),
        ("guided-diffusion steps=10 ratio=20", dict(steps=10, guide=guide_np, ratio=20.0)),
        ("stochastic steps=4 mcmc=4 ratio=60", dict(steps=4, guide=guide_np, ratio=60.0, mcmc=4)),
        ("biased-init steps=4 (skip t>50)", dict(steps=4, start_influence=50)),
    ]
    for name, kw in cases:
        diff = np.abs(torch_loop(**kw) - onnx_loop(**kw)).max()
        print(f"  loop parity [{name}]: max |diff| = {diff:.3e}")
        assert diff < 2e-2, (name, diff)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", required=True,
                    help="directory containing the `common` package (itps python sources)")
    ap.add_argument("--weights", default="itps/weights_dp/pretrained_model")
    ap.add_argument("--out", default="demo/dp_step.onnx")
    args = ap.parse_args()

    sys.path.insert(0, args.sources)
    from common.policies.diffusion.modeling_diffusion import DiffusionPolicy

    import onnx
    import onnxruntime as ort_py
    from onnxruntime.transformers.float16 import convert_float_to_float16

    torch.manual_seed(0)
    policy = DiffusionPolicy.from_pretrained(args.weights, alignment_strategy="post-hoc").eval()
    horizon = policy.config.horizon
    wrapper = FusedDDIMStep(policy.diffusion.unet, horizon).eval()

    B = 8
    inputs = (
        torch.randn(B, horizon, 2),
        torch.full((B,), 50, dtype=torch.long),
        torch.randn(B, 8),
        torch.randn(horizon, 2),
        torch.tensor([20.0]),
        torch.tensor([1.2]),
        torch.tensor([0.9]),
        torch.tensor([0.8]),
        torch.tensor([0.6]),
        torch.tensor([0.0]),
        torch.zeros(1, 1, 1),
    )
    input_names = [
        "sample", "timestep", "global_cond", "guide", "guide_ratio",
        "inv_sqrt_at", "som_over_sa", "c0", "c1", "c2", "noise",
    ]
    dynamic_axes = {
        "sample": {0: "batch"},
        "timestep": {0: "batch"},
        "global_cond": {0: "batch"},
        "noise": {0: "noise_b", 1: "noise_h", 2: "noise_d"},
        "sample_out": {0: "batch"},
    }

    fp32_path = args.out.replace(".onnx", "_fp32.onnx")
    torch.onnx.export(
        wrapper, inputs, fp32_path,
        opset_version=17,
        input_names=input_names,
        output_names=["sample_out"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    feeds = {n: t.numpy() for n, t in zip(input_names, inputs)}
    sess32 = ort_py.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        torch_out = wrapper(*inputs).numpy()
    diff32 = np.abs(sess32.run(None, feeds)[0] - torch_out).max()
    print(f"fp32 vs torch: max |diff| = {diff32:.3e}")
    assert diff32 < 1e-4

    model_fp16 = convert_float_to_float16(onnx.load(fp32_path), keep_io_types=True)
    onnx.save(model_fp16, args.out)
    os.remove(fp32_path)

    sess16 = ort_py.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    diff16 = np.abs(sess16.run(None, feeds)[0] - torch_out).max()
    print(f"fp16 vs torch: max |diff| = {diff16:.3e}  ({os.path.getsize(args.out)/1e6:.1f} MB)")
    assert diff16 < 5e-3

    alphas = policy.diffusion.noise_scheduler.alphas_cumprod.numpy()
    verify_loops(policy, sess16, horizon, alphas)
    print("all parity checks passed;", args.out)


if __name__ == "__main__":
    main()
