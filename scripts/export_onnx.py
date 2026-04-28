"""One-time export of ITPS policies (ACT or DP) to ONNX for browser inference.

Usage:
    python scripts/export_onnx.py --engine act        # writes ./act.onnx
    python scripts/export_onnx.py --engine dp         # writes ./dp_unet.onnx
    python scripts/export_onnx.py --engine dp --fp32  # also keep dp_unet_fp32.onnx

Both wrappers take fully positional inputs (ONNX hates dict feeds). The JS side
generates the random tensors that the original PyTorch code produces internally
(VAE latent for ACT, initial sample for DP) so that:

* the ONNX graph is fully deterministic given inputs,
* the JS can re-seed per call to match ``seeded_context(0)`` from the CLI, and
* we don't depend on ORT WebGPU's spotty ``RandomNormal`` support.

Normalization stats (mean/std for ACT, min/max for DP) are printed to stdout
after export so they can be pasted into ``app.js`` as JS constants -- we keep
normalization on the JS side so the ONNX graph is purely the trained network.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn

# Make ``from common.* import ...`` work the same way as ``interact_maze2d.py``.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "itps"))

from common.policies.act.modeling_act import ACTPolicy  # noqa: E402
from common.policies.diffusion.modeling_diffusion import DiffusionPolicy  # noqa: E402

# ---------------------------------------------------------------------------
# Engine wrappers
# ---------------------------------------------------------------------------


class ACTExport(nn.Module):
    """Thin wrapper around ACT's transformer that takes positional, fully
    deterministic inputs (state, env_state, latent).

    Mirrors the eval-mode path of ``ACT.forward`` (modeling_act.py:355-410)
    but skips the ``torch.randn`` line: the latent comes in as a tensor input
    instead, so the same (state, env_state, latent) triple always produces
    the same actions.
    """

    def __init__(self, policy: ACTPolicy):
        super().__init__()
        self.model = policy.model
        assert self.model.use_robot_state, "expected observation.state in trained config"
        assert self.model.use_env_state, "expected observation.environment_state in trained config"
        assert not self.model.use_images, "this exporter only supports the Maze2D state-only ACT config"

    def forward(self, state: torch.Tensor, env_state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        m = self.model

        encoder_in_tokens = [
            m.encoder_latent_input_proj(latent),
            m.encoder_robot_state_input_proj(state),
            m.encoder_env_state_input_proj(env_state),
        ]
        encoder_in_pos_embed = list(m.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        encoder_in_tokens = torch.stack(encoder_in_tokens, dim=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, dim=0)

        encoder_out = m.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        batch_size = env_state.shape[0]
        decoder_in = torch.zeros(
            (m.config.chunk_size, batch_size, m.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = m.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=m.decoder_pos_embed.weight.unsqueeze(1),
        )
        decoder_out = decoder_out.transpose(0, 1)  # (B, S, D)
        return m.action_head(decoder_out)  # (B, chunk_size=64, action_dim=2)


class DPExport(nn.Module):
    """Thin wrapper around DP's UNet for the DDIM denoising step. The JS side
    drives the loop (sample initialization, timestep schedule, DDIM update) so
    only one forward pass through the UNet is exported. This keeps the graph
    small and the number of inference steps a runtime-tunable JS knob rather
    than a property baked into the ONNX file.
    """

    def __init__(self, policy: DiffusionPolicy):
        super().__init__()
        self.unet = policy.diffusion.unet

    def forward(
        self, sample: torch.Tensor, timestep: torch.Tensor, global_cond: torch.Tensor
    ) -> torch.Tensor:
        # sample:      (B, horizon, action_dim) e.g. (B, 64, 2)
        # timestep:    (B,) int64
        # global_cond: (B, global_cond_dim)     e.g. (B, 8)
        # returns:     noise_pred same shape as sample
        return self.unet(sample, timestep, global_cond=global_cond)


# ---------------------------------------------------------------------------
# Engine specs
# ---------------------------------------------------------------------------


@dataclass
class EngineSpec:
    name: str
    weights_path: str
    out_basename: str  # filename for the FP16 onnx, e.g. "act.onnx"
    build: Callable[[], "tuple[nn.Module, tuple[torch.Tensor, ...], list[str], list[str], dict]"]
    print_stats: Callable[[object], None]  # takes the policy, prints the JS-pasteable constants


def _build_act() -> tuple:
    policy = ACTPolicy.from_pretrained("itps/weights_act/pretrained_model").eval()
    wrapper = ACTExport(policy).eval()
    B = 32
    state = torch.zeros(B, 2)
    env_state = torch.zeros(B, 2)
    latent = torch.randn(B, policy.config.latent_dim)
    inputs = (state, env_state, latent)
    input_names = ["state", "env_state", "latent"]
    output_names = ["actions"]
    dynamic_axes = {n: {0: "batch"} for n in input_names + output_names}
    return policy, wrapper, inputs, input_names, output_names, dynamic_axes


def _print_act_stats(policy: ACTPolicy) -> None:
    print("\n  JS constants (already in app.js):")
    bsm = policy.normalize_inputs.buffer_observation_state
    bem = policy.unnormalize_outputs.buffer_action
    print(f"    STATE_MEAN  = {bsm['mean'].tolist()}")
    print(f"    STATE_STD   = {bsm['std'].tolist()}")
    print(f"    ACTION_MEAN = {bem['mean'].tolist()}")
    print(f"    ACTION_STD  = {bem['std'].tolist()}")


def _build_dp() -> tuple:
    policy = DiffusionPolicy.from_pretrained(
        "itps/weights_dp/pretrained_model", alignment_strategy="post-hoc"
    ).eval()
    wrapper = DPExport(policy).eval()
    B = 32
    horizon = policy.config.horizon
    action_dim = policy.config.output_shapes["action"][0]
    n_obs_steps = policy.config.n_obs_steps
    state_dim = policy.config.input_shapes["observation.state"][0]
    env_state_dim = policy.config.input_shapes["observation.environment_state"][0]
    global_cond_dim = (state_dim + env_state_dim) * n_obs_steps  # = 8 for Maze2D

    # Pseudo-random non-zero inputs so the verification step exercises real
    # weights (a zero sample can mask broadcast / shape bugs).
    torch.manual_seed(0)
    sample = torch.randn(B, horizon, action_dim)
    timestep = torch.full((B,), 50, dtype=torch.long)
    global_cond = torch.randn(B, global_cond_dim)

    inputs = (sample, timestep, global_cond)
    input_names = ["sample", "timestep", "global_cond"]
    output_names = ["noise_pred"]
    dynamic_axes = {n: {0: "batch"} for n in input_names + output_names}
    return policy, wrapper, inputs, input_names, output_names, dynamic_axes


def _print_dp_stats(policy: DiffusionPolicy) -> None:
    print("\n  JS constants for app.js (DP):")
    bsm = policy.normalize_inputs.buffer_observation_state
    bem = policy.unnormalize_outputs.buffer_action
    print(f"    DP_STATE_MIN  = {bsm['min'].tolist()}")
    print(f"    DP_STATE_MAX  = {bsm['max'].tolist()}")
    print(f"    DP_ACTION_MIN = {bem['min'].tolist()}")
    print(f"    DP_ACTION_MAX = {bem['max'].tolist()}")
    print(f"    DP_HORIZON              = {policy.config.horizon}")
    print(f"    DP_N_OBS_STEPS          = {policy.config.n_obs_steps}")
    print(f"    DP_GLOBAL_COND_DIM      = {(policy.config.input_shapes['observation.state'][0] + policy.config.input_shapes['observation.environment_state'][0]) * policy.config.n_obs_steps}")
    print(f"    DP_NUM_TRAIN_TIMESTEPS  = {policy.config.num_train_timesteps}")
    # Print the alphas_cumprod table so it can be pasted as ALPHAS_CUMPROD literal.
    alphas = policy.diffusion.noise_scheduler.alphas_cumprod.tolist()
    print(f"    ALPHAS_CUMPROD ({len(alphas)} floats) =")
    for i in range(0, len(alphas), 5):
        chunk = ", ".join(f"{a:.10f}" for a in alphas[i : i + 5])
        print(f"      {chunk},")


ENGINES = {
    "act": EngineSpec(
        name="act",
        weights_path="itps/weights_act/pretrained_model",
        out_basename="act.onnx",
        build=_build_act,
        print_stats=_print_act_stats,
    ),
    "dp": EngineSpec(
        name="dp",
        weights_path="itps/weights_dp/pretrained_model",
        out_basename="dp_unet.onnx",
        build=_build_dp,
        print_stats=_print_dp_stats,
    ),
}


# ---------------------------------------------------------------------------
# Generic export pipeline
# ---------------------------------------------------------------------------


def export(spec: EngineSpec, fp32_path: str, fp16_path: str, keep_fp32: bool) -> None:
    print(f"[1/4] loading {spec.weights_path}")
    policy, wrapper, inputs, input_names, output_names, dynamic_axes = spec.build()

    print(f"[2/4] exporting FP32 ONNX -> {fp32_path}")
    # ``dynamo=False`` selects the legacy TorchScript-based exporter, which is
    # better-trodden for transformer/conv models and produces cleaner opset-17
    # output than the new dynamo path (which keeps emitting opset-18 ``Split``
    # even when opset 17 is requested).
    torch.onnx.export(
        wrapper,
        inputs,
        fp32_path,
        opset_version=17,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    print("[3/4] verifying FP32 ONNX vs PyTorch...")
    import onnxruntime as ort

    feeds = {name: t.numpy() for name, t in zip(input_names, inputs)}
    sess = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, feeds)[0]
    with torch.no_grad():
        torch_out = wrapper(*inputs).numpy()
    diff = np.abs(onnx_out - torch_out).max()
    print(f"      max |onnx - pytorch| = {diff:.3e}  (expecting < 1e-3)")
    assert diff < 1e-3, f"FP32 ONNX divergence too large: {diff}"
    print(f"      onnx output shape   = {onnx_out.shape}  (matches PyTorch)")

    print(f"[4/4] converting FP32 -> FP16 -> {fp16_path}")
    import onnx
    # Prefer ``onnxruntime.transformers.float16`` over ``onnxconverter_common.float16``:
    # the former patches ``Cast`` nodes' ``to`` attributes consistently, which
    # the DP UNet's int64 timestep -> float32 sinusoidal embedder needs. The
    # older converter leaves a type mismatch behind (cast output declared FP32
    # but downstream consumer wired up to FP16) that ORT refuses to load.
    from onnxruntime.transformers.float16 import convert_float_to_float16

    model_fp32 = onnx.load(fp32_path)
    model_fp16 = convert_float_to_float16(
        model_fp32,
        keep_io_types=True,  # keep float inputs/outputs as float32 for JS simplicity
    )
    onnx.save(model_fp16, fp16_path)

    sess16 = ort.InferenceSession(fp16_path, providers=["CPUExecutionProvider"])
    onnx16_out = sess16.run(None, feeds)[0]
    diff16 = np.abs(onnx16_out - torch_out).max()
    print(f"      max |fp16 onnx - pytorch| = {diff16:.3e}  (FP16 noise floor ~ 1e-2)")

    fp16_size_mb = os.path.getsize(fp16_path) / (1024 * 1024)
    fp32_size_mb = os.path.getsize(fp32_path) / (1024 * 1024)
    print("\nDone:")
    print(f"  FP16 ONNX: {fp16_path}  ({fp16_size_mb:.1f} MB)")
    if keep_fp32:
        print(f"  FP32 ONNX: {fp32_path}  ({fp32_size_mb:.1f} MB) -- kept for debugging")
    else:
        # ``onnx.save`` may externalize large initializers into a sibling
        # ``<model>.data`` file; remove that too so the working tree stays clean.
        os.remove(fp32_path)
        ext = fp32_path + ".data"
        if os.path.exists(ext):
            os.remove(ext)
        print("  (FP32 ONNX removed; pass --fp32 to keep it for debugging)")

    spec.print_stats(policy)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--engine",
        choices=sorted(ENGINES.keys()),
        default="act",
        help="which policy to export (default: act)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output FP16 ONNX path (default: <repo>/<engine>.onnx)",
    )
    parser.add_argument(
        "--fp32-out",
        default=None,
        help="intermediate FP32 ONNX path (default: <fp16_out>_fp32)",
    )
    parser.add_argument("--fp32", action="store_true", help="keep the intermediate FP32 ONNX file")
    args = parser.parse_args()

    spec = ENGINES[args.engine]
    fp16_path = args.out or os.path.join(_REPO_ROOT, spec.out_basename)
    fp32_path = args.fp32_out or fp16_path.replace(".onnx", "_fp32.onnx")
    export(spec, fp32_path, fp16_path, keep_fp32=args.fp32)


if __name__ == "__main__":
    main()
