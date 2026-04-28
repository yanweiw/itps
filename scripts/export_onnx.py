"""One-time export of the ACT policy to ONNX for browser-side inference.

Usage:
    python scripts/export_onnx.py            # writes ./act.onnx (FP16, ~75 MB)
    python scripts/export_onnx.py --fp32     # also keep act_fp32.onnx for debugging

The wrapper takes three positional inputs:

* ``state``     (B, 2)   pre-normalized agent xy in maze coords
* ``env_state`` (B, 2)   pre-normalized environment state (== state for Maze2D)
* ``latent``    (B, 32)  standard-normal noise vector that replaces the
                         ``torch.randn`` call inside ``ACT.forward``. Injecting
                         it as an input means the model is fully deterministic
                         given inputs, the JS side can re-seed it per call to
                         match ``seeded_context(0)`` from the original CLI, and
                         the ONNX graph contains no ``RandomNormal`` op (which
                         is sometimes patchy under WebGPU).

Normalization stays on the JS side (just 4 floats: mean=[3.6688, 5.3566],
std=[1.8173, 2.5588]) so the ONNX graph is purely the trained transformer.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch import nn

# Make ``from common.* import ...`` work the same way as ``interact_maze2d.py``.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "itps"))

from common.policies.act.modeling_act import ACTPolicy  # noqa: E402

DEFAULT_WEIGHTS = "itps/weights_act/pretrained_model"
DEFAULT_OUTPUT = os.path.join(_REPO_ROOT, "act.onnx")
DEFAULT_FP32_OUTPUT = os.path.join(_REPO_ROOT, "act_fp32.onnx")


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
        batch_size = env_state.shape[0]

        # Build the transformer-encoder input tokens in the same order as the
        # original forward pass: [latent, robot_state, env_state].
        encoder_in_tokens = [
            m.encoder_latent_input_proj(latent),
            m.encoder_robot_state_input_proj(state),
            m.encoder_env_state_input_proj(env_state),
        ]
        encoder_in_pos_embed = list(m.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        encoder_in_tokens = torch.stack(encoder_in_tokens, dim=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, dim=0)

        encoder_out = m.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

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


def export(weights_path: str, fp32_path: str, fp16_path: str, keep_fp32: bool) -> None:
    print(f"[1/4] loading {weights_path}")
    policy = ACTPolicy.from_pretrained(weights_path).eval()
    wrapper = ACTExport(policy).eval()

    B = 32
    state_dim = 2
    latent_dim = policy.config.latent_dim
    chunk_size = policy.config.chunk_size

    state = torch.zeros(B, state_dim)
    env_state = torch.zeros(B, state_dim)
    latent = torch.randn(B, latent_dim)

    print(f"[2/4] exporting FP32 ONNX -> {fp32_path}")
    # ``dynamo=False`` selects the legacy TorchScript-based exporter, which is
    # better-trodden for transformer models and produces cleaner opset-17 output
    # than the new dynamo path (which keeps emitting opset-18 ``Split`` even when
    # opset 17 is requested).
    torch.onnx.export(
        wrapper,
        (state, env_state, latent),
        fp32_path,
        opset_version=17,
        input_names=["state", "env_state", "latent"],
        output_names=["actions"],
        dynamic_axes={
            "state": {0: "batch"},
            "env_state": {0: "batch"},
            "latent": {0: "batch"},
            "actions": {0: "batch"},
        },
        dynamo=False,
    )

    print("[3/4] verifying FP32 ONNX vs PyTorch...")
    import onnxruntime as ort

    sess = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(
        None,
        {
            "state": state.numpy(),
            "env_state": env_state.numpy(),
            "latent": latent.numpy(),
        },
    )[0]
    with torch.no_grad():
        torch_out = wrapper(state, env_state, latent).numpy()
    diff = np.abs(onnx_out - torch_out).max()
    print(f"      max |onnx - pytorch| = {diff:.3e}  (expecting < 1e-3)")
    assert diff < 1e-3, f"FP32 ONNX divergence too large: {diff}"
    print(f"      onnx output shape   = {onnx_out.shape}  (expected (B={B}, T={chunk_size}, 2))")

    print(f"[4/4] converting FP32 -> FP16 -> {fp16_path}")
    import onnx
    from onnxconverter_common import float16

    model_fp32 = onnx.load(fp32_path)
    model_fp16 = float16.convert_float_to_float16(
        model_fp32,
        keep_io_types=True,  # keep state/env_state/latent/actions as float32 for JS simplicity
    )
    onnx.save(model_fp16, fp16_path)

    # Sanity-check FP16 output as well.
    sess16 = ort.InferenceSession(fp16_path, providers=["CPUExecutionProvider"])
    onnx16_out = sess16.run(
        None,
        {
            "state": state.numpy(),
            "env_state": env_state.numpy(),
            "latent": latent.numpy(),
        },
    )[0]
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="path to pretrained ACT checkpoint")
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help="output FP16 ONNX path")
    parser.add_argument("--fp32-out", default=DEFAULT_FP32_OUTPUT, help="intermediate FP32 ONNX path")
    parser.add_argument("--fp32", action="store_true", help="keep the intermediate FP32 ONNX file")
    args = parser.parse_args()

    export(args.weights, args.fp32_out, args.out, keep_fp32=args.fp32)


if __name__ == "__main__":
    main()
