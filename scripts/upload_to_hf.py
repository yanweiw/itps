"""Upload the pre-trained ITPS Maze2D checkpoints to the Hugging Face Hub.

Usage:
    huggingface-cli login                    # one-time, stores token in ~/.cache/huggingface
    python scripts/upload_to_hf.py           # uploads both ACT and DP repos
    python scripts/upload_to_hf.py --dry-run # show what would happen, do not push

The script uploads the contents of `itps/weights_act/pretrained_model/` and
`itps/weights_dp/pretrained_model/` to two public model repos and overwrites
the auto-generated PyTorchModelHubMixin README with a real model card.

It is idempotent: re-running updates the existing repos in place.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

HF_USERNAME = "felixw"
PAPER_ID = "2411.16627"
PAPER_URL = f"https://huggingface.co/papers/{PAPER_ID}"
PROJECT_URL = "https://yanweiw.github.io/itps/"
GITHUB_URL = "https://github.com/yanweiw/itps"


def model_card(policy_name: str, policy_long: str, repo_id: str) -> str:
    """Render the README/model card for one checkpoint repo."""
    return f"""---
library_name: lerobot
license: mit
tags:
  - lerobot
  - {policy_name}
  - robotics
  - maze2d
  - itps
  - pytorch_model_hub_mixin
pipeline_tag: robotics
---

# ITPS Maze2D — {policy_long} ({policy_name.upper()})

Pre-trained {policy_long} checkpoint used in
**Inference-Time Policy Steering through Human Interactions**
([paper]({PAPER_URL}), [project page]({PROJECT_URL}), [code]({GITHUB_URL})).

The model was trained on the [D4RL Maze2D](https://github.com/Farama-Foundation/D4RL)
dataset and is intended to be loaded with the
[LeRobot](https://github.com/huggingface/lerobot) policy classes.

## Usage

Clone the inference repo, then load this checkpoint directly from the Hub:

```bash
git clone https://github.com/yanweiw/itps.git && cd itps
pip install -e .
python interact_maze2d.py -p {policy_name} --hf
```

Or load it programmatically:

```python
from itps.common.policies.{ "diffusion" if policy_name == "dp" else policy_name }.modeling_{ "diffusion" if policy_name == "dp" else policy_name } import { "DiffusionPolicy" if policy_name == "dp" else "ACTPolicy" }

policy = { "DiffusionPolicy" if policy_name == "dp" else "ACTPolicy" }.from_pretrained("{repo_id}")
policy.eval()
```

## Citation

```bibtex
@article{{wang2024itps,
  title={{Inference-Time Policy Steering through Human Interactions}},
  author={{Wang, Yanwei and others}},
  journal={{arXiv preprint arXiv:{PAPER_ID}}},
  year={{2024}}
}}
```

## License

MIT — see [LICENSE]({GITHUB_URL}/blob/main/LICENSE).
"""


CHECKPOINTS = [
    {
        "policy_name": "act",
        "policy_long": "Action Chunking Transformer",
        "local_dir": REPO_ROOT / "itps" / "weights_act" / "pretrained_model",
        "repo_id": f"{HF_USERNAME}/itps-act",
    },
    {
        "policy_name": "dp",
        "policy_long": "Diffusion Policy",
        "local_dir": REPO_ROOT / "itps" / "weights_dp" / "pretrained_model",
        "repo_id": f"{HF_USERNAME}/itps-dp",
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print plan without uploading.")
    parser.add_argument(
        "--only",
        choices=["act", "dp"],
        help="Only upload one of the two checkpoints.",
    )
    args = parser.parse_args()

    targets = [c for c in CHECKPOINTS if args.only is None or c["policy_name"] == args.only]

    for ckpt in targets:
        if not ckpt["local_dir"].is_dir():
            print(f"ERROR: missing local checkpoint dir: {ckpt['local_dir']}", file=sys.stderr)
            print("       Download the weights as described in the project README first.", file=sys.stderr)
            return 1

    if args.dry_run:
        print("DRY RUN — no uploads will be performed.\n")

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("ERROR: huggingface_hub is not installed. Run: pip install huggingface_hub", file=sys.stderr)
        return 1

    api = HfApi()

    for ckpt in targets:
        repo_id = ckpt["repo_id"]
        local_dir = ckpt["local_dir"]
        card = model_card(ckpt["policy_name"], ckpt["policy_long"], repo_id)

        print(f"=== {repo_id} ===")
        print(f"  source:   {local_dir}")
        files = sorted(p.name for p in local_dir.iterdir() if p.is_file())
        print(f"  files:    {', '.join(files)}")
        print(f"  card:     {len(card)} chars (will replace existing README.md)")

        if args.dry_run:
            print()
            continue

        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)

        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / "stage"
            shutil.copytree(local_dir, stage)
            (stage / "README.md").write_text(card)

            api.upload_folder(
                folder_path=str(stage),
                repo_id=repo_id,
                repo_type="model",
                commit_message="Upload ITPS Maze2D pretrained checkpoint",
            )

        print(f"  uploaded: https://huggingface.co/{repo_id}\n")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
