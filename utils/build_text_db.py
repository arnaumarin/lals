"""Build the reference text database of a model (needed once before the demo notebook).

Corpus: 50,000 captions sampled (seed 42) from the 569,002 unique COCO train2017
captions, gender-balanced to 46,227 captions. Every caption is run through the model's
language backbone and the contextual embedding of each token is stored for every
requested layer.

    python -m utils.build_text_db --model qwen2vl \
        --coco-captions annotations/captions_train2017.json

Output: <db-root>/<model>/layer_<L>/embeddings_cache.pt (about 3.6 GB per layer for the
Qwen models, 4.4 GB for LLaVA / InternVL).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from utils.functions import (base_rates, build_text_db, gender_balance_sentences, label_entries,
                             layer_path, load_coco_captions)
from utils.models import MODELS, load_vlm

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=sorted(k for k, m in MODELS.items() if m.text_db == k))
    parser.add_argument("--coco-captions", type=Path, required=True,
                        help="COCO 2017 annotations/captions_train2017.json")
    parser.add_argument("--db-root", type=Path, default=REPO_ROOT / "embeddb")
    parser.add_argument("--layers", default=None, help="comma-separated; default: the model's sweep layers")
    args = parser.parse_args()

    spec = MODELS[args.model]
    out_dir = args.db_root / args.model
    layers = [int(x) for x in args.layers.split(",")] if args.layers else list(spec.sweep_layers)
    layers = [layer for layer in layers if not layer_path(out_dir, layer).exists()]
    if not layers:
        print(f"all requested layers already exist in {out_dir}")
        return

    captions = gender_balance_sentences(load_coco_captions(args.coco_captions))
    print(f"{len(captions)} captions after gender balancing")
    for layer, path in build_text_db(load_vlm(args.model), captions, layers, out_dir).items():
        print(f"layer {layer}: {path}")

    # Same metadata at every layer.
    metadata = torch.load(path, map_location="cpu", weights_only=False)["metadata"]
    rates = base_rates(*label_entries(metadata))
    print(f"{len(metadata):,} entries, {len({m['token_str'] for m in metadata}):,} unique tokens, "
          f"base rates female={rates.female!r} male={rates.male!r}")


if __name__ == "__main__":
    main()
