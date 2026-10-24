import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from configs.config import Config
from eval.retrieval import (
    TileEmbeddings,
    embed_tiles,
    load_model
)

# TODO: Consider adding a function to fend against common failure points

def default_out_path(
    tsv_path: str,
    model_name: str,
    pretrained: str,
) -> Path:
    """Construct a default out path using similar naming conventions to runs"""
    tsv = Path(tsv_path)
    return tsv.parent / f"embeds_{model_name}_{pretrained}_{tsv.stem}.npz"

def save_embeddings(
    path: Path,
    emb: TileEmbeddings,
    model_name: str,
    pretrained: str
) -> None:
    """Write embeddings to a .npz"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(
        tmp,
        embeds=emb.image_embeds.astype(np.float32),
        labels=emb.labels.astype(np.float32),
        tile_ids=np.array(emb.tile_ids, dtype=str),
        type_names=np.array(emb.type_names, dtype=str),
        model_name=np.array(model_name),
        pretrained=np.array(pretrained),
    )
    tmp.replace(path)

def load_embeddings(path):
    """Load a cache wrriten by save_embeddings"""
    with np.load(path, allow_pickle=False) as z:
        return TileEmbeddings(
            image_embeds=z["embeds"],
            labels=z["labels"],
            tile_ids=z["tile_ids"].tolist(),
            type_names=z["type_names"].tolist(),
        )

def main():
    cfg = Config()

    ap = argparse.ArgumentParser(description="Embed all tiles once, cache vectors")
    ap.add_argument("--tsv", default=cfg.tsv_path, help="TSV written by prep_data.py")
    ap.add_argument("--out", default=None, help="Where to write the .npz")
    ap.add_argument("--overwrite", action="store_true", help="Recompute even if the cache already exists")
    args = ap.parse_args()

    out = Path(args.out) if args.out else default_out_path(args.tsv, cfg.model_name, cfg.pretrained)

    # For when the cache already exists, overwrite not specified
    if out.exists() and not args.overwrite:
        emb = load_embeddings(out)
        n, d = emb.image_embeds.shape
        print(f"Cache already exists: {out} ({n} tiles, {d} dims). Nothing to do; use --overwrite to recompute.")

    device = cfg.device
    model, preprocess, _ = load_model(cfg.model_name, cfg.pretrained, device)
    emb = embed_tiles(model, preprocess, args.tsv, device, cfg.batch_size, cfg.num_workers)
    save_embeddings(out, emb, cfg.model_name, cfg.pretrained)

    # Confirmation
    n, d = emb.image_embeds.shape
    counts = dict(zip(emb.type_names, emb.labels.sum(axis=0).astype(int).tolist()))
    print(f"Saved {n} tiles x {d} dims to {out}")
    print(f"Tiles per type (multi-type tiles count once per type): {counts}")

if __name__ == "__main__":
    main()