import argparse
from pathlib import Path
import faiss
import numpy as np
import pandas as pd
from configs.config import Config
from embed_images import (
    default_out_path,
    load_embeddings
)

def top_k(
    embeds: np.ndarray,
    k: int
) -> tuple[np.ndarray, np.ndarray]:
    """For each row in the embeddings, retrieve the k most similar other rows by cosine similarity using FAISS"""
    x = np.ascontiguousarray(embeds, dtype=np.float32)
    n, dim = x.shape

    # Create an index that can search for vectors with the highest inner product in "dim"-dimensional space
    index = faiss.IndexFlatIP(dim)
    index.add(x)

    # k + 1 bc index.search includes itself
    found_scores, found_idx = index.search(x, k + 1)

    idx = np.empty((n, k), dtype=np.int64)
    scores = np.empty((n, k), dtype=np.float32)

    for i in range(n):
        # filter out self
        keep = [p for p in range(k+1) if found_idx[i, p] != i][:k]
        idx[i] = found_idx[i, keep]
        scores[i] = found_scores[i, keep]

    return idx, scores

def build_table(
    df: pd.DataFrame,
    idx: np.ndarray,
    scores: np.ndarray,
) -> pd.DataFrame:
    """Convert FAISS retrieval into DataFrame for easier analysis"""
    tile_ids = df["tile_id"].tolist()
    types = df["types"].tolist()
    type_sets = [set(t.split(",")) for t in types]

    rows = []
    for i in range(len(df)):
        for r in range(idx.shape[1]):
            j = int(idx[i, r])
            sim = float(scores[i, r])
            rows.append(
                {
                    "target_id": tile_ids[i],
                    "rank": r + 1, # Convert 0-index to 1-index
                    "retrieved_id": tile_ids[j],
                    "similarity": sim,
                    "target_types": types[i],
                    "retrieved_types": types[j],
                    "shares_type": bool(type_sets[i] & type_sets[j]),
                }
            )
    return pd.DataFrame(rows)

# Left out for now!
# def report(
#     table: pd.DataFrame,
#     scores: np.ndarray,
# ) -> None:
#     pass 

def main():
    cfg = Config()

    ap = argparse.ArgumentParser(description="Top-k image-to-image retrieval over cached embeddings")
    ap.add_argument("--tsv", default=cfg.tsv_path, help="TSV written by prep_data")
    ap.add_argument("--emb", default=None, help="embeddings .npz (default: the one embed_images.py writes)")
    ap.add_argument("--out", default=None, help="where to write the CSV (default: next to the TSV)")
    ap.add_argument("--k", type=int, default=max(cfg.ks), help="matches per target (default: largest of Config.ks)")
    args = ap.parse_args()

    emb_path = Path(args.emb) if args.emb else default_out_path(args.tsv, cfg.model_name, cfg.pretrained)
    emb = load_embeddings(emb_path)

    df = pd.read_csv(args.tsv, sep="\t", dtype={"tile_id": str})
    n = len(df)

    idx, scores = top_k(emb.image_embeds, args.k)
    table = build_table(df, idx, scores)

    tsv = Path(args.tsv)

    out = Path(args.out) if args.out else tsv.parent / f"retrieval_{cfg.model_name}_{cfg.pretrained}_{tsv.stem}_k{args.k}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    print(f"Retrieved top {args.k} matches for {n} targets ({len(table)} rows) -> {out}")
    # report(table, scores)

if __name__ == "__main__":
    main()
