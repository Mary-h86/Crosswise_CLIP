import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from crosswise_clip.config import Config
from crosswise_clip.embed_image import load_model
from sentence_transformers import SentenceTransformer

BGE_MODEL = "BAAI/bge-m3"
CLIP_CONTEXT = 77

@dataclass
class TextEmbeddings:
    tile_ids: list  # (N,) same order as the TSV
    descriptions: list  # (N,) the text that was embedded
    clip: np.ndarray  # (N, 512) CLIP text vectors, L2-normalized
    bge: np.ndarray  # (N, 1024) BGE-M3 vectors, L2-normalized
    clip_tokens: np.ndarray  # (N,) tokens CLIP needs for each description, start and end tokens included
    clip_model: str
    clip_pretrained: str
    bge_model: str # new

def load_descriptions(
    tsv: Path,
    db: Path,
) -> tuple[list[str], list[str]]:
    """Output Tile IDs in TSV order with corresponding descriptions"""
    ids = pd.read_csv(tsv, sep="\t", dtype={"tile_id": str})[["tile_id"]]
    with duckdb.connect(str(db), read_only=True) as con:
        con.register("ids", ids)
        rows = con.execute("""
            SELECT t.tile_id, t.tile_description
            FROM tiles t
            JOIN ids 
            USING (tile_id)
        """).fetchall()

        found = {tid: ("" if desc is None else str(desc).strip()) for tid, desc in rows}
        tile_ids = ids["tile_id"].tolist()

        return tile_ids, [found[t] for t in tile_ids]

def report_repeats(
    texts: list[str]
):
    """Show identical embedded vectors"""
    counts = Counter(t.lower() for t in texts)
    repeated = counts.most_common()
    repeated = [(t, c) for t, c in repeated if c > 1]
    print(f"Descriptions that appear on more than one tile (ignoring case): {len(repeated)} text(s), shared by {sum(c for _, c in repeated)} of {len(texts)} tiles")
    for text, count in repeated[:10]:
        print(f"  {count:4d} tiles  {text[:110]!r}")

@torch.no_grad()
def clip_text_embds(
    model,
    tokenizer,
    texts: list[str],
    device: str,
    batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Output CLIP text vectors and the token count of each text prior to truncation, if applicable"""
    vectors = []
    for start in tqdm(range(0, len(texts), batch_size), desc="CLIP text"):
        tokens = tokenizer(texts[start:start + batch_size]).to(device)
        f = model.encode_text(tokens)
        f = f / f.norm(dim=-1, keepdim=True)
        vectors.append(f.cpu())
        counts = np.array([len(tokenizer.encode(t))+ 2 for t in texts])

    return torch.cat(vectors).numpy().astype(np.float32), counts

def bge_embds(
    texts: list[str],
    device: str,
    batch_size: int
) -> np.ndarray:
    """Output BGE-M3 dense vectors (N, 1024) L2-norm"""
    model = SentenceTransformer(BGE_MODEL, device = device)
    return np.asarray(
        model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True),
        dtype=np.float32,
    )

def default_out_path(
    tsv: Path,
    model_name: str,
    pretrained: str,
) -> Path:
    return tsv.parent / f"text_embeds_{model_name}_{pretrained}_bge-m3_{tsv.stem}.npz"

def save_text_embeddings(
    path: Path,
    emb: TextEmbeddings
):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(
        tmp, 
        clip=emb.clip.astype(np.float32),
        bge=emb.bge.astype(np.float32),
        clip_tokens=emb.clip_tokens.astype(np.int32),
        tile_ids=np.array(emb.tile_ids, dtype=str),
        clip_model=np.array(emb.clip_model),
        clip_pretrained=np.array(emb.clip_pretrained),
        bge_model=np.array(emb.bge_model),
    )
    tmp.replace(path)

def load_text_embedds(
    path: Path,
) -> TextEmbeddings:
    with np.load(path, allow_pickle=False) as z:
        return TextEmbeddings(
            tile_ids=z["tile_ids"].tolist(),
            descriptions=z["descriptions"].tolist(),
            clip=z["clip"],
            bge=z["bge"],
            clip_tokens=z["clip_tokens"],
            clip_model=str(z["clip_model"]),
            clip_pretrained=str(z["clip_pretrained"]),
            bge_model=str(z["bge_model"]),
        )

def print_summary(emb: TextEmbeddings, path: Path) -> None:
    n = len(emb.tile_ids)
    print(f"{path.name}: {n} tiles; CLIP text {emb.clip.shape[1]} dims, BGE-M3 {emb.bge.shape[1]} dims")
    cut = emb.clip_tokens[emb.clip_tokens > CLIP_CONTEXT]
    print(f"CLIP cuts {len(cut)} of {n} descriptions (more than {CLIP_CONTEXT} tokens); BGE-M3 cuts none")
    if len(cut):
        kept = CLIP_CONTEXT / cut
        print(f"  a cut description keeps a median {np.median(kept):.0%} of its tokens in CLIP (smallest {kept.min():.0%})")

def main():
    cfg = Config()
    ap = argparse.ArgumentParser(description="Embed all tile descriptions with CLIP text and BGE-M3, cache the vectors")
    ap.add_argument("--tsv", default=cfg.tsv_path, help="TSV written by prep_data / exclude_types")
    ap.add_argument("--db", default=None, help="DuckDB file (default: annotations_per_tile.duckdb next to the TSV)")
    ap.add_argument("--out", default=None, help="Where to write the .npz")
    ap.add_argument("--overwrite", action="store_true", help="Recompute even if the cache already exists")
    ap.add_argument("--device", default=cfg.device, help="cuda, mps or cpu (default: the best available). Try cpu if mps fails")
    ap.add_argument("--batch-size", type=int, default=16, help="descriptions per batch")
    args = ap.parse_args()

    tsv = Path(args.tsv)
    db = Path(args.db) if args.db else tsv.parent / "annotations_per_tile.duckdb"
    out = Path(args.out) if args.out else default_out_path(tsv, cfg.model_name, cfg.pretrained)

    tile_ids, texts = load_descriptions(tsv, db)
    report_repeats(texts)

    if out.exists() and not args.overwrite:
        emb = load_text_embedds(out)
        print_summary(emb, out)
        return

    model, _, tokenizer = load_model(cfg.model_name, cfg.pretrained, args.device)
    clip, clip_tokens = clip_text_embds(model, tokenizer, texts, args.device, args.batch_size)

    # free CLIP from memory before loading BGE
    del model   

    bge = bge_embds(texts, args.device, args.batch_size)

    emb = TextEmbeddings(tile_ids, texts, clip, bge, clip_tokens, cfg.model_name, cfg.pretrained, BGE_MODEL)
    save_text_embeddings(out, emb)
    print(f"\nSaved to {out}")
    print_summary(emb, out)

if __name__ == "__main__":
    main()
