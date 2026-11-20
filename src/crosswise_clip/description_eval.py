import argparse
import random
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from crosswise_clip.config import Config
from crosswise_clip.embed_image import default_out_path as image_cache_path
from crosswise_clip.embed_image import load_embeddings
from crosswise_clip.embed_text import CLIP_CONTEXT
from crosswise_clip.embed_text import default_out_path as text_cache_path
from crosswise_clip.embed_text import load_text_embedds
from crosswise_clip.eval import type_masks

def neighbors(
    x: np.ndarray,
    k: int,
) -> np.ndarray:
    """Report indices of k most similar other rows to each row in x"""
    sim = np.round(x.astype(np.float64) @ x.astype(np.float64).T, 6)
    np.fill_diagonal(sim, -np.inf)
    return np.argsort(-sim, axis=1, kind="stable")[:, :k]

def overlap(
    a: np.ndarray,
    b: np.ndarray
) -> np.ndarray:
    k = a.shape[1]
    return np.array([len(set(x) &  set(y)) / k for x, y in zip(a, b)])

def description_similarity(
    text_x: np.ndarray,
    image_top: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Each target's mean description cosine to its image matchs, and to every other as well"""
    sim = text_x.astype(np.float64) @ text_x.astype(np.float64).T
    np.fill_diagonal(sim, 0.0) # nullify self-matches
    matched = np.take_along_axis(sim, image_top, axis=1).mean(axis=1)
    baseline = sim.sum(axis=1) / (len(sim) - 1)
    return matched, baseline

def md_table(
    headers: list[str],
    rows: list[list[str]],
) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|---|" + "---:|" * (len(headers) - 1)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)

def pct(values: np.ndarray) -> str:
    return f"{100 * values.mean():.1f}%" if len(values) else "n/a"

def report(label: str, pool: np.ndarray, d: dict, masks: dict, k: int) -> dict:
    """Print the two tables for the tiles in `pool` and return the per-target results."""
    n = len(pool)
    if n < k + 2:
        raise SystemExit(f"Only {n} tiles in '{label}'; need at least k + 2 = {k + 2}.")
    top = {name: neighbors(d[name][pool], k) for name in ("clip", "bge", "img")}
    cut = d["cut"][pool]
    ov = {"CLIP text vs BGE-M3": overlap(top["clip"], top["bge"]),
          "CLIP text vs image": overlap(top["clip"], top["img"]),
          "BGE-M3 vs image": overlap(top["bge"], top["img"])}
 
    print(f"\n## {label} (N = {n}, k = {k})")
    print(f"\nOverlap@{k}: share of the {k} tiles that appear in both lists (random lists give {100 * k / (n - 1):.1f}%)\n")
    rows = [[name, pct(v), pct(v[cut]), pct(v[~cut])] for name, v in ov.items()]
    rows.append(["random", f"{100 * k / (n - 1):.1f}%", f"{100 * k / (n - 1):.1f}%", f"{100 * k / (n - 1):.1f}%"])
    print(md_table(["Compared lists", f"all targets (n={n})", f"CLIP cut the description (n={cut.sum()})", f"not cut (n={(~cut).sum()})"], rows))
 
    sims = {}
    for name, title in (("clip", "CLIP text"), ("bge", "BGE-M3")):
        sims[name] = description_similarity(d[name][pool], top["img"])
    print(f"\nDescription similarity of each target's {k} image matches (cosine of the description vectors; random = average cosine to every other tile)\n")
    rows = []
    groups = [("all targets", np.ones(n, dtype=bool))] + [(t, m[pool]) for t, m in masks.items() if m[pool].any()]
    for group, m in groups:
        row = [f"{group} (n={m.sum()})"]
        for name in ("clip", "bge"):
            matched, baseline = sims[name]
            row += [f"{matched[m].mean():.3f}", f"{baseline[m].mean():.3f}", f"{matched[m].mean() - baseline[m].mean():+.3f}"]
        rows.append(row)
    print(md_table(["Target type", "CLIP: image matches", "CLIP: random", "CLIP: diff", "BGE-M3: image matches", "BGE-M3: random", "BGE-M3: diff"], rows))
    return {"top": top, "overlap": ov["CLIP text vs BGE-M3"]}

def show_examples(pool: np.ndarray, result: dict, d: dict, tiles: pd.DataFrame, k: int, count: int, seed: int) -> None:
    """Print a few targets whose CLIP and BGE-M3 lists differ, with the descriptions."""
    differ = np.flatnonzero(result["overlap"] < 1.0)
    if len(differ) == 0:
        print("\nThe CLIP and BGE-M3 lists are identical for every target; no examples to show.")
        return
    chosen = random.Random(seed).sample(list(differ), min(count, len(differ)))
    print(f"\n## Examples where the CLIP and BGE-M3 lists differ (random choice, {len(differ)} such targets)")
    print("* = also in the other model's list, i = also among the target's image matches\n")
 
    def line(i: int, j: int, sim: float, other: set, image: set) -> str:
        mark = ("*" if j in other else " ") + ("i" if j in image else " ")
        return f"    {tiles['tile_id'][pool[j]]}  {sim:.3f} {mark}  {d['text'][pool[j]][:150]!r}"
 
    for i in chosen:
        sets = {name: set(result["top"][name][i].tolist()) for name in ("clip", "bge", "img")}
        print(f"Target {tiles['tile_id'][pool[i]]} [{tiles['types'][pool[i]]}], CLIP-BGE-M3 overlap {100 * result['overlap'][i]:.0f}%")
        print(f"  description: {d['text'][pool[i]][:300]!r}")
        for name, title, other in (("clip", "CLIP text", "bge"), ("bge", "BGE-M3", "clip")):
            print(f"  {title} top-{k}:")
            vec = d[name][pool]
            for j in result["top"][name][i]:
                print(line(i, int(j), float(vec[i] @ vec[j]), sets[other], sets["img"]))
        print()

def main() -> None:
    cfg = Config()
    ap = argparse.ArgumentParser(description="Compare CLIP-text and BGE-M3 description rankings with each other and with image retrieval")
    ap.add_argument("--tsv", default=cfg.tsv_path, help="TSV written by prep_data / exclude_types")
    ap.add_argument("--k", type=int, default=5, help="matches per target (default 5)")
    ap.add_argument("--examples", type=int, default=3, help="targets to print as examples (default 3)")
    ap.add_argument("--seed", type=int, default=cfg.seed, help="seed for choosing the examples")
    ap.add_argument("--text-cache", default=None, help="description .npz from embed_text.py (default: next to the TSV)")
    ap.add_argument("--image-cache", default=None, help="image .npz from embed.py (default: next to the TSV)")
    args = ap.parse_args()
 
    tsv = Path(args.tsv)
    text_path = Path(args.text_cache) if args.text_cache else text_cache_path(tsv, cfg.model_name, cfg.pretrained)
    image_path = Path(args.image_cache) if args.image_cache else image_cache_path(str(tsv), cfg.model_name, cfg.pretrained)
    for path, script in ((text_path, "embed_text.py"), (image_path, "embed.py")):
        if not path.exists():
            raise SystemExit(f"No cache at {path}. Run {script} first.")
 
    tiles = pd.read_csv(tsv, sep="\t", dtype={"tile_id": str})
    text = load_text_embedds(text_path)
    image = load_embeddings(image_path)
    ids = tiles["tile_id"].tolist()
    if not (ids == text.tile_ids == image.tile_ids):
        raise SystemExit("The TSV, the image cache and the description cache list different tiles or a different order. Re-run embed.py / embed_text.py with --overwrite.")
 
    d = {"clip": text.clip, "bge": text.bge, "img": image.image_embeds, "text": text.descriptions, "cut": text.clip_tokens > CLIP_CONTEXT}
    counts = Counter(t.strip().lower() for t in text.descriptions)
    unique = np.array([counts[t.strip().lower()] == 1 for t in text.descriptions])
    masks = type_masks(tiles["types"].tolist())
 
    print(f"{tsv.name}: {len(ids)} tiles; {unique.sum()} have a description that no other tile shares; CLIP cuts {d['cut'].sum()} descriptions at {CLIP_CONTEXT} tokens")
    unique_pool, all_pool = np.flatnonzero(unique), np.arange(len(ids))
    result = report("Tiles with a unique description (headline)", unique_pool, d, masks, args.k)
    report("All tiles (identical descriptions tie, so overlap is inflated)", all_pool, d, masks, args.k)
    show_examples(unique_pool, result, d, tiles, args.k, args.examples, args.seed)
 
 
if __name__ == "__main__":
    main()
