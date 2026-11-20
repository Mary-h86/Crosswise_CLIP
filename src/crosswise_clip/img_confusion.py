import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from crosswise_clip.config import Config
from crosswise_clip.eval import type_masks

def membership(types: list[str]) -> tuple[list[str], np.ndarray]:
    """Type names and an (N, T) 0/1 matrix: 1 where the tile has that type."""
    masks = type_masks(types)
    return list(masks), np.stack(list(masks.values()), axis=1).astype(np.float64)
 
def confusion_counts(member: np.ndarray, target_idx: np.ndarray, match_idx: np.ndarray) -> np.ndarray:
    """(T, T): cell [a, b] = number of (target, match) pairs where the target has type a and the match has type b."""
    return member[target_idx].T @ member[match_idx]
 
def chance_counts(member: np.ndarray, k: int) -> np.ndarray:
    """(T, T): the same counts expected if every target were given k random OTHER tiles."""
    n = member.shape[0]
    others = member.sum(axis=0)[None, :] - member  # (N, T): tiles of each type other than the target itself
    return member.T @ (k * others / (n - 1))
 
def row_percent(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    return 100 * np.divide(counts, totals, out=np.zeros_like(counts, dtype=np.float64), where=totals > 0)
 
def short_names(names: list[str]) -> list[str]:
    """'Flow Intersection' -> 'Flow' for display, unless that would make two names identical."""
    short = [n.replace(" Intersection", "") for n in names]
    return short if len(set(short)) == len(short) else names
 
def markdown_table(df: pd.DataFrame, fmt) -> str:
    head = "| target type | " + " | ".join(df.columns) + " |"
    rule = "|---|" + "---:|" * len(df.columns)
    rows = ["| " + str(idx) + " | " + " | ".join(fmt(v) for v in row) + " |" for idx, row in zip(df.index, df.to_numpy())]
    return "\n".join([head, rule, *rows])
 
def main() -> None:
    cfg = Config()
    ap = argparse.ArgumentParser(description="Confusion matrix of image retrieval (target types x retrieved types)")
    ap.add_argument("--tsv", default=cfg.tsv_path, help="TSV written by prep_data (tile types)")
    ap.add_argument("--csv", default=None, help="retrieval CSV from retrieval.py (default: the one for the default k)")
    ap.add_argument("--k", type=int, default=5, help="matches per target to count (default 5)")
    args = ap.parse_args()
 
    tsv = Path(args.tsv)
    csv = Path(args.csv) if args.csv else tsv.parent / f"retrieval_{cfg.model_name}_{cfg.pretrained}_{tsv.stem}_k{max(cfg.ks)}.csv"
    if not csv.exists():
        raise SystemExit(f"No retrieval CSV at {csv}. Run retrieval.py first, or pass --csv.")
 
    tiles = pd.read_csv(tsv, sep="\t", dtype={"tile_id": str})
    table = pd.read_csv(csv, dtype={"target_id": str, "retrieved_id": str})
    tile_ids = tiles["tile_id"].tolist()
    n = len(tile_ids)
 
    if args.k < 1 or args.k > table["rank"].max() or args.k >= n:
        raise SystemExit(f"--k must be between 1 and {min(table['rank'].max(), n - 1)}, got {args.k}")
    if set(table["target_id"]) != set(tile_ids) or not set(table["retrieved_id"]) <= set(tile_ids):
        raise SystemExit(f"{csv.name} and {tsv.name} are for different tiles. Re-run retrieval.py.")
 
    top = table[table["rank"] <= args.k]
    if not (top.groupby("target_id").size() == args.k).all():
        raise SystemExit(f"{csv.name} does not have exactly {args.k} matches for every target.")
 
    names, member = membership(tiles["types"].tolist())
    pos = {t: i for i, t in enumerate(tile_ids)}
    counts = confusion_counts(member, top["target_id"].map(pos).to_numpy(), top["retrieved_id"].map(pos).to_numpy())
    observed = row_percent(counts)
    chance = row_percent(chance_counts(member, args.k))
    difference = observed - chance
 
    labels = short_names(names)
    frames = {
        "counts": pd.DataFrame(counts, index=labels, columns=labels),
        "row_pct": pd.DataFrame(observed, index=labels, columns=labels),
        "difference": pd.DataFrame(difference, index=labels, columns=labels),
    }
    n_targets = member.sum(axis=0).astype(int)  # tiles carrying each type
 
    base = tsv.parent / f"confusion_{cfg.model_name}_{cfg.pretrained}_{tsv.stem}_k{args.k}"
    full = {key: pd.DataFrame(df.to_numpy(), index=names, columns=names) for key, df in frames.items()}
    for key, df in full.items():
        df.index.name = "target_type"
        df.to_csv(base.with_name(base.name + f"_{key}.csv"))
 
    fmts = {"counts": lambda v: f"{int(round(v))}", "row_pct": lambda v: f"{v:.1f}", "difference": lambda v: f"{v:+.1f}"}
    titles = {
        "counts": f"Counts (retrieved matches in each cell, top {args.k})",
        "row_pct": "Row percentages (each row sums to 100; diagonal = match has the target's type)",
        "difference": "Row percentage minus what random picks would give (percentage points; positive = better than chance)",
    }
    md = [f"# Confusion matrix, top-{args.k} image retrieval ({tsv.name})", "",
          "Rows: target type. Columns: type of the retrieved match. A tile with several types counts under each.", ""]
    for key in ("counts", "row_pct", "difference"):
        out = frames[key].copy()
        if key == "row_pct":  # show how many targets each row is based on
            out.index = [f"{lab} (n={c})" for lab, c in zip(labels, n_targets)]
        print(f"\n{titles[key]}:")
        print(out.to_string(float_format=lambda v: fmts[key](v)) if key != "counts" else out.astype(int).to_string())
        md += [f"## {titles[key]}", "", markdown_table(out.rename_axis(None), lambda v, f=fmts[key]: f(v)), ""]
    base.with_name(base.name + ".md").write_text("\n".join(md))
 
    print(f"\nn = tiles carrying that type (the targets in that row). Each match is counted under every type it has, so rows are not exactly n x {args.k}.")
    print(f"Wrote {base.name}_counts.csv, _row_pct.csv, _difference.csv and {base.name}.md next to the TSV")

if __name__ == "__main__":
    main()
