import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.intersections.configs.config import Config

N_BOOTSTRAP = 2000

def type_masks(types: list[str]) -> dict[str, np.ndarray]:
    """type name -> (N,) bool, True for the tiles that have that type. `types` holds comma-joined strings."""
    names = sorted({t for s in types for t in s.split(",")})
    return {name: np.array([name in s.split(",") for s in types]) for name in names}
 
def share_matrix(masks: dict[str, np.ndarray]) -> np.ndarray:
    """(N, N) bool: True where two tiles share at least one type (the diagonal is True)."""
    m = np.stack(list(masks.values()), axis=1).astype(np.int32)
    return (m @ m.T) > 0
 
def clip_picks(table: pd.DataFrame, tile_ids: list[str], k: int) -> np.ndarray:
    """(N, k) bool from the retrieval CSV: does the rank-1..k match of each target share a type?"""
    top = table[table["rank"] <= k]
    wide = top.pivot(index="target_id", columns="rank", values="shares_type").reindex(tile_ids)
    if wide.isna().any().any() or wide.shape[1] != k:
        raise ValueError(f"the retrieval CSV does not have exactly {k} matches for every target")
    return wide.to_numpy(dtype=bool)
 
def random_picks(share: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """(N, k) bool: for each target, whether k randomly chosen OTHER tiles share a type with it."""
    n = share.shape[0]
    r = rng.random((n, n))
    np.fill_diagonal(r, 2.0)  # never pick the target itself
    chosen = np.argpartition(r, k, axis=1)[:, :k]
    return share[np.arange(n)[:, None], chosen]
 
def bootstrap_interval(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """95% interval for the mean of `values`, resampling the targets with replacement."""
    means = rng.choice(values, size=(N_BOOTSTRAP, len(values))).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)
 
def summarize(picks: np.ndarray, rand: np.ndarray, masks: dict[str, np.ndarray], rng: np.random.Generator) -> pd.DataFrame:
    """picks: (N, k) CLIP results. rand: (trials, N, k) random results. One row per overall / type / rank group."""
    k = picks.shape[1]
    per_target = picks.mean(axis=1)  # (N,)
    rand_per_target = rand.mean(axis=2)  # (trials, N)
 
    def row(section: str, group: str, clip_values: np.ndarray, rand_values: np.ndarray) -> dict:
        lo, hi = bootstrap_interval(clip_values, rng)
        return {
            "section": section,
            "group": group,
            "n_targets": len(clip_values),
            "clip": float(clip_values.mean()),
            "clip_lo": lo,
            "clip_hi": hi,
            "random_mean": float(rand_values.mean()),
            "random_std": float(rand_values.mean(axis=1).std()),
            "difference": float(clip_values.mean() - rand_values.mean()),
        }
 
    rows = [row("overall", "all targets", per_target, rand_per_target)]
    for name, mask in masks.items():
        rows.append(row("by target type", name, per_target[mask], rand_per_target[:, mask]))
    for r in range(k):
        rows.append(row("by rank", f"rank {r + 1}", picks[:, r].astype(float), rand[:, :, r].astype(float)))
    return pd.DataFrame(rows)
 
 
def show(summary: pd.DataFrame, k: int, trials: int) -> None:
    pct = lambda x: f"{100 * x:5.1f}%"
    for section in ("overall", "by target type", "by rank"):
        part = summary[summary["section"] == section]
        out = pd.DataFrame(
            {
                "group": part["group"],
                "n": part["n_targets"],
                "CLIP": part["clip"].map(pct),
                "95% interval": [f"{pct(a)} to {pct(b)}" for a, b in zip(part["clip_lo"], part["clip_hi"])],
                "random": [f"{pct(m)} +/- {100 * s:.1f}" for m, s in zip(part["random_mean"], part["random_std"])],
                "CLIP - random": [f"{100 * d:+.1f} pts" for d in part["difference"]],
            }
        )
        print(f"\n{section}:")
        print(out.to_string(index=False))
    print(
        f"\nAgreement = share of the top-{k} matches sharing at least one type with the target. Random baseline: "
        f"{trials} draws of {k} random other tiles per target. 'random +/-' is the spread across draws only; the CLIP "
        "interval (bootstrap over targets) is the one that widens for small groups, so read tiny groups with care."
    )
 
def main() -> None:
    cfg = Config()
    ap = argparse.ArgumentParser(description="Type agreement of image retrieval vs a random baseline")
    ap.add_argument("--tsv", default=cfg.tsv_path, help="TSV written by prep_data (tile types)")
    ap.add_argument("--csv", default=None, help="retrieval CSV from retrieve_images.py (default: the one for the default k)")
    ap.add_argument("--k", type=int, default=5, help="matches per target to score (default 5)")
    ap.add_argument("--trials", type=int, default=100, help="random-baseline draws (default 100)")
    ap.add_argument("--seed", type=int, default=cfg.seed, help="seed for the random baseline and bootstrap")
    ap.add_argument("--out", default=None, help="summary CSV path (default: next to the TSV)")
    args = ap.parse_args()
 
    tsv = Path(args.tsv)
    csv = Path(args.csv) if args.csv else tsv.parent / f"retrieval_{cfg.model_name}_{cfg.pretrained}_{tsv.stem}_k{max(cfg.ks)}.csv"
    if not csv.exists():
        raise SystemExit(f"No retrieval CSV at {csv}. Run retrieve_images.py first, or pass --csv.")
    if args.trials < 1:
        raise SystemExit(f"--trials must be at least 1, got {args.trials}")
 
    tiles = pd.read_csv(tsv, sep="\t", dtype={"tile_id": str})
    table = pd.read_csv(csv, dtype={"target_id": str, "retrieved_id": str})
    tile_ids = tiles["tile_id"].tolist()
    n = len(tile_ids)
 
    if args.k < 1 or args.k > table["rank"].max() or args.k >= n:
        raise SystemExit(f"--k must be between 1 and {min(table['rank'].max(), n - 1)}, got {args.k}")
    if set(table["target_id"]) != set(tile_ids):
        raise SystemExit(f"{csv.name} and {tsv.name} are for different tiles. Re-run retrieve_images.py.")
 
    masks = type_masks(tiles["types"].tolist())
    share = share_matrix(masks)
 
    # The CSV's shares_type must agree with the types in the TSV, or the two files are out of step
    pos = {t: i for i, t in enumerate(tile_ids)}
    recomputed = share[table["target_id"].map(pos).to_numpy(), table["retrieved_id"].map(pos).to_numpy()]
    if not (recomputed == table["shares_type"].to_numpy(dtype=bool)).all():
        raise SystemExit(f"shares_type in {csv.name} does not match the types in {tsv.name}. Re-run retrieve_images.py.")
 
    try:
        picks = clip_picks(table, tile_ids, args.k)
    except ValueError as e:
        raise SystemExit(str(e))
 
    rng = np.random.default_rng(args.seed)
    rand = np.stack([random_picks(share, args.k, rng) for _ in range(args.trials)])  # (trials, N, k)
    summary = summarize(picks, rand, masks, rng)
 
    # Per-target file: how many of its k matches share a type, and the exact chance level for that target
    chance = (share.sum(axis=1) - 1) / (n - 1)
    per_target = pd.DataFrame(
        {
            "target_id": tile_ids,
            "target_types": tiles["types"],
            "matches_sharing_type": picks.sum(axis=1),
            "k": args.k,
            "agreement": picks.mean(axis=1),
            "chance": chance,
        }
    )
 
    out = Path(args.out) if args.out else tsv.parent / f"type_agreement_{cfg.model_name}_{cfg.pretrained}_{tsv.stem}_k{args.k}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    per_target_path = out.with_name(out.stem + "_per_target.csv")
    per_target.to_csv(per_target_path, index=False)
 
    print(f"Type agreement @{args.k} for {n} targets, from {csv.name}")
    show(summary, args.k, args.trials)
 
    none = per_target[per_target["matches_sharing_type"] == 0]["target_id"].tolist()
    allk = per_target[per_target["matches_sharing_type"] == args.k]["target_id"].tolist()
    print(f"\nTargets with none of their {args.k} matches sharing a type: {len(none)}" + (f" (e.g. {', '.join(none[:5])})" if none else ""))
    print(f"Targets with all {args.k} matches sharing a type: {len(allk)}" + (f" (e.g. {', '.join(allk[:5])})" if allk else ""))
    print("Pass IDs from these lists to visualize_retrieval.py --targets to look at the best and worst cases.")
    print(f"\nWrote {out}\nWrote {per_target_path}")
 
if __name__ == "__main__":
    main()
