"""For Prompt -> Image retrieval: Precision, Recall, and F1"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from crosswise_clip.config import Config
from crosswise_clip.prompt_eval import FEATURES, has_feature, labels, load_annotations

LIST_COLUMNS = ["roadway_type", "street_typology", "land_use", "building_frontage", "intersection_shape", "crossing_control", "treatment"]
COLUMNS = ["prompt_id", "feature", "wording", "text", "n_tiles", "n_positives", "base_rate", "k", "precision", "recall", "f1",
           "best_f1", "best_k", "f1_retrieve_all"]

def outcome_counts(
    retrieved: set[str],
    positives: set[str], 
    all_tiles: set[str], # Cleaned to exclude tiles w/o labels
) -> tuple[int, int, int ,int]:  
    """
    Compute TP, FP, TN, FN (tuple returned in that order)

    Currently ignores tiles that don't have any labels
    """
    TP, FP, TN, FN = 0, 0, 0, 0 
    for tile in all_tiles:
        if tile in retrieved:
            if tile in positives:
                TP += 1
            else:
                FP += 1
        else: # tile not retrieved
            if tile not in positives:
                TN += 1
            else:
                FN += 1

    return TP, FP, TN, FN

def metrics(
    outcomes: tuple[int, int, int, int],
) -> tuple[float, float, float]:
    """
    Compute Precision, Recall, F1 (tuple returned in that order)

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 =  2TP / (2TP + FP + FN)
    """
    TP, FP, TN, FN = outcomes
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * TP / (2 * TP + FP + FN)

    return precision, recall, f1

def is_annotated(annotations: pd.DataFrame) -> np.ndarray:
    """(n,) bool: the tile has at least one annotation field filled in."""
    filled = [np.array([len(labels(v)) > 0 for v in annotations[c]]) for c in LIST_COLUMNS if c in annotations.columns]
    counts = [(pd.to_numeric(annotations[c], errors="coerce").fillna(0) > 0).to_numpy()
              for c in annotations.columns if c.startswith("confirmed_") and c.endswith("_count")]
    if not filled + counts:
        raise SystemExit("The database has none of the annotation columns, so no tile can count as annotated.")
    return np.logical_or.reduce(filled + counts)


def sweep(ranking: list[str], positives: set, universe: set) -> pd.DataFrame:
    """One row per k: the counts, precision, recall and F1 when the top k of `ranking` (every tile of `universe`, best first) is retrieved.

    metrics() never divides by 0 here: k >= 1, and only features with at least one positive are swept."""
    if len(ranking) != len(universe):
        raise ValueError("The ranking must list every tile of the universe once.")
    rows = []
    for k in range(1, len(ranking) + 1):
        tp, fp, tn, fn = outcome_counts(set(ranking[:k]), positives, universe)
        precision, recall, f1 = metrics((tp, fp, tn, fn))
        rows.append({"k": k, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = Config()
    ap = argparse.ArgumentParser(description="Precision, recall and F1 of each prompt's retrieval over the whole ranking")
    ap.add_argument("--tsv", default=cfg.tsv_path, help="TSV the retrieval ran on")
    ap.add_argument("--db", default=None, help="DuckDB file (default: annotations_per_tile.duckdb next to the TSV)")
    ap.add_argument("--csv-dir", required=True, help="run folder written by prompt_eval.py; the results are written there too")
    ap.add_argument("--k", type=int, default=5, help="cutoff for the precision/recall/F1 columns (default 5)")
    args = ap.parse_args()

    tsv, csv_dir = Path(args.tsv), Path(args.csv_dir)
    db = Path(args.db) if args.db else tsv.parent / "annotations_per_tile.duckdb"
    tiles = pd.read_csv(tsv, sep="\t", dtype={"tile_id": str})
    top = pd.read_csv(csv_dir / "prompt_top_tiles.csv", dtype={"tile_id": str}, keep_default_na=False)
    if top.groupby("prompt_id").size().min() < len(tiles):
        raise SystemExit("prompt_top_tiles.csv does not hold every tile for every prompt. Re-run prompt_eval.py without --k.")

    annotations = load_annotations(tiles, db)
    annotated = is_annotated(annotations)
    tile_ids = tiles["tile_id"].to_numpy()
    universe = set(tile_ids[annotated])
    if not 1 <= args.k <= len(universe):
        raise SystemExit(f"--k must be between 1 and the number of annotated tiles ({len(universe)}).")
    print(f"{len(universe)} of {len(tiles)} tiles have annotations. The other {len(tiles) - len(universe)} are left out.")

    rows, curves, skipped = [], [], []
    for pid, g in top.groupby("prompt_id", sort=False):
        feature = g["feature"].iloc[0]
        row = {"prompt_id": pid, "feature": feature, "wording": g["wording"].iloc[0], "text": g["text"].iloc[0]}
        rule = FEATURES[feature]
        positives = set() if rule["kind"] == "none" else set(tile_ids[has_feature(annotations, rule) & annotated])
        if 0 < len(positives) < len(universe):
            ranking = [t for t in g.sort_values("rank")["tile_id"] if t in universe]  # unannotated tiles dropped BEFORE cutting to k
            curve = sweep(ranking, positives, universe)
            best = curve.loc[curve["f1"].idxmax()]  # the smallest k among equal F1
            row.update(n_tiles=len(universe), n_positives=len(positives), base_rate=len(positives) / len(universe), k=args.k,
                       **curve.iloc[args.k - 1][["precision", "recall", "f1"]].to_dict(),
                       best_f1=best["f1"], best_k=best["k"], f1_retrieve_all=curve["f1"].iloc[-1])
            curves.append(curve.assign(prompt_id=pid, feature=feature))
        else:
            skipped.append(f"{pid} ({feature})")
        rows.append(row)

    summary = pd.DataFrame(rows, columns=COLUMNS).astype({"n_tiles": "Int64", "n_positives": "Int64", "k": "Int64", "best_k": "Int64"})
    summary.round(4).to_csv(csv_dir / "prompt_metrics.csv", index=False)
    if curves:
        pd.concat(curves, ignore_index=True).round(4).to_csv(csv_dir / "prompt_curves.csv", index=False)
    pd.DataFrame({"tile_id": tile_ids, "annotated": annotated}).to_csv(csv_dir / "annotated_tiles.csv", index=False)

    print(summary.drop(columns=["text", "n_tiles"]).round(2).astype(object).fillna("").to_string(index=False))
    print("\nbest_f1 only means something if it clearly beats f1_retrieve_all, the F1 of returning every tile.")
    if skipped:
        print("Not scored (no labels, or no tile with the feature among the annotated ones): " + ", ".join(skipped))
    print(f"Wrote prompt_metrics.csv, prompt_curves.csv and annotated_tiles.csv to {csv_dir}")


if __name__ == "__main__":
    main()
