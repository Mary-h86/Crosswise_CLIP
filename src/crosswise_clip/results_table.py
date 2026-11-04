import argparse
from pathlib import Path

import pandas as pd

from crosswise_clip.config import Config
from crosswise_clip.prompt_confusion import find_runs, load_runs
from crosswise_clip.prompt_eval import FEATURES, md_table

SUMMARY_COLUMNS = ["prompt_id", "feature", "text", "k", "hits", "chance", "auc", "tiles_with_feature", "n_tiles"]
TILE_COLUMNS = ["run", "prompt_id", "feature", "wording", "text", "rank", "tile_id", "similarity", "feature_confirmed", "tile_types"]


def esc(text: str) -> str:
    """A pipe would end a markdown table cell."""
    return str(text).replace("|", "\\|")


def load_summaries(folders: list[Path], kept: pd.DataFrame) -> pd.DataFrame:
    """The prompt_summary.csv row of every kept prompt, in the order of `kept`, with the run folder it came from."""
    frames = []
    for folder in folders:
        path = folder / "prompt_summary.csv"
        if not path.exists():
            raise SystemExit(f"No {path}. Run prompt_eval.py first, or pass another --csv-dir.")
        s = pd.read_csv(path)
        absent = [c for c in SUMMARY_COLUMNS if c not in s.columns]
        if absent:
            raise SystemExit(f"{path} has no column(s): {', '.join(absent)}. Is it a prompt_summary.csv from prompt_eval.py?")
        s["run"] = folder.name
        frames.append(s)
    merged = kept.merge(pd.concat(frames, ignore_index=True), on=["run", "prompt_id"], how="left", validate="one_to_one")
    lost = merged.loc[merged["feature"].isna(), ["run", "prompt_id"]]
    if not lost.empty:
        raise SystemExit("These prompts have retrieved tiles but no row in their prompt_summary.csv: "
                         + ", ".join(f"{r}/{p}" for r, p in lost.itertuples(index=False)) + ". Re-run prompt_eval.py for them.")
    merged[["hits", "tiles_with_feature"]] = merged[["hits", "tiles_with_feature"]].astype("Int64")  # whole numbers, blank for look-only
    return merged


def order_by_feature(summary: pd.DataFrame) -> pd.DataFrame:
    """Features in the order of FEATURES (the ones you registered), then any others; the order inside a feature is kept."""
    seen = list(dict.fromkeys(summary["feature"]))
    ranked = {f: i for i, f in enumerate([f for f in FEATURES if f in seen] + [f for f in seen if f not in FEATURES])}
    return summary.assign(_f=summary["feature"].map(ranked)).sort_values("_f", kind="stable").drop(columns="_f").reset_index(drop=True)


def prompt_table(summary: pd.DataFrame) -> str:
    """One row per prompt: its feature, ID and text, hits in the top k, the hits chance gives, and the AUC (n/a for look-only prompts)."""
    ks = sorted(summary["k"].unique())
    rows = []
    for r in summary.itertuples():
        if pd.isna(r.auc):  # no labels
            rows.append([f"{r.feature} (no labels)", r.prompt_id, esc(r.text), "n/a", "n/a", "n/a"])
        else:
            rows.append([f"{r.feature} ({int(r.tiles_with_feature)} tiles)", r.prompt_id, esc(r.text),
                         f"{int(r.hits)}/{int(r.k)}", f"{r.k * r.chance:.1f}/{int(r.k)}", f"{r.auc:.2f}"])
    hits = f"Hits in top-{int(ks[0])}" if len(ks) == 1 else "Hits in top-k"
    return md_table(["Feature", "Prompt", "Text", hits, "By chance", "AUC"], rows, text_columns=3)


def feature_table(summary: pd.DataFrame) -> str:
    """One row per feature: how many prompts, the spread of their AUCs and hits, and what a random ranking gives."""
    rows = []
    for feature, g in summary.groupby("feature", sort=False):
        scored = g[g["auc"].notna()]
        if scored.empty:
            rows.append([f"{feature} (no labels)", str(len(g)), "n/a", "n/a", "n/a", "n/a"])
            continue
        auc = scored["auc"]
        auc_range = f"{auc.min():.2f}" if f"{auc.min():.2f}" == f"{auc.max():.2f}" else f"{auc.min():.2f}-{auc.max():.2f}"
        ks = scored["k"].unique()
        if len(ks) == 1:  # the usual case: hits out of the same k
            k, lo, hi = int(ks[0]), int(scored["hits"].min()), int(scored["hits"].max())
            hits, chance = (f"{lo}/{k}" if lo == hi else f"{lo}-{hi}/{k}"), f"{k * scored['chance'].iloc[0]:.1f}/{k}"
        else:  # runs with different k: compare as shares
            rate = scored["hits"].astype(float) / scored["k"]
            hits = f"{rate.min():.0%}" if rate.min() == rate.max() else f"{rate.min():.0%}-{rate.max():.0%}"
            chance = f"{scored['chance'].iloc[0]:.0%}"
        rows.append([f"{feature} ({int(scored['tiles_with_feature'].iloc[0])} tiles)", str(len(g)), auc_range, f"{auc.mean():.2f}", hits, chance])
    return md_table(["Feature", "Prompts", "AUC range", "Mean AUC", "Hits range", "By chance"], rows)


def main() -> None:
    cfg = Config()
    ap = argparse.ArgumentParser(description="One results table from every prompt_eval run")
    ap.add_argument("--csv-dir", nargs="+", default=None, help="run folder(s) written by prompt_eval.py (default: every run under runs/prompt_eval)")
    ap.add_argument("--out-dir", default=None, help="folder for the results (default: runs/results/<run folder name>, or all_runs)")
    ap.add_argument("--overwrite", action="store_true", help="replace the results already in the folder")
    args = ap.parse_args()

    folders = [Path(f) for f in args.csv_dir] if args.csv_dir else find_runs(Path(cfg.runs_dir) / "prompt_eval")
    names = [f.name for f in folders]
    if len(set(names)) < len(names):
        raise SystemExit("Two run folders have the same name (" + ", ".join(sorted({x for x in names if names.count(x) > 1}))
                         + "), so their prompts cannot be told apart. Rename one or pass fewer folders.")
    name = names[0] if len(folders) == 1 else "all_runs"
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg.runs_dir) / "results" / name
    if (out_dir / "results_table.md").exists() and not args.overwrite:
        raise SystemExit(f"{out_dir} already has results (results_table.md). Pass --overwrite to replace them, "
                         "--out-dir to write somewhere else, or --csv-dir to use a different set of runs.")

    top, dropped = load_runs(folders)
    kept = top.drop_duplicates(["run", "prompt_id"])[["run", "prompt_id"]].reset_index(drop=True)
    summary = order_by_feature(load_summaries(folders, kept))
    n_tiles = sorted(summary["n_tiles"].unique())
    if len(n_tiles) > 1:
        raise SystemExit(f"These runs used different numbers of tiles ({', '.join(str(int(n)) for n in n_tiles)}), so their results do not "
                         "belong in one table. Pass --csv-dir with the runs that used the same tiles.")

    position = {(r, p): i for i, (r, p) in enumerate(zip(summary["run"], summary["prompt_id"]))}
    tiles = top.assign(_o=[position[(r, p)] for r, p in zip(top["run"], top["prompt_id"])]).sort_values(["_o", "rank"], kind="stable")
    tiles = tiles[[c for c in TILE_COLUMNS if c in tiles.columns]]

    lines = [f"{int(n_tiles[0])} tiles. {len(summary)} prompts from {len(folders)} run folder(s)."]
    if dropped:
        lines.append(f"{dropped} prompt(s) appeared in more than one folder with the same feature and text; only the newest copy is counted.")
    if summary["k"].nunique() > 1:
        lines.append("The runs used different k (" + ", ".join(str(int(k)) for k in sorted(summary['k'].unique())) + "): each row says hits out of its own k.")
    lines += [
        f"\n## By feature\n\n{feature_table(summary)}",
        f"\n## Every prompt\n\n{prompt_table(summary)}",
        "\nAUC 0.5 and the 'by chance' column are what a random ranking gives. A tile with an unknown label counts as a miss, so the scores are "
        "probably underestimates. The prompts of one feature rank the same tiles and retrieve overlapping tiles, so the ranges describe the "
        "spread, they are not independent evidence. With few tiles per feature, a difference of about 0.05 AUC or one or two hits is within noise. "
        "Every retrieved tile, with its rank and similarity, is in results_tiles.csv.",
    ]
    text = "\n".join(lines)
    print(text)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results_table.md").write_text(text + "\n", encoding="utf-8")
    summary.to_csv(out_dir / "results_summary.csv", index=False)
    tiles.to_csv(out_dir / "results_tiles.csv", index=False)
    print(f"\nWrote results_table.md, results_summary.csv and results_tiles.csv to {out_dir}")


if __name__ == "__main__":
    main()