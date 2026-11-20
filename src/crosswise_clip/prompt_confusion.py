import argparse
from pathlib import Path
 
import numpy as np
import pandas as pd
 
from crosswise_clip.config import Config
from crosswise_clip.prompt_eval import FEATURES, chance_tail, has_feature, load_annotations, md_table
 
NEEDED = ["prompt_id", "feature", "text", "rank", "tile_id"]
 
def find_runs(runs: Path) -> list[Path]:
    """Every folder under `runs` that holds a prompt_top_tiles.csv, oldest first."""
    found = sorted(runs.rglob("prompt_top_tiles.csv"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise SystemExit(f"No prompt_eval results under {runs}. Run prompt_eval.py first, or pass --csv-dir.")
    return [p.parent for p in found]
 
 
def load_runs(folders: list[Path]) -> tuple[pd.DataFrame, int]:
    """The retrieved tiles of every prompt in the folders, one row per prompt and tile, plus how many repeated prompts were dropped.
 
    The same feature and text always gives the same ranking, so when it appears in several folders only the newest copy is kept
    (counting it twice would tilt the pooled numbers). One prompt_id for two different texts is refused: the tables could not say which."""
    for folder in folders:  # before sorting by file time, which needs the file to exist
        if not (folder / "prompt_top_tiles.csv").exists():
            raise SystemExit(f"No {folder / 'prompt_top_tiles.csv'}. Run prompt_eval.py first, or pass another --csv-dir.")
    frames = []
    for folder in sorted(folders, key=lambda f: (f / "prompt_top_tiles.csv").stat().st_mtime):
        path = folder / "prompt_top_tiles.csv"
        top = pd.read_csv(path, dtype={"tile_id": str}, keep_default_na=False)
        absent = [c for c in NEEDED if c not in top.columns]
        if absent:
            raise SystemExit(f"{path} has no column(s): {', '.join(absent)}. Is it a prompt_top_tiles.csv from prompt_eval.py?")
        top["run"] = folder.name
        frames.append(top)
    top = pd.concat(frames, ignore_index=True)
    prompts = top.drop_duplicates(["run", "prompt_id"])[["run", "prompt_id", "feature", "text"]]
    newest = {}
    for p in prompts.itertuples():  # oldest folder first, so a later copy replaces an earlier one
        newest[(p.feature, p.text)] = (p.run, p.prompt_id)
    keep = set(newest.values())
    kept = prompts[[(r, i) in keep for r, i in zip(prompts["run"], prompts["prompt_id"])]]
    clash = kept[kept["prompt_id"].duplicated(keep=False)]
    if not clash.empty:
        what = "; ".join(f"{i} ({', '.join(g['run'] + ': ' + g['text'].str[:40])})" for i, g in clash.groupby("prompt_id", sort=False))
        raise SystemExit(f"The same prompt_id has different texts in different run folders: {what}. "
                         "Pass --csv-dir with just the folders you want, or rename one of the prompts.")
    top = top[[(r, i) in keep for r, i in zip(top["run"], top["prompt_id"])]]
    return top.reset_index(drop=True), len(prompts) - len(kept)
 
 
def hits_by_feature(top: pd.DataFrame, tile_index: dict, truth: dict, k: int, n: int) -> pd.DataFrame:
    """One row per prompt and scored feature: how many of the prompt's top-k tiles have that feature confirmed."""
    rows = []
    for pid, g in top.groupby("prompt_id", sort=False):
        g = g.sort_values("rank").head(k)
        idx = np.array([tile_index[t] for t in g["tile_id"]])
        for feature, mask in truth.items():
            hits, positives = int(mask[idx].sum()), int(mask.sum())
            rows.append({"prompt_id": pid, "run": g["run"].iloc[0], "prompted_feature": g["feature"].iloc[0], "text": g["text"].iloc[0],
                         "column_feature": feature, "k": k, "hits": hits, "share": hits / k, "tiles_with_feature": positives,
                         "base_rate": positives / n, "lift": hits / k / (positives / n), "p_chance": chance_tail(n, positives, k, hits)})
    return pd.DataFrame(rows)
 
 
def pool(by_prompt: pd.DataFrame) -> pd.DataFrame:
    """One row per prompted feature and scored feature: the mean share over that prompted feature's prompts, and its lift."""
    pooled = (by_prompt.groupby(["prompted_feature", "column_feature"], sort=False)
              .agg(n_prompts=("prompt_id", "nunique"), share=("share", "mean"), base_rate=("base_rate", "first")).reset_index())
    pooled["lift"] = pooled["share"] / pooled["base_rate"]
    return pooled
 
 
def cooccurrence(truth: dict) -> pd.DataFrame:
    """Among all tiles that have the row feature, the share that also have the column feature."""
    rows = [{"row_feature": a, "column_feature": b, "tiles_with_row_feature": int(ma.sum()),
             "share_also_with_column": float((ma & mb).sum() / ma.sum())}
            for a, ma in truth.items() for b, mb in truth.items()]
    return pd.DataFrame(rows)
 
 
def per_prompt_table(by_prompt: pd.DataFrame, columns: list[str], k: int) -> str:
    order = {f: i for i, f in enumerate(dict.fromkeys(by_prompt["prompted_feature"]))}
    wide = by_prompt.pivot(index="prompt_id", columns="column_feature", values="hits")
    info = by_prompt.drop_duplicates("prompt_id").set_index("prompt_id")
    file_order = list(dict.fromkeys(by_prompt["prompt_id"]))  # the pivot sorts its index, so start again from the order of the CSV
    ids = sorted(file_order, key=lambda p: order[info.loc[p, "prompted_feature"]])  # stable: the CSV's order inside a feature
    rows = []
    for pid in ids:
        own = info.loc[pid, "prompted_feature"]
        rows.append([pid, own] + [f"**{wide.loc[pid, c]}**" if c == own else str(wide.loc[pid, c]) for c in columns])
    base = by_prompt.drop_duplicates("column_feature").set_index("column_feature")["base_rate"]
    rows.append(["(random tiles)", ""] + [f"{k * base[c]:.1f}" for c in columns])
    return md_table(["Prompt", "Prompted feature"] + columns, rows, text_columns=2)
 
 
def pooled_table(pooled: pd.DataFrame, columns: list[str]) -> str:
    base = pooled.drop_duplicates("column_feature").set_index("column_feature")["base_rate"]
    rows = []
    for feature, g in pooled.groupby("prompted_feature", sort=False):
        g = g.set_index("column_feature")
        cells = []
        for c in columns:
            text = f"{g.loc[c, 'share']:.0%} ({g.loc[c, 'lift']:.1f}x)"
            cells.append(f"**{text}**" if c == feature else text)
        rows.append([f"{feature} ({g['n_prompts'].iloc[0]} prompt{'s' if g['n_prompts'].iloc[0] != 1 else ''})"] + cells)
    rows.append(["all tiles (base rate)"] + [f"{base[c]:.0%}" for c in columns])
    return md_table(["Prompted feature"] + columns, rows)
 
 
def cooccurrence_table(co: pd.DataFrame, columns: list[str], n: int, truth: dict) -> str:
    wide = co.pivot(index="row_feature", columns="column_feature", values="share_also_with_column")
    rows = []
    for a in columns:
        rows.append([f"{a} ({int(truth[a].sum())} tiles)"] + ["-" if a == b else f"{wide.loc[a, b]:.0%}" for b in columns])
    rows.append(["all tiles"] + [f"{truth[c].mean():.0%}" for c in columns])
    return md_table(["Tiles with"] + columns, rows)
 
 
def main() -> None:
    cfg = Config()
    ap = argparse.ArgumentParser(description="Which features do the tiles retrieved by each prompt have? A prompt x feature matrix")
    ap.add_argument("--tsv", default=cfg.tsv_path, help="TSV written by prep_data / exclude_types (the tiles the retrieval ran on)")
    ap.add_argument("--db", default=None, help="DuckDB file (default: annotations_per_tile.duckdb next to the TSV)")
    ap.add_argument("--csv-dir", nargs="+", default=None, help="run folder(s) written by prompt_eval.py (default: every run under runs/prompt_eval)")
    ap.add_argument("--k", type=int, default=None, help="use only the top k tiles of each prompt (default: all the CSVs have)")
    ap.add_argument("--out-dir", default=None, help="folder for the results (default: runs/prompt_confusion/<run folder name>)")
    ap.add_argument("--overwrite", action="store_true", help="replace the results already in the folder")
    args = ap.parse_args()
 
    tsv = Path(args.tsv)
    db = Path(args.db) if args.db else tsv.parent / "annotations_per_tile.duckdb"
    if not db.exists():
        raise SystemExit(f"No DuckDB file (--db) at {db}.")
    folders = [Path(f) for f in args.csv_dir] if args.csv_dir else find_runs(Path(cfg.runs_dir) / "prompt_eval")
    name = folders[0].name if len(folders) == 1 else "all_runs"
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg.runs_dir) / "prompt_confusion" / name
    if (out_dir / "confusion.md").exists() and not args.overwrite:
        raise SystemExit(f"{out_dir} already has results (confusion.md). Pass --overwrite to replace them, "
                         "--out-dir to write somewhere else, or --csv-dir to use a different set of runs.")
 
    top, dropped = load_runs(folders)
    tiles = pd.read_csv(tsv, sep="\t", dtype={"tile_id": str})
    tile_index = {t: i for i, t in enumerate(tiles["tile_id"])}
    unknown = sorted(set(top["tile_id"]) - set(tile_index))
    if unknown:
        raise SystemExit(f"{len(unknown)} retrieved tile(s) are not in {tsv.name} (for example {unknown[0]}). Is --tsv the one prompt_eval.py used?")
    n = len(tiles)
    counts = top.groupby("prompt_id").size()
    k = args.k if args.k is not None else int(counts.min())  # `is not None`: --k 0 must be refused, not read as "no --k"
    if k < 1 or k > counts.min():
        raise SystemExit(f"--k must be between 1 and {int(counts.min())}, the fewest tiles any prompt has saved.")
 
    annotations = load_annotations(tiles, db)
    truth, skipped = {}, []
    for feature, rule in FEATURES.items():
        if rule["kind"] == "none":
            continue  # no labels, so there is no column to count
        mask = has_feature(annotations, rule)
        if 0 < mask.sum() < n:
            truth[feature] = mask
        else:
            skipped.append(f"{feature} ({int(mask.sum())} of {n})")
    if not truth:
        raise SystemExit("No feature in FEATURES can be scored (each one is confirmed on none or on all of the tiles).")
    columns = list(truth)
 
    by_prompt = hits_by_feature(top, tile_index, truth, k, n)
    pooled, co = pool(by_prompt), cooccurrence(truth)
    n_prompts = by_prompt["prompt_id"].nunique()
 
    lines = [f"{tsv.name}: {n} tiles. {n_prompts} prompts from {len(folders)} run folder(s), top-{k} each."]
    if dropped:
        lines.append(f"{dropped} prompt(s) appeared in more than one folder with the same feature and text; only the newest copy is used.")
    if counts.nunique() > 1:
        lines.append(f"The prompts saved different numbers of tiles, so every prompt is cut to the fewest ({int(counts.min())}) unless --k is lower.")
    lines.append("Tiles with the feature confirmed (every other tile counts as not confirmed): "
                 + ", ".join(f"{f} {int(m.sum())} ({100 * m.mean():.1f}%)" for f, m in truth.items()))
    if skipped:
        lines.append("Not used as columns (confirmed on none or all of the tiles): " + ", ".join(skipped))
    look_only = [f for f in dict.fromkeys(by_prompt["prompted_feature"]) if f not in truth]
    if look_only:
        lines.append("Prompted features with no column of their own (no labels, so nothing in bold): " + ", ".join(look_only))
    lines += [
        f"\n## Per prompt: how many of the top-{k} tiles have each feature\n\nBold is the prompt's own feature. The last row is what {k} random tiles give.\n",
        per_prompt_table(by_prompt, columns, k),
        "\n## Pooled by prompted feature: share of retrieved tiles with each feature (lift)\n\n"
        "Lift = that share / the share of all tiles. Above 1 means over-represented. The bold cell is the prompts' own feature.\n",
        pooled_table(pooled, columns),
        "\n## For reference: how often the features occur together in the data\n\n"
        "Of the tiles that have the row feature, the share that also have the column feature.\n",
        cooccurrence_table(co, columns, n, truth),
        "\nHow to read it: a prompt is specific to its feature when its own (bold) lift clearly beats its lift on the other features. "
        "If another column is as high, the prompt may be retrieving whatever tends to occur with it. The third table shows how much "
        "that overlap exists in the data anyway. The wordings of one feature retrieve overlapping tiles, so pooled shares are descriptive, "
        f"not independent evidence, and top-{k} lists are small: a difference of one or two tiles is within chance.",
    ]
    text = "\n".join(lines)
    print(text)
 
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "confusion.md").write_text(text + "\n", encoding="utf-8")
    by_prompt.to_csv(out_dir / "confusion_by_prompt.csv", index=False)
    pooled.to_csv(out_dir / "confusion_pooled.csv", index=False)
    co.to_csv(out_dir / "cooccurrence.csv", index=False)
    print(f"\nWrote confusion.md, confusion_by_prompt.csv, confusion_pooled.csv and cooccurrence.csv to {out_dir}")
 
 
if __name__ == "__main__":
    main()
