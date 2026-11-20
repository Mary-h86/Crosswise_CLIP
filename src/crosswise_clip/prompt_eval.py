import argparse
import shutil
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch

from crosswise_clip.config import Config
from crosswise_clip.embed_image import default_out_path as image_cache_path
from crosswise_clip.embed_image import load_embeddings, load_model

CLIP_CONTEXT = 77  # CLIP's text window, including the start and end tokens
PROMPT_COLUMNS = ["prompt_id", "feature", "wording", "text"]

# How each feature is read from the database.
#   "count": the tile has it if confirmed_<class>_count is above 0.
#   "label": the tile has it if `value` is in the list column.
#   "none":  no labels exist. The prompt is run and its tiles are shown, but not scored.
# NULL, [] and 0 all mean "not confirmed". They do not prove the feature is absent.
FEATURES = {
    "crosswalk": {"kind": "count", "column": "confirmed_crosswalk_count"},
    "bike lane": {"kind": "count", "column": "confirmed_bike_lane_count"},
    "median": {"kind": "count", "column": "confirmed_median_count"},
    "T-shape": {"kind": "label", "column": "intersection_shape", "value": "T-Shape"},
    "roundabout": {"kind": "none"},
}


def labels(value) -> list[str]:
    """A VARCHAR[] value as a list of strings. NULL (None, NaN or pandas NA) becomes []."""
    missing = value is None or value is pd.NA or (isinstance(value, float) and value != value)
    return [] if missing else [str(x) for x in value]


def has_feature(df: pd.DataFrame, rule: dict) -> np.ndarray:
    """(n,) bool: which tiles have the feature confirmed."""
    column = rule["column"]
    if column not in df.columns:
        raise SystemExit(f"The tiles table has no column '{column}'.")
    if rule["kind"] == "count":
        return (pd.to_numeric(df[column], errors="coerce").fillna(0) > 0).to_numpy()
    return np.array([rule["value"] in labels(v) for v in df[column]], dtype=bool)


def load_annotations(tiles: pd.DataFrame, db: Path) -> pd.DataFrame:
    """The database rows of the tiles, in TSV order. Tiles missing from the database get empty values."""
    with duckdb.connect(str(db), read_only=True) as con:
        con.register("ids", tiles[["tile_id"]])
        rows = con.execute("SELECT t.* FROM tiles t JOIN ids USING (tile_id)").df()
    if rows["tile_id"].duplicated().any():
        raise SystemExit("The tiles table has more than one row for some tile_id; check before using these numbers.")
    if len(rows) < len(tiles):
        print(f"Warning: {len(tiles) - len(rows)} tile(s) are not in {db.name}; they count as having no features.")
    return tiles[["tile_id"]].merge(rows, on="tile_id", how="left")


def load_prompts(path: Path) -> pd.DataFrame:
    """prompts.csv with the columns prompt_id, feature, wording, text. Stops on anything that would make results unclear."""
    prompts = pd.read_csv(path, dtype=str, keep_default_na=False)
    prompts.columns = [c.strip() for c in prompts.columns]
    absent = [c for c in PROMPT_COLUMNS if c not in prompts.columns]
    if absent:
        raise SystemExit(f"{path.name} has no column(s): {', '.join(absent)}. Expected: {', '.join(PROMPT_COLUMNS)}.")
    prompts = prompts[PROMPT_COLUMNS].apply(lambda col: col.str.strip())
    if prompts.empty:
        raise SystemExit(f"{path.name} has no prompts.")
    if (prompts["text"] == "").any() or (prompts["prompt_id"] == "").any():
        raise SystemExit(f"{path.name} has a row with an empty prompt_id or text.")
    if prompts["prompt_id"].duplicated().any():
        raise SystemExit(f"{path.name} repeats a prompt_id: {', '.join(prompts.loc[prompts['prompt_id'].duplicated(), 'prompt_id'])}.")
    unknown = sorted(set(prompts["feature"]) - set(FEATURES))
    if unknown:
        raise SystemExit(f"{path.name} names feature(s) with no rule in FEATURES: {', '.join(unknown)}. Known: {', '.join(FEATURES)}.")
    return prompts


@torch.no_grad()
def encode_prompts(model, tokenizer, texts: list[str], device: str) -> tuple[np.ndarray, np.ndarray]:
    """CLIP text vectors (P, D), L2-normalized, and each prompt's token count before CLIP's 77-token cut."""
    f = model.encode_text(tokenizer(texts).to(device))
    f = f / f.norm(dim=-1, keepdim=True)
    counts = np.array([len(tokenizer.encode(t)) + 2 for t in texts])  # + start and end tokens
    return f.cpu().numpy().astype(np.float32), counts


def rank_tiles(scores: np.ndarray) -> np.ndarray:
    """(n,): tile indices from most to least similar. Scores are rounded first so float noise cannot break a tie; ties go to the lower index."""
    return np.argsort(-np.round(scores, 6), kind="stable")


def md_table(headers: list[str], rows: list[list[str]], text_columns: int = 1) -> str:
    """Markdown table. The first `text_columns` columns are left-aligned, the rest right-aligned (numbers)."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * text_columns + "---:|" * (len(headers) - text_columns)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    cfg = Config()
    ap = argparse.ArgumentParser(description="Text-to-image retrieval with short prompts, scored against the annotations")
    ap.add_argument("--tsv", default=cfg.tsv_path, help="TSV written by prep_data / exclude_types")
    ap.add_argument("--db", default=None, help="DuckDB file (default: annotations_per_tile.duckdb next to the TSV)")
    ap.add_argument("--prompts", default=None, help="prompts.csv (default: next to the TSV)")
    ap.add_argument("--image-cache", default=None, help="image .npz from embed.py (default: next to the TSV)")
    ap.add_argument("--k", type=int, default=10, help="tiles checked per prompt (default 10)")
    ap.add_argument("--show", type=int, default=5, help="top tiles to list for the best and worst prompt (default 5)")
    ap.add_argument("--out-dir", default=None, help="folder for the results (default: runs/prompt_eval/<name of the prompts file>)")
    ap.add_argument("--overwrite", action="store_true", help="replace the results already in the folder")
    ap.add_argument("--device", default=cfg.device, help="cuda, mps or cpu (default: the best available). Try cpu if mps fails")
    args = ap.parse_args()

    tsv = Path(args.tsv)
    db = Path(args.db) if args.db else tsv.parent / "annotations_per_tile.duckdb"
    prompts_path = Path(args.prompts) if args.prompts else tsv.parent / "prompts.csv"
    image_path = Path(args.image_cache) if args.image_cache else image_cache_path(str(tsv), cfg.model_name, cfg.pretrained)
    for path, what in ((db, "DuckDB file (--db)"), (prompts_path, "prompts file (--prompts)"), (image_path, "image cache (run embed.py first)")):
        if not path.exists():
            raise SystemExit(f"No {what} at {path}.")
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg.runs_dir) / "prompt_eval" / prompts_path.stem
    if (out_dir / "prompt_top_tiles.csv").exists() and not args.overwrite:  # checked now, before the slow part
        raise SystemExit(f"{out_dir} already has results (prompt_top_tiles.csv). Pass --overwrite to replace them, "
            "--out-dir to write somewhere else, or save the prompts under another file name.")

    tiles = pd.read_csv(tsv, sep="\t", dtype={"tile_id": str})
    image = load_embeddings(image_path)
    if image.tile_ids != tiles["tile_id"].tolist():
        raise SystemExit("The TSV and the image cache list different tiles or a different order. Re-run embed.py with --overwrite.")
    n = len(tiles)
    k = n if args.k is None else args.k
    if not 1 <= k <= n:
        raise SystemExit(f"--k must be between 1 and the number of tiles ({n}).")

    prompts = load_prompts(prompts_path)
    annotations = load_annotations(tiles, db)
    truth = {}
    for feature in dict.fromkeys(prompts["feature"]):  # in the order of the file
        if FEATURES[feature]["kind"] == "none":
            truth[feature] = None  # look only: nothing to score against
            continue
        truth[feature] = has_feature(annotations, FEATURES[feature])  # shown as feature_confirmed; prompt_metrics.py does the scoring

    model, _, tokenizer = load_model(cfg.model_name, cfg.pretrained, args.device)
    vectors, tokens = encode_prompts(model, tokenizer, prompts["text"].tolist(), args.device)
    if (tokens > CLIP_CONTEXT).any():
        print(f"Warning: {int((tokens > CLIP_CONTEXT).sum())} prompt(s) are longer than {CLIP_CONTEXT} tokens, so CLIP cuts them.")
    sims = image.image_embeds.astype(np.float64) @ vectors.astype(np.float64).T  # (n tiles, n prompts)

    tops = {}
    for j, p in enumerate(prompts.itertuples()):
        positive, scores = truth[p.feature], sims[:, j]
        order = rank_tiles(scores)[:k]
        tops[p.prompt_id] = pd.DataFrame({
            "prompt_id": p.prompt_id, "feature": p.feature, "wording": p.wording, "text": p.text, "rank": np.arange(1, k + 1),
            "tile_id": tiles["tile_id"].to_numpy()[order], "similarity": np.round(scores[order], 4),
            "feature_confirmed": [None] * k if positive is None else positive[order],  # blank for a look-only prompt
            "tile_types": tiles["types"].to_numpy()[order]})
    labeled = {f: m for f, m in truth.items() if m is not None}
    print(f"{tsv.name}: {n} tiles, {len(prompts)} prompts, top {k} saved per prompt, {cfg.model_name} ({cfg.pretrained})")
    if labeled:
        print("Tiles with the feature confirmed (shown as feature_confirmed; prompt_metrics.py decides what counts as a hit): "
              + ", ".join(f"{f} {int(m.sum())} ({100 * m.mean():.1f}%)" for f, m in labeled.items()))
    if len(labeled) < len(truth):
        print("No labels, so retrieved but not scored: " + ", ".join(f for f, m in truth.items() if m is None))
    if k < n:
        print(f"Warning: only the top {k} of {n} tiles are saved per prompt, so prompt_metrics.py cannot score this run. Leave out --k to save them all.")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(tops.values(), ignore_index=True).to_csv(out_dir / "prompt_top_tiles.csv", index=False)
    shutil.copyfile(prompts_path, out_dir / "prompts_used.csv")  # so the folder says which prompts made it
    print(f"\nWrote {out_dir / 'prompt_top_tiles.csv'} (one row per prompt and ranked tile)")
    print(f"Next: uv run python prompt_metrics.py --csv-dir {out_dir}")

if __name__ == "__main__":
    main()