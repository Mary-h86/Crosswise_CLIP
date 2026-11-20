import argparse
from pathlib import Path
import pandas as pd
from crosswise_clip.config import Config

DEF_EX_TYPES = ["Transit Intersection", "Complete Intersection"]

def type_counts(types) -> dict:
    """Dict with tiles per type"""
    counts = {}
    for s in types:
        for t in s.split(","):
            counts[t] = counts.get(t, 0) + 1
    return counts

def filter_tiles(
    tiles: pd.DataFrame,
    excluded: list[str],
    drop_tiles: bool = False
):
    """Return (kept tiles, ids of dropped tiles, (for RELABELED TILES:[(tile_id, type before, type after)]))"""
    excluded_set = set(excluded)
    kept, dropped, relabeled = [], [], []

    # Iterate over the dataframe rows as tuples
    for row in tiles.itertuples(index=False):
        types = row.types.split(",")
        remaining = [t for t in types if t not in excluded_set]
        
        #
        if not remaining or (drop_tiles and len(remaining) < len(types)):
            dropped.append(row.tile_id)
            continue

        # 
        if len(remaining) < len(types):
            relabeled.append((row.tile_id, row.types, ",".join(remaining)))

        kept.append(row._replace(types=",".join(remaining)))
    return pd.DataFrame(kept, columns=list(tiles.columns)), dropped, relabeled

def main():
    cfg = Config

    ap = argparse.ArgumentParser(description="Write a copy of the tiles TSV without some intersection types")
    ap.add_argument("--tsv", default=cfg.tsv_path, help="TSV written by prep_data")
    ap.add_argument("--exclude", nargs="+", default=DEF_EX_TYPES, help="type names to remove (default: Transit and Complete Intersection)")
    ap.add_argument("--drop-tiles", action="store_true", help="drop whole tiles that carry an excluded type, not just the label")
    ap.add_argument("--out", default=None, help="output TSV (default: {name}_filtered.tsv next to the input)")
    args = ap.parse_args()

    tsv = Path(args.tsv)
    out = Path(args.out) if args.out else tsv.with_name(f"{tsv.stem}_filtered.tsv")

    tiles = pd.read_csv(tsv, sep="\t", dtype=str)
    before = type_counts(tiles["types"])
    unknown = [t for t in args.exclude if t not in before]

    kept, dropped, relabeled = filter_tiles(tiles, args.exclude, args.drop_tiles)

    after = type_counts(kept["types"])

    out.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out, sep="\t", index=False)

    print(f"Tiles: {len(tiles)} -> {len(kept)}  ({len(dropped)} dropped, {len(relabeled)} relabeled)")
    for t in sorted(before):
        print(f"  {t}: {before[t]} -> {after.get(t, 0)}")
    for tid, old, new in relabeled:
        print(f"  relabeled {tid}: {old} -> {new}")
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()