import argparse
import re
import statistics as st
from collections import Counter
from pathlib import Path
import open_clip
import duckdb
import pandas as pd
from crosswise_clip.config import Config

CLIP_CONTEXT = 77 # Max tokens that CLIP accepts
LIST_COLS = ["roadway_type", "street_typology", "land_use", "building_frontage", "intersection_shape", "crossing_control", "treatment"]
STREET_SUFFS = {"st", "street", "ave", "avenue", "blvd", "boulevard", "rd", "road", "dr", "drive", "way", "ln", "lane", "ct", "court", "pl", "place", "hwy", "highway"}
REQUIRED = {"tile_id", "city", "tile_description", "street_ns", "street_ew"}

def is_missing(value):
    return value is None or value is pd.NA or (isinstance(value, float) and value != value)

def text_of(value):
    return "" if is_missing(value) else str(value).strip()

def labels(value) -> list[str]:
    """VARCHAR[] value as list of strings"""
    return [] if is_missing(value) else [str(x) for x in value]

def street_core(name):
    """String street names for consitency across lookup"""
    words = name.lower().replace(".", "").split()
    while len(words) > 1 and words[-1] in STREET_SUFFS:
        words.pop()

    return " ".join(words)

def clip_lengths(
    texts: list[str],
    model_name: str
) -> list[int]:
    """Tokens clip would need for each text description"""
    tokenizer = open_clip.get_tokenizer(model_name)

    # +2 for start and end tokens
    return [len(tokenizer.encode(t)) + 2 for t in texts]

def spread(values):
    return f"min {min(values)}, median {st.median(values):.0f}, max {max(values)}"

def describe_descriptions(
    df: pd.DataFrame,
    desc: pd.Series,
    cfg: Config,
):
    has = desc != ""
    texts = desc[has].tolist()

    print(f"Descriptions ({len(df)}) tiles \n")
    print(f"w/ description: {has.sum()}, w/o description: {(~has).sum()}")

    if (~has).any():
        shown = ", ".join(df.loc[~has, "tile_id"].head(5))
        print(f"  tiles without one (first 5): {shown}")

    words = [len(t.split()) for t in texts]
    tokens = clip_lengths(texts, cfg.model_name)
    cut = [n for n in tokens if n > CLIP_CONTEXT]
    print(f"  words per description:  {spread(words)}")
    print(f"  CLIP tokens per description:  {spread(tokens)}")
    print(f"  CLIP cuts {len(cut)} of {len(tokens)} descriptions (more than {CLIP_CONTEXT} tokens)")

    counts = Counter(t.lower() for t in texts)
    repeated = {t: c for t, c in counts.items() if c > 1}
    print(f"  identical descriptions: {len(repeated)} texts shared by {sum(repeated.values())} tiles")

def describe_place_names(
    df: pd.DataFrame,
    desc: pd.Series,
):
    has = desc != ""
    with_street, own_street, with_city, city_named = 0, 0, 0, 0

    for d, ns, ew, city in zip(desc[has], df.loc[has, "street_ns"], df.loc[has, "street_ew"], df.loc[has, "city"]):
        low = d.lower()
        cores = [street_core(text_of(s)) for s in (ns, ew)]
        cores = [c for c in cores if len(c) >= 3]

        if cores:
            with_street += 1
            own_street += any(c in low for c in cores)
        name = text_of(city).lower()

        if name:
            with_city += 1
            city_named += bool(re.search(rf"\b{re.escape(name)}\b", low))

    print(f"  tiles with a recorded street name: {with_street}; their description repeats one of them: {own_street}")
    print(f"  tiles with a recorded city: {with_city}; their description names the city: {city_named}")

def describe_cities(df: pd.DataFrame):
    city = df["city"].map(text_of).replace("", "(unknown)")
    print(city.value_counts().to_string())
    exploded = df.assign(type=df["types"].str.split(","), city_name=city).explode("type").reset_index(drop=True)
    table = pd.crosstab(exploded["type"], exploded["city_name"])
    print("\n  tiles per intersection type and city (a multi-type tile counts under each of its types):")
    print(table.to_string())

def describe_features(df: pd.DataFrame, min_tiles: int) -> None:
    print(f"\n Features (labels on at least {min_tiles} tiles; NULL or [] means unrecorded, not absent)")
    n = len(df)
    for col in LIST_COLS:
        if col not in df.columns:
            print(f"\n  {col}: not in the database")
            continue
        per_tile = [set(labels(v)) for v in df[col]]
        recorded = sum(bool(s) for s in per_tile)
        counts = Counter(label for s in per_tile for label in s)
        keep = [(lab, c) for lab, c in counts.most_common() if c >= min_tiles][:12]
        print(f"\n  {col}: recorded on {recorded} of {n} tiles")
        for lab, c in keep:
            print(f"      {c:4d}  {lab}")
        if not keep:
            print(f"      (no label on {min_tiles}+ tiles)")
 
    classes = sorted(
        c[len("confirmed_"):-len("_count")]
        for c in df.columns
        if c.startswith("confirmed_") and c.endswith("_count") and c != "confirmed_object_count"
    )
    print("\n  annotated objects (tiles with at least one confirmed object of the class):")
    if "confirmed_object_count" in df.columns:
        any_obj = int((pd.to_numeric(df["confirmed_object_count"], errors="coerce").fillna(0) > 0).sum())
        print(f"      any confirmed object: {any_obj} of {n} tiles (the other tiles have every class unknown)")
    rows = [(cls, int((pd.to_numeric(df[f"confirmed_{cls}_count"], errors="coerce").fillna(0) > 0).sum())) for cls in classes]
    for cls, c in sorted(rows, key=lambda r: -r[1]):
        print(f"      {c:4d}  {cls}")

def main() -> None:
    cfg = Config()
    ap = argparse.ArgumentParser(description="Profile descriptions and features of the tiles (read-only)")
    ap.add_argument("--tsv", default=cfg.tsv_path, help="TSV written by prep_data / exclude_types")
    ap.add_argument("--db", default=None, help="DuckDB file (default: annotations_per_tile.duckdb next to the TSV)")
    ap.add_argument("--min-tiles", type=int, default=15, help="list a feature label only if at least this many tiles have it")
    args = ap.parse_args()
 
    tsv = Path(args.tsv)
    db = Path(args.db) if args.db else tsv.parent / "annotations_per_tile.duckdb"
    if not db.exists():
        raise SystemExit(f"No DuckDB file at {db}. Pass --db.")
 
    tiles = pd.read_csv(tsv, sep="\t", dtype={"tile_id": str})
    with duckdb.connect(str(db), read_only=True) as con:
        con.register("ids", tiles[["tile_id"]])
        rows = con.execute("SELECT t.* FROM tiles t JOIN ids USING (tile_id)").df()
 
    absent = REQUIRED - set(rows.columns)
    if absent:
        raise SystemExit(f"The tiles table has no column(s): {', '.join(sorted(absent))}")
    if rows["tile_id"].duplicated().any():
        raise SystemExit("The tiles table has more than one row for some tile_id; check before using these numbers.")
 
    df = tiles[["tile_id", "types"]].merge(rows, on="tile_id", how="left")
    missing = len(tiles) - len(rows)
    print(f"{tsv.name}: {len(tiles)} tiles, {len(rows)} found in {db.name}" + (f" ({missing} missing from the database)" if missing else ""))
 
    desc = df["tile_description"].map(text_of)
    describe_descriptions(df, desc, cfg)
    describe_place_names(df, desc)
    describe_cities(df)
    describe_features(df, args.min_tiles)
 
 
if __name__ == "__main__":
    main()
