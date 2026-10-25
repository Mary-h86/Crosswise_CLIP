import argparse
import base64
import html
import io
import random
import re
from collections import Counter
from pathlib import Path
 
import duckdb
import pandas as pd
from PIL import Image
 
from src.crosswise_clip.config import Config
 
# Everyday words that are never highlighted: articles, prepositions, conjunctions, pronouns and
# determiners, auxiliary verbs. Every other word that two descriptions share is highlighted.
STOPWORDS = frozenset(
    """
    a an the and or but nor so yet if while although because whereas as than
    of in on at to from with without by for near between across along through into onto over under
    above below beside behind around about up down out off toward towards within via per upon
    against among beyond past
    this that these those it its they them their there here which who whom whose what where when
    each both all any some such
    is are was were be been being am has have had do does did can could will would should may might
    must also not no
    """.split()
)
WORD_SPLIT = re.compile(r"([A-Za-z][A-Za-z'-]*)")  # one capture group: split() alternates text, word, text, ...
 
CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--muted:#6b6b73;--card:#f6f6f8;--line:#d9d9df;--good:#1a7f37;--bad:#c62828;--target:#2457c5;--hl:#ffe08a}
@media (prefers-color-scheme: dark){:root{--bg:#16171a;--fg:#ececf0;--muted:#9a9aa6;--card:#202126;--line:#34353c;--good:#4cc16a;--bad:#ef6b6b;--target:#6f9bff;--hl:#6b5a1a}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}
main{max-width:1400px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 6px}
.meta{color:var(--muted);margin:0 0 24px;max-width:80ch}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:0 0 20px}
.card h2{font-size:16px;margin:0 0 10px}
.card h2 span{color:var(--muted);font-weight:400}
.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}
figure{margin:0;border:3px solid var(--line);border-radius:8px;background:var(--bg);overflow:hidden}
figure.target{border-color:var(--target)}
figure.good{border-color:var(--good)}
figure.bad{border-color:var(--bad)}
figure img,.missing{display:block;width:100%;aspect-ratio:1;object-fit:contain}
figure img{background:#111}
.missing{display:flex;align-items:center;justify-content:center;background:var(--line);color:var(--muted);font-size:13px}
figcaption{padding:6px 8px;font-size:12.5px}
.badge{font-weight:600}
figure.good .badge{color:var(--good)}
figure.bad .badge{color:var(--bad)}
.types{color:var(--muted)}
.desc{margin-top:14px}
.desc summary{cursor:pointer;font-weight:600}
.d{border-left:4px solid var(--line);border-radius:4px;background:var(--bg);margin:8px 0;padding:6px 10px}
.d.target{border-left-color:var(--target)}
.d.good{border-left-color:var(--good)}
.d.bad{border-left-color:var(--bad)}
.d b{font-size:13px}
.d p{margin:2px 0 0;white-space:pre-wrap;max-width:110ch}
.d p.none{color:var(--muted);font-style:italic}
mark{background:var(--hl);color:inherit;font-weight:600;border-radius:2px;padding:0 1px}
mark.soft{background:transparent;padding:0}
"""
 
 
def thumbnail_uri(path: str, size: int, cache: dict) -> str | None:
    """Shrink an image to at most size x size and return it as an inline data URI (None if unreadable)."""
    if path in cache:
        return cache[path]
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((size, size))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=80)
        uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except OSError:  # missing or unreadable file
        uri = None
    cache[path] = uri
    return uri
 
 
def pick_targets(tiles: pd.DataFrame, per_type: int, seed: int) -> list[str]:
    """per_type random tiles for each intersection type, reproducible for a given seed.
    A tile with several types can only be picked once."""
    rng = random.Random(seed)
    type_sets = {tid: set(t.split(",")) for tid, t in zip(tiles["tile_id"], tiles["types"])}
    chosen: list[str] = []
    for t in sorted({t for s in type_sets.values() for t in s}):
        pool = sorted(tid for tid, s in type_sets.items() if t in s and tid not in chosen)
        chosen += rng.sample(pool, min(per_type, len(pool)))
    return chosen
 
 
def load_descriptions(db: Path, tile_ids) -> dict[str, str]:
    """tile_id -> tile_description for the given tiles. Read-only; tiles with no row are just absent."""
    ids = sorted(tile_ids)
    marks = ",".join("?" * len(ids))
    with duckdb.connect(str(db), read_only=True) as con:
        rows = con.execute(f"SELECT tile_id, tile_description FROM tiles WHERE tile_id IN ({marks})", ids).fetchall()
    return {tid: desc for tid, desc in rows}
 
 
def figure(src: str | None, css_class: str, title: str, lines: list[str], badge: str = "") -> str:
    img = f'<img src="{src}" alt="{html.escape(title)}">' if src else '<div class="missing">image file not found</div>'
    text = f'<div class="badge">{html.escape(title)}</div>'
    text += "".join(f"<div>{html.escape(line)}</div>" for line in lines[:-1])
    text += f'<div class="types">{html.escape(lines[-1])}</div>'
    if badge:
        text += f'<div class="badge">{html.escape(badge)}</div>'
    return f'<figure class="{css_class}">{img}<figcaption>{text}</figcaption></figure>'
 
 
def pretty(types: str) -> str:
    return types.replace(",", ", ")
 
 
def normalize(word: str) -> str:
    """Lowercase, drop a possessive 's, and fold a plural s ("lanes" -> "lane"). Deliberately crude."""
    w = word.lower()
    if w.endswith("'s"):
        w = w[:-2]
    w = w.strip("'-")
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return w
 
 
def content_words(text: str) -> set[str]:
    """The normalized words of a description, without everyday words (STOPWORDS) or single letters."""
    found = set()
    for w in WORD_SPLIT.split(text or "")[1::2]:
        n = normalize(w)
        if len(n) > 1 and n not in STOPWORDS and w.lower() not in STOPWORDS:
            found.add(n)
    return found
 
 
def document_fractions(texts) -> dict[str, float]:
    """For each content word, the fraction of the (non-blank) descriptions that contain it."""
    docs = [content_words(t) for t in texts if (t or "").strip()]
    counts = Counter(w for d in docs for w in d)
    return {w: c / len(docs) for w, c in counts.items()}
 
 
def highlight(text: str, strong: set[str], soft: set[str]) -> str:
    """HTML-escape text. A word whose normalized form is in `strong` is wrapped in <mark> (bold on
    yellow), one in `soft` in <mark class="soft"> (bold only); everything else is left as is."""
    out = []
    for i, p in enumerate(WORD_SPLIT.split(text)):
        n = normalize(p) if i % 2 else None
        if n in strong:
            out.append(f"<mark>{html.escape(p)}</mark>")
        elif n in soft:
            out.append(f'<mark class="soft">{html.escape(p)}</mark>')
        else:
            out.append(html.escape(p))
    return "".join(out)
 
 
def description_block(
    target_id: str, matches: pd.DataFrame, descriptions: dict, fractions: dict, common_frac: float
) -> str:
    """Collapsible list of descriptions: the target first, then each match in rank order.
 
    A word is highlighted where it appears in both the target's description and a match's (in the
    target's own text: where it appears in at least one of the matches shown). Shared words that
    occur in fewer than common_frac of all descriptions get the strong highlight, the rest the soft one.
    """
 
    def text_of(tile_id: str) -> str:
        return (descriptions.get(tile_id) or "").strip()
 
    def split_tiers(words: set[str]) -> tuple[set[str], set[str]]:
        strong = {w for w in words if fractions.get(w, 0.0) < common_frac}
        return strong, words - strong
 
    def item(css_class: str, label: str, text: str, words: set[str]) -> str:
        strong, soft = split_tiers(words)
        body = f"<p>{highlight(text, strong, soft)}</p>" if text else '<p class="none">no description recorded</p>'
        return f'<div class="d {css_class}"><b>{html.escape(label)}</b>{body}</div>'
 
    target_text = text_of(target_id)
    target_words = content_words(target_text)
    shared_with = {m["retrieved_id"]: target_words & content_words(text_of(m["retrieved_id"])) for _, m in matches.iterrows()}
    in_any_match = set().union(*shared_with.values()) if shared_with else set()
 
    parts = [item("target", f"Target · tile {target_id}", target_text, in_any_match)]
    for _, m in matches.iterrows():
        shares = bool(m["shares_type"])
        shared = shared_with[m["retrieved_id"]]
        n_strong, n_all = len(split_tiers(shared)[0]), len(shared)
        label = (
            f"#{int(m['rank'])} · tile {m['retrieved_id']} · "
            + ("✓ shares a type" if shares else "✗ no shared type")
            + f" · {n_strong} distinctive / {n_all} shared word{'' if n_all == 1 else 's'}"
        )
        parts.append(item("good" if shares else "bad", label, text_of(m["retrieved_id"]), shared))
    return f'<details class="desc" open><summary>Descriptions</summary>{"".join(parts)}</details>'
 
 
def render_card(
    target_id: str,
    matches: pd.DataFrame,
    paths: dict,
    size: int,
    cache: dict,
    descriptions: dict | None = None,
    fractions: dict | None = None,
    common_frac: float = 0.25,
) -> str:
    """One card: the target, then its matches in rank order, then (optionally) their descriptions."""
    n_share = int(matches["shares_type"].sum())
    figs = [
        figure(
            thumbnail_uri(paths[target_id], size, cache),
            "target",
            "Target",
            [f"tile {target_id}", pretty(matches.iloc[0]["target_types"])],
        )
    ]
    for _, m in matches.iterrows():
        shares = bool(m["shares_type"])
        figs.append(
            figure(
                thumbnail_uri(paths[m["retrieved_id"]], size, cache),
                "good" if shares else "bad",
                f"#{int(m['rank'])}  similarity {m['similarity']:.3f}",
                [f"tile {m['retrieved_id']}", pretty(m["retrieved_types"])],
                "✓ shares a type" if shares else "✗ no shared type",
            )
        )
    head = (
        f"Target {html.escape(target_id)} "
        f'<span>· {n_share} of {len(matches)} matches share a type with it</span>'
    )
    desc = (
        description_block(target_id, matches, descriptions, fractions or {}, common_frac)
        if descriptions is not None
        else ""
    )
    return f'<section class="card"><h2>{head}</h2><div class="row">{"".join(figs)}</div>{desc}</section>'
 
 
def main() -> None:
    cfg = Config()
 
    ap = argparse.ArgumentParser(description="HTML report of targets beside their retrieved matches")
    ap.add_argument("--tsv", default=cfg.tsv_path, help="TSV written by prep_data (for image paths)")
    ap.add_argument("--csv", default=None, help="retrieval CSV from retrieve_images.py (default: the one for the default k)")
    ap.add_argument("--k", type=int, default=5, help="matches shown per target")
    ap.add_argument("--targets", nargs="+", default=None, help="tile IDs to show (default: random tiles per type)")
    ap.add_argument("--per-type", type=int, default=2, help="random targets per type when --targets is not given")
    ap.add_argument("--seed", type=int, default=cfg.seed, help="seed for the random targets")
    ap.add_argument("--db", default=None, help="DuckDB file with tile_description (default: annotations_per_tile.duckdb next to the TSV)")
    ap.add_argument("--no-descriptions", action="store_true", help="leave the descriptions out of the report")
    ap.add_argument(
        "--common-frac",
        type=float,
        default=0.25,
        help="a shared word is 'common' (bold only) if it appears in at least this fraction of all descriptions; "
        "rarer shared words get the strong highlight (default 0.25)",
    )
    ap.add_argument("--thumb", type=int, default=320, help="longest side of embedded thumbnails, in pixels")
    ap.add_argument("--out", default=None, help="output HTML (default: next to the TSV)")
    args = ap.parse_args()
 
    tsv = Path(args.tsv)
    csv = Path(args.csv) if args.csv else tsv.parent / f"retrieval_{cfg.model_name}_{cfg.pretrained}_{tsv.stem}_k{max(cfg.ks)}.csv"
    if not csv.exists():
        raise SystemExit(f"No retrieval CSV at {csv}. Run retrieve_images.py first, or pass --csv.")
 
    table = pd.read_csv(csv, dtype={"target_id": str, "retrieved_id": str})
    tiles = pd.read_csv(tsv, sep="\t", dtype={"tile_id": str})
    paths = dict(zip(tiles["tile_id"], tiles["filepath"]))
 
    if not 0 < args.common_frac <= 1:
        raise SystemExit(f"--common-frac must be above 0 and at most 1, got {args.common_frac}")
    if args.k < 1 or args.k > table["rank"].max():
        raise SystemExit(f"--k must be between 1 and {table['rank'].max()} (the ranks in {csv.name}), got {args.k}")
 
    targets = args.targets if args.targets else pick_targets(tiles, args.per_type, args.seed)
    unknown = [t for t in targets if t not in set(table["target_id"])]
    if unknown:
        raise SystemExit(f"Not targets in {csv.name}: {', '.join(unknown)}")
 
    shown = {tid: table[(table["target_id"] == tid) & (table["rank"] <= args.k)].sort_values("rank") for tid in targets}
 
    descriptions = None
    fractions: dict[str, float] = {}
    wanted: set[str] = set()
    if not args.no_descriptions:
        db = Path(args.db) if args.db else tsv.parent / "annotations_per_tile.duckdb"
        if not db.exists():
            raise SystemExit(f"No DuckDB file at {db}. Pass --db, or use --no-descriptions.")
        wanted = set(targets) | {rid for m in shown.values() for rid in m["retrieved_id"]}
        # Every tile's description is loaded, not just the ones shown: how common a word is is judged across all tiles
        descriptions = load_descriptions(db, wanted | set(tiles["tile_id"]))
        fractions = document_fractions(descriptions.get(t) for t in tiles["tile_id"])
 
    cache: dict = {}
    cards = [render_card(tid, shown[tid], paths, args.thumb, cache, descriptions, fractions, args.common_frac) for tid in targets]
 
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Image retrieval</title><style>{CSS}</style></head><body><main>"
        "<h1>Image-to-image retrieval</h1>"
        f'<p class="meta">{html.escape(cfg.model_name)} ({html.escape(cfg.pretrained)}) zero-shot. '
        f"Top {args.k} matches by image cosine similarity for {len(targets)} target tiles. "
        "Blue border: the target. Green: the match shares at least one intersection type with the target. "
        "Red: it shares none. Type agreement says nothing about whether two tiles look alike, so judge by eye too."
        + (
            " Under each card, the Descriptions section (click its title to collapse) shows the target's "
            "description followed by each match's, in rank order. Words that appear in both the target's "
            "description and the match's are highlighted: bold on yellow if the word is uncommon (under "
            f"{args.common_frac:.0%} of all tile descriptions), bold only if it is common. Everyday words like "
            "\"the\" and \"of\" are ignored, and matching is by exact word (plurals folded), so synonyms are not caught."
            if descriptions is not None
            else ""
        )
        + "</p>"
        + "".join(cards)
        + "</main></body></html>"
    )
 
    out = Path(args.out) if args.out else tsv.parent / f"retrieval_report_{cfg.model_name}_{cfg.pretrained}_{tsv.stem}_k{args.k}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
 
    n_missing = sum(1 for v in cache.values() if v is None)
    print(f"Wrote {len(targets)} cards ({args.k} matches each) to {out} ({out.stat().st_size / 1e6:.1f} MB)")
    if descriptions is not None:
        n_blank = sum(1 for t in wanted if not (descriptions.get(t) or "").strip())
        print(f"Descriptions: {len(wanted) - n_blank} of {len(wanted)} tiles shown have one; {n_blank} show 'no description recorded'")
        common = sorted(((f, w) for w, f in fractions.items() if f >= args.common_frac), reverse=True)
        n_docs = sum(1 for t in tiles["tile_id"] if (descriptions.get(t) or "").strip())
        print(f"Common words (in at least {args.common_frac:.0%} of the {n_docs} descriptions, so bold only): "
              + (", ".join(f"{w} {f:.0%}" for f, w in common[:25]) or "none")
              + (f" ... and {len(common) - 25} more" if len(common) > 25 else ""))
    if n_missing:
        print(f"Warning: {n_missing} image file(s) could not be read and show as placeholders: "
              + ", ".join(p for p, v in cache.items() if v is None))
 
 
if __name__ == "__main__":
    main()
