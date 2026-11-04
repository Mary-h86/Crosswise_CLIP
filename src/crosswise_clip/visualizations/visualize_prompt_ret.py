import argparse
import base64
import html
import io
from pathlib import Path

import pandas as pd
from PIL import Image

from crosswise_clip.config import Config

CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--muted:#6b6b73;--card:#f6f6f8;--line:#d9d9df;--good:#1a7f37;--unk:#b26a00}
@media (prefers-color-scheme: dark){:root{--bg:#16171a;--fg:#ececf0;--muted:#9a9aa6;--card:#202126;--line:#34353c;--good:#4cc16a;--unk:#e0a040}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}
main{max-width:1500px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 6px}
.meta{color:var(--muted);margin:0 0 24px;max-width:90ch}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:0 0 20px}
.card h2{font-size:16px;margin:0 0 4px}
.card h2 span,.card p{color:var(--muted);font-weight:400}
.card p{margin:0 0 10px}
.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}
figure{margin:0;border:3px solid var(--line);border-radius:8px;background:var(--bg);overflow:hidden}
figure.good{border-color:var(--good)}
figure.unk{border-color:var(--unk)}
figure.look{border-color:var(--muted)}
figure img,.missing{display:block;width:100%;aspect-ratio:1;object-fit:contain}
figure img{background:#111}
.missing{display:flex;align-items:center;justify-content:center;background:var(--line);color:var(--muted);font-size:13px}
figcaption{padding:6px 8px;font-size:12.5px}
.badge{font-weight:600}
figure.good .badge{color:var(--good)}
figure.unk .badge{color:var(--unk)}
.types{color:var(--muted)}
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


def figure(src: str | None, confirmed: bool | None, feature: str, row) -> str:
    """One tile: the picture, then rank and similarity, tile ID, intersection types and whether the feature is confirmed.
    `confirmed` is None when the feature has no labels (a look-only prompt): grey frame and no badge."""
    img = f'<img src="{src}" alt="{html.escape(row.tile_id)}">' if src else '<div class="missing">image file not found</div>'
    caption = (
        f'<div class="badge">#{int(row.rank)}  similarity {row.similarity:.3f}</div>'
        f"<div>tile {html.escape(row.tile_id)}</div>"
        f'<div class="types">{html.escape(str(row.tile_types).replace(",", ", "))}</div>'
    )
    if confirmed is not None:
        caption += f'<div class="badge">{html.escape(f"✓ {feature} confirmed" if confirmed else "not confirmed")}</div>'
    css_class = "look" if confirmed is None else "good" if confirmed else "unk"
    return f'<figure class="{css_class}">{img}<figcaption>{caption}</figcaption></figure>'


def card(rows: pd.DataFrame, stats, paths: dict, size: int, cache: dict) -> str:
    """One prompt: its title, one line of numbers (when prompt_summary.csv is available), then its tiles in rank order."""
    first = rows.iloc[0]
    head = f"{html.escape(first.prompt_id)} <span>· {html.escape(first.feature)} · {html.escape(first.wording)}</span>"
    line = f"“{html.escape(first.text)}”"
    states = [None if pd.isna(v) else bool(v) for v in rows["feature_confirmed"]]  # True / False / None (no labels)
    if all(s is None for s in states):
        line += f" &nbsp;·&nbsp; no labels exist for {html.escape(first.feature)}, so judge by eye"
    else:
        line += f" &nbsp;·&nbsp; {sum(s is True for s in states)} of {len(rows)} shown have {html.escape(first.feature)} confirmed"
        if stats is not None and not pd.isna(stats.auc):
            line += f" (chance {100 * stats.chance:.0f}%, AUC {stats.auc:.2f})"
    figs = [figure(thumbnail_uri(paths.get(r.tile_id, ""), size, cache), s, first.feature, r) for r, s in zip(rows.itertuples(), states)]
    return f'<section class="card"><h2>{head}</h2><p>{line}</p><div class="row">{"".join(figs)}</div></section>'


def main() -> None:
    cfg = Config()
    ap = argparse.ArgumentParser(description="HTML report of the tiles each prompt retrieved")
    ap.add_argument("--tsv", default=cfg.tsv_path, help="TSV written by prep_data (for image paths)")
    ap.add_argument("--csv-dir", default=None, help="run folder written by prompt_eval.py (default: the most recent run under runs/prompt_eval)")
    ap.add_argument("--prompt-ids", nargs="+", default=None, help="prompt IDs to show (default: all, in file order)")
    ap.add_argument("--k", type=int, default=None, help="tiles shown per prompt (default: all the CSV has)")
    ap.add_argument("--thumb", type=int, default=320, help="longest side of embedded thumbnails, in pixels")
    ap.add_argument("--out", default=None, help="output HTML (default: prompt_report.html in --csv-dir)")
    args = ap.parse_args()

    if args.csv_dir:
        csv_dir = Path(args.csv_dir)
    else:
        runs = Path(cfg.runs_dir) / "prompt_eval"
        found = sorted(runs.rglob("prompt_top_tiles.csv"), key=lambda p: p.stat().st_mtime)
        if not found:
            raise SystemExit(f"No prompt_eval results under {runs}. Run prompt_eval.py first, or pass --csv-dir.")
        csv_dir = found[-1].parent
        print(f"Using the most recent run: {csv_dir}")
    top_path, summary_path = csv_dir / "prompt_top_tiles.csv", csv_dir / "prompt_summary.csv"
    if not top_path.exists():
        raise SystemExit(f"No {top_path}. Run prompt_eval.py first, or pass --csv-dir.")
    top = pd.read_csv(top_path, dtype={"tile_id": str}, keep_default_na=False)
    # "True" / "False" / blank (a feature with no labels) -> True / False / None
    confirmed = top["feature_confirmed"].astype(str).str.strip().str.lower().map({"true": True, "false": False})
    top["feature_confirmed"] = confirmed.astype(object).where(confirmed.notna(), None)
    stats = pd.read_csv(summary_path).set_index("prompt_id") if summary_path.exists() else None

    available = list(dict.fromkeys(top["prompt_id"]))
    chosen = args.prompt_ids if args.prompt_ids else available
    unknown = [p for p in chosen if p not in available]
    if unknown:
        raise SystemExit(f"Not prompts in {top_path.name}: {', '.join(unknown)}. Available: {', '.join(available)}")
    if args.k is not None and args.k < 1:
        raise SystemExit("--k must be at least 1.")

    tiles = pd.read_csv(args.tsv, sep="\t", dtype={"tile_id": str})
    paths = dict(zip(tiles["tile_id"], tiles["filepath"]))
    cache: dict = {}
    cards = []
    for pid in chosen:
        rows = top[top["prompt_id"] == pid].sort_values("rank")
        if args.k is not None:
            rows = rows.head(args.k)
        cards.append(card(rows, stats.loc[pid] if stats is not None and pid in stats.index else None, paths, args.thumb, cache))

    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Text-to-image retrieval</title><style>{CSS}</style></head><body><main>"
        "<h1>Text-to-image retrieval</h1>"
        f'<p class="meta">{html.escape(cfg.model_name)} ({html.escape(cfg.pretrained)}) zero-shot. Each card is one prompt and the tiles '
        "whose image vectors are most similar to its text vector, best first. Green border: the database confirms the feature on that tile. "
        "Amber: it does not, which can mean the feature is absent or just not recorded, so judge by eye too. "
        "Grey: the feature has no labels at all, so there is nothing to confirm.</p>"
        + "".join(cards)
        + "</main></body></html>"
    )
    out = Path(args.out) if args.out else csv_dir / "prompt_report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")

    missing = [p for p, v in cache.items() if v is None]
    print(f"Wrote {len(cards)} cards to {out} ({out.stat().st_size / 1e6:.1f} MB)")
    if missing:
        print(f"Warning: {len(missing)} image file(s) could not be read and show as placeholders: " + ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else ""))


if __name__ == "__main__":
    main()