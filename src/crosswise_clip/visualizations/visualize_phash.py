import argparse
import base64
import html
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CSS = """
body { font-family: system-ui, sans-serif; margin: 0 auto; max-width: 1100px; padding: 16px; color: #1a1a1a; background: #fff; }
@media (prefers-color-scheme: dark) { body { color: #e8e8e8; background: #141414; } .card { border-color: #444; } th, td { border-color: #333; } }
h1 { font-size: 1.3rem; } h2 { font-size: 1.05rem; margin: 28px 0 8px; } p.note { color: #777; margin: 4px 0 12px; }
#panel { position: sticky; top: 0; background: inherit; padding: 8px 0 10px; border-bottom: 1px solid #8884; z-index: 2; }
#panel input { width: min(520px, 90%); vertical-align: middle; } #summary { margin-top: 6px; line-height: 1.5; }
table { border-collapse: collapse; font-size: .85rem; } th, td { border-bottom: 1px solid #ddd; padding: 2px 10px; text-align: right; }
td.bar div { background: #6a9bd8; height: 10px; } td.bar { width: 220px; padding: 2px 6px; text-align: left; }
tr.dup td { background: #2e9b4f22; }
.grid { display: flex; flex-wrap: wrap; gap: 12px; }
.card { border: 2px solid #ccc; border-radius: 6px; padding: 6px; } .card.dup { border-color: #2e9b4f; }
.imgs { display: flex; gap: 6px; } figure { margin: 0; } figcaption { font-size: .75rem; color: #777; max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.missing { background: #8884; color: #777; display: flex; align-items: center; justify-content: center; font-size: .8rem; }
.tag { font-size: .8rem; margin-top: 4px; } .card.dup .tag::after { content: "  duplicate"; color: #2e9b4f; font-weight: 600; }
"""

JS = """
const PAIRS = __PAIRS__, N_TILES = __N_TILES__, MAX_D = __MAX_D__;
const slider = document.getElementById('threshold'), summary = document.getElementById('summary');
function update() {
  const T = +slider.value;
  document.querySelectorAll('.card').forEach(c => c.classList.toggle('dup', +c.dataset.d <= T));
  document.querySelectorAll('tr[data-d]').forEach(r => r.classList.toggle('dup', +r.dataset.d <= T));
  const parent = new Map(), find = x => { while (parent.get(x) !== x) { parent.set(x, parent.get(parent.get(x))); x = parent.get(x); } return x; };
  let flagged = 0;
  for (const [a, b, d] of PAIRS) if (d <= T) {
    flagged++;
    for (const x of [a, b]) if (!parent.has(x)) parent.set(x, x);
    parent.set(find(a), find(b));
  }
  const groups = new Map();
  for (const x of parent.keys()) { const r = find(x); if (!groups.has(r)) groups.set(r, []); groups.get(r).push(x); }
  const sizes = [...groups.values()].map(g => g.length), inGroups = sizes.reduce((s, n) => s + n, 0);
  const big = [...groups.values()].filter(g => g.length >= 3).sort((a, b) => b.length - a.length);
  document.getElementById('t').textContent = T;
  summary.innerHTML = `Threshold <b>${T}</b>: <b>${flagged}</b> pairs flagged, <b>${inGroups}</b> tiles in <b>${groups.size}</b> groups` +
    (groups.size ? `, largest group ${Math.max(...sizes)}` : '') +
    `. Keeping one tile per group leaves <b>${N_TILES - (inGroups - groups.size)}</b> of ${N_TILES} tiles.` +
    (big.length ? `<br>Groups of 3 or more (check these for chains): ` + big.slice(0, 8).map(g => '[' + g.join(', ') + ']').join(' ') + (big.length > 8 ? ' ...' : '') : '');
}
slider.addEventListener('input', update); update();
"""


def thumbnail_uri(path: str, size: int, cache: dict) -> str:
    """A small inline JPEG of the image, or '' if the file can't be read."""
    if path not in cache:
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((size, size))
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=80)
            cache[path] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        except OSError:
            cache[path] = ""
    return cache[path]


def figure(tile_id: str, tiles: pd.DataFrame, size: int, cache: dict) -> str:
    row = tiles.loc[tile_id]
    uri = thumbnail_uri(row["filepath"], size, cache)
    caption = html.escape(str(tile_id) + (f" · {row['types']}" if "types" in tiles.columns and pd.notna(row["types"]) else ""))
    image = f'<img src="{uri}" width="{size}">' if uri else f'<div class="missing" style="width:{size}px;height:{size}px">image not found</div>'
    return f'<figure>{image}<figcaption title="{caption}">{caption}</figcaption></figure>'


def main() -> None:
    ap = argparse.ArgumentParser(description="HTML report of tile pairs by pHash distance, for choosing the duplicate threshold")
    ap.add_argument("--tsv", required=True, help="the TSV you gave phash.py (tile_id, filepath)")
    ap.add_argument("--pairs", default=None, help="pairs CSV from phash.py (default: phash_pairs_<tsv name>.csv next to the TSV)")
    ap.add_argument("--bits", type=int, default=64, help="hash length the pairs table was made with: 16 for a 4x4 block, 64 for 8x8 (default), 256 for 16x16")
    ap.add_argument("--max-distance", type=int, default=None, help="show pairs up to this distance (default: 31%% of --bits, i.e. 20 for 64 bits)")
    ap.add_argument("--per-distance", type=int, default=4, help="pairs shown at each distance (default 4)")
    ap.add_argument("--start", type=int, default=None, help="where the threshold slider starts (default: 8%% of --bits, i.e. 5 for 64 bits)")
    ap.add_argument("--thumb", type=int, default=170, help="thumbnail size in pixels (default 170)")
    ap.add_argument("--seed", type=int, default=0, help="seed for choosing which pairs to show (default 0)")
    ap.add_argument("--out", default=None, help="output HTML (default: phash_report_<tsv name>.html next to the TSV)")
    args = ap.parse_args()

    tsv = Path(args.tsv)
    pairs_path = Path(args.pairs) if args.pairs else tsv.parent / f"phash_pairs_{tsv.stem}.csv"
    if not pairs_path.exists():
        raise SystemExit(f"No pairs table at {pairs_path}. Run phash.py on the same TSV first, or point --pairs at the file it wrote.")
    tiles = pd.read_csv(tsv, sep="\t", dtype={"tile_id": str}).set_index("tile_id", drop=False)
    pairs = pd.read_csv(pairs_path, dtype={"tile_a": str, "tile_b": str})
    unknown = (set(pairs["tile_a"]) | set(pairs["tile_b"])) - set(tiles.index)
    if unknown:
        raise SystemExit(f"The pairs table has tiles that are not in the TSV (for example {sorted(unknown)[0]}). Use the TSV that phash.py ran on.")

    if pairs["distance"].max() > args.bits:
        raise SystemExit(f"The pairs table has distances up to {pairs['distance'].max()} but --bits is {args.bits}. Pass --bits with the hash length it was made with (16, 64 or 256).")
    top = args.max_distance if args.max_distance is not None else args.bits * 20 // 64
    even = bool((pairs["distance"] % 2 == 0).all())   # a median threshold gives every hash exactly half ones, so distances come in twos
    step = 2 if even else 1
    start = min(args.start if args.start is not None else args.bits * 5 // 64, top) // step * step
    counts = np.bincount(pairs["distance"], minlength=max(args.bits, top) + 1)
    near = pairs[pairs["distance"] <= top]
    rng = np.random.default_rng(args.seed)
    cache: dict = {}

    rows, cum, peak = [], 0, max(1, counts[: top + 1].max())
    for d in range(top + 1):
        cum += int(counts[d])
        if even and d % 2:
            continue
        rows.append(f'<tr data-d="{d}"><td>{d}</td><td>{counts[d]}</td><td>{cum}</td>'
                    f'<td class="bar"><div style="width:{100 * counts[d] / peak:.0f}%"></div></td></tr>')

    sections = []
    for d in range(top + 1):
        group = near[near["distance"] == d]
        if group.empty:
            continue
        shown = group if len(group) <= args.per_distance else group.iloc[np.sort(rng.choice(len(group), args.per_distance, replace=False))]
        cards = "".join(
            f'<div class="card" data-d="{d}"><div class="imgs">{figure(r.tile_a, tiles, args.thumb, cache)}{figure(r.tile_b, tiles, args.thumb, cache)}</div>'
            f'<div class="tag">distance {d}</div></div>' for r in shown.itertuples())
        sections.append(f"<h2>Distance {d}: {len(group)} pair{'s' if len(group) != 1 else ''}"
                        f"{'' if len(shown) == len(group) else f' (showing {len(shown)})'}</h2><div class=\"grid\">{cards}</div>")

    beyond = int(counts[top + 1:].sum())
    name = pairs_path.stem.replace("phash_pairs_", "phash_report_", 1) if pairs_path.stem.startswith("phash_pairs_") else f"phash_report_{pairs_path.stem}"
    out = Path(args.out) if args.out else pairs_path.parent / f"{name}.html"   # one report per pairs table, so sizes do not overwrite each other
    script = (JS.replace("__PAIRS__", json.dumps(near[["tile_a", "tile_b", "distance"]].values.tolist()))
                .replace("__N_TILES__", str(len(tiles))).replace("__MAX_D__", str(top)))
    even_note = f" Distances come in twos here: every hash has exactly {args.bits // 2} ones, so the number of differing bits is always even." if even else ""
    out.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>pHash duplicate threshold</title><style>{CSS}</style></head><body>
<h1>Where do near-copies stop? {html.escape(tsv.name)}</h1>
<p class="note">{len(tiles)} tiles, {len(pairs)} pairs. Distance = number of differing bits out of {args.bits} (0 = identical hash, unrelated tiles sit near {args.bits // 2}).
Scroll down from 0 and find where the two tiles stop being the same scene. Green = flagged as duplicates at the current threshold.{even_note}</p>
<div id="panel"><label>Threshold (flag pairs at or below): <b id="t">{start}</b>
<input id="threshold" type="range" min="0" max="{top}" step="{step}" value="{start}"></label><div id="summary"></div></div>
<h2>Pairs per distance</h2>
<table><tr><th>distance</th><th>pairs</th><th>pairs at or below</th><th></th></tr>{''.join(rows)}</table>
<p class="note">{beyond} more pairs sit above distance {top} and are not shown.</p>
{''.join(sections)}
<script>{script}</script></body></html>""", encoding="utf-8")
    print(f"{len(near)} pairs at distance {top} or less, {sum(len(s) > 0 for s in sections)} distances shown -> {out}")


if __name__ == "__main__":
    main()
