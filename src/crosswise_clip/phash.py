import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.fft
from PIL import Image

def phash(path, size=8) -> np.ndarray:
    """
    Compute a 64-bit perceptual hash of the image at 'path'
    
    Returns a flat boolean array
    """

    # Convert to grayscale + resize to 32x32 
    img = Image.open(path).convert("L")
    img = img.resize((32, 32), Image.Resampling.LANCZOS)

    pixels = np.asarray(img, dtype=float)
    
    # Apply 2D Discrete Cosine Transformation 
    dct = scipy.fft.dct(scipy.fft.dct(pixels, axis=0), axis=1)

    # Extract 64 lowest frequncies to capture the general structure of the image
    low = dct[:size, :size]

    # Binary mask => want relative ordering, not absolute differences
    return (low > np.median(low)).ravel()

def hamming(a, b) -> int:
    """Number of bits that differ between two hashes"""
    return int(np.count_nonzero(np.asarray(a) != np.asarray(b)))

# --- Binary <-> Hex Conversion for Convience and Readibility ---
def to_hex(bits: np.ndarray) -> str:
    return np.packbits(bits).tobytes().hex()

def from_hex(text: str) -> np.ndarray:
    return np.unpackbits(np.frombuffer(bytes.fromhex(text), dtype=np.uint8)).astype(bool)

def main():
    """Writes:
    1) phash_{tsv.stem}_{args.size}.csv => each images phash in hex

    2) phash_pairs_{tsv.stem}_{args.size}.csv => pairwise phash distances
    """
    ap = argparse.ArgumentParser(
        description="pHash tiles, measures how close each pair is"
    )
    ap.add_argument(
        "--tsv",
        required=True,
        help="TSV w/ tile-Id and filepath cols"
    )
    ap.add_argument(
        "--show",
        type=int,
        default=15,
        help="closest pairs to print"
    )
    ap.add_argument(
        "--size",
        type=int,
        help="shrink size => determines how much structural information to keep"        
    )
    args = ap.parse_args()

    tsv = Path(args.tsv)
    tiles = pd.read_csv(tsv, sep="\t", dtype={"tile_id": str})

    # Check missing tiles
    missing = [p for p in tiles["filepath"] if not Path(p).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} tile files not found")

    # Hash all images
    ids = tiles["tile_id"].tolist()
    bits = np.array([phash(p, args.size) for p in tiles["filepath"]])

    hashes_out = tsv.parent / f"phash_{tsv.stem}_{args.size}.csv"
    pd.DataFrame({
        "tile_id": ids,
        "phash": [to_hex(b) for b in bits]
    }).to_csv(hashes_out, index=False)

    # Compute distances (tile1, tile2, corresponding hash dist)
    dist = (bits[:, None, :] != bits[None, :, :]).sum(axis=2)

    # dist is symetric mat and diag is 0 => extract upper triangular (excluding diag), decompose into arrays of x and y coords
    a, b = np.triu_indices(len(ids), k=1)

    pairs = pd.DataFrame({
            "tile_a": np.array(ids)[a],
            "tile_b": np.array(ids)[b],
            "distance": dist[a, b]
    })

    # sort pairs by similaritiy
    pairs = pairs.sort_values(
        "distance", kind="stable"
    ).reset_index(drop=True)

    pairs_out = tsv.parent / f"phash_pairs_{tsv.stem}_{args.size}.csv"
    pairs.to_csv(pairs_out, index=False)

    print(f"Hashed {len(ids)} tiles -> {hashes_out}")
    print(f"{len(pairs)} pairs -> {pairs_out}")
    counts = np.bincount(pairs["distance"], minlength=65)
    for d in range(21):
        print(f"  {d:2d}  {counts[d]:6d}")
    print(f"  21+ {counts[21:].sum():6d}\n")
    print(f"Closest {args.show} pairs:")
    print(pairs.head(args.show).to_string(index=False))

if __name__ == "__main__":
    main()