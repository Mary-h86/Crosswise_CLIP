# TODO: Create another database, add another column with the modified data (ie minus the street names)
# and keep the original

import duckdb
from pathlib import Path
import pandas as pd

def read_duckdb(
    db: str, # duck_db path
    out: str, # where to write the tsv to
) -> None:
    """Reads the DuckDB file, extracts tile_id, image_path, and intersection_type into a tsv file"""

    db_path = Path(db).resolve()
    with duckdb.connect(str(db_path), read_only=True) as con:
        data = con.sql(
            """
            SELECT tile_id, image_path, intersection_type
            FROM tiles
            WHERE intersection_type != '[]'
            """
        ).fetchall()

    print(type(data)) # Expect List[Tuple["tile_id" -> str, "image_path" -> str, "intersection_type" -> List]]

    df = pd.DataFrame(
        {
            "tile_id": tile_id,
            "filepath": str((db_path.parent / image_path).resolve()),
            "types": ",".join(types),
        }
        for tile_id, image_path, types in data
    )

    # Print intersection type splits
    print(f"Tile Counts: {df['types'].str.split(',').explode().value_counts().to_string()}")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)

if __name__ == "__main__":
    read_duckdb("./data/annotations_per_tile.duckdb", "./data/tiles.tsv")

