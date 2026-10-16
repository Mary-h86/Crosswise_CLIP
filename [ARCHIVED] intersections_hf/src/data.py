import duckdb
import torch
import os
from torch.utils.data import Dataset
from transformers import CLIPProcessor
from PIL import Image
from sklearn.model_selection import train_test_split

# Create another database, add another column with the modified data (ie minus the street names)
# and keep the original

# template = ["a photo of {}, a type of intersection"]

template = "a photo of {}"

class IntersectionDataset(Dataset):
    """
    Pytorch Datset for intersection images and CLIP prompts
    """

    def __init__(
        self,
        data: list[dict[str, str]],
        img_root_dir: str,
        processor: CLIPProcessor,
        template: list[str] 
    ):
        self.data = data
        self.img_root_dir = img_root_dir
        self.processor = processor
        self.template = template

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data[idx]
        img_path = os.path.join(self.img_root_dir, row["image_path"])
        image = Image.open(img_path).convert("RGB")

        prompt = self.template.format(row["tile_description"])

        pixel_values = self.processor.image_processor(
            images=image, 
            return_tensors="pt"
        )["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values, # Shape (3x224x224)
            "prompt": prompt,
            "tile_id": row["tile_id"]
        }

def prep_data(
    db_path: str,
    val_size: float = 0.2, # default 80/20 split
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """
    Query DuckDB to extract image-description pairs, also performs train/val split
    
    CURRENTLY: (img, tile_annotation) pairs
    """

    seed = 123

    con = duckdb.connect(db_path)

    print(f"Reading data from {db_path}")

    query = """
        SELECT a.tile_id, t.image_path, a.tile_description
        FROM annotations AS A
        JOIN tiles AS t ON a.tile_id = t.tile_id
        WHERE a.tile_description IS NOT NULL 
            AND TRIM(a.tile_description) != ''
    """

    # Extract (tile_id, image_path, tile_description) tuples for each image
    results = con.sql(query).fetchall()
    con.close()

    # Convert results to a list of dictionaries
    records = [
        {"tile_id": r[0], "image_path": r[1], "tile_description": r[2]}
        for r in results
    ]

    # Train/Val Split
    train_records, val_records = train_test_split(
        records,
        test_size=val_size,
        random_state=seed,
    )

    print(f"Train Set: {len(train_records)} images")
    print(f"Val Set: {len(val_records)} images")

    return train_records, val_records

if __name__ == "__main__":

    # --- Experimentation ---

    con = duckdb.connect("../annotations.duckdb")

    for table in ["annotations", "bounding_boxes", "progress", "reference_segment_dismissals", "reference_segments", "reviews", "tiles"]:
        print(f"\n==={table}===")
        con.sql(f"DESCRIBE {table}").show()

    con.sql("""
        DESCRIBE annotations
    """).show()

    con.sql("""
        SELECT tile_id, tile_description, section1_tags
        FROM annotations AS a
        WHERE section1_tags->>'roadway_type' IS NOT NULL
        LIMIT 10
    """).show()

    # t.image_path, t.city, a.roadway_type, a.land_use, a.existing_bike, a.new_bike, a.existing_ped, a.new_ped, a.treatment, a.tile_description, a.is_complete


