import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class TileDataset(Dataset):
    def __init__(
        self, 
        tsv_path: str, 
        transform: transforms
    ):
        self.df = pd.read_csv(tsv_path, sep="\t", dtype={"tile_id": str})
        self.type_names = sorted({t for s in self.df["types"] for t in s.split(",")})
        self.indexes = {t: i for i, t in enumerate(self.type_names)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:

        # Get the idx-th row from the dataframe
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")

        # Per tile type-tensor, with indexes aligning with self.indexes
        # 1 if that type is present, 0 otherwise
        labels = torch.zeros(len(self.type_names))
        for t in row["types"].split(","):
            labels[self.indexes[t]] = 1.0

        return {
            "image": self.transform(image),
            "labels": labels,
            "tile_id": row["tile_id"],
        }

if __name__ == "__main__":
    test_ds = TileDataset("./data/tiles.tsv", transforms.Lambda(lambda x: x))
    print(test_ds.type_names)
    print(len(test_ds))