from data_prep.dataset import TileDataset
import open_clip
from configs.config import Config
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class TileEmbeddings:
    def __init__(
        self, 
        image_embeds: np.ndarray, 
        labels: np.ndarray, 
        tile_ids: list, 
        type_names: list,
    ):
        self.image_embeds = image_embeds # (N tiles, D #s per embedding)
        self.labels = labels # (N tiles, T types) => binary values corresponding to intersection type
        self.tile_ids = tile_ids # (N tiles) => pair image with its index
        self.type_names = type_names # (T types)

def load_model(
    model_name: str,
    pretrained: str,
    device: str,
):
    # Returns ("...", train_transform, val_transform) => don't need train
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()

    return model, preprocess, tokenizer

@torch.no_grad()
def embed_tiles(
    model,
    preprocess,
    tsv_path: str,
    device: str,
    batch_size=32,
    num_workers=0,
):
    """Push tsv tiles through image encoder"""
    dataset = TileDataset(tsv_path, preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    feats, labels, tile_ids = [], [], []
    for batch in tqdm(loader, desc="Embedding Tiles"):
        f = model.encode_image(batch["image"].to(device))
        f = f / f.norm(dim=-1, keepdim=True) # norm cosine similarity
        feats.append(f.cpu()) # copy into normal mem
        labels.append(batch["labels"])
        tile_ids.extend(batch["tile_id"])

    image_embeds = torch.cat(feats).numpy()
    labels = torch.cat(labels).numpy()
    return TileEmbeddings(image_embeds, labels, tile_ids, dataset.type_names)

@torch.no_grad()
def embed_text(
    model,
    tokenizer,
    prompts: list[str],
    device: str,
) -> np.ndarray:
    """Push prompts through text embedder"""
    tokens = tokenizer(prompts).to(device)
    f = model.encode_text(tokens)
    f = f / f.norm(dim=-1, keepdim=True)
    return f.cpu().numpy()

def score_prompts(
    model, 
    tokenizer,
    emb: TileEmbeddings,
    template: str, # Sample template: "an aerial image of a {}, a type of intersection"
    device: str,
) -> np.ndarray:
    """Generate one prompt per type, return (num_types, num_tiles) similarity matrix"""
    prompts = [template.format(t) for t in emb.type_names]
    text_embeds = embed_text(model, tokenizer, prompts, device) # (T, D)
    scores = text_embeds @ emb.image_embeds.T # (T, D) @ (D, N) = (T, N)
    return scores