import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from model import (
    ZeroShotCLIP,
)
from data import (
    IntersectionDataset,
    prep_data,
    template,
)

def test_baseline_train(
    model: nn.Module,
    loader: DataLoader, 
    loss_fn,
    device: str,
):
    """
    Run one pass through ZeroShotBaseline, return avg contrastive loss
    """

    # ZeroShotBaseline doesn't need gradient updates => set to eval
    model.eval()
    total_loss = 0.0

    # TODO: Update this based on how data is implemented
    for batch in tqdm(loader, desc="Baseline Pass"):

        # Move data to devices
        pixel_values = batch["pixel_values"].to(device)
        prompts = batch["prompt"]

        # --- Forward Pass ---
        image_embeds = model.encode_image(pixel_values)
        text_embeds = model.encode_text(prompts)

        # logits = np.dot(I_e, T_e.T) * np.exp(t)
        logits = model.model.logit_scale.exp() * image_embeds @ text_embeds.T
        # --------------------

        # --- Loss ---
        labels = torch.arange(len(pixel_values), device=device)
        loss = (loss_fn(logits, labels) + loss_fn(logits.T, labels)) / 2
        # -----------------------------------------------

        # --- Back Propogation (Not Relevant in Baseline) ---
        # loss.backward()
        # optimizer.step()
        # ---------------------------------------------------

        total_loss += loss.item()

    # return average loss
    return total_loss / len(loader)

def main():

    cfg = Config()
    device = cfg.device
    torch.manual_seed(cfg.seed)

    model = ZeroShotCLIP(cfg.model_name).to(device)

    train_records, val_records = prep_data(cfg.db_path, cfg.val_size)
    train_dataset = IntersectionDataset(train_records, cfg.img_root_dir, model.processor, template)
    val_dataset = IntersectionDataset(val_records, cfg.img_root_dir, model.processor, template)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    loss_fn = nn.CrossEntropyLoss() 

    # Check that each we have unique captions per image, as assumed of diagnoal loss
    unique = len({r["tile_description"] for r in train_records})
    print(f"Unique Desc in Train: {unique}/{len(train_records)}")

    train_loss = test_baseline_train(model, train_loader, loss_fn, device)
    val_loss = test_baseline_train(model, val_loader, loss_fn, device)

    print(f"Baseline train loss: {train_loss:.4f}")
    print(f"Baseline val loss:   {val_loss:.4f}")

if __name__ == "__main__":
    main()