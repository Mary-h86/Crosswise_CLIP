from dataclasses import dataclass
import torch

@dataclass
class Config:
    # --- Path + Directory Management ---
    db_path: str = "annotations.duckdb"
    img_root_dir: str = "FILL IN ONCE DETERMINE WHERE IMAGES ARE"
    val_size: float = 0.2
    seed: int = 123
    
    # --- Model + Training Hyperparams --- 
    model_name: str = "openai/clip-vit-base-patch32"
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-4
    num_workers: int = 4

    # --- Device ---
    @property
    def device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.mps.is_available():
            return "mps"
        return "cpu"