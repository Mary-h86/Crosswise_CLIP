from dataclasses import dataclass
import torch

PROMPTS = {
    "p1": "an aerial image of a {}",
    "p2": "an aerial image of a {}, a type of intersection"
}

@dataclass
class Config:
    # --- Data ---
    tsv_path: str = "data_prep/out"
    seed: int = 123
    
    # --- Model --- 
    model_name: str = "ViT-B-32"
    pretrained: str = "openai"

    # --- Eval --
    ks: tuple = (5, 10)
    batch_size: int = 32
    num_workers: int = 4
    runs_dir: str = "runs"

    # --- Fine-tuning --
    val_size: float = 0.2
    epochs: int = 10
    learning_rate: float = 1e-5

    # --- Device ---
    @property
    def device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.mps.is_available():
            return "mps"
        return "cpu"