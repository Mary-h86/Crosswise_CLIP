import torch
from torch import nn
from transformers import CLIPModel, CLIPProcessor

class ZeroShotCLIP(nn.Module):
    """Baseline CLIP implenmentation for benchmarking other methods"""

    def __init__(self, model_name: str):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        # Freeze weights
        for p in self.model.parameters():
            p.requires_grad_(False)

    # Decorator to disable gradient calculations
    @torch.no_grad()
    def encode_image(self, pixel_values):
        features = self.model.get_image_features(pixel_values=pixel_values)

        # Embedding vector normalization 
        return features / features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_text(self, prompts: str):
        device = next(self.model.parameters()).device
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True)
        
        # Configure tensors to active device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        features = self.model.get_text_features(**inputs)
        return features / features.norm(dim=-1, keepdim=True)

if __name__ == "__main__":
    # TODO: Write tests
    pass
