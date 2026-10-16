from PIL import Image
from transformers import CLIPModel, CLIPProcessor

model_name = "openai/clip-vit-base-patch32"
img_path = "./test_intersection.jpg" # hardcoded for now

prompts = [
    "a photo of a dirt path",
    "a photo of a four way intersection",
    "a photo of a highway",
]

def main():
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    image = Image.open(img_path).convert("RGB")

    inputs = processor(
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=-1)[0]

    for prompt, prob in zip(prompts, probs):
        print(f"{prob.item():3f} {prompt}")

if __name__ == "__main__":
    main()