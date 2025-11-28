from transformers import CLIPModel, CLIPProcessor
import torch

def load_model_to_train(model_name="openai/clip-vit-large-patch14", device="cuda" if torch.cuda.is_available() else "cpu"):
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
    return model, processor