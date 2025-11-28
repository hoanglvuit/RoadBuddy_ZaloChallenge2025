import torch
import torch.nn.functional as F
from PIL import Image

def sim_m(top_4_frames, true_frames, model, processor, sim_threshold=0.95):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Encode với DINOv2
    def encode(img):
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :]   # CLS embedding
            emb = F.normalize(emb, p=2, dim=-1)
        return emb

    top_embs = [encode(img) for img in top_4_frames]
    true_embs = [encode(img) for img in true_frames]

    for emb_true in true_embs:
        ok = False
        for emb_top in top_embs:
            sim = (emb_true @ emb_top.T).item()
            if sim >= sim_threshold:
                ok = True
                break
        if not ok:
            return 0

    return 1