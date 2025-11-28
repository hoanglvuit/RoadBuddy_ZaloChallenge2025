import torch
import torch.nn.functional as F
from PIL import Image

def sim_m(top_4_frames: list[Image.Image], true_frames: list[Image.Image], model, processor, sim_threshold=0.95):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    # Hàm encode ảnh để lấy embedding CLIP
    def encode(img):
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
        return F.normalize(emb, p=2, dim=-1)

    # Encode trước cho nhanh
    top_embs = [encode(img) for img in top_4_frames]
    true_embs = [encode(img) for img in true_frames]

    # Với mỗi true_frame → phải tìm được ít nhất 1 top_frame đạt sim
    for emb_true in true_embs:
        ok = False

        for emb_top in top_embs:
            sim = (emb_true @ emb_top.T).item()
            if sim >= sim_threshold:
                ok = True
                break

        if not ok:
            # Chỉ cần 1 true_frame không đạt → fail toàn bộ
            return 0

    # Nếu tất cả true_frame đều match → pass
    return 1
