from transformers import CLIPModel, CLIPProcessor
import torch
import cv2
from PIL import Image
import gc



def read_video_frames(video_path, fps_target=15):
    """Đọc và trích xuất frame theo FPS target."""
    cap = cv2.VideoCapture(video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if fps_video <= 0:
        fps_video = 30.0
    frame_interval = max(1, int(round(fps_video / fps_target)))

    frames, frame_count = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        frame_count += 1
    cap.release()
    return frames


def get_top4_frames(
    video_path: str,
    caption: str,
    model,
    processor,
    device: str = "cuda",
    fps_target: int = 15,
    batch_size: int = 112,
):

    # get text embedding
    model.eval()
    text_inputs = processor(text=[caption], return_tensors="pt", truncation=True).to(device)
    text_emb = model.get_text_features(**text_inputs)
    text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)

    # get frames
    frames = read_video_frames(video_path, fps_target)
    print(f"✅ Model+caption loaded | Frames extracted: {len(frames)}")

    # ---------------------------------------------------------
    # --- Embed frames in batches ---
    # ---------------------------------------------------------
    all_img_embs = []

    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]

            inputs = processor(
                images=batch,
                return_tensors="pt",
                truncation=True
            ).to("cuda")

            img_embs = model.get_image_features(**inputs)
            img_embs = img_embs / img_embs.norm(p=2, dim=-1, keepdim=True)

            all_img_embs.append(img_embs.cpu())

    img_embs = torch.cat(all_img_embs, dim=0)
    print(f"🧩 Embedded frames shape: {img_embs.shape}")

    # ---------------------------------------------------------
    # --- Compute similarity ---
    # ---------------------------------------------------------
    text_emb_cpu = text_emb.cpu()
    scores = (img_embs @ text_emb_cpu.T).squeeze(1).tolist()

    # ---------------------------------------------------------
    # 🚀 Chia thành 4 đoạn bằng nhau
    # ---------------------------------------------------------
    N = len(frames)
    
    # --- Nhóm 1: cố định 3 frames ---
    seg1_start = 0
    seg1_end = min(3, N)   # đề phòng N < 3
    segments = [(seg1_start, seg1_end)]
    
    # --- 3 nhóm còn lại ---
    remaining = N - seg1_end
    if remaining > 0:
        segment_size = remaining // 3
    
        # Tạo 3 nhóm tiếp theo
        s2 = seg1_end
        s3 = s2 + segment_size
        s4 = s3 + segment_size
        s5 = N  # group 4 lấy hết phần còn lại
    
        segments += [
            (s2, s3),
            (s3, s4),
            (s4, s5)
        ]
    
    # --- Chọn frame tốt nhất mỗi nhóm ---
    top_indices = []
    for (start, end) in segments:
        if start >= end:
            continue
    
        idxs = list(range(start, end))
        best_idx = max(idxs, key=lambda i: scores[i])
        top_indices.append(best_idx)
    
    # Sort theo index
    top_indices = sorted(top_indices)
    
    # Lấy kết quả
    top_frames = [frames[i].resize((1920, 1080)) for i in top_indices]
    top_scores = [scores[i] for i in top_indices]

    return top_frames, top_scores