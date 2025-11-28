import os
import math
import random
from PIL import Image
import torch
from torch.nn.functional import cosine_similarity
import cv2
from transformers import CLIPModel, CLIPProcessor
import json
from utils import set_seed
set_seed(42)

def extract_true_and_negative_frames_fast(
    input_folder,
    output_folder,
    device="cuda" if torch.cuda.is_available() else "cpu",
    default_fps=30,
    num_segments=5,
    neg_sim_threshold=0.95,
):
    """
    Lưu ý:
    - True frames lấy theo item["support_frames"] (đã có).
    - Negative: chia các frame (exclude true frames) thành `num_segments` khoảng theo thời gian,
      mỗi khoảng random 1 frame (nếu có). Nếu cosine(true,neg) < neg_sim_threshold thì lưu.
    - Lưu PNG RGB (không metadata).
    """

    json_in = os.path.join(input_folder, "train/train.json")
    with open(json_in, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]

    os.makedirs(output_folder, exist_ok=True)
    true_root = os.path.join(output_folder, "frames_true")
    neg_root = os.path.join(output_folder, "frames_neg")
    os.makedirs(true_root, exist_ok=True)
    os.makedirs(neg_root, exist_ok=True)

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)

    for item in data:
        video_path = item.get("video_path")
        if not video_path:
            continue
        full_path = os.path.join(input_folder, video_path) if not os.path.isabs(video_path) else video_path
        if not os.path.exists(full_path):
            print(f"[WARN] Missing: {full_path}")
            continue

        cap = cv2.VideoCapture(full_path)
        if not cap.isOpened():
            print(f"[WARN] Cannot open: {full_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or default_fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames == 0:
            cap.release()
            continue

        vid_id = item.get("id", "unknown_id")
        times = item.get("support_frames", [])
        # convert times (seconds) to frame indices, clamp to [0, total_frames-1]
        true_indices = sorted({ min(int(math.floor(float(t) * fps)), total_frames - 1) for t in times })

        true_dir = os.path.join(true_root, vid_id)
        neg_dir = os.path.join(neg_root, vid_id)
        os.makedirs(true_dir, exist_ok=True)
        os.makedirs(neg_dir, exist_ok=True)

        # precompute set of true indices for fast exclusion
        true_set = set(true_indices)

        # --- save and embed true frames (we'll compute embeddings per true frame) ---
        true_embeddings = {}  # idx -> normalized embedding tensor (1, D)
        for i, idx in enumerate(true_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                print(f"[WARN] can't read true frame {idx} for {vid_id}")
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # embedding true
            inputs = processor(images=frame_rgb, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = model.get_image_features(**inputs)  # (1, D)
                emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            true_embeddings[idx] = emb

            # lưu true frame
            Image.fromarray(frame_rgb).save(
                os.path.join(true_dir, f"{vid_id}_true{i:02d}_f{idx:06d}.png"),
                format="PNG",
                icc_profile=None,
            )

        # --- prepare candidate frame indices excluding true frames ---
        all_indices = list(range(total_frames))
        candidate_indices = [fi for fi in all_indices if fi not in true_set]
        if len(candidate_indices) == 0:
            cap.release()
            continue

        # --- chia candidate_indices thành num_segments theo thứ tự thời gian ---
        # We'll split by index ranges on the original timeline so segments are contiguous time ranges.
        n_seg = int(num_segments)
        seg_size = total_frames // n_seg if n_seg > 0 else total_frames
        selected_negatives = []  # list of (seg_idx, chosen_frame_idx)
        for seg in range(n_seg):
            start = seg * seg_size
            # last segment goes to end
            end = (seg + 1) * seg_size if seg < n_seg - 1 else total_frames
            # collect candidates inside this time range and not true
            seg_candidates = [x for x in range(start, end) if x not in true_set]
            if not seg_candidates:
                continue
            chosen = random.choice(seg_candidates)
            selected_negatives.append((seg, chosen))

        # --- evaluate and save negatives: compare each chosen negative to ALL true frames of the example
        # Keep a negative if its similarity to EVERY true frame is < neg_sim_threshold
        for seg_idx, neg_idx in selected_negatives:
            cap.set(cv2.CAP_PROP_POS_FRAMES, neg_idx)
            ret2, neg = cap.read()
            if not ret2:
                continue
            neg_rgb = cv2.cvtColor(neg, cv2.COLOR_BGR2RGB)

            # embed negative
            neg_inputs = processor(images=neg_rgb, return_tensors="pt")
            neg_inputs = {k: v.to(device) for k, v in neg_inputs.items()}
            with torch.no_grad():
                neg_emb = model.get_image_features(**neg_inputs)
                neg_emb = neg_emb / neg_emb.norm(p=2, dim=-1, keepdim=True)

            # compare to all true embeddings (if no true embeddings available, skip)
            if len(true_embeddings) == 0:
                continue

            sims = []
            for t_idx, t_emb in true_embeddings.items():
                # cosine_similarity returns tensor; specify dim=1 to compare batch dims
                sim_val = cosine_similarity(t_emb, neg_emb, dim=-1).item()
                sims.append(sim_val)

            # require similarity to ALL true frames < threshold to accept as negative
            if all(s < neg_sim_threshold for s in sims):
                # use segment index and neg frame index in filename
                Image.fromarray(neg_rgb).save(
                    os.path.join(neg_dir, f"{vid_id}_neg_seg{seg_idx:02d}_f{neg_idx:06d}.png"),
                    format="PNG",
                    icc_profile=None,
                )

        cap.release()

    print("✅ Done! Saved true + negative PNG frames (negatives chosen per-segment)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="../dataset")
    parser.add_argument("--output_folder", type=str, default="dataset")

    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    extract_true_and_negative_frames_fast(input_folder, output_folder)