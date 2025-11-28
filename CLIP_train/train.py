from dataset import load_data
from utils import set_seed
from dataset import PosNegDataset, custom_collate_fn
from torch.utils.data import DataLoader
from model import load_model_to_train
import os
import torch
import math
from transformers import get_cosine_schedule_with_warmup
from evaluate import sim_m
from PIL import Image
from predict import get_top4_frames
from loss import clip_loss_with_negatives
from tqdm import tqdm


set_seed(22520465)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="../dataset/train/train.json")
    parser.add_argument("--image_path", type=str, default="dataset")
    parser.add_argument("--setting", type=int, default=1)
    parser.add_argument("--output_path", type=str, default="saved_models/clip-finetuned-posneg")
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-large-patch14")

    args = parser.parse_args()
    json_path = args.json_path
    image_path = args.image_path
    setting = args.setting
    output_path = args.output_path

    # create output path if not exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # load dataset
    train_list, val_list = load_data(json_path, image_path, setting, ratio)
    train_dataset = PosNegDataset(train_list)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    model, processor = load_model_to_train()
    model.to(device)

    # load optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # define hyperparameters
    num_epochs = 1
    accumulation_steps = 16
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accumulation_steps)
    num_training_steps = num_update_steps_per_epoch * num_epochs

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    model.train()

    # training
    for epoch in range(num_epochs):
        total_loss = 0
        count = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_dataloader, start=1):
            caption = batch["caption"][0]
            pos_imgs = batch["pos_imgs"][0]
            neg_imgs = batch["neg_imgs"][0]

            loss = clip_loss_with_negatives(model, processor, caption, pos_imgs, neg_imgs, device)
            if loss is None:
                continue

            # Chia loss ra để tránh tích gradient quá lớn
            loss = loss / accumulation_steps
            loss.backward()

            total_loss += loss.item() * accumulation_steps  # để giữ nguyên giá trị thực tế
            count += 1

            # ✅ Chỉ update sau mỗi accumulation_steps batch
            if batch_idx % accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # In ra loss sau mỗi 25 batch
            if batch_idx % 25 == 0:
                avg_loss = total_loss / max(count, 1)
                print(f"Batch {batch_idx} - Current Avg Loss: {avg_loss:.4f}")
                
        # Epoch summary
        avg_loss = total_loss / max(count, 1)
        print(f"✅ Epoch {epoch+1}/{num_epochs} - Epoch Avg Loss: {avg_loss:.4f}")

        
    # evaluation 
    if val_list:
        sim_95_score = 0 
        sim_96_score = 0 
        sim_97_score = 0 
        sim_98_score = 0 
        sim_99_score = 0 
        for val in val_list: 
            caption = val["caption"]
            video_path = val["video_path"]
            pos_path = val["pos_path"]

            # get true frames
            true_frames = [] 
            for file in os.listdir(pos_path):
                true_frames.append(Image.open(os.path.join(pos_path, file)).convert("RGB"))

            # get top 4 frames
            top_4_frames = get_top4_frames(video_path, caption, model, processor, device)
            sim_95_score += sim_m(top_4_frames, true_frames, model, processor, 0.95)
            sim_96_score += sim_m(top_4_frames, true_frames, model, processor, 0.96)
            sim_97_score += sim_m(top_4_frames, true_frames, model, processor, 0.97)
            sim_98_score += sim_m(top_4_frames, true_frames, model, processor, 0.98)
            sim_99_score += sim_m(top_4_frames, true_frames, model, processor, 0.99)
        print(f"Sim 95 Score: {sim_95_score / len(val_list)}")
        print(f"Sim 96 Score: {sim_96_score / len(val_list)}")
        print(f"Sim 97 Score: {sim_97_score / len(val_list)}")
        print(f"Sim 98 Score: {sim_98_score / len(val_list)}")
        print(f"Sim 99 Score: {sim_99_score / len(val_list)}")

    # Lưu model
    model.save_pretrained(output_path)
    processor.save_pretrained(output_path)
    print(f"🎉 Fine-tune CLIP xong, model lưu tại {output_path}")

