import json
from PIL import Image
import os
from torch.utils.data import Dataset
import random

def load_data(json_path, image_path,video_path, setting=1, val_ratio=0.2, seed=42):
    """
    setting = 1: only question 
    setting = 2: only choices 
    setting = 3: question + choices
    
    Return:
        train_list, val_list
    """

    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)["data"]

    # Shuffle trước khi split
    if val_ratio != 1:
        random.seed(seed)
        random.shuffle(data)

    # Nếu val_ratio = 1 → bỏ qua val luôn
    if val_ratio == 1:
        train_data = data
        val_data = []
    else:
        # Split train/val
        val_size = int(len(data) * val_ratio)
        val_data = data[:val_size]
        train_data = data[val_size:]

    base_pos = os.path.join(image_path, "frames_true")
    base_neg = os.path.join(image_path, "frames_neg")

    train_list = []
    val_list = []

    # ------ TRAIN ------
    for item in train_data:
        choices_str = " ".join(item["choices"])

        if setting == 1:
            caption = item["question"]
        elif setting == 2:
            caption = choices_str
        else:
            caption = item["question"] + " " + choices_str

        train_list.append({
            "pos_path": os.path.join(base_pos, item['id']),
            "neg_path": os.path.join(base_neg, item['id']) if os.path.exists(os.path.join(base_neg, item['id'])) else "",
            "caption": caption
        })

    # ------ VAL (bỏ qua nếu val_ratio = 1) ------
    for item in val_data:
        choices_str = " ".join(item["choices"])

        if setting == 1:
            caption = item["question"]
        elif setting == 2:
            caption = choices_str
        else:
            caption = item["question"] + " " + choices_str

        val_list.append({
            "caption": caption,
            "video_path": os.path.join(video_path, item.get("video_path", "")),
            "pos_path": os.path.join(base_pos, item['id'])
        })

    return train_list, val_list

class PosNegDataset(Dataset):
    def __init__(self, data_list):
        """
        data_list: list of dicts như:
          {'pos_path': '...', 'neg_path': '...', 'caption': '...'}
        """
        self.data = data_list

    def load_images_from_folder(self, folder_path):
        if not folder_path or not os.path.exists(folder_path):
            return []
        imgs = []
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = Image.open(os.path.join(folder_path, fname)).convert("RGB")
                imgs.append(img)
        return imgs

    def __getitem__(self, idx):
        sample = self.data[idx]
        pos_imgs = self.load_images_from_folder(sample['pos_path'])
        neg_imgs = self.load_images_from_folder(sample['neg_path']) if sample['neg_path'] else []
        return {
            "caption": sample['caption'],
            "pos_imgs": pos_imgs,
            "neg_imgs": neg_imgs
        }

    def __len__(self):
        return len(self.data)

def custom_collate_fn(batch):
    captions = [item["caption"] for item in batch]
    pos_imgs = [item["pos_imgs"] for item in batch]
    neg_imgs = [item["neg_imgs"] for item in batch]
    return {"caption": captions, "pos_imgs": pos_imgs, "neg_imgs": neg_imgs}