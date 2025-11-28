import json
from PIL import Image
import os

def load_data(json_path, image_path, setting=1): 
    """
    setting = 1: only question 
    setting = 2: only choices 
    setting = 3: question and choices
    """
    with open(json_path, 'r') as f:
        data = json.load(f)["data"]

    base_pos = os.path.join(image_path, "true_frames")
    base_neg = os.path.join(image_path, "false_frames")

    data_list = []
    for item in data:
        choices_str = " ".join(item["choices"])  

        if setting == 1:
            caption = item["question"]
        elif setting == 2:
            caption = choices_str
        else:
            caption = item["question"] + " " + choices_str

        data_list.append({
            "pos_path": os.path.join(base_pos, item['id']),
            "neg_path": os.path.join(base_neg, item['id']) if os.path.exists(os.path.join(base_neg, item['id'])) else "",
            "caption": caption
        })

    return data_list


