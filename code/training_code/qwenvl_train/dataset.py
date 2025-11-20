import json
import os
from PIL import Image
from datasets import Dataset
from prompt import instruction


def load_data(PATH, mode="train"):
    with open(os.path.join(PATH, f"{mode}.json"), "r") as f:
        data = json.load(f)['data']

    dataset_list = []
    for item in data:
        frames_folder = os.path.join(PATH, "dataset", mode, item['id'])
        item_dict = {
            'id': item['id'],
            'frames_folder': frames_folder,
            'question': item['question'],
            'choices': item['choices']
        }

        # Chỉ thêm answer nếu tồn tại
        if 'answer' in item:
            item_dict['answer'] = item['answer']

        dataset_list.append(item_dict)

    dataset = Dataset.from_list(dataset_list)
    return dataset

def make_conversation(item, mode="train"):

    folder_path = item["frames_folder"]

    # Load all images from the folder
    images = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.lower().endswith(".png"):
            img_path = os.path.join(folder_path, fname)
            with Image.open(img_path) as img:
                images.append(img.copy())

    # Build content list
    content = []

    # === 1. Instruction ===
    content.append({
        "type": "text",
        "text": instruction
    })

    # === 2. Images ===
    for img in images:
        content.append({
            "type": "image",
            "image": img
        })

    # === 3. Question ===
    content.append({
        "type": "text",
        "text": f"Question: {item['question']}"
    })

    # === 4. Choices ===
    for choice in item["choices"]:
        content.append({
            "type": "text",
            "text": choice
        })

    # Build user message
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]

    if "answer" in item:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": item["answer"]}]
        })
    
    return { "messages" : messages, "id": item["id"] }