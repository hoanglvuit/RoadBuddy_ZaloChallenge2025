import json
from PIL import Image


def load_data(json_path, image_path): 
    with open(json_path, 'r') as f:
        data = json.load(f)["data"]
