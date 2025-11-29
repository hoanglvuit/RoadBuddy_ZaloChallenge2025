import os
from PIL import Image

# Instruction for the model
instruction = """
You are provided with four key frames captured sequentially from a camera mounted on the front of a car. An object may appear in multiple frames.
Based on these frames and according to Vietnamese traffic laws, select the choice to answer the question below — paying special attention to all traffic signs, traffic lights, road markings, lanes, and other vehicles.
"""

def make_conversation(item, top_frames):

    # Load all images from the folder
    images = []
    for i, frame in enumerate(top_frames):
        images.append(frame)

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
    return messages