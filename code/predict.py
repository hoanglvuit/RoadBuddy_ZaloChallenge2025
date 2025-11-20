# Standard library imports
import os
import json
import gc
import time
import csv
import torch
from glob import glob


# Local module imports
from seed import set_seed
from load_model import load_model
from dataset import make_conversation
from retrieval import get_top3_diverse_frames_parallel


def inference(messages, model_path="saved_models/qwen3vl-8b"):
    # load model 
    model, tokenizer = load_model(model_path)

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to("cuda")

    tokens_per_example = inputs['input_ids'].shape[1]
    print(f'Tokens per example: {tokens_per_example}')
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs,max_new_tokens=2)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = tokenizer.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # release gpu memory 
    del model, tokenizer, inputs, generated_ids, generated_ids_trimmed
    torch.cuda.empty_cache()
    gc.collect()
    return output_text

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("result"):
        os.makedirs("result")
    # define 
    set_seed()
    DATA_PATH = "data"

    # Lấy tất cả file JSON trong thư mục data
    json_files = glob(os.path.join(DATA_PATH, "*.json"))

    test_cases = []

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)["data"]
            for case in data:
                # Chia path thành các phần
                parts = case["video_path"].split(os.sep)
                # Thay phần đầu tiên bằng 'data'
                parts[0] = DATA_PATH
                # Ghép lại thành path mới
                case["video_path"] = os.sep.join(parts)
            test_cases.extend(data)

    print(f"Tổng số test case: {len(test_cases)}")

    all_time = []
    all_result = []
    # sửa ở đây ---------------------------------------------fsaasdadsad
    model_path = "code/saved_models/qwen3vl-8b"
    # run 
    for item in test_cases:
        start_time = time.time()
        # retrieve top 4 frames 
        top_frames, top_scores = get_top3_diverse_frames_parallel(item["video_path"], item["question"])
        print(top_scores)

        # make conversation
        messages = make_conversation(item, top_frames)

        # inference 
        output_text = inference(messages, model_path)

        # Lấy kí tự đầu tiên của output_text
        pred = output_text[0][0]
        if pred not in ["A", "B", "C", "D"] or pred is None: 
            pred = "A"

        all_time.append(time.time() - start_time)
        all_result.append(pred)

        print(item["id"], pred)


    # save result 
    # prepare data for saving
    time_submission_data = []
    submission_data = []

    for item, pred, t in zip(test_cases, all_result, all_time):
        time_ms = int(t * 1000)  # convert seconds to milliseconds
        time_ms = t
        time_submission_data.append([item["id"], pred, time_ms])
        submission_data.append([item["id"], pred])

    
    # save time_submission.csv
    with open("result/time_submission.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "answer", "time"])  # header
        writer.writerows(time_submission_data)

    # save submission.csv
    with open("result/submission.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "answer"])  # header
        writer.writerows(submission_data)

    print(f"CSV files saved successfully! at result")


