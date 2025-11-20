# Standard library imports
import os
import json
import gc
import time
import csv
from collections import Counter
import torch

# Local module imports
from seed import set_seed
from load_model import load_model
from dataset import make_conversation
from retrieval import get_top3_diverse_frames_parallel


def inference(messages, model_path="../saved_models/qwen3vl-8b"):
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
        generated_ids = model.generate(**inputs,max_new_tokens=128)

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
    # define 
    set_seed()
    DATA_PATH = "data"

    # read all test cases
    with open(os.path.join(DATA_PATH, "public_test.json")) as f:
        test_cases = json.load(f)["data"][:20]
    for case in test_cases:
        case["video_path"] = case["video_path"].replace("public_test", "data")
    
    print(test_cases[0])

    all_time = []
    all_result = []
    model_path =  "saved_models/qwen25vl-7b"
    # run 
    for item in test_cases:
        start_time = time.time()
        # retrieve top 4 frames 
        top_frames, top_scores = get_top3_diverse_frames_parallel(item["video_path"], item["question"])
        print(top_scores)

        # make conversation
        messages = make_conversation(item, top_frames)

        # inference 
        output = []
        output_text = inference(messages, model_path)
        # Lấy kí tự đầu tiên của output_text
        pred = output_text[0][0]
        if pred not in ["A", "B", "C", "D"] or pred is None: 
            pred = "A"
        output.append(pred)

        # voting


        all_time.append(time.time() - start_time)
        all_result.append(pred)
    
    # save result 
    # prepare data for saving
    time_submission_data = []
    submission_data = []
    PATH_SAVE = "results"
    os.makedirs(PATH_SAVE, exist_ok=True)

    for item, pred, t in zip(test_cases, all_result, all_time):
        time_ms = int(t * 1000)  # convert seconds to milliseconds
        time_submission_data.append([item["id"], pred, time_ms])
        submission_data.append([item["id"], pred])

    # save time_submission.csv
    with open("time_submission.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "answer", "time"])  # header
        writer.writerows(time_submission_data)

    # save submission.csv
    with open("submission.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "answer"])  # header
        writer.writerows(submission_data)

    print(f"CSV files saved successfully! at {PATH_SAVE}")




    


