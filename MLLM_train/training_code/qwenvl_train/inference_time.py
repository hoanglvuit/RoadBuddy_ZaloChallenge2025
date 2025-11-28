from tqdm import tqdm
import pandas as pd
from seed import set_seed
from dataset import load_data, make_conversation
from model import load_model_to_inference
import time
import torch
import gc
set_seed()

def inference_one_model(model, tokenizer, messages):
    with torch.no_grad():
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        tokens_per_example = inputs['input_ids'].shape[1]
        print(f'Tokens per example: {tokens_per_example}')

        generated_ids = model.generate(**inputs, max_new_tokens=128)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    return output_text

# def inference(messages):
#     model, tokenizer = load_model_to_inference()
#     # Tokenization and preparing inputs
#     inputs = tokenizer.apply_chat_template(
#         messages,
#         tokenize=True,
#         add_generation_prompt=True,
#         return_dict=True,
#         return_tensors="pt"
#     )
#     inputs = inputs.to(model.device)
#     tokens_per_example = inputs['input_ids'].shape[1]
#     print(f'Tokens per example: {tokens_per_example}')
    
#     # Inference: Generation of the output
#     generated_ids = model.generate(**inputs,max_new_tokens=128)
#     generated_ids_trimmed = [
#         out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#     ]
#     output_text = tokenizer.batch_decode(
#         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )
#     del model, tokenizer, inputs, generated_ids, generated_ids_trimmed
#     gc.collect()
#     torch.cuda.empty_cache()
#     return output_text


if __name__ == "__main__":
    dataset = load_data("./", mode="public_test")
    dataset = [make_conversation(sample, mode="public_test") for sample in tqdm(dataset, desc="make conversation")]
    print(dataset[0])
    
    id_list = []
    answer_list = []
    time_list = []
    for example in dataset:
        print(f"ID: {example['id']}")
        time_start = time.time()
        model, tokenizer = load_model_to_inference()
        output_text = inference_one_model(model, tokenizer,example["messages"])
    
        # Lấy ký tự đầu tiên
        pred = output_text[0][0]
    
        # Nếu không phải A/B/C/D thì đặt thành A
        if pred not in ["A", "B", "C", "D"] or pred is None:
            pred = "A"
    
        id_list.append(example["id"])
        answer_list.append(pred)
        time_list.append(time.time() - time_start)
        print(f"Inference time: {time_list[-1]:.2f} seconds")
        print("Raw output:", output_text)
        print("Final answer:", pred)
        print("-" * 100)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    
    # save csv with id and answer
    df = pd.DataFrame({"id": id_list, "answer": answer_list, "time": time_list})
    df.to_csv("public_test_answer_time.csv", index=False)