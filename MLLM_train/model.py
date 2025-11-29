from unsloth import FastVisionModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def load_model_to_train(model_name="Qwen/Qwen3-VL-8B-Instruct", finetune_vision=True):
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit = True, 
        use_gradient_checkpointing = "unsloth",
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = finetune_vision, 
        finetune_language_layers   = True, 
        finetune_attention_modules = True,
        finetune_mlp_modules       = True, 

        r = 32,           
        lora_alpha = 32,  
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
        use_rslora = False,  
        loftq_config = None, 
    )
    return model, tokenizer


def load_model_to_inference(model_path="unsloth_finetune"):
    # default: Load the model on the available device(s)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,  device_map="auto", dtype="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor 