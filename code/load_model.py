from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

def load_model(model_path="code/saved_models/qwen3vl-8b"):
    # Load model
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, device_map="auto", dtype="auto")
    # model.to("cuda").eval()

    # Load processor
    processor = AutoProcessor.from_pretrained(model_path)

    return model, processor