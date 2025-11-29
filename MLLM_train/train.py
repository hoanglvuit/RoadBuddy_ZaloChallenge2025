from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm
from seed import set_seed
from dataset import load_data, make_conversation
from model import load_model_to_train
set_seed()

if __name__ == "__main__":

    # load dataset
    dataset = load_data("./", mode="train")
    print(dataset[0])
    dataset_2 = load_data("./", mode="public_test")
    print(dataset_2[0])

    # make conversation
    dataset = [make_conversation(item) for item in tqdm(dataset, desc="make conversation")]
    dataset_2 = [make_conversation(item) for item in tqdm(dataset_2, desc="make conversation")]
    print(dataset[0])
    print(dataset[0])
    dataset = dataset + dataset_2

    # load model
    model, tokenizer = load_model_to_train(model_name="Qwen/Qwen3-VL-8B-Instruct", finetune_vision=True)

    # define trainer
    FastVisionModel.for_training(model) # Enable for training!
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer, max_seq_length=8500, resize = "max"),
        train_dataset = dataset,
        args = SFTConfig(
            save_strategy ="no",
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 8,
            warmup_steps = 5,
            num_train_epochs = 2,
            learning_rate = 4e-4,
            logging_steps = 5,
            optim = "adamw_8bit",
            weight_decay = 0.001,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none",

            # You MUST put the below items for vision finetuning:
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
        max_seq_length = 8500,
        ),
    )

    trainer_stats = trainer.train()

    # save model
    if True: model.save_pretrained_merged("unsloth_finetune", tokenizer)