from datasets import load_dataset, Dataset
from PIL import Image
import pandas as pd
import os
from unsloth import FastVisionModel
import torch
from transformers import AutoTokenizer, TextStreamer
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig


############################################## FORMATTING DATASET ######################################################

# Load Json dataset
df = pd.read_json("./data/train_labels.jsonl", lines=True)
dataset = Dataset.from_pandas(df)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df)

# Function to load images
def load_image(example, raw_data_dir: str = './data'):
    example["image"] = Image.open(os.path.join(raw_data_dir, example["path"])).convert("RGB")
    return example

# Apply image loading function
dataset = dataset.map(load_image)

# Select the model that we want to fine-tune and set maximum sequence length
model_name = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit"
max_seq_length = 512

# Load tokenizer & model
model, tokenizer = FastVisionModel.from_pretrained(
    model_name,
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
)

# Sets the model up for PEFT
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,

    r = 8,
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    random_state = 42
)

# Set our prompt
instruct_prompt = 'You are an expert at inspecting power grid infrastructure. Specifically, you analyze images and determine if there is or is not damage from a woodpecker to the wooden utility/power pole(s). Your output must be a valid Json object, with only one key, "has_woodpecker_damage", mapping to a boolean true or false.'

def generate_conversation(sample):
    """
    Converts a sample from our Json dataset into a conversational sample
    :param sample: sample from Json dataset
    :return: conversational dictionary sample
    """
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : instruct_prompt},
            {"type" : "image", "image" : sample["image"]} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : f'{{"has_woodpecker_damage": {sample["has_woodpecker_damage"]}}}'} ]
        },
    ]
    return { "messages" : conversation }


# Convert our whole dataset into conversation format
conversational_dataset = [generate_conversation(sample) for sample in dataset]
# Show our dataset
print(conversational_dataset[0])


###################################### TEST SOME SAMPLES WITH BASE MODEL ###############################################

def infer(my_model, image_sample):
    """
    Run an image sample through the model
    :param my_model: A PEFT-formatted model instance
    :param image_sample: An image sample (PIL Image)
    :return: None, prints reponse
    """
    # set model to inference mode
    FastVisionModel.for_inference(my_model)
    # create our messages list
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": instruct_prompt},
            {"type": "image"}
        ]}
    ]

    # Convert to appropriate chat/instruct style
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
    # tokenize input
    inputs = tokenizer(
        image_sample,
        input_text,
        add_special_tokens = False,
        return_tensors = "pt",
    ).to("cuda")

    # Setup token streamer (dumps to stdout)
    text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    # Run the model
    _ = my_model.generate(**inputs, streamer = text_streamer, max_new_tokens = max_seq_length,
                       use_cache = True, temperature = 0.6, min_p = 0.1)


# Test the sample on an image without woodpecker damage
print("Response on Test Image (No Damage): ")
infer(model, Image.open("./data/normal/TEST.jpg").convert("RGB"))
# Test the sample on an image with woodpecker damage
print("Response on Test Image (Has Woodpecker Damage): ")
infer(model, Image.open("./data/woodpecker/TEST.jpg").convert("RGB"))
exit()
############################################### TRAINING ###############################################################
# Enable training mode
FastVisionModel.for_training(model)

# Setup supervised fine tuning trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = conversational_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 30,  # only training for 30 steps (only use this for debugging)
        # num_train_epochs = 1, # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",     # For Weights and Biases

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4,
        max_seq_length = max_seq_length,
    ),
)
# Actually run the training
trainer_stats = trainer.train()
print(trainer_stats)

# Save the tokenizer and finetuned model
model.save_pretrained("qwen2.5vlinstruct-woodpecker")
tokenizer.save_pretrained("qwen2.5vlinstruct-llama-3.2-11B-Vision-Instruct-unsloth-bnb-4bit-woodpecker")


################################### RUN TEST SAMPLES THROUGH FINETUNED MODEL ###########################################
# Test the sample on an image without woodpecker damage
print("Response on Test Image (No Damage): ")
infer(model, Image.open("./data/normal/TEST.jpg").convert("RGB"))
# Test the sample on an image with woodpecker damage
print("Response on Test Image (Has Woodpecker Damage): ")
infer(model, Image.open("./data/woodpecker/TEST.jpg").convert("RGB"))
