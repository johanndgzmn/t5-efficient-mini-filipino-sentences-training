import os
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments
)
import torch

# ======================================================
# CONFIG
# ======================================================
MODEL_NAME = "google/t5-efficient-mini"

TRAIN_FILE = "training_new_augmented.tsv"
VAL_FILE = "validation_new_augmented.tsv"
CHECKPOINT_DIR = "./t5_filspell_mini"     # Folder where checkpoints & final model save

MAX_INPUT = 128
MAX_TARGET = 64
LR = 5e-5
EPOCHS = 30
BATCH_SIZE = 32
FP16 = False
WANDB = False

# ======================================================
# DISABLE WANDB
# ======================================================
if not WANDB:
    os.environ["WANDB_MODE"] = "offline"

# ======================================================
# LOAD DATASET
# ======================================================
dataset = load_dataset(
    "csv",
    data_files={"train": TRAIN_FILE, "validation": VAL_FILE},
    delimiter="\t"
)

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

# ======================================================
# PREPROCESSING
# ======================================================
def preprocess(batch):
    inputs = ["fix: " + (str(x) if x is not None else "") for x in batch["input"]]
    targets = [(str(x) if x is not None else "") for x in batch["target"]]

    enc = tokenizer(
        inputs,
        max_length=MAX_INPUT,
        truncation=True,
        padding="max_length"
    )

    dec = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET,
        truncation=True,
        padding="max_length"
    )

    enc["labels"] = dec["input_ids"]
    return enc

tokenized = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# ======================================================
# CHECK FOR CHECKPOINT
# ======================================================
checkpoint_path = None

if os.path.isdir(CHECKPOINT_DIR):
    subfolders = [
        f.path for f in os.scandir(CHECKPOINT_DIR)
        if f.is_dir() and "checkpoint-" in f.path
    ]
    if subfolders:
        def extract_step(path):
            return int(os.path.basename(path).replace("checkpoint-", ""))
        checkpoint_path = max(subfolders, key=extract_step)
        print(f"🔄 Found checkpoint: {checkpoint_path}")
    else:
        print("⚠️ No checkpoint found — training from scratch.")
else:
    print("⚠️ Model directory does not exist — training from scratch.")

# ======================================================
# LOAD MODEL
# ======================================================
if checkpoint_path:
    print("🔄 Resuming from last checkpoint...")
    model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)
else:
    print("🆕 Starting new training...")
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# ======================================================
# TRAINING ARGUMENTS
# ======================================================
args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    fp16=FP16,
    load_best_model_at_end=True,
)

# ======================================================
# TRAINER
# ======================================================
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
)
# Patch torch.load inside Trainer to disable weights_only restriction
orig_torch_load = torch.load

def patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return orig_torch_load(*args, **kwargs)

torch.load = patched_load

# ======================================================
# TRAIN OR RESUME
# ======================================================
if checkpoint_path:
    trainer.train(resume_from_checkpoint=checkpoint_path)
else:
    trainer.train()

# ======================================================
# SAVE FINAL MODEL
# ======================================================
trainer.save_model(CHECKPOINT_DIR)
tokenizer.save_pretrained(CHECKPOINT_DIR)

print("✅ Training completed successfully!")
