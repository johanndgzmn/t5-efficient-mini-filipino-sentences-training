import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import argparse
import os

def convert_model(model_path, output_path):
    print(f"\n🔹 Loading original T5 model from: {model_path}")
    model = T5ForConditionalGeneration.from_pretrained(model_path)

    print("🔹 Casting model weights to float32...")
    model = model.to(torch.float32)

    print("🔹 Saving float32 CPU model to:", output_path)
    model.save_pretrained(output_path)

    print("🔹 Saving tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(output_path)

    print("\n✅ Conversion complete!")
    print(f"👉 CPU Float32 model saved at: {output_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a T5 model into pure CPU Float32 format.")
    parser.add_argument("--model", type=str, required=True, help="Path to the original T5 model directory.")
    parser.add_argument("--out", type=str, required=True, help="Output directory for the CPU float32 model.")

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    convert_model(args.model, args.out)
