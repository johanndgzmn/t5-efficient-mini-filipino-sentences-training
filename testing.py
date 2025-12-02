from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# --------------------------
# Load 8-bit Model on GPU
# --------------------------
model = T5ForConditionalGeneration.from_pretrained(
    "./t5_filspell_mini_8bit",
    device_map="auto",
    load_in_8bit=True
)

tokenizer = T5Tokenizer.from_pretrained("./t5_filspell_mini_8bit")


# --------------------------
# Correction Function
# --------------------------
def correct(text_line):
    if not text_line.strip():
        return ""  # skip empty lines

    prompt = "fix: " + text_line
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        inputs["input_ids"],
        max_length=64,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# --------------------------
# File I/O
# --------------------------
input_file = "./test.txt"     # your text file
output_file = "./corrected_output.txt"

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

corrected_lines = []
for line in lines:
    corrected = correct(line.strip())
    corrected_lines.append(corrected)
    print("Input:   ", line.strip())
    print("Corrected:", corrected)
    print()

with open(output_file, "w", encoding="utf-8") as f:
    for c in corrected_lines:
        f.write(c + "\n")

print(f"✅ Done! Corrected text saved to: {output_file}")
