from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_id = "./output_3"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    device_map="auto"
)

model.save_pretrained("./t5_filspell_mini_8bit")
tokenizer.save_pretrained("./t5_filspell_mini_8bit")

print("✅ 8-bit GPU quantization done!")
