# Setup for T5 Model
1. Create a virtual environment using python 3.10 ``` py -3.10 -m venv t5 ```
2. Activate virtual environment ``` Scripts\activate```
3. Install dependencies
```
pip install transformers datasets sentencepiece accelerate optimum-intel neural-compressor pydantic
```

Use t5-train.py for training

Use quantize.py for quantisizing the trained model (makes the model smaller)

Use testing.py for testing

Sentence dataset credit - https://huggingface.co/datasets/jfernandez/cebuano-filipino-sentences
