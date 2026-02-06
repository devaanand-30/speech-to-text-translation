import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ✅ Force CPU (avoid CUDA/cuDNN errors)

import torch
from transformers import MarianMTModel, MarianTokenizer

class Translator:
    def __init__(self, direction: str):
        """
        direction must be:
        - "te-en"  → Telugu to English
        - "en-te"  → English to Telugu
        """
        self.direction = direction

        if direction == "te-en":
            model_name = "Helsinki-NLP/opus-mt-mul-en"   # Multi-language → English
        elif direction == "en-te":
            model_name = "Helsinki-NLP/opus-mt-en-mul"   # English → Multi-language (we force Telugu)
        else:
            raise ValueError("direction must be 'te-en' or 'en-te'")

        # ✅ Download models to local folder models/marian/
        cache_dir = os.path.join("models", "marian")

        self.tokenizer = MarianTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = MarianMTModel.from_pretrained(model_name, cache_dir=cache_dir).to("cpu")

    def __call__(self, text: str, max_length: int = 256) -> str:
        """
        Translate text based on direction.
        For English → Telugu, we force the model to output Telugu by using '>>tel<<' token.
        """
        # ✅ Force output in Telugu language for English → Telugu
        if self.direction == "en-te":
            text = ">>tel<< " + text

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to("cpu")

        # Generate translation
        outputs = self.model.generate(**inputs, max_length=max_length, num_beams=5)

        # Decode translated output
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
