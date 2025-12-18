import numpy as np
import time
import torch
from transformers import BertForSequenceClassification, BertTokenizer

class BertModel:
    def __init__(self, model_dir="../../src/models/bert_IMDB",
                 model_class=BertForSequenceClassification,
                 tokenizer_class = BertTokenizer

    ):
        self.model = model_class.from_pretrained(model_dir)
        self.tokenizer = tokenizer_class.from_pretrained(model_dir, model_max_length=512)
        self.mask_token_id = self.tokenizer.mask_token_id

    def get_tokens(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        return inputs

    def predict(self, text, padded=False):
        inputs = self.get_tokens(text)
        input_ids = inputs['input_ids'][0]  # Remove batch dimension
        attention_mask = inputs['attention_mask'][0].unsqueeze(0)
        token_ids = input_ids.cpu().numpy()

        if isinstance(token_ids, np.ndarray):
            token_ids = torch.tensor(token_ids).unsqueeze(0)
        elif isinstance(token_ids, list):
            token_ids = torch.tensor(token_ids).unsqueeze(0)
        else:
            token_ids = token_ids.unsqueeze(0)

        if padded:
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = (token_ids != self.tokenizer.pad_token_id).long()

        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = self.model(input_ids=token_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=-1)

        end_time = time.perf_counter()

        return {
            "probabilities": probabilities[0].cpu().numpy(),
            "prediction_time": end_time - start_time
        }
