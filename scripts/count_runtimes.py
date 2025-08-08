import shap
import transformers
from transformers import BertForSequenceClassification, BertTokenizer
import json
import os

short_sentence = "Absolutely terrible."
longer_text = "The acting was wooden, and the plot was painfully predictable."
model_dir = "../src/models/bert_IMDB"
# config_class = BertConfig
model_class = BertForSequenceClassification
tokenizer_class = BertTokenizer
text_num = 0
results = [{ 'tokens': [], 'PartitionExplainer' : {} }]

model = model_class.from_pretrained(model_dir)
tokenizer = tokenizer_class.from_pretrained(model_dir)

pred = transformers.pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    top_k=None
)

# Count the model forward passes (this is what SHAP actually calls)
forward_count = 0
original_forward = pred.model.forward

def counting_forward(*args, **kwargs):
    global forward_count
    forward_count += 1
    return original_forward(*args, **kwargs)

# Patch the model's forward method
pred.model.forward = counting_forward

# Now use with SHAP
explainer = shap.Explainer(pred)
shap_values = explainer([short_sentence])

print(f"Model forward passes during SHAP: {forward_count}")
results[text_num]['tokens'] = list(shap_values.data[0])
results[text_num]['PartitionExplainer']['prediction_calls'] = forward_count
results[text_num]['PartitionExplainer']['feature_importance'] = list(shap_values.values[:, :, 0][0])

# Restore original if needed
pred.model.forward = original_forward

structure_file = "../results/runtime_analysis.json"

with open(structure_file, 'w') as f:
    json.dump(results, f, indent=2)

json.dumps(results, indent=2)
print()