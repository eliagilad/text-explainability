
import json
import numpy as np
import pandas as pd
import shap
import transformers
from transformers import BertForSequenceClassification, BertTokenizer

from partitions.balanced_partition import BalancedTextMasker
import aopc

sst_sub_file = "../../datasets/sst2_sampled_with_tokens.csv"
df_sst = pd.read_csv(sst_sub_file)
df_sst = df_sst[df_sst["num_tokens"] >= 30]

positive_example = "i have always appreciated a smartly written motion picture , and , whatever flaws igby goes down may possess , it is undeniably that . "
negative_example = "is without doubt an artist of uncompromising vision , but that vision is beginning to feel , if not morally bankrupt , at least terribly monotonous "

# Load model and tokenizer
model_dir = "../models/bert_IMDB"
model_class = BertForSequenceClassification
tokenizer_class = BertTokenizer
model = model_class.from_pretrained(model_dir)
tokenizer = tokenizer_class.from_pretrained(model_dir)

device = next(model.parameters()).device
print("Model device:", device)

pred = transformers.pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None
    )

def get_fi(explainer, text, model, tokenizer):
    shap_values = explainer([text], max_evals=6000)
    device2 = next(model.parameters()).device
    print("Model device:", device2)
    predicted_class = np.argmax(np.sum(shap_values.values[0], axis=0))
    fi = list(shap_values.values[:, :, predicted_class][0])

    # Compute AOPC
    model.eval()
    model = model.to(device)
    aopc_score, _, _ = aopc.bert_aopc(model, tokenizer, text, fi)

    model = model.to(device2)
    return fi, predicted_class, aopc_score

sample_size = 10

all_results = []
aopc_default = []
aopc_balanced = []

for i, row in df_sst.iterrows():
    if sample_size == 0:
        break

    short_sentence = row["sentence"]

    # Default explainer
    explainer = shap.Explainer(pred)

    # Balanced explainer
    masker = BalancedTextMasker(tokenizer)
    explainer2 = shap.Explainer(pred, masker)

    sample_size -= 1

    fi, predicted_class, aopc_score = get_fi(explainer, positive_example, model, tokenizer)
    fi2, predicted_class2, aopc_score2 = get_fi(explainer2, positive_example, model, tokenizer)

    aopc_default.append(aopc_score)
    aopc_balanced.append(aopc_score2)

    results = { "text": positive_example,
                "predicted_class": int(predicted_class),
                "default": {
                    "fi": fi,
                    "aopc": aopc_score
                },
                "balanced": {
                    "fi": fi2,
                    "aopc": aopc_score2}
                }
    all_results.append(results)

# Save results
file = "../../results/partitions_aopc.json"
with open(file, "w") as f:
    json.dump(all_results, f, indent=2)

print("Default AOPC: ", np.mean(np.array(aopc_default)))
print("Balanced partition AOPC: ", np.mean(np.array(aopc_balanced)))