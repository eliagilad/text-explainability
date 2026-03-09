
import json
import numpy as np
import pandas as pd
import shap
import transformers
from transformers import BertForSequenceClassification, BertTokenizer

from partitions.balanced_partition import BalancedTextMasker
from partitions.random_partition import RandomTextMasker
import aopc

add_to_existing = True



root_dir = "" #"../../"
sst_sub_file = f"{root_dir}datasets/sst2_sampled_with_tokens.csv"
df_sst = pd.read_csv(sst_sub_file)
df_sst = df_sst[df_sst["num_tokens"] >= 20]

if add_to_existing:
    with open(f"{root_dir}results/partitions_aopc.json", "r") as f:
        all_results = json.load(f)
else:
    all_results = []

# Load model and tokenizer
model_dir = f"{root_dir}src/models/bert_IMDB"
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
    aopc_score, confidence, _ = aopc.bert_aopc(model, tokenizer, text, fi)

    model = model.to(device2)
    return fi, predicted_class, aopc_score, confidence

sample_size = 100

aopc_default = []
aopc_balanced = []

ind = 0

for i, row in df_sst.iterrows():
    if sample_size == 0:
        break

    sentence = row["sentence"]

    if not add_to_existing:
        # Default explainer
        explainer = shap.Explainer(pred)
        fi, predicted_class, aopc_score, confidence = get_fi(explainer, sentence, model, tokenizer)
        aopc_default.append(aopc_score)

    # Balanced explainer
    #masker = BalancedTextMasker(tokenizer)
    masker = RandomTextMasker(tokenizer)
    explainer2 = shap.Explainer(pred, masker)
    fi2, predicted_class2, aopc_score2, confidence2 = get_fi(explainer2, sentence, model, tokenizer)
    aopc_balanced.append(aopc_score2)

    sample_size -= 1

    

    if add_to_existing:
        all_results[ind]["random"] = {
                        "fi": fi2,
                        "aopc": aopc_score2
                    }
        
    else:
        results = { "text": sentence,
                    "num_tokens": row["num_tokens"],
                    "confidence": float(confidence),
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

    ind += 1
                    

# Save results
file = f"{root_dir}results/partitions_aopc.json"
with open(file, "w") as f:
    json.dump(all_results, f, indent=2)

print("Default AOPC: ", np.mean(np.array(aopc_default)))
print("Balanced partition AOPC: ", np.mean(np.array(aopc_balanced)))