import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import shap
import transformers
from transformers import BertForSequenceClassification, BertTokenizer

import aopc

# Change this per run to avoid overriding old results.
EXPERIMENT_VERSION = "v1"
ADD_TO_EXISTING = False

sample_size = 10

THIS_FILE = Path(__file__).resolve()

# Make imports robust to different launch directories.
for parent in THIS_FILE.parents:
    if (parent / "src").exists() and str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    if parent.name == "utils" and str(parent) not in sys.path:
        sys.path.insert(0, str(parent))

try:
    from src.utils.partitions.balanced_partition import BalancedTextMasker
    from src.utils.partitions.random_partition import RandomTextMasker
    from src.utils.partitions.intercation_partition import InteractionTextMasker
except ModuleNotFoundError:
    from partitions.balanced_partition import BalancedTextMasker
    from partitions.random_partition import RandomTextMasker
    from partitions.intercation_partition import InteractionTextMasker




repo_root = next((p for p in THIS_FILE.parents if (p / "src").exists()), THIS_FILE.parents[0])
root_dir = str(repo_root) + "/"

sst_sub_file = f"{root_dir}datasets/sst2_sampled_with_tokens.csv"
df_sst = pd.read_csv(sst_sub_file)
df_sst = df_sst[df_sst["num_tokens"] >= 20]

results_file = f"{root_dir}results/partitions_aopc_{EXPERIMENT_VERSION}.json"

if ADD_TO_EXISTING:
    with open(results_file, "r") as f:
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

PARTITIONS = {
    "default": lambda: None,
    "balanced": lambda: BalancedTextMasker(tokenizer),
    "random": lambda: RandomTextMasker(tokenizer),
    "interaction": lambda: InteractionTextMasker(tokenizer, model, win_size=1, max_length=250),
}

aopc_by_partition = {k: [] for k in PARTITIONS.keys()}

ind = 0

for i, row in df_sst.iterrows():
    if sample_size == 0:
        break

    sentence = row["sentence"]

    # Ensure base record exists.
    if ADD_TO_EXISTING:
        results = all_results[ind]
    else:
        results = {
            "text": sentence,
            "num_tokens": int(row["num_tokens"]),
        }

    for pname, masker_factory in PARTITIONS.items():
        if ADD_TO_EXISTING and pname in results:
            continue

        masker = masker_factory()
        explainer = shap.Explainer(pred) if masker is None else shap.Explainer(pred, masker)
        fi, predicted_class, aopc_score, confidence = get_fi(explainer, sentence, model, tokenizer)
        aopc_by_partition[pname].append(aopc_score)

        results[pname] = {
            "fi": fi,
            "aopc": aopc_score,
            "predicted_class": int(predicted_class),
            "confidence": float(confidence),
        }

    sample_size -= 1

    if not ADD_TO_EXISTING:
        all_results.append(results)

    ind += 1
                    

# Save results
with open(results_file, "w") as f:
    json.dump(all_results, f, indent=2)

for pname, scores in aopc_by_partition.items():
    if len(scores) > 0:
        print(f"{pname} AOPC: {float(np.mean(np.array(scores)))}")