import shap
import transformers
from transformers import BertForSequenceClassification, BertTokenizer
import json
import numpy as np
import os
import pandas as pd

#structure_file = "../results/runtime_analysis3.json"
add = False
override_existing = True
structure_file = "../results/runtime_analysis4.json"

sst_sub_file = "../datasets/sst2_sampled_with_tokens.csv"
df_sst = pd.read_csv(sst_sub_file)
# short_sentences = ["Very, boring, film, terrible",
#                    "Bad film terrible and very boring."]

#setences = list(df_sst['sentence'])

short_sentences = ["Absolutely terrible.",
                   "This movie was terrible.",
                   "The acting was dull and boring.",
                   "The plot was weak and the acting terrible.",
                   "This film was boring and slow with a predictable ending.",
                   "The movie was slow, the dialogue wooden, and the ending predictable.",
                   "The characters were flat, the dialogue awkward, and the pacing unbearably slow throughout the film .",
                    "I found the acting unconvincing, the story predictable, and the constant overuse of clichés made the entire film tedious .",
                   "The film dragged on far too long, with weak performances, a predictable script, and no real emotional weight to keep me engaged ."]
#longer_text = "The acting was wooden, and the plot was painfully predictable."
model_dir = "../src/models/bert_IMDB"
# config_class = BertConfig
model_class = BertForSequenceClassification
tokenizer_class = BertTokenizer
text_num = 0
results = []

model = model_class.from_pretrained(model_dir)
tokenizer = tokenizer_class.from_pretrained(model_dir)


#for short_sentence in short_sentences:
for i, row in df_sst.iterrows():
    short_sentence = row["sentence"]

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
        forward_count += len(kwargs['input_ids'])
        #print(f"\n=== model.forward call #{forward_count} ===")
        # traceback.print_stack(limit=15)
        return original_forward(*args, **kwargs)

    # Patch the model's forward method
    pred.model.forward = counting_forward

    # Now use with SHAP
    explainer = shap.Explainer(pred)

    results.append({'tokens': [], 'num_tokens': row["num_tokens"], 'PartitionExplainer': {}})
    shap_values = explainer([short_sentence], max_evals=6000)
    predicted_class = np.argmax(np.sum(shap_values.values[0], axis=0))

    print(f"Model forward passes during SHAP: {forward_count}")
    results[text_num]['tokens'] = list(shap_values.data[0])
    results[text_num]['PartitionExplainer']['prediction_calls'] = forward_count
    results[text_num]['PartitionExplainer']['feature_importance'] = list(shap_values.values[:, :, predicted_class][0])

    # Restore original if needed
    pred.model.forward = original_forward
    text_num += 1



 # Load existing results if the file exists
def get_exiting(structure_file):
    if os.path.exists(structure_file):
        with open(structure_file, "r") as f:
            existing_results = json.load(f)
    else:
        existing_results = []

    return existing_results

if add:
    existing_results = get_exiting(structure_file)

    # Append the new results
    existing_results.extend(results)  # `results` is your new array

    # Save back to file
    with open(structure_file, "w") as f:
        json.dump(existing_results, f, indent=2)
elif override_existing:
    existing_results = get_exiting(structure_file)

    for i in range(len(results)):
        partition_result = results[i]["PartitionExplainer"]
        existing_results[i]['PartitionExplainer'] = partition_result

    with open(structure_file, "w") as f:
        json.dump(existing_results, f, indent=2)
else:
    with open(structure_file, 'w') as f:
        json.dump(results, f, indent=2)

    json.dumps(results, indent=2)

print()