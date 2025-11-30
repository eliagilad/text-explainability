import json
import numpy as np
import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer

should_save = False
aopc_file = "../../results/aopc.json"

def aopc(feature_importance_scores, tokens, prediction_func, mask_value):
    full_prediction = prediction_func(tokens)
    fp_max, fp_max_ind = np.max(full_prediction), np.argmax(full_prediction)
    n_features = len(feature_importance_scores)

    # Features indices highest to lowest
    sorted_indices = np.argsort(feature_importance_scores)[::-1]
    sum_diff = 0

    for k in range(1, n_features + 1):
        modified_input = tokens.copy()
        top_k_indices = sorted_indices[:k]
        modified_input[top_k_indices] = mask_value
        k_prediction = prediction_func(modified_input)
        k_prediction = k_prediction[fp_max_ind]
        sum_diff += (fp_max - k_prediction)

    aopc = sum_diff / n_features
    return aopc

def bert_aopc(model, tokenizer, text, feature_importance_scores):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs['input_ids'][0]  # Remove batch dimension
    attention_mask = inputs['attention_mask'][0]

    def prediction_func(token_ids):
        """Returns class probabilities for given token IDs."""
        # Convert to tensor if needed
        if isinstance(token_ids, np.ndarray):
            token_ids = torch.tensor(token_ids).unsqueeze(0)
        elif isinstance(token_ids, list):
            token_ids = torch.tensor(token_ids).unsqueeze(0)
        else:
            token_ids = token_ids.unsqueeze(0)

        # Create attention mask (1 for real tokens, 0 for padding)
        attn_mask = (token_ids != tokenizer.pad_token_id).long()

        with torch.no_grad():
            outputs = model(input_ids=token_ids, attention_mask=attn_mask)
            probabilities = torch.softmax(outputs.logits, dim=-1)

        return probabilities[0].cpu().numpy()

    # Calculate AOPC
    mask_token_id = tokenizer.mask_token_id
    tokens_numpy = input_ids.cpu().numpy()

    aopc_score = aopc(
        feature_importance_scores,
        tokens_numpy,
        prediction_func,
        mask_token_id
    )

    return aopc_score

def get_tokens_and_text(tokenizer, text):
    """Helper function to see the actual tokens."""
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    return tokens, inputs['input_ids'][0].numpy()

def generic_aopc_example():
    """
    For non-BERT models, you just need different masking strategies.
    """
    print("\n" + "=" * 60)
    print("Masking strategies for different model types:\n")

    strategies = {
        "BERT/Transformers": "Use tokenizer.mask_token_id (usually 103 for BERT)",
        "CNN on images": "Use 0 (black pixel) or mean pixel value",
        "Tabular data": "Use feature mean, median, or 0",
        "Time series": "Use 0, mean value, or carry-forward last value",
        "Graph neural networks": "Remove edges or set node features to 0"
    }

    for model_type, strategy in strategies.items():
        print(f"• {model_type}:")
        print(f"  {strategy}\n")


# Load model
model_dir = "../models/bert_IMDB"
# config_class = BertConfig
model_class = BertForSequenceClassification
tokenizer_class = BertTokenizer


model = model_class.from_pretrained(model_dir)
tokenizer = tokenizer_class.from_pretrained(model_dir)

#bert_aopc(model, tokenizer)

#model_name = "distilbert-base-uncased-finetuned-sst-2-english"
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Example text
# text = "This movie is absolutely wonderful and amazing!"
#
# # Get tokens to see what we're working with
# tokens, token_ids = get_tokens_and_text(tokenizer, text)
# print(f"Text: {text}")
# print(f"Tokens: {tokens}")
# print(f"Token IDs: {token_ids}")
# print(f"Number of tokens: {len(tokens)}\n")

# Example attribution scores (you'd get these from your explanation method)
# For demonstration, let's say we have importance scores for each token
# Higher score = more important
# NOTE: This should match the length of token_ids
# feature_importance_scores = np.array([
#     0.1,  # [CLS]
#     0.3,  # this
#     0.5,  # movie
#     0.2,  # is
#     0.9,  # absolutely
#     0.8,  # wonderful
#     0.4,  # and
#     0.7,  # amazing
#     0.05,  # !
#     0.1  # [SEP]
# ])

sst_sub_file = "../../datasets/sst2_sampled_with_tokens.csv"
df_sst = pd.read_csv(sst_sub_file)

results_file = "../../results/runtime_analysis4.json"
with open(results_file, "r") as f:
    fi_file = json.load(f)

aopcs = []

for i, row in df_sst.iterrows():
    fi_row = fi_file[i]

    if fi_row["num_tokens"] % 3 > 0:
        continue

    if fi_row["num_tokens"] > 18:
        break

    text = row["sentence"]
    print(text)
    fi_partition = fi_row["PartitionExplainer"]["feature_importance"]
    fi_hedge = fi_row["HEDGE"]["feature_importance"]

    # Calculate AOPC
    aopc_score_partition = bert_aopc(
        model,
        tokenizer,
        text,
        fi_partition
    )

    aopc_score_hedge = bert_aopc(
        model,
        tokenizer,
        text,
        fi_hedge
    )

    print(f"aopc score partition: {aopc_score_partition}")
    print(f"aopc score hedge: {aopc_score_partition}")

    aopcs.append({"text": text,
                 "tokens": fi_row["tokens"],
                    "num_tokens": fi_row["num_tokens"],
                    "aopc_partition": aopc_score_partition,
                  "aopc_hedge": aopc_score_hedge})

    if should_save:
        with open(aopc_file, 'w') as f:
            json.dump(aopcs, f, indent=2)

# print(f"AOPC Score: {aopc_score:.4f}")
# print("\nInterpretation:")
# print("- Higher AOPC = Better explanation (removing important features hurts prediction)")
# print("- Lower AOPC = Worse explanation (important features don't affect prediction)")
