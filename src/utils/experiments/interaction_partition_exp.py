import sys
from pathlib import Path

import shap
import transformers
from transformers import BertForSequenceClassification, BertTokenizer

THIS_FILE = Path(__file__).resolve()

# Make imports robust to different launch directories:
# - repo root in sys.path  -> import via `src...`
# - src/utils in sys.path  -> import via `partitions...`
for parent in THIS_FILE.parents:
    if (parent / "src").exists() and str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    if parent.name == "utils" and str(parent) not in sys.path:
        sys.path.insert(0, str(parent))

try:
    from src.utils.partitions.intercation_partition import InteractionTextMasker
except ModuleNotFoundError:
    from partitions.intercation_partition import InteractionTextMasker


text = "It is not good"

# Load your fine-tuned model/tokenizer (same as other experiments)
repo_root = next((p for p in THIS_FILE.parents if (p / "src").exists()), THIS_FILE.parents[0])
model_dir = str(repo_root / "src" / "models" / "bert_IMDB")
model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)

pred = transformers.pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    top_k=None,
)

# Interaction-based partition (HEDGE)
masker = InteractionTextMasker(tokenizer, model, win_size=1, max_length=250)
explainer = shap.Explainer(pred, masker)

# Example run
_ = explainer([text], max_evals=2000)