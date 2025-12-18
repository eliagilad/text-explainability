# ### Based on SHAP text masker in order to use the text hierarchy ###
#
import math
import numpy as np
import pandas as pd
import shap
import transformers
from transformers import BertForSequenceClassification, BertTokenizer

# Load text
df = pd.read_json("../../../results/runtime_analysis4.json")
s = df.iloc[190]
sent = "".join(s["tokens"][1:])

model_dir = "../../models/bert_IMDB"
model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)


pred = transformers.pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None
    )

explainer = shap.Explainer(pred)
s_cluster = explainer.masker.clustering(sent)
print(s_cluster)



#### Clusters data structure:
# Each item in the list is a merge
# 0 - index of left token
# 1 - index of right token
# 2 - Height
# 3 - How many tokens in the group
