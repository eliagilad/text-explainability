import shap
import transformers
from transformers import BertForSequenceClassification, BertTokenizer
import json

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

class SHAPexplainer2:
    def __init__(self, model, tokenizer, words_dict, words_dict_reverse):
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cpu"
        self.tweet_tokenizer = TweetTokenizer()
        self.words_dict = words_dict
        self.words_dict_reverse = words_dict_reverse

    def split_string(self, string):
        data_raw = self.tweet_tokenizer.tokenize(string)
        data_raw = [x for x in data_raw if x not in ".,:;'"]
        return data_raw

    def softmax(self, it):
        exps = np.exp(np.array(it))
        return exps / np.sum(exps)

    def dt_to_idx(self, data, max_seq_len=None):
        idx_dt = [[self.words_dict_reverse[xx] for xx in x] for x in data]
        if not max_seq_len:
            max_seq_len = min(max(len(x) for x in idx_dt), 512)
        for i, x in enumerate(idx_dt):
            if len(x) < max_seq_len:
                idx_dt[i] = x + [0] * (max_seq_len - len(x))
        return np.array(idx_dt), max_seq_len

    def predict(self, indexed_words):
        # self.model.to(self.device)

        sentences = [[self.words_dict[xx].lower() if xx != 0 else "" for xx in x] for x in indexed_words]
        sentences = [" ".join(s) for s in sentences]
        tokens_tensor = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)

        with torch.no_grad():
            outputs = self.model(**tokens_tensor)
            predictions = outputs.detach().cpu().numpy()

        final = [self.softmax(x) for x in predictions]
        return np.array(final)

predictor = SHAPexplainer(model, tokenizer, words_dict, words_dict_reverse)
explainer = shap.KernelExplainer(model=predictor.predict, data=shap.kmeans(idx_train_data, k=50))
to_use = idx_texts[-1:]
shap_values = explainer.shap_values(X=to_use, nsamples=64, l1_reg="aic")

#explainer = shap.KernelExplainer(f, np.reshape(reference, (1, len(reference))))
#shap_values = explainer.shap_values(x)

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