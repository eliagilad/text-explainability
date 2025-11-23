from datasets import load_dataset
from transformers import BertTokenizer
import pandas as pd

def count_tokens(example):
    # Load your BERT tokenizer (replace with your specific tokenizer)
    model_dir = "../models/bert_IMDB"
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    #tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer.encode(example['sentence'], add_special_tokens=False)
    example['num_tokens'] = len(tokens)
    return example

# Load SST-2 dataset
dataset = load_dataset("glue", "sst2")

# Access splits
train_data = dataset['train']
train_data = train_data.select(range(5000))

# Add token count to train split
train_with_tokens = train_data.map(count_tokens)
train_df = train_with_tokens.to_pandas()
max_length = 33
train_df = train_df[train_df['num_tokens'] <= max_length]

# Sample 10 sentences from each length
sampled_df = train_df.groupby('num_tokens', group_keys=False).apply(
    lambda x: x.sample(min(len(x), 10), random_state=42)
).reset_index(drop=True)

# Sort by token length
sampled_df = sampled_df.sort_values('num_tokens').reset_index(drop=True)

print("Files saved!")
print(f"Total samples: {len(sampled_df)}")
print(f"\nSentences per token length:")
print(sampled_df['num_tokens'].value_counts().sort_index())
print(f"\nToken length statistics:")
print(sampled_df['num_tokens'].describe())

print(sampled_df['num_tokens'].unique())

# Save CSV with all columns
sampled_df.to_csv('../../datasets/sst2_sampled_with_tokens.csv', index=False)

# Save TSV with only sentence and label
tsv_df = sampled_df[['sentence', 'label']]
tsv_df.to_csv('../../datasets/sentences/sst2-sampled/test.tsv', sep='\t', index=False)

# Example: View first item
print()