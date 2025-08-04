import os
#from load_repositories import *
#import load_repositories
import pandas as pd
from transformers import BertTokenizer

import sys
external_repos_path = '../../external_repos/eraserbenchmark-master'
sys.path.append(external_repos_path)

from rationale_benchmark.utils import load_documents, load_datasets, annotations_from_jsonl, Annotation

class Eraser:
    def __init__(self):
        self.data_root = os.path.join('..', '..', 'external_repos', 'eraserbenchmark-master', 'data', 'movies')
        self.documents = load_documents(self.data_root)
        self.train, self.val, self.test = load_datasets(self.data_root)
        self.dataset = self.test

    def select_dataset(self, dataset_type="test"):
        if dataset_type == "test":
            dataset = self.test
        elif dataset_type == "train":
            dataset = self.train
        else:
            dataset = self.val

        return dataset

    def get_shorter_doc(self, doc_text, sentences, tokenizer):
        total_tokens = tokenizer(doc_text)
        n_tokens = len(total_tokens['input_ids'])

        if n_tokens > 512:
            rel_tokens = 512 / n_tokens
            n_sents = int(rel_tokens * len(sentences)) - 1
            shorter_text = ' '.join(sentences[:n_sents])
            tokens1 = tokenizer(shorter_text)

            while len(tokens1['input_ids']) > 512:
                n_sents -= 1
                shorter_text = ' '.join(sentences[:n_sents])
                tokens1 = tokenizer(shorter_text)
        else:
            shorter_text = doc_text

        return shorter_text

    def to_tsv(self, tokenizer, rows=200, sentences_limit=None, output_file="eraser/eraser_movies_test.tsv"):
        reviews = []
        labels = []

        for i in range(rows):
            annotation = self.dataset[i]
            evidences = annotation.all_evidences()
            (docid,) = set(ev.docid for ev in evidences)
            doc = self.documents[docid]
            sentences = []

            if sentences_limit:
                for sent in doc:
                    sentence = ' '.join(sent)
                    sentences.append(sentence)

                    if len(sentences) >= sentences_limit:
                        break
            else:
                for sent in doc:
                    sentence = ' '.join(sent)
                    sentences.append(sentence)

            doc_text = ' '.join(sentences)
            shorter_text = self.get_shorter_doc(doc_text, sentences, tokenizer)
            reviews.append(shorter_text)
            labels.append(int(annotation.classification != 'NEG'))

        # Save TSV
        # Sample data
        data = {'sentence': reviews, 'label': labels}

        # Create DataFrame
        df = pd.DataFrame(data)

        # Save as TSV
        df.to_csv(f"../../datasets/{output_file}", sep='\t', index=False)

eraser = Eraser()
pretrained_model = "../models/bert_tweets"
tokenizer = BertTokenizer.from_pretrained(pretrained_model, do_lower_case=False)
#eraser.to_tsv(tokenizer, rows=10)
eraser.to_tsv(tokenizer, rows=50, output_file="eraser/test.tsv")
