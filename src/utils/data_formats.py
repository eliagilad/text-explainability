import pandas as pd

short_sentences = ["Absolutely terrible.",
                   "This movie was terrible .",
                   "The acting was dull and boring .",
                   "The plot was weak and the acting terrible .",
                   "This film was boring and slow with a predictable ending .",
                   "The movie was slow, the dialogue wooden, and the ending predictable .",
                   ]

def save_tsv(texts, labels, output_file):
    data = {'sentence': texts, 'label': labels}

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save as TSV
    df.to_csv(f"../../datasets/{output_file}", sep='\t', index=False)


save_tsv(short_sentences, [0] * len(short_sentences), "sentences/test.tsv")