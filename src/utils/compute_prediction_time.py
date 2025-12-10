import src.utils.bert_model as bm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load SST sub sample
root_dic = "../../"
sst_sub_file = f"{root_dic}datasets/sst2_sampled_with_tokens.csv"
df_sst = pd.read_csv(sst_sub_file)
results_dict = f"{root_dic}results/"

def plot_prediction_time(stats_df):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot mean with std as shaded area
    x = stats_df['num_tokens']
    y = stats_df['mean']
    std = stats_df['std']

    ax.plot(x, y, marker='o', linewidth=2, label='Mean', color='#2E86AB')
    ax.fill_between(x, y - std, y + std, alpha=0.3, label='±1 std', color='#2E86AB')

    # Optionally add min/max as dashed lines
    ax.plot(x, stats_df['min'], linestyle='--', alpha=0.5, label='Min', color='gray')
    ax.plot(x, stats_df['max'], linestyle='--', alpha=0.5, label='Max', color='gray')

    ax.set_xlabel('Number of Tokens', fontsize=12)
    ax.set_ylabel('Prediction Time (seconds)', fontsize=12)
    ax.set_title('Prediction Time by Number of Tokens', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{results_dict}prediction_time_by_tokens.png", dpi=300, bbox_inches='tight')
    print("Plot saved to prediction_time_by_tokens.png")
    plt.show()

def measure_prediction_time():
    model = bm.BertModel()
    times = []

    for i, row in df_sst.iterrows():
        result = model.predict(row["sentence"])
        times.append(result["prediction_time"])

    df_times = pd.DataFrame({"num_tokens": df_sst["num_tokens"], "prediction_time": times})
    df_times.to_csv(f"{results_dict}prediction_times.csv", index=False)

    # Calculate stats grouped by number of tokens
    stats_df = df_times.groupby('num_tokens')['prediction_time'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('count', 'count')  # How many samples per group
    ]).reset_index()

    stats_df.to_csv(f"{results_dict}prediction_time_stats.csv", index=False)
    plot_prediction_time(stats_df)


measure_prediction_time()
