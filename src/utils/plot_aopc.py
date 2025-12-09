import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

results_file = "../../results/aopc.json"

df = pd.read_json(results_file)
print(df.head())

def plot_aopc_scatter(df):
    num_tokens, aopc_partition = df["num_tokens"], df["aopc_partition"]
    plt.scatter(num_tokens, aopc_partition)
    plt.xlabel("Number of tokens")
    plt.ylabel("AOPC")
    plt.title("AOPC by number of tokens")
    plot_file = "../../results/aopc_by_ntokens.png"
    plt.savefig(plot_file, dpi=300)

def plot_aopc_mean(df):
    plot_file2 = "../../results/compare_aopc_by_ntokens.png"

    ax = df.groupby("num_tokens")[[
        "aopc_partition", "aopc_hedge"]].mean().plot()

    fig = ax.get_figure()
    fig.savefig(plot_file2, dpi=300)

def plot_aopc_std(df):
    # Calculate mean and std
    grouped = df.groupby("num_tokens")[["aopc_partition", "aopc_hedge"]]
    means = grouped.mean()
    stds = grouped.std()

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot for each method
    for col in ["aopc_partition", "aopc_hedge"]:
        x = means.index
        y = means[col]
        std = stds[col]

        # Plot mean line
        ax.plot(x, y, marker='o', label=col, linewidth=2)

        # Add shaded confidence interval (mean ± std)
        ax.fill_between(x, y - std, y + std, alpha=0.3)

    ax.set_xlabel("Number of Tokens")
    ax.set_ylabel("AOPC Score")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_file3 = "../../results/compare_aopc_std_by_ntokens.png"
    fig.savefig(plot_file3, dpi=300)

    plt.tight_layout()
    plt.show()

    # # For 95% CI using standard error
    # n = grouped.count()
    # sem = stds / np.sqrt(n)  # Standard error of mean
    # ci = 1.96 * sem  # 95% confidence interval
    #
    # # Then use:
    # ax.fill_between(x, y - ci[col], y + ci[col], alpha=0.3)

def plot_aopc_by_confidence(df, explainer="aopc_partition"):
    df['confidence_level'] = pd.cut(df['prediction_confidence'],
                                    bins=[0.5, 0.8, 0.95, 0.99, 0.999, 1.0])

    fig, ax = plt.subplots(figsize=(10, 6))
    classes = ["Negative", "Positive"]

    # Plot each class separately
    for class_label in df['prediction_class'].unique():
        class_data = df[df['prediction_class'] == class_label]

        # Group by confidence and calculate mean
        grouped_mean = class_data.groupby('confidence_level')[explainer].mean()
        grouped_count = class_data.groupby('confidence_level').size()

        # Convert intervals to midpoints for plotting
        x_values = [interval.mid for interval in grouped_mean.index]

        ax.plot(x_values, grouped_mean.values, marker='o',
                label=classes[class_label], linewidth=2)

        for x, y, count in zip(x_values, grouped_mean.values, grouped_count.values):
            ax.annotate(f'n={count}', xy=(x, y), xytext=(5, 5),
                        textcoords='offset points', fontsize=8, alpha=0.7)

    ax.set_xlabel('Confidence Level')
    ax.set_ylabel('AOPC Score')
    ax.set_title('AOPC by Prediction Confidence (by Class) - {}'.format(explainer.split('_')[1]))
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_file4 = f"../../results/compare_aopc_by_conf_{explainer}.png"
    fig.savefig(plot_file4, dpi=300)

    plt.show()

# plot_aopc_scatter(df)
# plot_aopc_mean(df)
# plot_aopc_std(df)
#plot_aopc_by_confidence(df)
plot_aopc_by_confidence(df, explainer="aopc_partition")
plot_aopc_by_confidence(df, explainer="aopc_hedge")

