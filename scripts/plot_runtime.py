import json
import matplotlib.pyplot as plt
import pandas as pd

include_hedge = True
results_file = "../results/runtime_analysis4.json"
plot_file = "../results/prediction_calls_vs_tokens3.png"

# Load your JSON file
with open(results_file, "r") as f:
    data = json.load(f)

# Extract number of tokens and prediction calls
num_tokens = [example["num_tokens"] for example in data]
prediction_calls = [example["PartitionExplainer"]["prediction_calls"] for example in data]

#df = pd.read_json(results_file)
data_partition = {
    "num_tokens": num_tokens,
    "prediction_calls": prediction_calls, }
df = pd.DataFrame(data=data_partition)
df_stats = df.groupby(['num_tokens'])["prediction_calls"].mean().reset_index()

# Plot
plt.figure(figsize=(6, 4))
plt.plot(df_stats["num_tokens"], df_stats["prediction_calls"], marker="o", linestyle="-",
         label="SHAP Partition Explainer")

if include_hedge:
    prediction_calls2 = [example.get("HEDGE", {}).get("prediction_calls")
                         #if not type(example.get("HEDGE", {}) == str) else 0
                         for example in data]

    data_hedge = {
        "num_tokens": num_tokens,
        "prediction_calls": prediction_calls2, }
    df2 = pd.DataFrame(data=data_hedge)
    df_stats2 = df2.groupby(['num_tokens'])["prediction_calls"].mean().reset_index()

    plt.plot(df_stats2["num_tokens"], df_stats2["prediction_calls"], marker="o", linestyle="-",
             label="HEDGE")

# Labels & title
plt.xlabel("Number of tokens")
plt.ylabel("Mean Prediction calls")
plt.title("Prediction Calls vs. Number of Tokens")
plt.legend()

# Make it clean
plt.grid(True)
plt.tight_layout()

plt.savefig(plot_file, dpi=300)
plt.show()
