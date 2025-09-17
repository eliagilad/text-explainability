import json
import matplotlib.pyplot as plt

# Load your JSON file
with open("../results/runtime_analysis.json", "r") as f:
    data = json.load(f)

# Extract number of tokens and prediction calls
num_tokens = [len(example["tokens"]) for example in data]
prediction_calls = [example["PartitionExplainer"]["prediction_calls"] for example in data]

# Plot
plt.figure(figsize=(6, 4))
plt.plot(num_tokens, prediction_calls, marker="o", linestyle="-")

# Labels & title
plt.xlabel("Number of tokens")
plt.ylabel("Prediction calls")
plt.title("Prediction Calls vs. Number of Tokens")

# Make it clean
plt.grid(True)
plt.tight_layout()

plt.savefig("../results/prediction_calls_vs_tokens.png", dpi=300)
plt.show()
