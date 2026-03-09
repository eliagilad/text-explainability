from ast import Num
import json
import numpy as np
import matplotlib.pyplot as plt

compared_partition = "random" # "balanced" or "default"

# Load results
with open("results/partitions_aopc.json", "r") as f:
    data = json.load(f)

# Extract AOPC scores
aopc_default = np.array([d["default"]["aopc"] for d in data])
aopc_balanced = np.array([d[compared_partition]["aopc"] for d in data])
aopc_diff = aopc_default - aopc_balanced
num_tokens = np.array([len(d["default"]["fi"]) for d in data])
confidence = np.array([d["confidence"] for d in data])

print("Default AOPC mean:", np.mean(aopc_default))
print(f"{compared_partition} AOPC mean:", np.mean(aopc_balanced))

# Boxplot comparison
plt.figure(figsize=(5, 4))
plt.boxplot(
    [aopc_default, aopc_balanced],
    labels=["Default partition", f"{compared_partition} partition"],
    showmeans=True
)
plt.ylabel("AOPC")
plt.title(f"AOPC comparison: default vs {compared_partition} partitions")

# Save to file in results directory
plt.tight_layout()
plt.savefig(f"results/{compared_partition}_partitions_aopc_comparison.png", dpi=300)
plt.close()

# Plot AOPC diff by the default AOPC
plt.figure(figsize=(5, 4))
plt.scatter(aopc_default, aopc_diff)
plt.xlabel("Default AOPC")
plt.ylabel("AOPC diff")
plt.title("AOPC diff by default AOPC")
plt.tight_layout()
plt.savefig(f"results/{compared_partition}_aopc_diff_by_default_aopc.png", dpi=300)
plt.close()


# Plot AOPC diff by the number of tokens
plt.figure(figsize=(5, 4))
plt.scatter(num_tokens, aopc_diff)
plt.xlabel("Number of tokens")
plt.ylabel("AOPC diff")
plt.title("AOPC diff by number of tokens")
plt.tight_layout()
plt.savefig(f"results/{compared_partition}_aopc_diff_by_num_tokens.png", dpi=300)
plt.close()

# Plot AOPC diff by the confidence
plt.figure(figsize=(5, 4))
plt.scatter(confidence, aopc_diff)
plt.xlabel("Confidence")
plt.ylabel("AOPC diff")
plt.title("AOPC diff by confidence")
plt.tight_layout()
plt.savefig(f"results/{compared_partition}_aopc_diff_by_confidence.png", dpi=300)
plt.close()