import json
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

with open("results/partitions_aopc.json", "r") as f:
    data = json.load(f)


def plot_distribution(data, partition_type1="default", partition_type2="balanced"):
    aopc_default = np.array([d[partition_type1]["aopc"] for d in data])
    aopc_balanced = np.array([d[partition_type2]["aopc"] for d in data])
    diff = aopc_default - aopc_balanced

    partition_type1_mean = np.mean(aopc_default)
    partition_type2_mean = np.mean(aopc_balanced)

    print(f"{partition_type1} AOPC mean: {partition_type1_mean}")
    print(f"{partition_type2} AOPC mean: {partition_type2_mean}")
    print(f"Difference: {np.mean(diff)}")

    sns.histplot(diff, kde=True)
    plt.show()

    # Save the plot
    plt.savefig(f"results/distr_aopc_diff_{partition_type1}_vs_{partition_type2}.png", dpi=300)

    stats = {"partition_type1": partition_type1, 
        "partition_type2": partition_type2,
        "partition_type1_mean": partition_type1_mean, 
    "partition_type2_mean": partition_type2_mean, 
    "difference": np.mean(diff),
    "difference_std": np.std(diff),
    "difference_median": np.median(diff),
    "difference_min": np.min(diff),
    "difference_max": np.max(diff),
    "difference_25": np.percentile(diff, 25),
    "difference_75": np.percentile(diff, 75),
    }
    return stats

stats_balanced = plot_distribution(data, "default", "balanced")
stats_random = plot_distribution(data, "default", "random")
print(stats_balanced)
print(stats_random)

all_stats = [stats_balanced, stats_random]


# Save the stats to a json file
with open("results/distributions_stats.json", "w") as f:
    json.dump(all_stats, f)