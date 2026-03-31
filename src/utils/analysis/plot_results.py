import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = Path("results/partitions_aopc_v1.json")

PARTITIONS = ["default", "balanced", "random", "interaction"]

with RESULTS_PATH.open("r") as f:
    data = json.load(f)

def extract_aopc_matrix(records, partitions):
    """
    Returns:
      xs: observation indices
      mat: (n_obs, n_partitions) float array with NaN for missing
    """
    xs = np.arange(len(records))
    mat = np.full((len(records), len(partitions)), np.nan, dtype=float)
    for i, obs in enumerate(records):
        for j, p in enumerate(partitions):
            val = obs.get(p, {}).get("aopc", None)
            if val is None:
                continue
            mat[i, j] = float(val)
    return xs, mat


xs, mat = extract_aopc_matrix(data, PARTITIONS)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

# 1) Per-observation AOPC curves
ax = axes[0]
for j, p in enumerate(PARTITIONS):
    ax.plot(xs, mat[:, j], marker="o", linewidth=1.5, markersize=3, label=p)
ax.set_title(f"AOPC per observation ({RESULTS_PATH.name})")
ax.set_xlabel("Observation index")
ax.set_ylabel("AOPC")
ax.grid(True, alpha=0.3)
ax.legend(ncol=2)

# 2) Distribution comparison (boxplot)
ax = axes[1]
series = [mat[:, j][~np.isnan(mat[:, j])] for j in range(len(PARTITIONS))]
ax.boxplot(series, labels=PARTITIONS, showmeans=True)
ax.set_title("AOPC distribution by partition")
ax.set_ylabel("AOPC")
ax.grid(True, axis="y", alpha=0.3)

plt.show()