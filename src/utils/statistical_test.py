import json
import numpy as np
from scipy import stats

# Load experimental results
with open("results/partitions_aopc.json", "r") as f:
    data = json.load(f)

# Extract AOPC scores
aopc_default = [d["default"]["aopc"] for d in data]
aopc_balanced = [d["balanced"]["aopc"] for d in data]

print(f"Default AOPC mean: {np.mean(aopc_default)}")
print(f"Balanced AOPC mean: {np.mean(aopc_balanced)}")

print(f"Default AOPC std: {np.std(aopc_default)}")
print(f"Balanced AOPC std: {np.std(aopc_balanced)}")

# Perform t-test
t_stat, p_value = stats.ttest_rel(aopc_default, aopc_balanced)
print(f"T-statistic: {t_stat}, P-value: {p_value}")
print()