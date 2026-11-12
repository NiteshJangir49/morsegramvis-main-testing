import matplotlib.pyplot as plt
import pandas as pd

# 1. Load the whitespace/tab-separated file without headers
csv_file = "anisotropy_values.csv"
df = pd.read_csv(csv_file, sep=r"\s+", header=None)

# 2. Extract values (up to 8 timesteps)
col1_data = df[0].iloc[:8]  # lower threshold 1.00
col2_data = df[1].iloc[:8]  # lower threshold 2.00

# 3. Define X-axis ticks from 10 to 80 in intervals of 10
x_ticks = list(range(10, 10 + 10 * len(col1_data), 10))  # [10, 20, 30, ..., 80]

# 4. Create the plot
plt.figure(figsize=(9, 5))

plt.plot(
    x_ticks,
    col1_data,
    marker="o",
    linewidth=2,
    color="#1f77b4",
    label="lower threshold 1.00",
)
plt.plot(
    x_ticks,
    col2_data,
    marker="s",
    linewidth=2,
    color="#ff7f0e",
    label="lower threshold 2.00",
)

# 5. Configure labels, title, ticks, and legend
plt.title(
    "Edge Fabric Anisotropy Across Timesteps",
    fontsize=13,
    fontweight="bold",
)
plt.xlabel("Timestep", fontsize=11)
plt.ylabel("Anisotropy Value", fontsize=11)
plt.xticks(x_ticks)  # Forces exact ticks at [10, 20, 30, 40, 50, 60, 70, 80]
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="best", fontsize=10)

plt.tight_layout()
plt.show()