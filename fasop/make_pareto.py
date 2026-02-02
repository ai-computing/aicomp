import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('your_file.csv')  # Modify CSV file path

# Filter only is_oom == False
df = df[df['is_oom'] == False].reset_index(drop=True)

# Extract cost and throughput
costs = df['cost($)'].values
throughputs = df['throughput(samples/s)'].values

# Pareto frontier computation function
def compute_pareto_frontier(costs, throughputs):
    n = len(costs)
    is_pareto = np.ones(n, dtype=bool)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                if (costs[j] <= costs[i] and throughputs[j] >= throughputs[i]) and \
                   (costs[j] < costs[i] or throughputs[j] > throughputs[i]):
                    is_pareto[i] = False
                    break
    
    return is_pareto

# Compute First Pareto frontier
pareto_mask = compute_pareto_frontier(costs, throughputs)

# Find highest throughput point
max_throughput_idx = np.argmax(throughputs)
max_point = df.iloc[max_throughput_idx]

# Draw graph
fig, ax = plt.subplots(figsize=(12, 8))

# Non-Pareto points (dark gray)
non_pareto_mask = ~pareto_mask
ax.scatter(costs[non_pareto_mask], throughputs[non_pareto_mask], 
           c='#888888', alpha=0.6, s=30, label='Other configurations', zorder=1)

# Pareto frontier points (blue)
ax.scatter(costs[pareto_mask], throughputs[pareto_mask], 
           c='#3b82f6', s=100, label='Pareto frontier', zorder=3, 
           edgecolors='#1d4ed8', linewidths=1.5)

# Add annotation to highest throughput point
annotation_text = f"mbs={int(max_point['mbs'])} tp={int(max_point['tp'])} pp={int(max_point['pp'])} dp={int(max_point['dp'])}\ntotal_time(s)={max_point['total_time(s)']:.2f}"
ax.annotate(annotation_text,
            xy=(costs[max_throughput_idx], throughputs[max_throughput_idx]),
            xytext=(costs[max_throughput_idx] + 15, throughputs[max_throughput_idx] - 0.5),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor='#3b82f6', linewidth=1.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', 
                           color='#3b82f6'),
            zorder=5)

# Styling
ax.set_xlabel('Training Cost ($)', fontsize=14, fontweight='bold')
ax.set_ylabel('Throughput (samples/s)', fontsize=14, fontweight='bold')
ax.set_title('LLaMA 70B training pareto frontier analysis', 
             fontsize=16, fontweight='bold', pad=20)

ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')

# Adjust axis range
ax.set_xlim(0, costs.max() * 1.05)
ax.set_ylim(0, throughputs.max() * 1.1)

# Background color
ax.set_facecolor('#f8fafc')
fig.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig('llama70b_pareto_frontier.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')