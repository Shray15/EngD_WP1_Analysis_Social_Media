# Smoothed timeline visualization aggregated by quarter (Q1 2018 to Q1 2023)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from matplotlib.ticker import MultipleLocator
import numpy as np

# Load data
df = pd.read_csv(r"C:\Users\20245179\OneDrive - TU Eindhoven\Research Paper\Clustering\Probabilties\Clustered_Comments_Probabilities_KMeans.csv")

# Ensure datetime format
df["Comments_time"] = pd.to_datetime(df["Comments_time"])

# Filter data from Q1 2018 to Q1 2023
start_date = datetime(2018, 1, 1)
end_date = datetime(2023, 3, 31)
df_filtered = df[(df["Comments_time"] >= start_date) & (df["Comments_time"] <= end_date)].copy()

# Group by quarter
df_filtered["quarter"] = df_filtered["Comments_time"].dt.to_period("Q").dt.to_timestamp()

# Count clusters per quarter
cluster_time_quarterly = df_filtered.groupby(["quarter", "Cluster_KMeans"]).size().reset_index(name="count")

# Pivot for plotting
pivot_df_quarterly = cluster_time_quarterly.pivot(index="quarter", columns="Cluster_KMeans", values="count").fillna(0)

# Apply moving average smoothing but preserve first and last points
smoothed_quarterly = pivot_df_quarterly.rolling(window=2, center=False, min_periods=1).mean()

# Cluster name mapping
cluster_names = {
    1: 'On-topic Criticism',
    5: 'Information Seeking',
    3: 'Off-topic Complaints',
    4: 'Content Sharing',
    0: 'On-topic Feedbacks',
    2: 'On-topic Praise', 
}
# Define custom color palette for clusters
cluster_colors = {
    'On-topic Criticism': '#1f77b4',   # Blue
    'Information Seeking': '#2ca02c',      # Green
    'Off-topic Complaints': '#d62728',    # Red
    'Content Sharing': '#ff7f0e',          # Orange
    'On-topic Feedbacks': '#8c564b',        # Brown
    'On-topic Praise': '#9467bd',        # Purple
    
}

# Calculate growth indices for each cluster using SMOOTHED data
def calculate_growth_indices(data, raw_data):
    """
    Calculate growth indices for each cluster:
    - Total Growth Index: final_value / initial_value
    - Average Quarterly Growth Index: (final_value / initial_value)^(1/number_of_quarters)
    """
    growth_stats = {}
    for cluster in data.columns:
        # Use smoothed values for both display and calculation
        smoothed_values = data[cluster].values
        first_smoothed = smoothed_values[0]
        last_smoothed = smoothed_values[-1]
        
        # Use smoothed values for growth calculation
        first_value = first_smoothed if first_smoothed > 0 else 1  # Avoid division by zero
        last_value = last_smoothed
        
        # Total growth index (ratio of final to initial)
        total_growth_index = last_value / first_value
        
        # Number of quarters for average calculation
        num_quarters = len(smoothed_values) - 1  # Number of transitions
        
        # Average quarterly growth index (geometric mean)
        avg_quarterly_growth_index = total_growth_index ** (1 / num_quarters) if num_quarters > 0 else 1.0
        
        growth_stats[cluster] = {
            'total_growth_index': total_growth_index,
            'avg_quarterly_growth_index': avg_quarterly_growth_index,
            'first_raw': first_value,
            'last_raw': last_value,
            'first_smoothed': first_smoothed,
            'last_smoothed': last_smoothed,
            'cluster_name': cluster_names.get(cluster, f"Cluster {cluster}")
        }
    
    return growth_stats

# Calculate growth statistics
growth_stats = calculate_growth_indices(smoothed_quarterly, pivot_df_quarterly)

# Create smoothed visualization
plt.figure(figsize=(18, 8))
sns.set_style("whitegrid")

# Sort clusters by their order in cluster_colors dictionary
clusters_sorted_by_color = [k for k, v in cluster_names.items() if v in cluster_colors.keys()]

for cluster in clusters_sorted_by_color:
    cluster_name = cluster_names.get(cluster, f"Cluster {cluster}")
    color = cluster_colors.get(cluster_name, '#888888')
    plt.plot(
        pivot_df_quarterly.index,
        smoothed_quarterly[cluster],
        marker='o',
        linewidth=2.5,
        markersize=6,
        label=cluster_name,
        alpha=0.85,
        color=color
    )

# Add growth indices at the end of each line with offset positioning to avoid overlap
cluster_end_values = [(cluster, smoothed_quarterly[cluster].iloc[-1]) for cluster in clusters_sorted_by_color]

# Sort clusters by their final y-values to position annotations without overlap
cluster_end_values = [(cluster, smoothed_quarterly[cluster].iloc[-1]) for cluster in clusters_sorted_by_name]
cluster_end_values.sort(key=lambda x: x[1])  # Sort by end value

# Define vertical offsets with extra spacing for orange and blue lines
vertical_offsets = [-80, -40, -10, 20, 50, 80, 110, 20]  # Extra spacing to avoid overlap

for i, (cluster, last_y) in enumerate(cluster_end_values):
    total_growth_idx = growth_stats[cluster]['total_growth_index']
    avg_growth_idx = growth_stats[cluster]['avg_quarterly_growth_index']
    start_raw = growth_stats[cluster]['first_raw']
    end_raw = growth_stats[cluster]['last_raw']
    
    # Format growth text with simplified start:end format
    growth_text = f"{start_raw:.0f}:{end_raw:.0f}\nTotal Index: {total_growth_idx:.2f}\nAvg Q Index: {avg_growth_idx:.3f}"
    
    # Use different vertical offsets for each cluster, with special treatment for cluster 1 (orange line)
    if cluster == 1:  # Orange line (Constructive Criticism) - bring closer to the line
        vertical_offset = 15
    else:
        vertical_offset = vertical_offsets[i] if i < len(vertical_offsets) else 160 + (i - len(vertical_offsets)) * 40
    
    plt.annotate(
        growth_text,
        xy=(last_x, last_y),
        xytext=(25, vertical_offset),  # Increased horizontal distance and use calculated vertical offset
        textcoords='offset points',
        fontsize=12,
        fontweight='bold',
        verticalalignment='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray'),
        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6, lw=1.5)  # Add small arrow to connect to line
    )

plt.title("Evolution of Discourse Clusters Over Time (Quarterly Aggregation)\nFrom Q1 2018 to Q1 2023", 
          fontsize=20, pad=25, fontweight='bold')
plt.xlabel("Quarter", fontsize=16, fontweight='bold')
plt.ylabel("Number of Comments", fontsize=16, fontweight='bold')

plt.legend(title="Discourse Cluster", title_fontsize=14, fontsize=16,
           bbox_to_anchor=(1.15, 1), loc='upper left')  # Moved legend further right

quarter_labels = [f"{date.year}-Q{((date.month-1)//3)+1}" 
                  for date in pivot_df_quarterly.index]

plt.xticks(pivot_df_quarterly.index, quarter_labels,
           rotation=45, ha='right', fontsize=16)
plt.yticks(fontsize=16)

plt.gca().yaxis.set_major_locator(MultipleLocator(5))
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Print growth statistics summary with detailed calculation explanation
print("\n" + "="*100)
print("CLUSTER GROWTH INDICES (Q1 2018 to Q1 2023)")
print("="*100)
print("Growth Index Calculation Methods:")
print("• Total Growth Index: Final Value ÷ Initial Value")
print("• Avg Quarterly Growth Index: (Total Growth Index)^(1/number_of_quarters)")
for cluster in clusters_sorted_by_color:
    print("-" * 100)

for cluster in clusters_sorted_by_name:
    stats = growth_stats[cluster]
    print(f"{stats['cluster_name']:25} | {stats['first_raw']:4.0f}:{stats['last_raw']:4.0f} | "

          f"Total Index: {stats['total_growth_index']:6.2f} | Avg Q Index: {stats['avg_quarterly_growth_index']:6.3f}")
plt.show()
