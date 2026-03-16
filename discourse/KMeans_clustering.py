import pandas as pd

# =============================================================================
# CONFIGURATION — update this path for your environment
# =============================================================================
COMBINED_SENTIMENT_INTENT_CSV = r"PATH_TO_COMBINED_SENTIMENT_INTENT_CSV"  # e.g. data/Combined_Sentiment_Intent_all_data.csv
# =============================================================================

df = pd.read_csv(COMBINED_SENTIMENT_INTENT_CSV)


# similarity of comments to its respective post
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Encode posts from Affordability
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Encode posts and comments
post_embeddings = model.encode(df["Post"].tolist(), normalize_embeddings=False)
comment_embeddings = model.encode(df["Comment"].tolist(), normalize_embeddings=False)

# Compute cosine similarity per row (post-comment pair)
similarities = [
    util.cos_sim(post_embeddings[i], comment_embeddings[i]).item()
    for i in range(len(df))
]

df["Similarity"] = similarities
print(f"Average similarity: {np.mean(similarities):.4f}")

mean_sim = df["Similarity"].mean()
std_sim = df["Similarity"].std()

print(f"Mean similarity: {mean_sim:.3f}")
print(f"Standard deviation: {std_sim:.3f}")

import matplotlib.pyplot as plt

plt.hist(df["Similarity"], bins=30, color='skyblue', edgecolor='black')
plt.axvline(mean_sim, color='red', linestyle='--', label='Mean')
plt.axvline(mean_sim + 1* std_sim, color='green', linestyle='--', label='+1σ')
plt.axvline(mean_sim - 1* std_sim, color='orange', linestyle='--', label='-1σ')
plt.legend()
plt.title('Distribution of Comment–Post Similarities')
plt.xlabel('Cosine Similarity')
plt.ylabel('Count')
plt.show()


def categorize_similarity(x, mean, std):
    if x > mean + 1* std:
        return "High"
    elif x < mean - 1* std:
        return "Low"
    else:
        return "Average"

df["Similarity_Category"] = df["Similarity"].apply(lambda x: categorize_similarity(x, mean_sim, std_sim))
df["Similarity_Category"].value_counts()


# assuming your DataFrame is called df
intents = ['Appreciation', 'Criticism', 'Inquiry', 'Statement']

for intent in intents:
    df[f"avg_prob_{intent}"] = df[[f"GRO_NLP_prob_{intent}",
                                   f"roberta_prob_{intent}",
                                   f"debertaV3_prob_{intent}"]].mean(axis=1)
    
#create new column avg_prob_Forward and fill with 1 where final_label is Forward else 0
df["avg_prob_Forward"] = df["final_label_three_models_final"].apply(lambda x: 1 if x == "Forward" else 0)

features = [
    "avg_prob_Appreciation",
    "avg_prob_Criticism",
    "avg_prob_Inquiry",
    "avg_prob_Statement",
    "avg_prob_Forward",
    "Similarity",
    "Sentiment over comment from models",
    
]

X = df[features].copy()


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


scores = []
for k in range(2, 20):
    km = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
    scores.append(silhouette_score(X_scaled, km.labels_))
pd.DataFrame({"k": range(2, 20), "silhouette": scores})


k = 6 # pick based on elbow/ silhouette
kmeans = KMeans(n_clusters=k, random_state=42)
df["Cluster_KMeans"] = kmeans.fit_predict(X_scaled)


cluster_summary = (
    df.groupby("Cluster_KMeans")[features]
    .mean()
    .sort_values("avg_prob_Criticism", ascending=False)
)
print(cluster_summary)


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


df.rename(columns={'Sentiment over comment from models': 'Sentiment'}, inplace=True)

# 1️⃣ Create cluster summary with original column names (without Cluster Name in aggregation)
cluster_summary = df.groupby('Cluster_KMeans')[[
    'avg_prob_Appreciation',
    'avg_prob_Criticism',
    'avg_prob_Inquiry',
    'avg_prob_Statement',
    'avg_prob_Forward',
    'Similarity',
    'Sentiment'
]].mean().round(3)

# Add support column
cluster_summary["Support"] = df.groupby('Cluster_KMeans').size()

# 2️⃣ Rename columns to shorter, cleaner names
cluster_summary.rename(columns={
    'avg_prob_Appreciation': 'Appreciation',
    'avg_prob_Criticism': 'Criticism', 
    'avg_prob_Inquiry': 'Inquiry',
    'avg_prob_Statement': 'Statement',
    'avg_prob_Forward': 'Forward',
}, inplace=True)

# 3️⃣ Reorder clusters in the specified sequence: 1, 5, 3, 4, 2, 0
desired_order = [1, 5, 3, 4, 2, 0]
existing_clusters = [c for c in desired_order if c in cluster_summary.index]
print(f"Clusters in data (ordered as requested): {existing_clusters}")
cluster_summary = cluster_summary.reindex(existing_clusters)

# 4️⃣ Add cluster names as the first column
cluster_name_map = {
    1: 'On-topic Criticism',
    5: 'Information Seeking',
    3: 'Off-topic Complaints',
    4: 'Content Sharing',
    0: 'On-topic Feedbacks',
    2: 'On-topic Praise'
}

# Insert cluster names as first column
cluster_summary.insert(0, 'Cluster Name', [cluster_name_map[idx] for idx in cluster_summary.index])

# 5️⃣ Create custom annotations with decimals for all columns except Support and Cluster Name
annotations = cluster_summary.copy()
for col in annotations.columns:
    if col == 'Cluster Name':
        # Keep cluster names as text
        annotations[col] = annotations[col].astype(str)
    elif col != 'Support':
        # Format numeric columns with 3 decimals
        annotations[col] = annotations[col].apply(lambda x: f"{x:.3f}")
    else:
        # Support as integers
        annotations[col] = annotations[col].astype(int).astype(str)

# 6️⃣ Create mask to hide coloring for Cluster Name and Support columns
mask = np.zeros_like(cluster_summary.select_dtypes(include=[np.number]), dtype=bool)
numeric_cols = cluster_summary.select_dtypes(include=[np.number]).columns
support_idx = list(numeric_cols).index('Support')
mask[:, support_idx] = True

# Create data for coloring (only numeric columns, neutralize Support)
plot_data = cluster_summary.select_dtypes(include=[np.number]).copy()
plot_data['Support'] = 0  # Set to neutral value for coloring

# Create annotations for numeric columns only
numeric_annotations = annotations.select_dtypes(include=[object]).copy()
# Get numeric column annotations in same order as plot_data
numeric_annotations = annotations[numeric_cols].copy()

# 7️⃣ Create the plot with custom formatting
vmin, vmax = -1, 1   # Color range for sentiment and probabilities

fig, ax = plt.subplots(figsize=(32, 15))
sns.heatmap(
    plot_data,
    cmap="coolwarm",      # diverging: blue (-1) → white (0) → red (1)
    vmin=vmin,
    vmax=vmax,
    annot=numeric_annotations,    # use numeric annotations only
    fmt="",               # empty format since we're providing pre-formatted strings
    cbar_kws={'label': 'Value Range', 'shrink': 0.8},
    mask=mask,            # mask the Support column
    annot_kws={"size": 26, "weight": "bold"},  # larger, bold text
    linewidths=1,         # add grid lines
    linecolor='white',
    ax=ax
)

# Customize colorbar font sizes
cbar = ax.collections[0].colorbar
cbar.set_label('Value Range', fontsize=26, fontweight='bold')
cbar.ax.tick_params(labelsize=26)

# Add Cluster Name column manually at the end (after Support)
cluster_name_col_idx = len(plot_data.columns)  # Position after all existing columns
for i in range(len(cluster_summary)):
    # Add light gray background for Cluster Name column
    rect = plt.Rectangle((cluster_name_col_idx, i), 1, 1,
                    facecolor='lightgray', alpha=0.3,
                    edgecolor='none')
    ax.add_patch(rect)
    
    # Add cluster name text
    cluster_name = cluster_summary.iloc[i]['Cluster Name']
    # Wrap text to fit in cell (adjust width as needed)
    wrapped_text = '\n'.join(textwrap.wrap(cluster_name, width=12))
    ax.text(cluster_name_col_idx + 0.5, i + 0.5, wrapped_text,
           ha='center', va='center', fontsize=25, fontweight='bold', 
           color='black')

# Add Support column background manually
support_col_idx = list(plot_data.columns).index('Support')
for i in range(len(cluster_summary)):
    rect = plt.Rectangle((support_col_idx, i), 1, 1, 
                        facecolor='lightgray', alpha=0.5, 
                        edgecolor='white', linewidth=1)
    ax.add_patch(rect)
    
    # Add Support text manually
    support_value = cluster_summary.iloc[i]['Support']
    ax.text(support_col_idx + 0.5, i + 0.5, str(int(support_value)),
           ha='center', va='center', fontsize=26, fontweight='bold', 
           color='black', zorder=10)
# Extend x-axis to include Cluster Name column at the end
ax.set_xlim(0, len(plot_data.columns) + 1)

# Set title and labels
plt.title("Cluster Summary", 
          fontsize=28, fontweight='bold', pad=30)
plt.xlabel('Features', fontsize=28, fontweight='bold')
plt.ylabel('Clusters', fontsize=28, fontweight='bold')

# Customize tick labels
x_labels = list(plot_data.columns) + ['Cluster Name']
ax.set_xticks(np.arange(0, len(plot_data.columns) + 1) + 0.5)  # Center the x-ticks in cells
ax.set_xticklabels(x_labels, fontsize=28, fontweight='bold', rotation=90, ha='center')
ax.set_yticks(np.arange(len(cluster_summary)) + 0.5)  # Center the y-ticks in cells
ax.set_yticklabels([f"{idx}" for idx in existing_clusters], fontsize=28, fontweight='bold', ha='center')

plt.tight_layout()
plt.show()

# Show the final cluster summary for reference
print("\nFinal cluster summary:")
print(cluster_summary)

