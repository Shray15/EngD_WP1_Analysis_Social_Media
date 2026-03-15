# posts features to be created: length, number of comments, topics covered,  nu_words
import pandas as pd

df1 = pd.read_csv(r"C:\Users\20245179\OneDrive - TU Eindhoven\Research Paper\Data\final_data_with_author_names.csv")

posts_df = df.copy()   # your original posts dataframe

#number pf comments per post
comments_per_post = posts_df.groupby('Post').size().reset_index(name='num_comments')


# post length
posts_df['post_length'] = posts_df['Post'].apply(lambda x: len(str(x).split()))
bins = [0, 50, 150, 300, np.inf]          # adjust as you like
labels = ["tweet (<50)", "short (50-150)", "abstract (150-300)", "long (>300)"]

posts_df["length_category"] = pd.cut(posts_df["post_length"], bins=bins, labels=labels,
                               right=True, include_lowest=True)

# number of unique words in posts
posts_df['Unique Words'] = posts_df['Post'].apply(lambda x: len(set(str(x).split())))
posts_df["Num Hashtags"] = posts_df["Post"].str.count(r"#\w+")
posts_df["Question Present"] = posts_df["Post"].str.contains(r"\?", regex=True).astype(int)
posts_df["URL Present"] = posts_df["Post"].str.contains(r"http\S+", regex=True).astype(int)

# Calculate unique words as percentage of post length
posts_df['Unique Words %'] = ((posts_df['Unique Words'] / posts_df['post_length']) * 100).round(2)
import subprocess, sys; subprocess.check_call([sys.executable, "-m", "pip", "install", "textstat"])
import textstat
posts_df["Readability"] = posts_df["Post"].apply(lambda x: textstat.flesch_reading_ease(x))
# sustainability_keywords = ["energ", "isolat", "warmt", "zonne", "duurz", "renov"]


# Ensure 'post' column is lowercase and handles missing values
posts_df['Post'] = posts_df['Post'].fillna('')

# # Create a binary column for each keyword
# for kw in sustainability_keywords:
#     posts_df[kw] = posts_df['Post'].apply(lambda x: int(kw in x))

# merge author name from df2 on Post

df2 = pd.read_csv(r"C:\Users\20245179\OneDrive - TU Eindhoven\Research Paper\Data\final_data_with_author_names.csv")
df2 = df2[['text_ha', 'author_name']]

# Make sure both columns are lowercase strings
posts_df['Post'] = posts_df['Post'].astype(str).str.lower()
df2['text_ha'] = df2['text_ha'].astype(str).str.lower()

# Drop duplicates in df2 to prevent row duplication
df2_unique = df2.drop_duplicates(subset='text_ha')

# Merge only author_name into posts_df (row count will stay the same)
posts_df = posts_df.merge(
    df2_unique[['text_ha', 'author_name']],
    left_on='Post',
    right_on='text_ha',
    how='left'
)


# count the number of unique posts per length category
unique_posts_per_length = posts_df.groupby("length_category")["Post"].nunique().reset_index()
unique_posts_per_length.columns = ["length_category", "num_unique_posts"]
print(unique_posts_per_length)


features_kde = ["Num Hashtags", "Unique Words %"]   # numeric features → KDE
features_bar = ["Question Present", "URL Present"]       # binary features → bar chart

# Cluster name mapping
cluster_names = {
    1: 'On-topic Criticism',
    5: 'Information Seeking',
    3: 'Off-topic Complaints',
    4: 'Content Sharing',
    0: 'On-topic Feedbacks',
    2: 'On-topic Praise'
}

# Build cluster name labels for plotting (use names only, not "ID: name" format)
cluster_labels = {k: v for k, v in cluster_names.items()}

# Add a label column to the dataframe using cluster names
posts_df["Cluster_Label"] = posts_df["Cluster_KMeans"].map(cluster_labels).astype("category")

# Palette matching the timeline visualization colors exactly
cluster_palette_labeled = {
    cluster_labels[1]: "#1f77b4",  # blue - On-topic Constructive Criticism  
    cluster_labels[5]: "#2ca02c",  # green - Information Seeking
    cluster_labels[3]: "#d62728",  # red - Off-topic Complaints
    cluster_labels[4]: "#ff7f0e",  # orange - Content Sharing
    cluster_labels[0]: "#8c564b",  # brown - On-topic Feedbacks
    cluster_labels[2]: "#9467bd",  # purple - On-topic Praise
}

# --- Length Category Cumulative Bar Chart ---
plt.figure(figsize=(14, 10))

# Create cross-tabulation for length categories
length_crosstab = pd.crosstab(posts_df['length_category'], posts_df['Cluster_Label'])

# Reorder columns according to cluster color sequence
cluster_order = ['On-topic Criticism', 'Information Seeking', 'Off-topic Complaints', 
                'Content Sharing', 'On-topic Feedbacks', 'On-topic Praise']
available_clusters = [cluster for cluster in cluster_order if cluster in length_crosstab.columns]
length_crosstab = length_crosstab.reindex(columns=available_clusters)

# Create color list in the same order as columns
colors = [cluster_palette_labeled[col] for col in length_crosstab.columns]

# Create stacked bar chart
ax = length_crosstab.plot(kind='bar', stacked=True, color=colors, figsize=(14, 10))

# Add numbers on top of each segment
for i, category in enumerate(length_crosstab.index):
    cumulative_height = 0
    for j, cluster in enumerate(length_crosstab.columns):
        value = length_crosstab.loc[category, cluster]
        if value > 0:  # Only add text if value is greater than 0
            # Position text in the middle of each segment
            text_y = cumulative_height + value / 2
            ax.text(i, text_y, str(value), ha='center', va='center', 
                   fontsize=14, fontweight='bold', color='white')
        cumulative_height += value

plt.title("Cumulative Distribution of Length Categories per Cluster", fontsize=18)
plt.xlabel("Length Category", fontsize=16)
plt.ylabel("Cumulative Count", fontsize=16)
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=16)
plt.legend(title="Discourse Cluster", title_fontsize=14, fontsize=12, 
           bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --- Length Category Percentage Stacked Bar Chart ---
plt.figure(figsize=(14, 10))

# Calculate percentages for length categories
length_crosstab_pct = pd.crosstab(posts_df['length_category'], posts_df['Cluster_Label'], normalize='index') * 100
length_crosstab_pct = length_crosstab_pct.reindex(columns=available_clusters)

# Create color list in the same order as columns
colors = [cluster_palette_labeled[col] for col in length_crosstab_pct.columns]
# Create percentage stacked bar chart
ax = length_crosstab_pct.plot(kind='bar', stacked=True, color=colors, figsize=(14, 10))

# Add percentage labels on each segment
for i, category in enumerate(length_crosstab_pct.index):
    cumulative_height = 0
    for j, cluster in enumerate(length_crosstab_pct.columns):
        value = length_crosstab_pct.loc[category, cluster]
        if value > 2:  # Only add text if percentage is greater than 2%
            # Position text in the middle of each segment
            text_y = cumulative_height + value / 2
            ax.text(i, text_y, f'{value:.1f}%', ha='center', va='center', 
                   fontsize=14, fontweight='bold', color='white')
        cumulative_height += value

plt.title("Percentage Distribution of Length Categories per Cluster", fontsize=18)
plt.xlabel("Length Category", fontsize=16)
plt.ylabel("Percentage (%)", fontsize=16)
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=16)
plt.legend(title="Discourse Cluster", title_fontsize=14, fontsize=14, 
           bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --- Cumulative Bar charts for binary features ---
for feature in features_bar:
    plt.figure(figsize=(8, 6))
    
    # Create cross-tabulation for the current feature
    feature_crosstab = pd.crosstab(posts_df[feature], posts_df['Cluster_Label'])
    
    # Reorder columns according to cluster color sequence
    feature_crosstab = feature_crosstab.reindex(columns=available_clusters)
    
    # Create color list in the same order as columns
    colors = [cluster_palette_labeled[col] for col in feature_crosstab.columns]
    
    # Create stacked bar chart
    ax = feature_crosstab.plot(kind='bar', stacked=True, color=colors, figsize=(8, 6))
    
    # Add numbers on top of each segment
    for i, feature_value in enumerate(feature_crosstab.index):
        cumulative_height = 0
        for j, cluster in enumerate(feature_crosstab.columns):
            value = feature_crosstab.loc[feature_value, cluster]
            if value > 0:  # Only add text if value is greater than 0
                # Position text in the middle of each segment
                text_y = cumulative_height + value / 2
                ax.text(i, text_y, str(value), ha='center', va='center', 
                       fontsize=11, fontweight='bold', color='white')
            cumulative_height += value
    
    plt.title(f"Cumulative Distribution of {feature} per Cluster", fontsize=16)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel("Cumulative Count", fontsize=18)
    plt.xticks(rotation=0, fontsize=14)
    plt.yticks(fontsize=14)
    
    # LEGEND OUTSIDE TO AVOID OVERLAPPING
    plt.legend(
        title="Discourse Cluster",
        title_fontsize=12,
        fontsize=16,
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )
    
    plt.tight_layout()
    plt.show()

    
for feature in features_bar:
    plt.figure(figsize=(8, 6))
# --- Percentage Bar charts for binary features ---
    # Reorder columns according to cluster color sequence
    feature_crosstab_pct = feature_crosstab_pct.reindex(columns=available_clusters)
    
    # Create color list in the same order as columns
    colors = [cluster_palette_labeled[col] for col in feature_crosstab_pct.columns]
    
    # Create percentage stacked bar chart
    ax = feature_crosstab_pct.plot(kind='bar', stacked=True, color=colors, figsize=(8, 6))
    
    # Add percentage labels on each segment
    for i, feature_value in enumerate(feature_crosstab_pct.index):
        cumulative_height = 0
        for j, cluster in enumerate(feature_crosstab_pct.columns):
            value = feature_crosstab_pct.loc[feature_value, cluster]
            if value > 3:  # Only add text if percentage is greater than 3%
                # Position text in the middle of each segment
                text_y = cumulative_height + value / 2
                ax.text(i, text_y, f'{value:.1f}%', ha='center', va='center', 
                       fontsize=12, fontweight='bold', color='white')
            cumulative_height += value
    
    plt.title(f"Percentage Distribution of Clusters within {feature}", fontsize=16)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=18)
    plt.xticks(rotation=0, fontsize=16)
    plt.yticks(fontsize=16)
    
    # LEGEND OUTSIDE TO AVOID OVERLAPPING
    plt.legend(
        title="Discourse Cluster",
        title_fontsize=12,
        fontsize=10,
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# BLACK & WHITE PATTERN VERSIONS FOR PRINTING
# =============================================================================

# Define patterns for each cluster (for black and white printing)
cluster_patterns = {
    cluster_labels[0]: '///',     # Relevant Comments - diagonal lines
    cluster_labels[1]: '\\\\\\',   # Constructive Criticism - reverse diagonal  
    cluster_labels[2]: '|||',     # Positive Feedback - vertical lines
    cluster_labels[3]: '---',     # Irrelevant Statements - horizontal lines
    cluster_labels[4]: '+++',     # Content Sharing - crosses
    cluster_labels[5]: '...',     # Information Seeking - dots
}

# --- Length Category Cumulative Bar Chart (B&W Pattern Version) ---
plt.figure(figsize=(14, 10))

# Reorder columns according to cluster color sequence
length_crosstab = length_crosstab.reindex(columns=available_clusters)

# Create pattern list in the same order as columns
patterns = [cluster_patterns[col] for col in length_crosstab.columns]

# Create stacked bar chart with patterns
ax = plt.subplots(figsize=(14, 10))[1]
bottom = np.zeros(len(length_crosstab.index))

for j, cluster in enumerate(length_crosstab.columns):
    values = length_crosstab[cluster].values
    ax.bar(length_crosstab.index, values, bottom=bottom, 
           label=cluster, hatch=patterns[j], color='white', edgecolor='black', linewidth=1)
    
    # Add numbers on segments
    for i, value in enumerate(values):
        if value > 0:
            text_y = bottom[i] + value / 2
            ax.text(i, text_y, str(value), ha='center', va='center', 
                   fontsize=14, fontweight='bold', color='black')
    bottom += values

plt.title("Cumulative Distribution of Length Categories per Cluster (B&W)", fontsize=18)
plt.xlabel("Length Category", fontsize=16)
plt.ylabel("Cumulative Count", fontsize=16)
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=16)
plt.legend(title="Discourse Cluster", title_fontsize=14, fontsize=12, 
           bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --- Length Category Percentage Stacked Bar Chart (B&W Pattern Version) ---
plt.figure(figsize=(14, 10))

# Reorder columns according to cluster color sequence
length_crosstab_pct = length_crosstab_pct.reindex(columns=available_clusters)

# Create stacked bar chart with patterns
ax = plt.subplots(figsize=(14, 10))[1]
bottom = np.zeros(len(length_crosstab_pct.index))

for j, cluster in enumerate(length_crosstab_pct.columns):
    values = length_crosstab_pct[cluster].values
    ax.bar(length_crosstab_pct.index, values, bottom=bottom, 
           label=cluster, hatch=patterns[j], color='white', edgecolor='black', linewidth=1)
    
    # Add percentage labels on segments
    for i, value in enumerate(values):
        if value > 2:
            text_y = bottom[i] + value / 2
            ax.text(i, text_y, f'{value:.1f}%', ha='center', va='center', 
                   fontsize=14, fontweight='bold', color='black')
    bottom += values

plt.title("Percentage Distribution of Length Categories per Cluster (B&W)", fontsize=18)
plt.xlabel("Length Category", fontsize=16)
plt.ylabel("Percentage (%)", fontsize=16)
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=16)
plt.legend(title="Discourse Cluster", title_fontsize=14, fontsize=14, 
           bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

    
for feature in features_bar:
    plt.figure(figsize=(8, 6))
# --- Cumulative Bar charts for binary features (B&W Pattern Version) ---
    # Reorder columns according to cluster color sequence
    feature_crosstab = feature_crosstab.reindex(columns=available_clusters)
    
    # Create stacked bar chart with patterns
    ax = plt.subplots(figsize=(8, 6))[1]
    bottom = np.zeros(len(feature_crosstab.index))
    
    for j, cluster in enumerate(feature_crosstab.columns):
        values = feature_crosstab[cluster].values
        ax.bar(feature_crosstab.index, values, bottom=bottom, 
               label=cluster, hatch=patterns[j], color='white', edgecolor='black', linewidth=1)
        
        # Add numbers on segments
        for i, value in enumerate(values):
            if value > 0:
                text_y = bottom[i] + value / 2
                ax.text(i, text_y, str(value), ha='center', va='center', 
                       fontsize=11, fontweight='bold', color='black')
        bottom += values
    
    plt.title(f"Cumulative Distribution of {feature} per Cluster (B&W)", fontsize=16)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel("Cumulative Count", fontsize=18)
    plt.xticks(rotation=0, fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.legend(
        title="Discourse Cluster",
        title_fontsize=12,
        fontsize=16,
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )
    
    plt.tight_layout()
    plt.show()

    
for feature in features_bar:
    plt.figure(figsize=(8, 6))
# --- Percentage Bar charts for binary features (B&W Pattern Version) ---
    # Reorder columns according to cluster color sequence
    feature_crosstab_pct = feature_crosstab_pct.reindex(columns=available_clusters)
    
    # Create stacked bar chart with patterns
    ax = plt.subplots(figsize=(8, 6))[1]
    bottom = np.zeros(len(feature_crosstab_pct.index))
    
    for j, cluster in enumerate(feature_crosstab_pct.columns):
        values = feature_crosstab_pct[cluster].values
        ax.bar(feature_crosstab_pct.index, values, bottom=bottom, 
               label=cluster, hatch=patterns[j], color='white', edgecolor='black', linewidth=1)
        
        # Add percentage labels on segments
        for i, value in enumerate(values):
            if value > 3:
                text_y = bottom[i] + value / 2
                ax.text(i, text_y, f'{value:.1f}%', ha='center', va='center', 
                       fontsize=12, fontweight='bold', color='black')
        bottom += values
    
    plt.title(f"Percentage Distribution of Clusters within {feature} (B&W)", fontsize=16)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=18)
    plt.xticks(rotation=0, fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.legend(
        title="Discourse Cluster",
        title_fontsize=12,
        fontsize=10,
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )
    
    plt.tight_layout()
    plt.show()
    
    plt.tight_layout()


    plt.show()





















