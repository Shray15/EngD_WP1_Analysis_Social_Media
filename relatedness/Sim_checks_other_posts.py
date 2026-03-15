#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv(r"C:\Users\20245179\OneDrive - TU Eindhoven\Research Paper\Data\final_data_cleaned_with_author_names.csv")

# In[2]:


df

# In[ ]:


from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# Group comments by post to capture comment counts per post (sustainability)
grouped = df.groupby("text_ha")["text"].apply(list).reset_index(name="comments")

# Encode posts from Sustainability
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
post_embeddings = model.encode(grouped["text_ha"].tolist(), convert_to_tensor=True, normalize_embeddings=True)
# Encode comments from Other Topics

# Compute real similarities (average per post)
real_avg_similarities = []

comment_cursor = 0

for i, comments in enumerate(grouped["comments"]):
    n_comments = len(comments)
    post_embed = post_embeddings[i].unsqueeze(0)
    
    # Comments are the actual sustainability comments for the post
    comment_embeds = model.encode(comments, convert_to_tensor=True, normalize_embeddings=True)

    sims = util.cos_sim(post_embed, comment_embeds).squeeze().cpu().numpy()
    real_avg_similarities.append(sims.mean())

real_avg_similarities = np.array(real_avg_similarities)

print(f"Real Avg Similarity: {real_avg_similarities.mean():.4f}")

# Random Sampling
num_simulations = 1
random_avg_similarities = []
all_random_similarities = []

for _ in tqdm(range(num_simulations)):
    sim_list = []

    for i, comments in enumerate(grouped["comments"]):
        n_comments = len(comments)
        post_embed = post_embeddings[i].unsqueeze(0)

        # Randomly sample comments
        random_indices = np.random.choice(len(df["text"]), size=n_comments, replace=False)
        sampled_comments = model.encode(df["text"].iloc[random_indices].tolist(), convert_to_tensor=True, normalize_embeddings=True)

        # Compute similarities
        sims = util.cos_sim(post_embed, sampled_comments).squeeze().cpu().numpy()
        sims = np.atleast_1d(sims)

        # Collect individual similarities
        all_random_similarities.extend(sims)
        sim_list.append(sims.mean())

    random_avg_similarities.append(np.mean(sim_list))

null_distribution = np.array(random_avg_similarities)
from scipy.stats import ttest_ind

# Independent t-test (equal variances)
ttest_t_stat, ttest_p_value = ttest_ind(real_avg_similarities, null_distribution, equal_var=True)

# Print results
print(f"Independent t-test - t-statistic: {ttest_t_stat:.4f}, p-value: {ttest_p_value:.4f}")
# Calculate p-value
p_value = np.mean(null_distribution >= real_avg_similarities.mean())

# Convert to numpy array
all_random_similarities = np.array(all_random_similarities)

# Plot the distribution
plt.figure(figsize=(14, 7))
sns.kdeplot(all_random_similarities, fill=True, color='skyblue', alpha=0.6, label=' Posts & Other Comments')
plt.axvline(all_random_similarities.mean(), color='blue', linestyle='--', label=f'Random Mean: {all_random_similarities.mean():.4f}')

plt.title('Distribution of Random Similarities =(Posts & Other Comments)')
plt.xlabel('Cosine Similarity')
plt.ylabel('Density')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Random Avg Similarity (Posts & Other Comments): {null_distribution.mean():.4f}")
print(f"Empirical p-value: {p_value:.4f}")

import matplotlib.pyplot as plt
import seaborn as sns

# Calculate means
real_mean = real_avg_similarities.mean()
random_mean = all_random_similarities.mean()

# Plot setup
plt.figure(figsize=(14, 7))

# KDE for All Random Similarities
sns.kdeplot(all_random_similarities, fill=True, color='skyblue', alpha=0.6, label='All Random Similarities')
plt.axvline(random_mean, color='blue', linestyle='--', label=f'Random Mean: {random_mean:.4f}')

# KDE for Real Similarities
sns.kdeplot(real_avg_similarities, fill=True, color='orange', alpha=0.6, label='Real Similarities')
plt.axvline(real_mean, color='orange', linestyle='--', label=f'Real Mean: {real_mean:.4f}')

# Plot settings
plt.title('Distribution of Real vs. All Random Similarities')
plt.xlabel('Cosine Similarity')
plt.ylabel('Density (Normalized)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



# In[ ]:



