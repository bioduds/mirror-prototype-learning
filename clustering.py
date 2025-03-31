# clustering.py

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- Load mirror_attention_output.npy and clip_features.npy ---
attention_output = np.load("mirror_attention_output.npy")  # [N, D]
clip_features = np.load("clip_features.npy")  # [N, D]

# --- Clustering with KMeans ---
kmeans = KMeans(n_clusters=5, random_state=0).fit(attention_output)
attention_labels = kmeans.labels_

# --- Analyze clusters using CLIP features ---
correlations = []
for cluster_id in np.unique(attention_labels):
    indices = np.where(attention_labels == cluster_id)[0]
    cluster_attention_vectors = attention_output[indices]
    cluster_clip_vectors = clip_features[indices]

    # Compare attention vectors and CLIP vectors using cosine similarity
    similarity_matrix = pairwise_distances(cluster_attention_vectors, cluster_clip_vectors, metric="cosine")
    avg_similarity = np.mean(1 - similarity_matrix)
    correlations.append((cluster_id, avg_similarity))

# Save correlations for later use
correlation_df = pd.DataFrame(correlations, columns=["Cluster ID", "Average Similarity"])
correlation_df.to_csv("cluster_correlation.csv", index=False)

# Plot the correlations
plt.figure(figsize=(10, 6))
sns.barplot(x="Cluster ID", y="Average Similarity", data=correlation_df)
plt.title("Average Similarity Between Clusters and CLIP Embeddings")
plt.xlabel("Cluster ID")
plt.ylabel("Average Similarity")
plt.tight_layout()
plt.savefig("correlation_plot.png")
plt.show()

print("[INFO] Clustering and correlation analysis completed successfully.")
