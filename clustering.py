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

    # Since dimensions don't match (attention: 128D, CLIP: 512D),
    # we'll analyze them separately and compute internal coherence
    if len(cluster_attention_vectors) > 1:
        # Internal coherence of attention vectors within cluster
        attention_coherence = 1 - \
            np.mean(pairwise_distances(
                cluster_attention_vectors, metric="cosine"))
        correlations.append((cluster_id, attention_coherence))
    else:
        # Single vector has perfect coherence
        correlations.append((cluster_id, 1.0))

# Save correlations for later use
correlation_df = pd.DataFrame(
    correlations, columns=["Cluster ID", "Attention Coherence"])
correlation_df.to_csv("cluster_correlation.csv", index=False)

# Also save clustering results
clustering_results = {
    'attention_labels': attention_labels,
    'cluster_centers': kmeans.cluster_centers_,
    'correlations': correlations,
    'n_clusters': len(np.unique(attention_labels))
}
np.save("clustering_results.npy", clustering_results)

# Plot the correlations
plt.figure(figsize=(10, 6))
sns.barplot(x="Cluster ID", y="Attention Coherence", data=correlation_df)
plt.title("Attention Vector Coherence Within Clusters")
plt.xlabel("Cluster ID")
plt.ylabel("Average Coherence")
plt.tight_layout()
plt.savefig("correlation_plot.png")
plt.show()

print("[INFO] Clustering and correlation analysis completed successfully.")
