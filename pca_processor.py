"""PCA processor for feature dimensionality reduction."""
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import Optional, Tuple


class PCAProcessor:
    """Handles PCA processing of feature vectors."""

    def __init__(self, n_components: int = 128):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.is_fitted = False

    def fit_transform(self, features: np.ndarray) -> Optional[np.ndarray]:
        """Fit PCA and transform features if enough data points exist."""
        if features.shape[0] < 2:
            print("[INFO] Not enough data points for PCA. Need at least 2 samples.")
            return None

        if features.shape[1] < self.n_components:
            print(
                f"[INFO] Input dimension ({features.shape[1]}) is smaller than n_components ({self.n_components})")
            self.n_components = features.shape[1]
            self.pca = PCA(n_components=self.n_components)

        try:
            transformed = self.pca.fit_transform(features)
            self.is_fitted = True
            return transformed
        except Exception as e:
            print(f"[INFO] PCA failed: {e}")
            return None

    def transform(self, features: np.ndarray) -> Optional[np.ndarray]:
        """Transform features using fitted PCA."""
        if not self.is_fitted:
            print("[INFO] PCA not fitted yet. Call fit_transform first.")
            return None

        try:
            return self.pca.transform(features)
        except Exception as e:
            print(f"[INFO] PCA transform failed: {e}")
            return None

    def plot_explained_variance(self, save_path: str = "pca_variance.png"):
        """Plot explained variance ratio."""
        if not self.is_fitted:
            print("[INFO] PCA not fitted yet. Cannot plot variance.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance vs. Number of Components')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        print(f"[INFO] Saved PCA variance plot to {save_path}")


def process_features(features: np.ndarray, n_components: int = 128) -> Tuple[Optional[np.ndarray], PCAProcessor]:
    """Process features through PCA if possible."""
    processor = PCAProcessor(n_components=n_components)
    transformed = processor.fit_transform(features)

    if transformed is not None:
        processor.plot_explained_variance()

    return transformed, processor
