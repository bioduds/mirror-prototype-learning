import os
import numpy as np
import torch
import torch.nn as nn
import random

class ForgettingMechanism:
    """
    Implements an organic forgetting mechanism inspired by hair growth and decay.
    """

    def __init__(self, max_capacity: int, decay_rate: float, regrowth_rate: float):
        """
        Initializes the forgetting mechanism.

        Args:
            max_capacity (int): Maximum number of vectors the system can hold before applying forgetting.
            decay_rate (float): Probability of a vector decaying (being forgotten) at each iteration.
            regrowth_rate (float): Probability of a vector being consolidated (strengthened) at each iteration.
        """
        self.max_capacity = max_capacity
        self.decay_rate = decay_rate
        self.regrowth_rate = regrowth_rate
        self.vectors = []

    def add_vector(self, vector: np.ndarray):
        """
        Adds a new vector to the system.

        Args:
            vector (np.ndarray): The new vector to be added.
        """
        if len(self.vectors) >= self.max_capacity:
            self.apply_forgetting()
        self.vectors.append(vector)

    def apply_forgetting(self):
        """
        Applies forgetting and consolidation mechanisms to the stored vectors.
        """
        # Decay mechanism: randomly forget vectors with probability `decay_rate`
        self.vectors = [v for v in self.vectors if random.random() > self.decay_rate]

        # Regrowth mechanism: consolidate similar vectors
        new_vectors = []
        for i, vec in enumerate(self.vectors):
            if random.random() < self.regrowth_rate:
                # Find a random similar vector to merge with
                similar_vec = random.choice(self.vectors)
                if np.dot(vec, similar_vec) > 0.8:  # Similarity threshold
                    merged_vec = (vec + similar_vec) / 2
                    new_vectors.append(merged_vec)
                else:
                    new_vectors.append(vec)
            else:
                new_vectors.append(vec)
        
        self.vectors = new_vectors

    def save_vectors(self, file_path: str):
        """
        Saves the current vectors to a file.

        Args:
            file_path (str): The path to save the vectors.
        """
        np.save(file_path, np.array(self.vectors))
        print(f"[INFO] Saved {len(self.vectors)} vectors to {file_path}")

    def load_vectors(self, file_path: str):
        """
        Loads vectors from a file.

        Args:
            file_path (str): The path to load the vectors from.
        """
        self.vectors = np.load(file_path, allow_pickle=True).tolist()
        print(f"[INFO] Loaded {len(self.vectors)} vectors from {file_path}")


if __name__ == "__main__":
    # Parameters for the forgetting mechanism
    max_capacity = 1000
    decay_rate = 0.05  # Probability of decay (forgetting)
    regrowth_rate = 0.1  # Probability of consolidation (strengthening)

    forgetting_mechanism = ForgettingMechanism(max_capacity, decay_rate, regrowth_rate)

    # Load existing vectors if available
    vectors_file = "engram_vectors.npy"
    if os.path.exists(vectors_file):
        forgetting_mechanism.load_vectors(vectors_file)
    
    # Example: Adding new vectors
    for _ in range(500):
        new_vector = np.random.randn(128)  # Example vector of size 128
        forgetting_mechanism.add_vector(new_vector)

    # Apply forgetting mechanism
    forgetting_mechanism.apply_forgetting()

    # Save the modified vectors
    forgetting_mechanism.save_vectors(vectors_file)
