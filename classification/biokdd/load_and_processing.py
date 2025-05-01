import numpy as np
import os
import re

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_gene_embeddings(gene_list, base_dir="/pscratch/sd/h/hazely/NESAP/caduceus/classification/2nd_mdd/input_data"):
    
    # Load demographic data and labels
    age = np.load(os.path.join(base_dir, "age.npy"))
    sex = np.load(os.path.join(base_dir, "sex.npy"))
    labels = np.load(os.path.join(base_dir, "labels.npy"))

    def numerical_sort(file_name):
        match = re.search(r"embeddings_(\d+).npy", file_name)
        return int(match.group(1))

    # Load embeddings for each gene
    embeddings_list = []
    for gene in gene_list:
        # Get sorted file paths
        directory = os.path.join(base_dir, gene)
        file_paths = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.startswith("embeddings")
        ]
        sorted_file_paths = sorted(file_paths, key=numerical_sort)

        # Load and concatenate embeddings for current gene
        gene_embeddings = []
        for file_path in sorted_file_paths:
            chunk = np.load(file_path)
            gene_embeddings.append(chunk)

        # Combine chunks for current gene
        gene_embeddings = np.concatenate(gene_embeddings, axis=0)
        embeddings_list.append(gene_embeddings)

    # Stack all gene embeddings
    embeddings_combined = np.stack(embeddings_list, axis=1)

    # Process demographics
    age_reshaped = age.reshape(-1, 1).astype(np.float32)
    sex_reshaped = sex.reshape(-1, 1).astype(np.float32)
    demographics = np.hstack([age_reshaped, sex_reshaped])

    return embeddings_combined, demographics, labels


def process_embeddings(embeddings_combined, demographics, labels, train_val_idx, test_idx, method="pca"):
    
    if method == "concat":
        n_samples = embeddings_combined.shape[0]
        embeddings_reshaped = embeddings_combined.reshape(n_samples, -1)
        embeddings_transformed = embeddings_reshaped.copy()

    elif method == "pca":
        n_samples = embeddings_combined.shape[0]
        embeddings_reshaped = embeddings_combined.reshape(n_samples, -1)

        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings_reshaped)

        pca = PCA(n_components=256, random_state=98)
        embeddings_transformed = pca.fit_transform(embeddings_scaled).astype(np.float32)

    elif method == "max_pool":
        embeddings_transformed = np.max(embeddings_combined, axis=1)

    else:  # mean_pool
        embeddings_transformed = np.mean(embeddings_combined, axis=1)

    X = np.hstack([embeddings_transformed, demographics])
    X_train_val, X_test = X[train_val_idx], X[test_idx]
    y_train_val, y_test = labels[train_val_idx], labels[test_idx]

    return X_train_val, X_test, y_train_val, y_test
