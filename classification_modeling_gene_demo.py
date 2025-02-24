import numpy as np
import os
import re
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import cuml
from cuml.ensemble import RandomForestClassifier as cuRFC
from sklearn.neural_network import MLPClassifier
from cuml import LogisticRegression as cuLR
import xgboost as xgb
import cupy as cp


def optimize_ensemble_weights(gene_pred, demo_pred, y_true):
    # Simple grid search for weights
    best_auc = 0
    best_weights = (0.5, 0.5)

    for w1 in np.linspace(0, 1, 11):
        w2 = 1 - w1
        combined = w1 * gene_pred + w2 * demo_pred
        auc = roc_auc_score(y_true, combined)
        if auc > best_auc:
            best_auc = auc
            best_weights = (w1, w2)

    return best_weights


def load_gene_embeddings(gene_list, base_dir="../"):
    """
    Load and concatenate embeddings for all genes.

    Args:
        gene_list (list): List of gene names
        base_dir (str): Base directory containing gene folders

    Returns:
        tuple: (embeddings_combined, demographics, labels)
    """
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


def process_embeddings(embeddings_combined, method="pca", n_components=256):
    """
    Process the combined embeddings using various methods.

    Args:
        embeddings_combined (np.ndarray): Combined embeddings of shape (n_samples, n_genes, n_features)
        method (str): One of 'pca', 'max_pool', 'mean_pool', or 'concat'
        n_components (int): Number of components for PCA

    Returns:
        np.ndarray: Processed embeddings
    """
    n_samples = embeddings_combined.shape[0]

    if method == "concat":
        return embeddings_combined.reshape(n_samples, -1)

    if method == "pca":
        embeddings_reshaped = embeddings_combined.reshape(n_samples, -1)
        pca = PCA(n_components=n_components, random_state=98)
        return pca.fit_transform(embeddings_reshaped).astype(np.float32)

    if method == "max_pool":
        return np.max(embeddings_combined, axis=1)

    if method == "mean_pool":
        return np.mean(embeddings_combined, axis=1)

    raise ValueError(f"Unknown method: {method}")


# Modeling (using four ML models) & Manual cross-validation
models = {
    "rf": lambda: cuRFC(random_state=98, n_streams=1),
    "mlp": lambda: MLPClassifier(random_state=98, max_iter=500),
    "xgb": lambda: xgb.XGBClassifier(
        n_estimators=100, random_state=98, eval_metric="error", device="cuda"
    ),
    "lr": lambda: cuLR(tol=0.001),
}


def train_evaluate_model(
    gene_embeddings_transformed, demographics, labels, model_name, model_code
):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=98)
    auc_scores = {"gene": [], "demo": [], "combined": []}
    weights_history = []

    for fold, (train_val_idx, test_idx) in enumerate(
        skf.split(gene_embeddings_transformed, labels)
    ):
        # Create inner split for validation
        inner_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=98)
        train_idx, val_idx = next(
            inner_split.split(
                gene_embeddings_transformed[train_val_idx], labels[train_val_idx]
            )
        )

        # Map inner indices to original data indices
        train_idx = train_val_idx[train_idx]
        val_idx = train_val_idx[val_idx]

        # Scale features using only training data
        scaler_gene = StandardScaler()
        scaler_demo = StandardScaler()

        # Prepare training data
        X_gene_train = scaler_gene.fit_transform(gene_embeddings_transformed[train_idx])
        X_demo_train = scaler_demo.fit_transform(demographics[train_idx])
        y_train = labels[train_idx]

        # Prepare validation data
        X_gene_val = scaler_gene.transform(gene_embeddings_transformed[val_idx])
        X_demo_val = scaler_demo.transform(demographics[val_idx])
        y_val = labels[val_idx]

        # Prepare test data
        X_gene_test = scaler_gene.transform(gene_embeddings_transformed[test_idx])
        X_demo_test = scaler_demo.transform(demographics[test_idx])
        y_test = labels[test_idx]

        try:
            # Convert to GPU if needed
            if model_name != "mlp":
                X_gene_train = cp.asarray(X_gene_train)
                X_gene_val = cp.asarray(X_gene_val)
                X_gene_test = cp.asarray(X_gene_test)
                X_demo_train = cp.asarray(X_demo_train)
                X_demo_val = cp.asarray(X_demo_val)
                X_demo_test = cp.asarray(X_demo_test)

            # Train models
            model_gene = model_code()
            model_demo = model_code()
            model_gene.fit(X_gene_train, y_train)
            model_demo.fit(X_demo_train, y_train)

            # Get predictions for validation set to optimize weights
            gene_pred_val = model_gene.predict_proba(X_gene_val)[:, 1]
            demo_pred_val = model_demo.predict_proba(X_demo_val)[:, 1]

            # Convert to numpy if needed
            gene_pred_val = (
                cp.asnumpy(gene_pred_val)
                if isinstance(gene_pred_val, cp.ndarray)
                else gene_pred_val
            )
            demo_pred_val = (
                cp.asnumpy(demo_pred_val)
                if isinstance(demo_pred_val, cp.ndarray)
                else demo_pred_val
            )

            # Optimize weights using validation set
            weights = optimize_ensemble_weights(gene_pred_val, demo_pred_val, y_val)
            weights_history.append(weights)

            # Get predictions for test set
            gene_pred_test = model_gene.predict_proba(X_gene_test)[:, 1]
            demo_pred_test = model_demo.predict_proba(X_demo_test)[:, 1]

            # Convert to numpy if needed
            gene_pred_test = (
                cp.asnumpy(gene_pred_test)
                if isinstance(gene_pred_test, cp.ndarray)
                else gene_pred_test
            )
            demo_pred_test = (
                cp.asnumpy(demo_pred_test)
                if isinstance(demo_pred_test, cp.ndarray)
                else demo_pred_test
            )

            # Apply optimized weights to test predictions
            combined_pred = weights[0] * gene_pred_test + weights[1] * demo_pred_test

            # Calculate final metrics
            auc_scores["gene"].append(roc_auc_score(y_test, gene_pred_test))
            auc_scores["demo"].append(roc_auc_score(y_test, demo_pred_test))
            auc_scores["combined"].append(roc_auc_score(y_test, combined_pred))

        except Exception as e:
            print(f"Error in fold {fold}: {str(e)}")
            continue

    return auc_scores, weights_history


if __name__ == "__main__":
    # Define gene list
    gene_list = [
        "CRHR1",
        "ESR1",
        "ESR2",
        "PCLO",
        "FHIT",
        "CACNA1C",
        "DRD2",
        "GRM7",
        "EHD3",
        "BICC1",
        "PLOD1",
        "LINC00687",
        "CSMD1",
        "LHPP",
        "APC",
        "ARHGAP8",
        "LOC100996549",
        "CNTNAP2",
        "CRY1",
        "COMT",
        "FKBP5",
        "HTR2A",
        "BDNF",
        "SLC6A4",
        "ACE",
        "SLC6A2",
        "KCNK2",
        "NR3C1",
        "MTHFR",
        "TPH1",
        "TPH2",
        "SOD2",
        "CNR1",
        "TNF",
        "HTR1A",
        "ABCB1",
        "GNB3",
        "GSK3B",
    ]
    # Load data
    embeddings_combined, demographics, labels = load_gene_embeddings(gene_list)

    # Process embeddings with different methods
    embedding_methods = ["pca", "max_pool", "mean_pool", "concat"]
    results = {}

    for method in embedding_methods:
        print(f"\nProcessing embeddings with method: {method}")
        gene_embeddings_transformed = process_embeddings(
            embeddings_combined, method=method
        )

        # Train and evaluate models
        for model_name, model_code in models.items():
            print(f"Training {model_name.upper()}")
            scores, weights = train_evaluate_model(
                gene_embeddings_transformed,
                demographics,
                labels,
                model_name,
                model_code,
            )

            results[f"{method}_{model_name}"] = {"scores": scores, "weights": weights}

            avg_weights = np.mean(weights, axis=0)
            print(
                f"Gene-only AUC = {np.mean(scores['gene']):.3f} ± {np.std(scores['gene']):.3f}"
            )
            print(
                f"Demographics-only AUC = {np.mean(scores['demo']):.3f} ± {np.std(scores['demo']):.3f}"
            )
            print(
                f"Combined AUC = {np.mean(scores['combined']):.3f} ± {np.std(scores['combined']):.3f}"
            )
            print(
                f"Average optimal weights (gene, demo) = ({avg_weights[0]:.2f}, {avg_weights[1]:.2f})"
            )
            print()
