import numpy as np
import os
import re
import scipy.stats

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import xgboost as xgb
import cupy as cp


def load_gene_embeddings(gene, base_dir='/global/cfs/projectdirs/m4244/heesun/NESAP/caduceus/classification/2nd_mdd'):
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

    # Process demographics
    age_reshaped = age.reshape(-1, 1).astype(np.float32)
    sex_reshaped = sex.reshape(-1, 1).astype(np.float32)
    demographics = np.hstack([age_reshaped, sex_reshaped])

    return gene_embeddings, demographics, labels


def process_embeddings(X_gene_train, X_demo_train, X_gene_test, X_demo_test, y_train):
    """
    Process gene embeddings and demographic information as a single input array
    """
    X_train_idx, X_val_idx, y_train_split, y_val_split = train_test_split(
        np.arange(len(y_train)), y_train, test_size=0.1, random_state=98
    )
    X_gene_train_split = X_gene_train[X_train_idx]
    X_gene_val_split = X_gene_train[X_val_idx]
    X_demo_train_split = X_demo_train[X_train_idx]
    X_demo_val_split = X_demo_train[X_val_idx]

    n_samples_train = X_gene_train_split.shape[0]
    n_samples_val = X_gene_val_split.shape[0]
    n_samples_test = X_gene_test.shape[0]

    X_gene_train_reshaped = X_gene_train_split.reshape(n_samples_train, -1)
    X_gene_val_reshaped = X_gene_val_split.reshape(n_samples_val, -1)
    X_gene_test_reshaped = X_gene_test.reshape(n_samples_test, -1)

    X_train = np.hstack([X_gene_train_reshaped, X_demo_train_split])
    X_val = np.hstack([X_gene_val_reshaped, X_demo_val_split])
    X_test = np.hstack([X_gene_test_reshaped, X_demo_test])
    y_train = y_train_split.copy()
    y_val = y_val_split.copy()

    return X_train, X_val, X_test, y_train, y_val


def train_evaluate_model(
    gene_embeddings, demographics, labels, model
):
    """
    Train and evaluate an XGB model
    """
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=98)
    auc_scores = []

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(gene_embeddings, labels)
    ):
        # Process embeddings & prepare data
        X_gene_train = gene_embeddings[train_idx]
        X_demo_train = demographics[train_idx]
        y_train = labels[train_idx]
        X_gene_test = gene_embeddings[test_idx]
        X_demo_test = demographics[test_idx]
        y_test = labels[test_idx]

        X_train, X_val, X_test, y_train, y_val = process_embeddings(
            X_gene_train, X_demo_train, X_gene_test, X_demo_test, y_train
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        X_train = X_train_scaled.copy()
        X_val = X_val_scaled.copy()
        X_test = X_test_scaled.copy()

        try:
            # # Convert to GPU
            X_train = cp.asarray(X_train)
            X_val = cp.asarray(X_val)
            X_test = cp.asarray(X_test)
            y_train = cp.asarray(y_train)
            y_val = cp.asarray(y_val)

            # Train an XGB model with validation and early stopping
            model.fit(X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False)
                
            # Get predictions for test set
            pred_test = model.predict_proba(X_test)[:, 1]

            # Convert to numpy if needed
            pred_test = (
                cp.asnumpy(pred_test)
                if isinstance(pred_test, cp.ndarray)
                else pred_test
            )

            # Calculate final metrics
            auc_scores.append(roc_auc_score(y_test, pred_test))

        except Exception as e:
            print(f"Error in fold {fold}: {str(e)}")
            continue

    return auc_scores



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

    model = xgb.XGBClassifier(
        n_estimators=100,
        random_state=98,
        early_stopping_rounds=10,
        eval_metric="auc",
        device="cuda"
    )

    for gene in gene_list:
        print('['+gene+']')

        # Load data
        embeddings_combined, demographics, labels = load_gene_embeddings(gene)

        # Train and evaluate models
        scores = train_evaluate_model(
            embeddings_combined,
            demographics,
            labels,
            model
        )

        print(
            f"AUC = {np.mean(scores):.3f} ± {np.std(scores):.3f}"
        )
        print()
