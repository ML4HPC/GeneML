import numpy as np
import os
import re

from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import cuml
from cuml.ensemble import RandomForestClassifier as cuRFC
from cuml import LogisticRegression as cuLR
import xgboost as xgb
import cupy as cp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


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


def load_gene_embeddings(gene_list, base_dir="/global/cfs/projectdirs/m4244/heesun/NESAP/caduceus/classification/simple", emb_dir="embgen_mean"):
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
        directory = os.path.join(base_dir, emb_dir, gene)
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


# Defining an MLP classification model to leverage predefined validation sets for early stopping
class MLPClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)
    
# MLP - Training with early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs=200, patience=5, device='cuda'):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        # Evaluation on validation set
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)

        print(f'Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()

        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print('Early stopping triggered!')
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
    
    return model


torch.manual_seed(98)
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# Modeling (using four ML models) & Manual cross-validation
models = {
    "rf": lambda: cuRFC(random_state=98, n_streams=1),
    "mlp": lambda input_dim: MLPClassifier(
        input_dim=input_dim,
        hidden_dim=100
    ).to(device),
    "xgb": lambda: xgb.XGBClassifier(
        n_estimators=100,
        random_state=98,
        early_stopping_rounds=10,
        eval_metric="auc",
        device="cuda"
    ),
    "lr": lambda: cuLR(tol=0.001),
}


def get_predictions(model, X_tensor, device='cuda'):
    """
    Returns the positive class probabilities for the given input tensor
    """
    model.eval()
    with torch.inference_mode():
        outputs = model(X_tensor.to(device))
        probs = torch.sigmoid(outputs)
        return probs.cpu().numpy().flatten()


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
            # Convert to tensors or GPU if needed
            if model_name == 'mlp':
                X_gene_train = torch.tensor(X_gene_train, dtype=torch.float32)
                X_gene_val = torch.tensor(X_gene_val, dtype=torch.float32)
                X_gene_test = torch.tensor(X_gene_test, dtype=torch.float32)
                X_demo_train = torch.tensor(X_demo_train, dtype=torch.float32)
                X_demo_val = torch.tensor(X_demo_val, dtype=torch.float32)
                X_demo_test = torch.tensor(X_demo_test, dtype=torch.float32)
                y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
                y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
                y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

                gene_train_set = TensorDataset(X_gene_train, y_train)
                gene_val_set = TensorDataset(X_gene_val, y_val)
                gene_test_set = TensorDataset(X_gene_test, y_test)
                demo_train_set = TensorDataset(X_demo_train, y_train)
                demo_val_set = TensorDataset(X_demo_val, y_val)
                demo_test_set = TensorDataset(X_demo_test, y_test)

                gene_train_loader = DataLoader(gene_train_set, batch_size=10, shuffle=True)
                gene_val_loader = DataLoader(gene_val_set, batch_size=10, shuffle=False)
                gene_test_loader = DataLoader(gene_test_set, batch_size=10, shuffle=False)
                demo_train_loader = DataLoader(demo_train_set, batch_size=10, shuffle=True)
                demo_val_loader = DataLoader(demo_val_set, batch_size=10, shuffle=False)
                demo_test_loader = DataLoader(demo_test_set, batch_size=10, shuffle=False)

            else:
                X_gene_train = cp.asarray(X_gene_train)
                X_gene_val = cp.asarray(X_gene_val)
                X_gene_test = cp.asarray(X_gene_test)
                X_demo_train = cp.asarray(X_demo_train)
                X_demo_val = cp.asarray(X_demo_val)
                X_demo_test = cp.asarray(X_demo_test)
                
            # Train models with validation where applicable
            if model_name == 'mlp':
                input_dim_gene = X_gene_train.shape[1]
                input_dim_demo = X_demo_train.shape[1]
                model_gene = models[model_name](input_dim_gene)
                model_demo = models[model_name](input_dim_demo)

            else:
                model_gene = model_code()
                model_demo = model_code()

            if model_name == 'mlp':
                # MLP with early stopping using validation set
                criterion = nn.BCEWithLogitsLoss()
                optimizer_gene = optim.Adam(model_gene.parameters(), lr=0.001)
                optimizer_demo = optim.Adam(model_demo.parameters(), lr=0.001)

                n_epochs = 500
                patience = 10

                print('Training gene model started!')
                model_gene = train_model(
                    model_gene,
                    gene_train_loader,
                    gene_val_loader,
                    criterion,
                    optimizer_gene,
                    n_epochs=n_epochs,
                    patience=patience,
                    device=device
                )

                print('Training demo model started!')
                model_demo = train_model(
                    model_demo,
                    demo_train_loader,
                    demo_val_loader,
                    criterion,
                    optimizer_demo,
                    n_epochs=n_epochs,
                    patience=patience,
                    device=device
                )
                
            elif model_name == "xgb":
                # XGBoost with early stopping using validation set
                model_gene.fit(X_gene_train, y_train,
                             eval_set=[(X_gene_val, y_val)],
                             verbose=False)
                model_demo.fit(X_demo_train, y_train,
                             eval_set=[(X_demo_val, y_val)],
                             verbose=False)
                
            else:
                # RF and LR don't use validation during training
                model_gene.fit(X_gene_train, y_train)
                model_demo.fit(X_demo_train, y_train)

            # Get predictions for validation set to optimize weights
            if model_name == 'mlp':
                gene_pred_val = get_predictions(model_gene, X_gene_val, device)
                demo_pred_val = get_predictions(model_demo, X_demo_val, device)

            else:
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
            if model_name == 'mlp':
                gene_pred_test = get_predictions(model_gene, X_gene_test, device)
                demo_pred_test = get_predictions(model_demo, X_demo_test, device)
            
            else:
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
