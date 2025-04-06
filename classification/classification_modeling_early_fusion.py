import numpy as np
import os
import re
import scipy.stats

from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import cuml
from cuml.ensemble import RandomForestClassifier as cuRFC
from cuml import LogisticRegression as cuLR
import xgboost as xgb
import cupy as cp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

def load_gene_embeddings(gene_list, base_dir='/global/cfs/projectdirs/m4244/heesun/NESAP/caduceus/classification/2nd_mdd'):
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

def process_embeddings_with_val(X_gene_train, X_demo_train, X_gene_test, X_demo_test, y_train, method="pca", n_components=256):
    """
    Process gene embeddings for modeling with MLP or XGB
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

    if method == "concat":
        X_gene_train_transformed = X_gene_train_reshaped.copy()
        X_gene_val_transformed = X_gene_val_reshaped.copy()
        X_gene_test_transformed = X_gene_test_reshaped.copy()

    elif method == "pca":
        scaler = StandardScaler()
        X_gene_train_scaled = scaler.fit_transform(X_gene_train_reshaped)
        X_gene_val_scaled = scaler.transform(X_gene_val_reshaped)
        X_gene_test_scaled = scaler.transform(X_gene_test_reshaped)

        pca = PCA(n_components=n_components, random_state=98)
        X_gene_train_transformed = pca.fit_transform(X_gene_train_scaled).astype(np.float32)
        X_gene_val_transformed = pca.transform(X_gene_val_scaled).astype(np.float32)
        X_gene_test_transformed = pca.transform(X_gene_test_scaled).astype(np.float32)

    elif method == "max_pool":
        X_gene_train_transformed = np.max(X_gene_train_split, axis=1)
        X_gene_val_transformed = np.max(X_gene_val_split, axis=1)
        X_gene_test_transformed = np.max(X_gene_test, axis=1)

    else:  # mean_pool
        X_gene_train_transformed = np.mean(X_gene_train_split, axis=1)
        X_gene_val_transformed = np.mean(X_gene_val_split, axis=1)
        X_gene_test_transformed = np.mean(X_gene_test, axis=1)

    X_train = np.hstack([X_gene_train_transformed, X_demo_train_split])
    X_val = np.hstack([X_gene_val_transformed, X_demo_val_split])
    X_test = np.hstack([X_gene_test_transformed, X_demo_test])
    y_train = y_train_split.copy()
    y_val = y_val_split.copy()

    return X_train, X_val, X_test, y_train, y_val


def process_embeddings_no_val(X_gene_train, X_demo_train, X_gene_test, X_demo_test, method="pca", n_components=256):
    """
    Process gene embeddings for modeling with RF or LR
    """
    n_samples_train = X_gene_train.shape[0]
    n_samples_test = X_gene_test.shape[0]

    X_gene_train_reshaped = X_gene_train.reshape(n_samples_train, -1)
    X_gene_test_reshaped = X_gene_test.reshape(n_samples_test, -1)

    if method == "concat":
        X_gene_train_transformed = X_gene_train_reshaped.copy()
        X_gene_test_transformed = X_gene_test_reshaped.copy()

    elif method == "pca":
        scaler = StandardScaler()
        X_gene_train_scaled = scaler.fit_transform(X_gene_train_reshaped)
        X_gene_test_scaled = scaler.transform(X_gene_test_reshaped)

        pca = PCA(n_components=n_components, random_state=98)
        X_gene_train_transformed = pca.fit_transform(X_gene_train_scaled).astype(np.float32)
        X_gene_test_transformed = pca.transform(X_gene_test_scaled).astype(np.float32)

    elif method == "max_pool":
        X_gene_train_transformed = np.max(X_gene_train, axis=1)
        X_gene_test_transformed = np.max(X_gene_test, axis=1)

    else:  # mean_pool
        X_gene_train_transformed = np.mean(X_gene_train, axis=1)
        X_gene_test_transformed = np.mean(X_gene_test, axis=1)
    
    X_train = np.hstack([X_gene_train_transformed, X_demo_train])
    X_test = np.hstack([X_gene_test_transformed, X_demo_test])

    return X_train, X_test


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
def train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs=200, patience=10, device='cuda'):

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    "lr": lambda: cuLR(tol=0.001, max_iter=5000),
}

def get_predictions(model, X_tensor, device='cuda'):
    """
    Returns the positive class probabilities for the given input tensor (MLP)
    """
    model.eval()
    with torch.inference_mode():
        outputs = model(X_tensor.to(device))
        probs = torch.sigmoid(outputs)
        return probs.cpu().numpy().flatten()


def train_evaluate_model(
    embeddings_combined, demographics, labels, model_name, model_code, method
):
    """
    Train and evaluate models for each gene embedding processing method
    """
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=98)
    auc_scores = []

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(embeddings_combined, labels)
    ):
        # Process embeddings & prepare data
        X_gene_train = embeddings_combined[train_idx]
        X_demo_train = demographics[train_idx]
        y_train = labels[train_idx]
        X_gene_test = embeddings_combined[test_idx]
        X_demo_test = demographics[test_idx]
        y_test = labels[test_idx]

        if model_name in ['mlp', 'xgb']:
            X_train, X_val, X_test, y_train, y_val = process_embeddings_with_val(
                X_gene_train, X_demo_train, X_gene_test, X_demo_test, y_train, method=method, n_components=256
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)

            X_train = X_train_scaled.copy()
            X_val = X_val_scaled.copy()
            X_test = X_test_scaled.copy()

        else:
            X_train, X_test = process_embeddings_no_val(
                X_gene_train, X_demo_train, X_gene_test, X_demo_test, method=method, n_components=256
                )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            X_train = X_train_scaled.copy()
            X_test = X_test_scaled.copy()

        try:
            # Convert to tensors or GPU if needed
            if model_name == 'mlp':
                X_train = torch.tensor(X_train, dtype=torch.float32)
                X_val = torch.tensor(X_val, dtype=torch.float32)
                X_test = torch.tensor(X_test, dtype=torch.float32)
                y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
                y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

                train_set = TensorDataset(X_train, y_train)
                val_set = TensorDataset(X_val, y_val)

                train_loader = DataLoader(train_set, batch_size=10, shuffle=True)
                val_loader = DataLoader(val_set, batch_size=10, shuffle=False)

            elif model_name in ['rf', 'lr']:
                X_train = cp.asarray(X_train)
                X_test = cp.asarray(X_test)
                y_train = cp.asarray(y_train)

            else:
                X_train = cp.asarray(X_train)
                X_val = cp.asarray(X_val)
                X_test = cp.asarray(X_test)
                y_train = cp.asarray(y_train)
                y_val = cp.asarray(y_val)

            # Train models with validation where applicable
            if model_name == 'mlp':
                input_dim = X_train.shape[1]
                clf = models[model_name](input_dim)

            else:
                clf = model_code()

            if model_name == 'mlp':
                criterion = nn.BCEWithLogitsLoss()
                optimizer = optim.Adam(clf.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001)

                n_epochs = 200
                patience = 10

                print('Training started!')
                clf = train_model(
                    clf,
                    train_loader,
                    val_loader,
                    criterion,
                    optimizer,
                    n_epochs=n_epochs,
                    patience=patience,
                    device=device
                )

            elif model_name == "xgb":
                # XGBoost with early stopping using validation set
                clf.fit(X_train, y_train,
                             eval_set=[(X_val, y_val)],
                             verbose=False)
                
            else:
                # RF and LR don't use validation during training
                clf.fit(X_train, y_train)

            # Get predictions for test set
            if model_name == 'mlp':
                pred_test = get_predictions(clf, X_test, device)
            
            else:
                pred_test = clf.predict_proba(X_test)[:, 1]

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
    # Load data
    embeddings_combined, demographics, labels = load_gene_embeddings(gene_list)

    # Process embeddings with different methods
    embedding_methods = ["pca", "max_pool", "mean_pool", "concat"]
    results = {}

    for method in embedding_methods:
        print(f"\nProcessing embeddings with method: {method}")

        # Train and evaluate models
        for model_name, model_code in models.items():
            print(f"Training {model_name.upper()}")
            scores = train_evaluate_model(
                embeddings_combined,
                demographics,
                labels,
                model_name,
                model_code,
                method
            )

            results[f"{method}_{model_name}"] = scores

            print(
                f"AUC = {np.mean(scores):.3f} ± {np.std(scores):.3f}"
            )
            print()
