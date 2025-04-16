import numpy as np
import os
import re
from MLstatkit.stats import Delong_test
from statsmodels.stats.multitest import fdrcorrection

from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import cuml
from cuml.ensemble import RandomForestClassifier as cuRFC
from cuml import LogisticRegression as cuLR
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import cupy as cp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

#################################
## Gene embeddings processing ##

def load_gene_embeddings(gene_list, base_dir='/global/cfs/projectdirs/m4244/heesun/NESAP/caduceus/classification/2nd_mdd'):
    """
    Load and concatenate embeddings for all genes.

    Args:
        gene_list (list): List of gene names

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


def process_embeddings(embeddings_combined, demographics, labels, train_val_idx, test_idx, method="pca"):
    """
    Process gene embeddings and concatenate it with demographic information & Split to train and test sets
    """
    n_samples = embeddings_combined.shape[0]
    embeddings_reshaped = embeddings_combined.reshape(n_samples, -1)

    if method == "concat":
        embeddings_transformed = embeddings_reshaped.copy()

    elif method == "pca":
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

#######################
## Model Definition ##

# 1) MLPClassifier
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
    

# 2) 1D CNNClassifier
class CNNClassifier(nn.Module):

    def __init__(self, input_dim, dropout=0.5, random_seed=None):
        """
        1D CNN for processing gene embeddings and combining them with demographic data.

        Args:
            input_dim (int): Total number of features in the 1D input 
            dropout (float): dropout rate applied after flattening and between the FC layers.
            random_seed (int): seed for reproducibility
        """
        # Set the random seed
        if random_seed is not None:
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)

        super(CNNClassifier, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        with torch.inference_mode():
            dummy_input = torch.zeros(1, 1, input_dim)
            x = self.pool(F.relu(self.bn1(self.conv1(dummy_input))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            flattened_size = x.reshape(1, -1).shape[-1]
        
        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension: (batch, 1, input_dim)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.reshape(x.size(0), -1)
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)

        return logits
    
######################################
## MLP/CNN - Training & Prediction ##

# Training & validation with early stopping
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

        #print(f'Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()

        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                #print('Early stopping triggered!')
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
    
    return model


# Getting predicted probabilities for the positive class
def get_predictions(model, X_test, device='cuda'):
    """
    Return the positive class probabilities for the given input tensor.
    """
    model.eval()
    with torch.inference_mode():
        outputs = model(X_test.to(device))
        probs = torch.sigmoid(outputs)

        return probs.cpu().numpy().flatten()

######################################
## Cross-validated Model Evaluation ##

def train_evaluate_model(
    embeddings_combined, demographics, labels, model_name, model_code, method
):
    outer_split = StratifiedKFold(n_splits=10, shuffle=True, random_state=98)
    auc_scores = []
    all_y_test = []
    all_preds = []

    for outer_fold, (train_val_idx, test_idx) in enumerate(
        outer_split.split(embeddings_combined, labels)
    ):
        X_train_val, X_test, y_train_val, y_test = process_embeddings(
            embeddings_combined, demographics, labels, train_val_idx, test_idx, method=method
            )

        inner_preds = []
        inner_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=98)
        for inner_fold, (train_idx, val_idx) in enumerate(
            inner_split.split(X_train_val, y_train_val)
        ):
            X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
            y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)

            X_train = X_train_scaled.copy()
            X_val = X_val_scaled.copy()
            X_test = X_test_scaled.copy()

            try:
                # Convert to tensors or GPU if needed
                if model_name in ['mlp','cnn']:
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

                elif model_name == 'xgb':
                    X_train = cp.asarray(X_train)
                    X_val = cp.asarray(X_val)
                    X_test = cp.asarray(X_test)
                    y_train = cp.asarray(y_train)
                    y_val = cp.asarray(y_val)

                # Train models with validation where applicable
                if model_name in ['mlp','cnn']:
                    input_dim = X_train.shape[1]
                    clf = models[model_name](input_dim)

                else:
                    clf = model_code()

                if model_name in ['mlp','cnn']:
                    criterion = nn.BCEWithLogitsLoss()
                    optimizer = optim.Adam(clf.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001)

                    n_epochs = 200
                    patience = 10

                    #print('Training started!')
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

                elif model_name == 'xgb':
                    # XGBoost with early stopping using validation set
                    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                    
                elif model_name == 'lgb':
                    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc')

                elif model_name == 'cb':
                    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, metric_period=1, verbose=False)
                    
                else:
                    # RF and LR don't use validation during training
                    clf.fit(X_train, y_train)

                # Get predictions for test set
                if model_name in ['mlp','cnn']:
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
                inner_preds.append(np.array(pred_test).flatten())

            except Exception as e:
                print(f"Error in fold {outer_fold}-{inner_fold}: {str(e)}")
                continue

        avg_inner_preds = np.mean(inner_preds, axis=0)
        all_y_test.append(np.array(y_test).flatten())
        all_preds.append(avg_inner_preds)
    
    all_y_test = np.concatenate(all_y_test)
    all_preds = np.concatenate(all_preds)
    
    return {"auc_scores": auc_scores, "y_true": all_y_test, "y_pred": all_preds}

####################
## Delong's test ##

def delong_test(method_results):

    comparison_results = []
    model_names = list(method_results.keys())
    print("\n[DeLong's Test]")

    # Delong's test
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            model1 = model_names[i]
            model2 = model_names[j]
            print(f"\n({model1.upper()} vs. {model2.upper()})")

            try:
                # Get aggregated predictions and true labels for each model
                y_true = method_results[model1]["y_true"]
                preds1 = method_results[model1]["y_pred"]
                preds2 = method_results[model2]["y_pred"]

                overall_auc_model1 = roc_auc_score(y_true, preds1)
                overall_auc_model2 = roc_auc_score(y_true, preds2)
                print(f"Overall AUC ({model1.upper()}) = {overall_auc_model1:.3f}")
                print(f"Overall AUC ({model2.upper()}) = {overall_auc_model2:.3f}")

                z, p = Delong_test(y_true, preds1, preds2)
                comparison_results.append((model1, model2, z, p))

            except Exception as e:
                print(f"Error in DeLong's test comparing {model1.upper()} and {model2.upper()}: {e}")

    return comparison_results


torch.manual_seed(98)
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
    final_results = {}

    # Modeling (using four ML models) & Manual cross-validation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {
        "rf": lambda: cuRFC(random_state=98, n_streams=1),
        "mlp": lambda input_dim: MLPClassifier(input_dim=input_dim, hidden_dim=100).to(device),
        'cnn': lambda input_dim: CNNClassifier(input_dim=input_dim, dropout=0.5, random_seed=98).to(device),
        "xgb": lambda: xgb.XGBClassifier(n_estimators=100, random_state=98, early_stopping_rounds=10, eval_metric="auc", device="cuda"),
        "lgb": lambda: lgb.LGBMClassifier(n_estimators=100, random_state=98, early_stopping_round=10, device='cuda', verbose=0),
        "cb": lambda: cb.CatBoostClassifier(iterations=100, random_seed=98, task_type='GPU', eval_metric='AUC'),
        "lr": lambda: cuLR(tol=0.001, max_iter=5000),
}

    for method in embedding_methods:
        print(f"\nProcessing embeddings with method: {method}")
        method_results = {}
        
        # Train and evaluate models
        for model_name, model_code in models.items():
            print(f"\n[Training {model_name.upper()}]")

            results = train_evaluate_model(
                embeddings_combined,
                demographics,
                labels,
                model_name,
                model_code,
                method
            )
            torch.cuda.empty_cache()

            method_results[model_name] = results
            model_aucs = results["auc_scores"]
            print(f"AUC = {np.mean(model_aucs):.3f} ± {np.std(model_aucs):.3f}")

        final_results[method] = method_results
        
        # Pairwise DeLong's tests between models for each method
        comparison_results = delong_test(method_results)

        print("\nDelong's test completed. Now proceeding to FDR correction...")

        # FDR correction
        if comparison_results:
            raw_p = [x[3] for x in comparison_results]
            h0_rejected, p_corrected = fdrcorrection(raw_p, alpha=0.05)
            
            for idx, (model1, model2, z, p) in enumerate(comparison_results):
                print(f"\n{model1.upper()} vs. {model2.upper()}:")
                print(f"Z-score (absolute) = {abs(z):.4f}")
                print(f"Raw p-value (-log10-transformed) = {-np.log10(p+1e-300):.4f}")
                print(f"FDR-adjusted p-value (-log10-transformed) = {-np.log10(p_corrected[idx]+1e-300):.4f}")
                if h0_rejected[idx] == True:
                    print("AUC values are significantly different.")
                else:
                    print("AUC values are NOT significantly different.")
                
        else:
            print("\nNo pairwise comparisons available.")

        print()
