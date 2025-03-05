import numpy as np
import pandas as pd
import os
import re

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import xgboost as xgb
import cupy as cp

import optuna


def load_gene_embeddings(gene_list, base_dir="embgen_mean"):  # embgen_mean, embgen_max
    """
    Load and concatenate embeddings for all genes.

    Args:
        gene_list (list): List of gene names
        base_dir (str): Base directory containing gene folders

    Returns:
        tuple: (embeddings_combined, demographics, labels)
    """
    # Load demographic data and labels
    age = np.load("age.npy")
    sex = np.load("sex.npy")
    labels = np.load("labels.npy")

    def numerical_sort(file_name):
        match = re.search(r"embeddings_(\d+).npy", file_name)  # embeddings_, embeddings_max_
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


def process_embeddings(embeddings_combined, demographics):
    """
    Process the combined embeddings using concatenation.

    Args:
        embeddings_combined (np.ndarray): Combined gene embeddings of shape (n_samples, n_genes, n_features)
        demographics (np.ndarray): Combined demographic embeddings of shape (n_samples, n_features)
    
    Returns:
        np.ndarray: Processed embeddings
    """
    n_samples = embeddings_combined.shape[0]
    all_gene_embeddings = embeddings_combined.reshape(n_samples, -1)
    
    return np.hstack([all_gene_embeddings, demographics])



def tune_train_evaluate_model(embeddings_transformed, labels):
    outer_split = StratifiedKFold(n_splits=10, shuffle=True, random_state=98)
    outer_auc_scores = []

    # Outer split - train/val vs. test
    for fold, (train_val_idx, test_idx) in enumerate(
        outer_split.split(embeddings_transformed, labels)
    ):
        X_train_val, X_test = embeddings_transformed[train_val_idx], embeddings_transformed[test_idx]
        y_train_val, y_test = labels[train_val_idx], labels[test_idx]

        def objective(trial):
            """
            Objective function for Optuna
            """
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 1),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10, log=True)
            }

            # Inner split - train vs. val
            inner_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=98)
            inner_auc_scores = []
            best_iterations = []

            for train_idx, val_idx in inner_split.split(X_train_val, y_train_val):
                X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
                y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

                # Scale features using only training data
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

                # Convert to cupy arrays
                X_train = cp.asarray(X_train)
                X_val = cp.asarray(X_val)

                # Define and train the model with early stopping
                model = xgb.XGBClassifier(
                    **params,
                    random_state=98,
                    early_stopping_rounds=10,
                    eval_metric="auc",
                    device="cuda"
                )
                model.fit(X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            verbose=False)
        
                # Get predictions for validation set
                val_preds = model.predict_proba(X_val)[:, 1]
        
                # Convert to numpy
                val_preds = (
                    cp.asnumpy(val_preds)
                    if isinstance(val_preds, cp.ndarray)
                    else val_preds
                )

                # Compute AUC
                inner_auc_scores.append(roc_auc_score(y_val, val_preds))

                # Best iteration
                best_iter = model.best_iteration
                best_iterations.append(best_iter)

            trial.set_user_attr('avg_best_iter', np.mean(best_iterations))

            return np.mean(inner_auc_scores)

        # Hyperparameter tuning using Optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)

        # Save trial information as a csv file
        trials_info = []
        for each_trial in study.trials:
            trial_info = {
                'trial_number': each_trial.number,
                'mean_auc': each_trial.value,
                'avg_best_iter': each_trial.user_attrs.get('avg_best_iter', None)
            }

            for key, value in each_trial.params.items():
                trial_info[key] = value
            
            trials_info.append(trial_info)

        trials_df = pd.DataFrame(trials_info)
        trials_df.to_csv(f'optuna_trial/mean/fold_{fold}_trial_info.csv', index=False)  # mean, max

        best_params = study.best_trial.params
        best_auc = study.best_trial.value
        print(f'Best hyperparameters in fold {fold}: {best_params}')
        print(f'Best mean AUC in fold {fold}: {best_auc}')

        # Train the final model on the entire training set (train & validation sets) using the best parameters
        best_best_iter = int(study.best_trial.user_attrs.get('avg_best_iter', best_params.get('n_estimators', None)))
        best_params['n_estimators'] = best_best_iter
        print(f'Best iterations in fold {fold}: {best_best_iter}')

        # Scale features only on train/val set
        scaler = StandardScaler()
        X_train_val = scaler.fit_transform(X_train_val)
        X_test = scaler.transform(X_test)

        # Convert to cupy arrays
        X_train_val = cp.asarray(X_train_val)
        X_test = cp.asarray(X_test)

        final_model = xgb.XGBClassifier(
            **best_params,
            random_state=98,
            device="cuda"
        )
        final_model.fit(X_train_val, y_train_val)

        # Evaluate the model on the test set
        test_preds = final_model.predict_proba(X_test)[:, 1]

        # Convert to numpy
        test_preds = (
            cp.asnumpy(test_preds)
            if isinstance(test_preds, cp.ndarray)
            else test_preds
        )
        
        # Compute AUC
        outer_auc_scores.append(roc_auc_score(y_test, test_preds))

    return outer_auc_scores



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

    # Concatenate gene embeddings and demographic information
    embeddings_transformed = process_embeddings(embeddings_combined, demographics)

    # Train and evaluate model
    scores = tune_train_evaluate_model(embeddings_transformed, labels)
    print(f'AUC = {np.mean(scores):.3f} ± {np.std(scores):.3f}')
