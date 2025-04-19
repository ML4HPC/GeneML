import numpy as np
import pandas as pd
import os
import re
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import catboost as cb
import cupy as cp
import optuna

#################################
## Gene embeddings processing ##

def load_gene_embeddings(gene_list, base_dir='/global/cfs/projectdirs/m4244/heesun/NESAP/caduceus/classification/2nd_mdd'):

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

def max_pool_embeddings(embeddings_combined, demographics, labels, train_val_idx, test_idx):
    """
    Process (max pooling) gene embeddings and concatenate it with demographic information & Split to train and test sets
    """
    embeddings_transformed = np.max(embeddings_combined, axis=1)

    X = np.hstack([embeddings_transformed, demographics])
    X_train_val, X_test = X[train_val_idx], X[test_idx]
    y_train_val, y_test = labels[train_val_idx], labels[test_idx]

    return X_train_val, X_test, y_train_val, y_test

#################################################################
## Cross-validated Model Evaluation with Hyperparameter Tuning ##

def tune_train_model(embeddings_combined, demographics, labels):

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=98)
    train_val_idx, test_idx = next(
            skf.split(embeddings_combined, labels)
        )

    X_train_val, X_test, y_train_val, y_test = max_pool_embeddings(
        embeddings_combined, demographics, labels, train_val_idx, test_idx
    )

    def objective(trial):
        """
        Objective function for Optuna
        """
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_strength': trial.suggest_float('random_strength', 0, 1)
        }

        # Inner split - train vs. val
        inner_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=98)
        inner_auc_scores = []
        best_iterations = []

        for train_idx, val_idx in inner_split.split(X_train_val, y_train_val):
            X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
            y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

            # Define and train the model with early stopping
            model = cb.CatBoostClassifier(
                **params, random_seed=98, task_type='GPU', eval_metric='AUC'
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, metric_period=1, verbose=False)
    
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
            best_iter = model.best_iteration_
            best_iterations.append(best_iter)

        trial.set_user_attr('avg_best_iter', np.mean(best_iterations))

        return np.mean(inner_auc_scores)

    # Hyperparameter tuning using Optuna
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=98))
    study.optimize(objective, n_trials=500)

    # Save trial information as a csv file
    trials_info = []
    for each_trial in study.trials:
        trial_info = {
            'trial_number': int(each_trial.number),
            'mean_auc': float(each_trial.value),
            'avg_best_iter': int(each_trial.user_attrs.get('avg_best_iter', None))
        }

        for key, value in each_trial.params.items():
            trial_info[key] = float(value)
        
        trials_info.append(trial_info)

    trials_df = pd.DataFrame(trials_info)
    trials_df.to_csv('/global/cfs/projectdirs/m4244/heesun/NESAP/caduceus/classification/2nd_mdd/optuna_results/final_trials_info.csv', index=False)

    best_params = study.best_trial.params
    best_auc = float(study.best_trial.value)
    print(f'\nBest hyperparameters: {best_params}')
    print(f'Best mean AUC: {best_auc}')

    trial = int(study.best_trial.number)
    iterations = int(best_params['iterations'])
    learning_rate = float(best_params['learning_rate'])
    depth = int(best_params['depth'])
    l2_leaf_reg = int(best_params['l2_leaf_reg'])
    bagging_temperature = float(best_params['bagging_temperature'])
    random_strength = float(best_params['random_strength'])
    best_trial_info = {'trial': trial, 'iterations': iterations, 'learning_rate': learning_rate, 'depth': depth,
                        'l2_leaf_reg': l2_leaf_reg, 'bagging_temperature': bagging_temperature, 'random_strength': random_strength,
                        'auc_val': best_auc}

    # Train the final model on the entire training set (train & validation sets) using the best parameters
    best_best_iter = int(study.best_trial.user_attrs.get('avg_best_iter', best_params.get('iterations', None)))
    best_params['iterations'] = best_best_iter
    print(f'Best iterations: {best_best_iter}')

    best_trial_info['best_iterations'] = best_best_iter

    final_model = cb.CatBoostClassifier(
        **best_params, random_seed=98, task_type='GPU'
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
    outer_auc = float(roc_auc_score(y_test, test_preds))
    best_trial_info['auc_test'] = outer_auc

    # Save the information of best trial of each fold
    best_trial_df = pd.DataFrame([best_trial_info])
    best_trial_df.to_csv('/global/cfs/projectdirs/m4244/heesun/NESAP/caduceus/classification/2nd_mdd/optuna_results/final_best_trial_info.csv', index=False)

    return final_model


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

    # Train a model
    trained_model = tune_train_model(embeddings_combined, demographics, labels)
    joblib.dump(trained_model, '/global/cfs/projectdirs/m4244/heesun/NESAP/caduceus/classification/2nd_mdd/trained_models/cb_max.pkl')
    print('Trained model saved!')
