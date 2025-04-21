import random
import numpy as np
import pandas as pd
import cupy as cp
import optuna

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import cuml
from cuml.ensemble import RandomForestClassifier as cuRFC
from cuml import LogisticRegression as cuLR
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from biokdd.neural_nets import MLPClassifier, CNNClassifier, train_model, get_predictions
from biokdd.load_and_processing import process_embeddings

seed = 98
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

models = {
        "rf": lambda **hp: cuRFC(**hp, random_state=98, n_streams=1),
        "mlp": lambda input_dim, **hp: MLPClassifier(input_dim=input_dim, **hp).to(device),
        'cnn': lambda input_dim, **hp: CNNClassifier(input_dim=input_dim, **hp, random_seed=98).to(device),
        "xgb": lambda **hp: xgb.XGBClassifier(**hp, random_state=98, early_stopping_rounds=10, eval_metric="auc", device="cuda"),
        "lgb": lambda **hp: lgb.LGBMClassifier(**hp, random_state=98, early_stopping_round=10, device='cuda', verbose=0),
        "cb": lambda **hp: cb.CatBoostClassifier(**hp, random_seed=98, task_type='GPU', eval_metric='AUC'),
        "lr": lambda **hp: cuLR(**hp, max_iter=5000, solver='saga'),
}

final_models = {
    "rf": lambda **hp: cuRFC(**hp, random_state=98, n_streams=1),
    "mlp": lambda input_dim, **hp: MLPClassifier(input_dim=input_dim, **hp).to(device),
    'cnn': lambda input_dim, **hp: CNNClassifier(input_dim=input_dim, **hp, random_seed=98).to(device),
    "xgb": lambda **hp: xgb.XGBClassifier(**hp, random_state=98, device="cuda"),
    "lgb": lambda **hp: lgb.LGBMClassifier(**hp, random_state=98, device='cuda', verbose=0),
    "cb": lambda **hp: cb.CatBoostClassifier(**hp, random_seed=98, task_type='GPU'),
    "lr": lambda **hp: cuLR(**hp, max_iter=5000, solver='saga')
}


generator = torch.Generator()
generator.manual_seed(seed)

def make_objective(X_train_val, y_train_val, model_name):

    def objective(trial):
        """
        Objective function for Optuna
        """
        if model_name == 'rf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 32),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7])
            }
        elif model_name == 'mlp':
            params = {
                'hidden_dim': trial.suggest_int('hidden_dim', 50, 200, step=50),
                'lr': trial.suggest_float('lr', 1e-5, 1e-1, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
            }
        elif model_name == 'cnn':
            params = {
                'conv1_out': trial.suggest_categorical('conv1_out', [16, 32, 64]),
                'conv2_out': trial.suggest_categorical('conv2_out', [32, 64, 128]),
                'conv3_out': trial.suggest_categorical('conv3_out', [64, 128, 256]),
                'fc1_units': trial.suggest_categorical('fc1_units', [256, 512, 1024]),
                'fc2_units': trial.suggest_categorical('fc2_units', [64, 128, 256]),
                'dropout': trial.suggest_float('dropout', 0.2, 0.5),
                'lr': trial.suggest_float('lr', 1e-5, 1e-1, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
            }
        elif model_name == 'xgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1)
            }
        elif model_name == 'lgb':
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 20, 127),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'n_estimators': trial.suggest_int('n_estimators', 100, 3000, step=100),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'subsample_freq': trial.suggest_int('subsample_freq', 0, 10),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
            }
        elif model_name == 'cb':
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                'random_strength': trial.suggest_float('random_strength', 0, 1)
            }
        else:
            params = {
                'C': trial.suggest_float('C', 0.0001, 1000, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', None])
            }

        # Inner split - train vs. val
        inner_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=98)
        inner_auc_scores = []
        best_iterations = []

        for train_idx, val_idx in inner_split.split(X_train_val, y_train_val):
            X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
            y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

            fit_idx, eval_idx = train_test_split(np.arange(len(train_idx)), test_size=0.1, stratify=y_train, random_state=98)
            X_fit, X_eval = X_train[fit_idx], X_train[eval_idx]
            y_fit, y_eval = y_train[fit_idx], y_train[eval_idx]

            # Scale features using only training data
            if model_name in ['mlp', 'cnn', 'lr']:
                scaler = StandardScaler()
                X_fit_scaled = scaler.fit_transform(X_fit)
                X_eval_scaled = scaler.transform(X_eval)
                X_val_scaled = scaler.transform(X_val)

            # Convert to tensors or GPU if needed
            if model_name in ['mlp','cnn']:
                X_fit_conv = torch.tensor(X_fit_scaled, dtype=torch.float32)
                X_eval_conv = torch.tensor(X_eval_scaled, dtype=torch.float32)
                X_val_conv = torch.tensor(X_val_scaled, dtype=torch.float32)
                y_fit_conv = torch.tensor(y_fit, dtype=torch.float32).unsqueeze(1)
                y_eval_conv = torch.tensor(y_eval, dtype=torch.float32).unsqueeze(1)

                fit_set = TensorDataset(X_fit_conv, y_fit_conv)
                eval_set = TensorDataset(X_eval_conv, y_eval_conv)

                fit_loader = DataLoader(fit_set, batch_size=10, shuffle=True, generator=generator)
                eval_loader = DataLoader(eval_set, batch_size=10, shuffle=False)

            elif model_name in ['rf', 'xgb']:
                X_fit_conv = cp.asarray(X_fit)
                X_eval_conv = cp.asarray(X_eval)
                X_val_conv = cp.asarray(X_val)
                y_fit_conv = cp.asarray(y_fit)
                y_eval_conv = cp.asarray(y_eval)

            elif model_name == 'lr':
                X_fit_conv = cp.asarray(X_fit_scaled)
                X_val_conv = cp.asarray(X_val_scaled)
                y_fit_conv = cp.asarray(y_fit)

            # Call model codes
            if model_name in ['mlp','cnn']:
                input_dim = X_fit_conv.shape[1]
                model_kwargs = {
                    key: value
                    for key, value in params.items()
                    if key not in ['lr','weight_decay']
                }
                clf = models[model_name](input_dim, **model_kwargs)

            else:
                clf = models[model_name](**params)

            # Train models with validation where applicable
            if model_name in ['mlp','cnn']:
                criterion = nn.BCEWithLogitsLoss()
                optimizer = optim.Adam(clf.parameters(), lr=params['lr'], betas=(0.9, 0.999), eps=1e-8, weight_decay=params['weight_decay'])
                n_epochs = 200
                patience = 10
                #print('Training started!')
                clf = train_model(
                    clf,
                    fit_loader,
                    eval_loader,
                    criterion,
                    optimizer,
                    n_epochs=n_epochs,
                    patience=patience,
                    device=device
                )

            elif model_name == 'xgb':
                clf.fit(X_fit_conv, y_fit_conv, eval_set=[(X_eval_conv, y_eval_conv)], verbose=False)
                
            elif model_name == 'lgb':
                clf.fit(X_fit_conv, y_fit_conv, eval_set=[(X_eval_conv, y_eval_conv)], eval_metric='auc')

            elif model_name == 'cb':
                clf.fit(X_fit, y_fit, eval_set=[(X_eval, y_eval)], early_stopping_rounds=10, metric_period=1, verbose=False)
                
            else:  # lr
                clf.fit(X_fit_conv, y_fit_conv)

            # Get predictions for validation set
            if model_name in ['mlp','cnn']:
                pred_val = get_predictions(clf, X_val_conv, device)

            elif model_name in ['lgb', 'cb']:
                pred_val = clf.predict_proba(X_val)[:, 1]
            
            else:  # rf, xgb, lr
                pred_val = clf.predict_proba(X_val_conv)[:, 1]

            # Convert to numpy if needed
            pred_val = (
                cp.asnumpy(pred_val)
                if isinstance(pred_val, cp.ndarray)
                else pred_val
            )

            # Calculate final metrics
            inner_auc_scores.append(roc_auc_score(y_val, pred_val))

            # Best iteration
            if model_name in ['xgb', 'lgb', 'cb']:
                
                if model_name in ['lgb','cb']:
                    best_iter = clf.best_iteration_

                elif model_name == 'xgb':
                    best_iter = clf.best_iteration

                best_iterations.append(best_iter)
                trial.set_user_attr('avg_best_iter', np.mean(best_iterations))

        return np.mean(inner_auc_scores)
    
    return objective



def tune_train_evaluate_model(embeddings_combined, demographics, labels, model_name, method):

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=98)
    outer_auc_scores = []
    best_trials_info = []
    all_y_test = []
    all_pred_test = []

    # Outer split - train/val vs. test
    for fold, (train_val_idx, test_idx) in enumerate(
        skf.split(embeddings_combined, labels)
    ):
        X_train_val, X_test, y_train_val, y_test = process_embeddings(
            embeddings_combined, demographics, labels, train_val_idx, test_idx, method
            )

        # Hyperparameter tuning using Optuna
        objective = make_objective(X_train_val, y_train_val, model_name)

        if model_name == 'mlp':
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=98), pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=n_epochs, reduction_factor=3))
            study.optimize(objective, n_trials=30)  # 150 in total
        elif model_name == 'cnn':
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=98), pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=n_epochs, reduction_factor=3))
            study.optimize(objective, n_trials=20)  # 100 in total
        else:
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=98), pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0))
            study.optimize(objective, n_trials=50)  # 250 in total

        # Save trial information as a csv file
        trials_info = []
        for each_trial in study.trials:

            if model_name in ['xgb', 'lgb', 'cb']:
                trial_info = {
                    'trial_number': each_trial.number,
                    'mean_auc': each_trial.value,
                    'avg_best_iter': each_trial.user_attrs.get('avg_best_iter', None)
                }
            else:
                trial_info = {
                    'trial_number': each_trial.number,
                    'mean_auc': each_trial.value
                }

            for key, value in each_trial.params.items():
                trial_info[key] = value
            
            trials_info.append(trial_info)

        trials_df = pd.DataFrame(trials_info)
        trials_df.to_csv(f'optuna_results/{model_name}/fold_{fold}_trial_info.csv', index=False)

        best_params = study.best_trial.params
        best_auc = study.best_trial.value
        best_trial_info = {
            'fold': fold,
            'auc_val': best_auc
        }

        for key, value in best_params.items():
            best_trial_info[key] = value
        
        if model_name in ['xgb', 'lgb', 'cb']:
            if model_name == 'cb':
                best_best_iter = int(study.best_trial.user_attrs.get('avg_best_iter', best_params.get('iterations', None)))
                best_params['iterations'] = best_best_iter
            else:
                best_best_iter = int(study.best_trial.user_attrs.get('avg_best_iter', best_params.get('n_estimators', None)))
                best_params['n_estimators'] = best_best_iter

            best_trial_info['best_iterations'] = best_best_iter

        # Train a final model with the best hyperparameters
        ffit_idx, feval_idx = train_test_split(np.arange(len(y_train_val)), test_size=0.1, stratify=y_train_val, random_state=98)
        X_ffit, X_feval = X_train_val[ffit_idx], X_train_val[feval_idx]
        y_ffit, y_feval = y_train_val[ffit_idx], y_train_val[feval_idx]

        if model_name in ['mlp', 'cnn', 'lr']:
            scaler = StandardScaler()
            X_ffit_scaled = scaler.fit_transform(X_ffit)
            X_feval_scaled = scaler.fit_transform(X_feval)
            X_test_scaled = scaler.transform(X_test)

        # Convert to tensors or GPU if needed
        if model_name in ['mlp','cnn']:
            X_ffit_conv = torch.tensor(X_ffit_scaled, dtype=torch.float32)
            X_feval_conv = torch.tensor(X_feval_scaled, dtype=torch.float32)
            X_test_conv = torch.tensor(X_test_scaled, dtype=torch.float32)
            y_ffit_conv = torch.tensor(y_ffit, dtype=torch.float32).unsqueeze(1)
            y_feval_conv = torch.tensor(y_feval, dtype=torch.float32).unsqueeze(1)

            ffit_set = TensorDataset(X_ffit_conv, y_ffit_conv)
            feval_set = TensorDataset(X_feval_conv, y_feval_conv)

            ffit_loader = DataLoader(ffit_set, batch_size=10, shuffle=True, generator=generator)
            feval_loader = DataLoader(feval_set, batch_size=10, shuffle=False)

        elif model_name in ['rf', 'xgb']:
            X_ffit_conv = cp.asarray(X_ffit)
            X_test_conv = cp.asarray(X_test)
            y_ffit_conv = cp.asarray(y_ffit)

        elif model_name == 'lr':
            X_ffit_conv = cp.asarray(X_ffit_scaled)
            X_test_conv = cp.asarray(X_test_scaled)
            y_ffit_conv = cp.asarray(y_ffit)

        # Call model codes
        if model_name in ['mlp','cnn']:
            input_dim = X_ffit_conv.shape[1]
            best_model_kwargs = {
                key: value
                for key, value in best_params.items()
                if key not in ['lr','weight_decay']
            }
            clf = final_models[model_name](input_dim, **best_model_kwargs)

        else:
            clf = final_models[model_name](**best_params)

        # Train models with validation where applicable
        if model_name in ['mlp','cnn']:
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(clf.parameters(), lr=best_params['lr'], betas=(0.9, 0.999), eps=1e-8, weight_decay=best_params['weight_decay'])
            n_epochs = 200
            patience = 10
            #print('Training started!')
            clf = train_model(
                clf,
                ffit_loader,
                feval_loader,
                criterion,
                optimizer,
                n_epochs=n_epochs,
                patience=patience,
                device=device
            )

        elif model_name == 'xgb':
            clf.fit(X_ffit_conv, y_ffit_conv, verbose=False)
            
        elif model_name == 'lgb':
            clf.fit(X_ffit_conv, y_ffit_conv)

        elif model_name == 'cb':
            clf.fit(X_ffit, y_ffit, verbose=False)
            
        else:  # lr
            clf.fit(X_ffit_conv, y_ffit_conv)

        # Get predictions for validation set
        if model_name in ['mlp','cnn']:
            pred_test = get_predictions(clf, X_test_conv, device)

        elif model_name in ['lgb', 'cb']:
            pred_test = clf.predict_proba(X_test)[:, 1]
        
        else:  # rf, xgb, lr
            pred_test = clf.predict_proba(X_test_conv)[:, 1]

        # Convert to numpy if needed
        pred_test = (
            cp.asnumpy(pred_test)
            if isinstance(pred_test, cp.ndarray)
            else pred_test
        )

        # Compute AUC
        outer_auc = roc_auc_score(y_test, pred_test)
        outer_auc_scores.append(outer_auc)
        best_trial_info['auc_test'] = outer_auc
        best_trials_info.append(best_trial_info)

        # Save y_test and pred_test for Delong's test
        all_y_test.append(np.array(y_test).flatten())
        all_pred_test.append(np.array(pred_test).flatten())

    all_y_test = np.concatenate(all_y_test)
    all_pred_test = np.concatenate(all_pred_test)
    
    # Save the information of best trial of each fold
    best_trials_df = pd.DataFrame(best_trials_info)
    best_trials_df.to_csv(f'optuna_results/{model_name}/best_trials_info.csv', index=False)

    return {"auc_scores": outer_auc_scores, "y_true": all_y_test, "y_pred": all_pred_test}