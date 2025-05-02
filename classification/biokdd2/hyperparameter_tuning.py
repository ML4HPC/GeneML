import random
import numpy as np
import pandas as pd
import cupy as cp
import optuna

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score
from collections import defaultdict

import cuml
from cuml.ensemble import RandomForestClassifier as cuRFC
from cuml import LogisticRegression as cuLR
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

import torch
import torch.nn as nn
import torch.optim as optim

from biokdd2.neural_nets import MLPClassifier, CNNClassifier, train_model, get_predictions
from biokdd2.load_and_processing import process_embeddings
from biokdd2.preprocessing import scale_data, create_dataloader
from biokdd2.metrics import tune_threshold

seed = 98
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gb_models = {
        "xgb": lambda **hp: xgb.XGBClassifier(**hp, random_state=98, early_stopping_rounds=10, eval_metric="auc", device="cuda"),
        "lgb": lambda **hp: lgb.LGBMClassifier(**hp, random_state=98, early_stopping_round=10, device='cuda', verbose=-1),
        "cb": lambda **hp: cb.CatBoostClassifier(**hp, random_seed=98, task_type='GPU', eval_metric='AUC')
}

models = {
        "rf": lambda **hp: cuRFC(**hp, random_state=98, n_streams=1),
        "mlp": lambda input_dim, **hp: MLPClassifier(input_dim=input_dim, **hp).to(device),
        'cnn': lambda input_dim, **hp: CNNClassifier(input_dim=input_dim, **hp, random_seed=98).to(device),
        "lr": lambda **hp: cuLR(**hp, max_iter=100000),
}
models.update(gb_models)

############################
## Quick tune (GB models) ##
############################

def quick_tune_gb(X, y, n_trials=10, csv_path='quick_tuning_trials_info.csv'):

    gb_quick_params = {
        'xgb': lambda trial: {
            'n_estimators': trial.suggest_categorical('n_estimators', [100, 300, 500]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-2, log=True),
            'max_depth': trial.suggest_categorical('max_depth', [4, 8]),
            'subsample': trial.suggest_categorical('subsample', [0.6, 0.8]),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.6, 0.8])
        },
        'lgb': lambda trial: {
            'num_leaves': trial.suggest_categorical('num_leaves', [31, 63]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-2, log=True),
            'n_estimators': trial.suggest_categorical('n_estimators', [200, 500, 1000]),
            'subsample': trial.suggest_categorical('subsample', [0.7, 0.9]),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.7, 0.9])
        },
        'cb': lambda trial: {
            'iterations': trial.suggest_categorical('iterations', [200, 500, 1000]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-2, log=True),
            'depth': trial.suggest_categorical('depth', [4, 8]),
            'l2_leaf_reg': trial.suggest_categorical('l2_leaf_reg', [1, 5])
        }
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=98)
    qt_results = {}

    for name, search in gb_quick_params.items():
        
        def obj(trial, name=name, search=search):
            params = search(trial)
            model = gb_models[name](**params)

            aucs = []
            for tidx, vidx in cv.split(X, y):
                X_tr, X_v = X[tidx], X[vidx]
                y_tr, y_v = y[tidx], y[vidx]

                eidx, pidx = train_test_split(np.arange(len(y_v)), test_size=0.5, stratify=y_v, random_state=98)
                X_e, X_p = X_v[eidx],X_v[pidx]
                y_e, y_p = y_v[eidx],y_v[pidx]

                if name == 'xgb':
                    X_tr_conv = cp.asarray(X_tr)
                    X_e_conv = cp.asarray(X_e)
                    X_p_conv = cp.asarray(X_p)
                    y_tr_conv = cp.asarray(y_tr)
                    y_e_conv = cp.asarray(y_e)

                    model.fit(X_tr_conv, y_tr_conv, eval_set=[(X_e_conv, y_e_conv)], verbose=False)
                    torch.cuda.empty_cache()
                    preds = model.predict_proba(X_p_conv)[:,1]

                elif name == 'lgb':
                    model.fit(X_tr, y_tr, eval_set=[(X_e, y_e)], eval_metric='auc')
                    torch.cuda.empty_cache()
                    preds = model.predict_proba(X_p)[:,1]

                else:
                    model.fit(X_tr, y_tr, eval_set=[(X_e, y_e)], early_stopping_rounds=10, metric_period=1, verbose=False)
                    torch.cuda.empty_cache()
                    preds = model.predict_proba(X_p)[:,1]
            
                aucs.append(roc_auc_score(y_p, preds))
            
            return float(np.mean(aucs))
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=98), pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1))
        study.optimize(obj, n_trials=n_trials)
        qt_results[name] = {
            'best_score': study.best_value,
            'best_params': study.best_params
        }

    qt_df = pd.DataFrame([
        {'model': name, 'best_score': info['best_score'], **info['best_params']} for name, info in qt_results.items()
    ])
    qt_df.to_csv(csv_path, index=False)

    return qt_results

def get_best_gb(embeddings_combined, demographics, labels, method):

    qt_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=98)
    qt_aucs = defaultdict(list)

    for qt_fold, (qt_tv_idx, qt_test_idx) in enumerate(qt_skf.split(embeddings_combined, labels), 1):
        
        qt_X_tv, _, qt_y_tv, _ = process_embeddings(
            embeddings_combined, demographics, labels, qt_tv_idx, qt_test_idx, method
            )
        
        out_path = f"/pscratch/sd/h/hazely/NESAP/caduceus/classification/2nd_mdd/py/optuna_results/qt_gb/{method}_fold{qt_fold}_best_info.csv"
        qt_results = quick_tune_gb(qt_X_tv, qt_y_tv, n_trials=10, csv_path=out_path)

        for gb_name, qt_info in qt_results.items():
            qt_aucs[gb_name].append(qt_info['best_score'])

    mean_auc = {m: np.mean(scores) for m, scores in qt_aucs.items()}
    best_gb = max(mean_auc, key=mean_auc.get)

    return best_gb

########################
## Objective function ##
########################

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
                'gamma': trial.suggest_float('gamma', 0.0, 0.5),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0)
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
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
            }
        elif model_name == 'cb':
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'random_strength': trial.suggest_float('random_strength', 0.0, 1.0)
            }
        else:
            penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', None])
            params = {
                'C': trial.suggest_float('C', 0.0001, 1000, log=True),
                'penalty': penalty
            }
            if penalty == 'elasticnet':
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)

        # Inner split - train vs. val
        inner_split = StratifiedKFold(n_splits=3, shuffle=True, random_state=98)
        inner_auc_scores = []

        for train_idx, val_idx in inner_split.split(X_train_val, y_train_val):
            X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
            y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

            # Extra split for early stopping
            tr_idx, ev_idx = train_test_split(np.arange(len(y_train)), test_size=0.1, stratify=y_train, random_state=98)
            X_tr, X_ev = X_train[tr_idx], X_train[ev_idx]
            y_tr, y_ev = y_train[tr_idx], y_train[ev_idx]

            # Scale features using only training data
            if model_name in ['mlp', 'cnn', 'lr']:
                X_tr_scaled, X_ev_scaled, X_val_scaled = scale_data(X_tr, X_ev, X_val)

            # Convert to tensors or GPU if needed
            if model_name in ['mlp','cnn']:
                tr_loader = create_dataloader(X_tr_scaled, y_tr)
                ev_loader = create_dataloader(X_ev_scaled, y_ev, shuffle=False)

                X_tr_conv = torch.tensor(X_tr_scaled, dtype=torch.float32)
                X_val_conv = torch.tensor(X_val_scaled, dtype=torch.float32)

            elif model_name in ['rf', 'xgb']:
                X_tr_conv = cp.asarray(X_tr)
                X_ev_conv = cp.asarray(X_ev)
                X_val_conv = cp.asarray(X_val)
                y_tr_conv = cp.asarray(y_tr)
                y_ev_conv = cp.asarray(y_ev)

            elif model_name == 'lr':
                X_tr_conv = cp.asarray(X_tr_scaled)
                X_val_conv = cp.asarray(X_val_scaled)
                y_tr_conv = cp.asarray(y_tr)

            # Call model codes
            if model_name in ['mlp','cnn']:
                input_dim = X_tr_conv.shape[1]
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
                clf = train_model(
                    clf,
                    tr_loader,
                    ev_loader,
                    criterion,
                    optimizer,
                    n_epochs=200,
                    patience=10,
                    device=device
                )
                torch.cuda.empty_cache()

            elif model_name == 'xgb':
                clf.fit(X_tr_conv, y_tr_conv, eval_set=[(X_ev_conv, y_ev_conv)], verbose=False)
                torch.cuda.empty_cache()
                
            elif model_name == 'lgb':
                clf.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], eval_metric='auc')
                torch.cuda.empty_cache()

            elif model_name == 'cb':
                clf.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], early_stopping_rounds=10, metric_period=1, verbose=False)
                torch.cuda.empty_cache()
                
            else:  # lr, rf
                clf.fit(X_tr_conv, y_tr_conv)
                torch.cuda.empty_cache()

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

        return np.mean(inner_auc_scores)
    
    return objective

##################################
## Tuning, training, evaluation ##
##################################

def tune_train_evaluate_model(embeddings_combined, demographics, labels, model_name, method):

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=98)
    outer_auc_scores, auprc_scores, precision_scores, recall_scores, f1_scores, best_thresholds, best_trials_info, all_y_test, all_pred_test = [],[],[],[],[],[],[],[],[]

    # Outer split - train/val vs. test
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(embeddings_combined, labels), 1):

        X_train_val, X_test, y_train_val, y_test = process_embeddings(
            embeddings_combined, demographics, labels, train_val_idx, test_idx, method
            )

        # Hyperparameter tuning using Optuna
        objective = make_objective(X_train_val, y_train_val, model_name)

        if model_name == 'mlp':
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=98), pruner=optuna.pruners.HyperbandPruner(min_resource=5, max_resource=200, reduction_factor=2))
            n_trials = 15  # 150 in total
        elif model_name == 'cnn':
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=98), pruner=optuna.pruners.HyperbandPruner(min_resource=30, max_resource=200, reduction_factor=2))
            n_trials = 10  # 100 in total
        else:
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=98), pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0))
            n_trials = 30  # 300 in total

        study.optimize(objective, n_trials=n_trials)

        # Save trial information as a csv file
        trials_info = []
        for t in study.trials:

            info = {'trial_number': t.number, 'mean_auc': t.value}
            info.update(t.params)
            trials_info.append(info)

        trials_df = pd.DataFrame(trials_info)
        trials_df.to_csv(f'/pscratch/sd/h/hazely/NESAP/caduceus/classification/2nd_mdd/py/optuna_results/{model_name}/{method}/2nd/fold_{fold}_trial_info.csv', index=False)

        # Retrieve the best params
        best_params = study.best_trial.params.copy()
        best_auc = study.best_trial.value

        best_trial_info = {
            'fold': fold,
            'auc_val': best_auc
        }

        for key, value in best_params.items():
            best_trial_info[key] = value

        # Train a final model with the best hyperparameters
        otr_idx, oev_idx = train_test_split(np.arange(len(y_train_val)), test_size=0.1, stratify=y_train_val, random_state=98)
        X_otr, X_oev = X_train_val[otr_idx], X_train_val[oev_idx]
        y_otr, y_oev = y_train_val[otr_idx], y_train_val[oev_idx]

        if model_name in ['mlp', 'cnn', 'lr']:
            X_otr_scaled, X_oev_scaled, X_test_scaled = scale_data(X_otr, X_oev, X_test)

        # Convert to tensors or GPU if needed
        if model_name in ['mlp','cnn']:
            otr_loader = create_dataloader(X_otr_scaled, y_otr)
            oev_loader = create_dataloader(X_oev_scaled, y_oev, shuffle=False)

            X_otr_conv = torch.tensor(X_otr_scaled, dtype=torch.float32)
            X_oev_conv = torch.tensor(X_oev_scaled, dtype=torch.float32)
            X_test_conv = torch.tensor(X_test_scaled, dtype=torch.float32)

        elif model_name in ['rf', 'xgb']:
            X_otr_conv = cp.asarray(X_otr)
            X_oev_conv = cp.asarray(X_oev)
            X_test_conv = cp.asarray(X_test)
            y_otr_conv = cp.asarray(y_otr)
            y_oev_conv = cp.asarray(y_oev)

        elif model_name == 'lr':
            X_otr_conv = cp.asarray(X_otr_scaled)
            X_oev_conv = cp.asarray(X_oev_scaled)
            X_test_conv = cp.asarray(X_test_scaled)
            y_otr_conv = cp.asarray(y_otr)

        # Call model codes
        if model_name in ['mlp','cnn']:
            input_dim = X_otr_conv.shape[1]
            best_model_kwargs = {
                key: value
                for key, value in best_params.items()
                if key not in ['lr','weight_decay']
            }
            clf = models[model_name](input_dim, **best_model_kwargs)

        else:
            clf = models[model_name](**best_params)

        # Train models with early stopping where applicable
        if model_name in ['mlp','cnn']:
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(clf.parameters(), lr=best_params['lr'], betas=(0.9, 0.999), eps=1e-8, weight_decay=best_params['weight_decay'])
            clf = train_model(
                clf,
                otr_loader,
                oev_loader,
                criterion,
                optimizer,
                n_epochs=200,
                patience=10,
                device=device
            )
            torch.cuda.empty_cache()

        elif model_name == 'xgb':
            clf.fit(X_otr_conv, y_otr_conv, eval_set=[(X_oev_conv, y_oev_conv)], verbose=False)
            torch.cuda.empty_cache()
            
        elif model_name == 'lgb':
            clf.fit(X_otr, y_otr, eval_set=[(X_oev, y_oev)], eval_metric='auc')
            torch.cuda.empty_cache()

        elif model_name == 'cb':
            clf.fit(X_otr, y_otr, eval_set=[(X_oev, y_oev)], early_stopping_rounds=10, metric_period=1, verbose=False)
            torch.cuda.empty_cache()
            
        else:  # lr, rf
            clf.fit(X_otr_conv, y_otr_conv)
            torch.cuda.empty_cache()

        # Get predictions for evaluation (validation) set for threshold tuning
        if model_name in ['mlp','cnn']:
            pred_oev = get_predictions(clf, X_oev_conv, device)

        elif model_name in ['lgb', 'cb']:
            pred_oev = clf.predict_proba(X_oev)[:, 1]
        
        else:  # rf, xgb, lr
            pred_oev = clf.predict_proba(X_oev_conv)[:, 1]

        # Convert to numpy if needed
        pred_oev = (
            cp.asnumpy(pred_oev)
            if isinstance(pred_oev, cp.ndarray)
            else pred_oev
        )
        best_threshold = tune_threshold(y_oev, pred_oev)
        best_trial_info['threshold'] = best_threshold
        best_thresholds.append(best_threshold)

        # Get predictions for test set for performance evaluation
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
        pred_value_test = (pred_test >= best_threshold).astype(int)

        # Compute metrics
        outer_auc = roc_auc_score(y_test, pred_test)
        auprc = average_precision_score(y_test, pred_test)
        precision = precision_score(y_test, pred_value_test)
        recall = recall_score(y_test, pred_value_test)
        f1 = f1_score(y_test, pred_value_test)

        outer_auc_scores.append(outer_auc)
        auprc_scores.append(auprc)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        best_trial_info.update({
            'auc_test': outer_auc,
            'auprc': auprc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        best_trials_info.append(best_trial_info)

        # Save y_test and pred_test for Delong's test
        y_test = (
            y_test.values
            if hasattr(y_test, "values")
            else np.array(y_test)
        )

        all_y_test.append(y_test)
        all_pred_test.append(pred_test)

    all_y_test = np.concatenate(all_y_test)
    all_pred_test = np.concatenate(all_pred_test)
    
    # Save the information of best trial of each fold
    best_trials_df = pd.DataFrame(best_trials_info)
    best_trials_df.to_csv(f'optuna_results/{model_name}/{method}/2nd/best_trials_info.csv', index=False)

    return {
        "auc_scores": outer_auc_scores,
        "precision_scores": precision_scores,
        "recall_scores": recall_scores,
        "f1_scores": f1_scores,
        "best_thresholds": best_thresholds,
        "auprc_scores": auprc_scores,
        "y_true": all_y_test,
        "y_pred": all_pred_test
    }
