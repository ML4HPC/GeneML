import random
import pickle
import numpy as np
import pandas as pd

import torch

from biokdd2.load_and_processing import load_gene_embeddings
from biokdd2.hyperparameter_tuning import tune_train_evaluate_model, get_best_gb

seed = 98
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


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
    embedding_methods = ["pca","max_pool","mean_pool"]  # concat
    model_names = ['rf','xgb','lgb','cb','lr','mlp','cnn']

    # Modeling (using four ML models) & Manual cross-validation
    metric_records = []

    for method in embedding_methods:
        print(f"\nProcessing embeddings with method: {method}")
        method_results = {}

        # Quick tuning (GB models) - highest mean AUC
        print("\nSearching for the best GB model...")
        best_gb = get_best_gb(embeddings_combined, demographics, labels, method)
        print(f"Best GB model: {best_gb.upper()}")
        
        # Train and evaluate models
        for model_name in model_names:
            
            if model_name not in ['xgb','lgb','cb'] or model_name == best_gb:
                print(f"\n[Training {model_name.upper()}]")

                results = tune_train_evaluate_model(
                    embeddings_combined,
                    demographics,
                    labels,
                    model_name,
                    method
                )
                torch.cuda.empty_cache()

            else:
                results = {
                    "auc_scores": [],
                    "precision_scores": [],
                    "recall_scores": [],
                    "f1_scores": [],
                    "best_thresholds": [],
                    "auprc_scores": [],
                    "y_true": np.array([]),
                    "y_pred": np.array([])
                    }

            method_results[model_name] = results

            # metrics - mean & std
            for key, value in results.items():

                if key in ['y_true', 'y_pred']:
                    continue

                mean = np.mean(value)
                std = np.std(value)

                if key.endswith('scores'):
                    metric = key.replace('_scores','')
                else:
                    metric = 'thresholds'

                metric_records.append({'method': method, 'model': model_name, f'mean_{metric}': mean, f'std_{metric}': std})
                metric_df = pd.DataFrame(metric_records)
                metric_df.to_csv(f'modeling_results/2nd/{method}_{model_name}_{metric}_mean_std.csv', index=False)

            with open(f'modeling_results/2nd/results/{method}_{model_name}_metrics_ytrue_ypred.pkl', 'wb') as f:
                pickle.dump(results, f)
