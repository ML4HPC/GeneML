import random
import pickle
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection

import torch

from biokdd.load_and_processing import load_gene_embeddings
from biokdd.hyperparameter_tuning import tune_train_evaluate_model
from biokdd.delong_test import delong_test

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
    embedding_methods = ["pca", "max_pool", "mean_pool", "concat"]
    model_names = ['rf','mlp','cnn','xgb','lgb','cb','lr']
    # final_results = {}

    # Modeling (using four ML models) & Manual cross-validation
    auc_records = []
    delong_records = []

    for method in embedding_methods:
        print(f"\nProcessing embeddings with method: {method}")
        method_results = {}
        
        # Train and evaluate models
        for model_name in model_names:
            print(f"\n[Training {model_name.upper()}]")

            results = tune_train_evaluate_model(
                embeddings_combined,
                demographics,
                labels,
                model_name,
                method
            )
            torch.cuda.empty_cache()

            method_results[model_name] = results
            model_aucs = results["auc_scores"]
            mean_auc = np.mean(model_aucs)
            std_auc = np.std(model_aucs)
            auc_records.append({'method': method, 'model': model_name, 'mean_auc': mean_auc, 'std_auc': std_auc})

            inter_auc_df = pd.DataFrame(auc_records)
            inter_auc_df.to_csv(f'modeling_results/{method}_{model_name}_auc_mean_std.csv', index=False)
            with open(f'modeling_results/results/{method}_{model_name}_auc_ytrue_ypred.pkl', 'wb') as f:
                pickle.dump(results, f)
        
        print("Tuning, training, and evaluation completed. Now proceeding to Delong's test...")
        # final_results[method] = method_results
        
        # Pairwise DeLong's tests between models for each method
        comparison_results = delong_test(method_results)

        print("\nDelong's test completed. Now proceeding to FDR correction...")

        # FDR correction
        if comparison_results:
            raw_p = [x[5] for x in comparison_results]
            h0_rejected, p_corrected = fdrcorrection(raw_p, alpha=0.05)
            
            for idx, (model1, model2, auc_model1, auc_model2, z, p) in enumerate(comparison_results):
                abs_z = abs(z)
                neglog_p = -np.log10(p+1e-300)
                neglog_padj = -np.log10(p_corrected[idx]+1e-300)
                test_result = ("AUC values are significantly different." if h0_rejected[idx] == True else "AUC values are NOT significantly different.")
                delong_records.append({'method': method, 'model1': model1, 'model2': model2, 'auc1': auc_model1, 'auc2': auc_model2,
                                       'abs_z': abs_z, 'neg_log_p': neglog_p, 'neg_log_p_adj': neglog_padj, 'test_result': test_result})
                
        else:
            print("\nNo pairwise comparisons available.")

        print()

    # Create and save dataframes with the information
    auc_df = pd.DataFrame(auc_records)
    delong_df = pd.DataFrame(delong_records)
    auc_df.to_csv('modeling_results/auc_mean_std.csv', index=False)
    delong_df.to_csv('modeling_results/delong_fdr.csv', index=False)

    print('Successfully saved result files!')