import numpy as np
import os
import re
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import cuml
from cuml.ensemble import RandomForestClassifier as cuRFC
from sklearn.neural_network import MLPClassifier
from cuml import LogisticRegression as cuLR
import xgboost as xgb
import cupy as cp

# Custom sorting function to extract numerical part of file name
def numerical_sort(file_name):
    match = re.search(r'embeddings_(\d+).npy', file_name) # Find the first number in the file name
    return int(match.group(1)) # Sort numerically

age = np.load('../age.npy')
sex = np.load('../sex.npy')
labels = np.load('../labels.npy')

gene_list = ['CRHR1','ESR1','ESR2','PCLO','FHIT','CACNA1C','DRD2','GRM7','EHD3','BICC1','PLOD1','LINC00687','CSMD1','LHPP','APC','ARHGAP8','LOC100996549','CNTNAP2','CRY1','COMT','FKBP5','HTR2A','BDNF','SLC6A4','ACE','SLC6A2','KCNK2','NR3C1','MTHFR','TPH1','TPH2','SOD2','CNR1','TNF','HTR1A','ABCB1','GNB3','GSK3B']

# 1. gene embeddings & covariates
embeddings_list = []
for gene in gene_list:

    # Directory containing the saved .npy files
    directory = gene
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith('embeddings')]
    sorted_file_paths = sorted(file_paths, key=numerical_sort)
    
    # Load and concatenate all embeddings
    all_embeddings = []
    for file_path in sorted_file_paths:
        chunk = np.load(file_path)
        all_embeddings.append(chunk)
        
    # Combine all chunks into a single NumPy array
    # & Append embeddings of each gene to a single list
    all_embeddings_cc = np.concatenate(all_embeddings, axis=0)
    embeddings_list.append(all_embeddings_cc)
   
# Convert the list to a numpy array with shape of (22000, 38, 256)
embeddings_combined = np.stack(embeddings_list, axis=1)

# Approach 1: PCA for gene embeddings
n_samples = embeddings_combined.shape[0]
embeddings_reshaped = embeddings_combined.reshape(n_samples, -1) # (22000, 38*256)
pca = PCA(n_components=256, random_state=98)
gene_embeddings_transformed = pca.fit_transform(embeddings_reshaped).astype(np.float32)

# Approach 2: max pooling for gene embeddings
gene_embeddings_transformed = np.max(embeddings_combined, axis=1)

# Approach 3: mean pooling for gene embeddings
gene_embeddings_transformed = np.mean(embeddings_combined, axis=1)

# Convert the shape of the covariate arrays from (22000,) -> (22000, 1)
age_reshaped = age.reshape(-1, 1).astype(np.float32)
sex_reshaped = sex.reshape(-1, 1).astype(np.float32)
demographics = np.hstack([age_reshaped, sex_reshaped])  # (22000, 2)

# 2. Predictive modeling

# Modeling (using four ML models) & Manual cross-validation
models = {
    'rf': lambda: cuRFC(random_state=98, n_streams=1),
    'mlp': lambda: MLPClassifier(random_state=98, max_iter=500),
    'xgb': lambda: xgb.XGBClassifier(n_estimators=100, random_state=98, eval_metric='error', device='cuda'),
    'lr': lambda: cuLR(tol=0.001)
}

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=98)
auc_gene_results = {}
auc_demo_results = {}
auc_comb_results = {}

for model_name, model_code in models.items():
    print(f'Model: {model_name.upper()}')
    auc_gene_scores = []
    auc_demo_scores = []
    auc_comb_scores = []

    for train_idx, test_idx in skf.split(gene_embeddings_transformed, labels):
    
        # Split features and labels
        X_gene_train, X_gene_test = gene_embeddings_transformed[train_idx], gene_embeddings_transformed[test_idx]
        X_demo_train, X_demo_test = demographics[train_idx], demographics[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        # Scale the data
        scaler_gene = StandardScaler()
        scaler_demo = StandardScaler()
        
        X_gene_train_scaled = scaler_gene.fit_transform(X_gene_train)
        X_gene_test_scaled = scaler_gene.transform(X_gene_test)
        
        X_demo_train_scaled = scaler_demo.fit_transform(X_demo_train)
        X_demo_test_scaled = scaler_demo.transform(X_demo_test)
        
        # Convert to GPU arrays
        if model_name == 'mlp':
            X_gene_train_conv = X_gene_train_scaled
            X_gene_test_conv = X_gene_test_scaled
            X_demo_train_conv = X_demo_train_scaled
            X_demo_test_conv = X_demo_test_scaled
        else:
            X_gene_train_conv = cp.asarray(X_gene_train_scaled)
            X_gene_test_conv = cp.asarray(X_gene_test_scaled)
            X_demo_train_conv = cp.asarray(X_demo_train_scaled)
            X_demo_test_conv = cp.asarray(X_demo_test_scaled)
    
        # Initialize & train separate classifiers
        model_gene = model_code()
        model_demo = model_code()
        model_gene.fit(X_gene_train_conv, y_train)
        model_demo.fit(X_demo_train_conv, y_train)
    
        # Get predictive probabilities
        gene_pred = model_gene.predict_proba(X_gene_test_conv)[:, 1]
        demo_pred = model_demo.predict_proba(X_demo_test_conv)[:, 1]
        
        # Convert to numpy arrays
        gene_pred = cp.asnumpy(gene_pred) if isinstance(gene_pred, cp.ndarray) else gene_pred
        demo_pred = cp.asnumpy(demo_pred) if isinstance(demo_pred, cp.ndarray) else demo_pred
        
        comb_pred = (gene_pred + demo_pred) / 2  # combine predictions
                
        # AUC calculation
        auc_gene = roc_auc_score(y_test, gene_pred)
        auc_demo = roc_auc_score(y_test, demo_pred)
        auc_comb = roc_auc_score(y_test, comb_pred)
        auc_gene_scores.append(auc_gene)
        auc_demo_scores.append(auc_demo)
        auc_comb_scores.append(auc_comb)
        
    auc_gene_results[model_name] = auc_gene_scores
    auc_demo_results[model_name] = auc_demo_scores
    auc_comb_results[model_name] = auc_comb_scores
    print(f"Gene-only AUC = {np.mean(auc_gene_scores):.2f}")
    print(f"Demographics-only AUC = {np.mean(auc_demo_scores):.2f}")
    print(f"Combined AUC = {np.mean(auc_comb_scores):.2f}")
    print()
