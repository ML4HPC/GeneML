import numpy as np
import os
import re
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import cuml
from cuml.ensemble import RandomForestClassifier as cuRFC
from cuml.svm import SVC as cuSVC
from cuml import LogisticRegression as cuLR
import xgboost as xgb
import cupy as cp

# Custom sorting function to extract numerical part of file name
def numerical_sort(file_name):
    if 'embeddings' in file_name:
        match = re.search(r'embeddings_(\d+).npy', file_name) # Find the first number in the file name
    else:
        match = re.search(r'labels_(\d+).npy', file_name)
    return int(match.group(1)) # Sort numerically

gene_list = ['CRHR1','ESR1','ESR2','PCLO','FHIT','CACNA1C','DRD2','GRM7','EHD3','BICC1','PLOD1','LINC00687','CSMD1','LHPP','APC','ARHGAP8','LOC100996549','CNTNAP2','CRY1','COMT','FKBP5','HTR2A','BDNF','SLC6A4','ACE','SLC6A2','KCNK2','NR3C1','MTHFR','TPH1','TPH2','SOD2','CNR1','TNF','HTR1A','ABCB1','GNB3','GSK3B']

for gene in gene_list:

    # Directory containing the saved .npy files
    directory = gene
    print('['+directory+']')
    
    for substr in ['embeddings','labels']:
    
        # Get a sorted list of file paths
        file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith(substr)]
        sorted_file_paths = sorted(file_paths, key=numerical_sort)

        if substr == 'embeddings':
            # Load and concatenate all embeddings
            all_embeddings = []
            for file_path in sorted_file_paths:
                chunk = np.load(file_path)
                all_embeddings.append(chunk)

            # Combine all chunks into a single NumPy array
            all_embeddings = np.concatenate(all_embeddings, axis=0)
            #print('embeddings: '+str(all_embeddings.shape))
        
        else:
            # Load and concatenate all labels
            all_labels = []
            for file_path in sorted_file_paths:
                chunk = np.load(file_path)
                all_labels.append(chunk)

            # Combine all chunks into a single NumPy array
            all_labels = np.concatenate(all_labels, axis=0)
            #print('labels: '+str(all_labels.shape))
    
    # Scale the embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(all_embeddings)
            
    # Initialize the model
    #model = cuRFC(random_state=98) # random forest
    #model = cuSVC(kernel='rbf', probability=False, random_state=98, gamma=0.1) # svm
    #model = xgb.XGBClassifier(n_estimators=100, random_state=98, eval_metric='error', device="cuda") # xgboost
    model = cuLR(tol=0.001) # logistic regression

    # 5-fold cross-validation - accuracy, AUC
    #embeddings_scaled_gpu = cp.asarray(embeddings_scaled).get() # for xgboost on gpu
    #all_labels_gpu = cp.asarray(all_labels).get() # for xgboost on gpu
    
    cv_acc = cross_val_score(model, embeddings_scaled, all_labels, cv=5, scoring='accuracy') # embeddings_scaled_gpu, all_labels_gpu for xgboost on gpu
    cv_auc = cross_val_score(model, embeddings_scaled, all_labels, cv=5, scoring='roc_auc')

    print(f"Cross-Validation Accuracy: {np.mean(cv_acc):.2f} ± {np.std(cv_acc):.2f}")
    print(f"Cross-Validation AUC: {np.mean(cv_auc):.2f} ± {np.std(cv_auc):.2f}")
    print()
