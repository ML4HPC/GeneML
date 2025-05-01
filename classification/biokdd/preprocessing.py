from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import TensorDataset, DataLoader

#################################
## Scaling train-val-test sets ##
#################################

def scale_data(X_train, X_val, X_test):
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled

###############################
## Create dataloader (torch) ##
###############################

def create_dataloader(X, y, batch_size=15, shuffle=True):

    generator = torch.Generator().manual_seed(98)
    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(tensor_x, tensor_y)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)
