import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

seed = 98
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class MLPClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim=100):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)
    

class CNNClassifier(nn.Module):

    def __init__(self, input_dim, conv1_out=32, conv2_out=64, conv3_out=128, fc1_units=512, fc2_units=128, dropout=0.5, random_seed=98):

        # Set the random seed
        if random_seed is not None:
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)

        super(CNNClassifier, self).__init__()

        self.conv1 = nn.Conv1d(1, conv1_out, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(conv1_out)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(conv1_out, conv2_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(conv2_out)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(conv2_out, conv3_out, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(conv3_out)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        with torch.inference_mode():
            dummy_input = torch.zeros(1, 1, input_dim)
            x = self.pool(F.relu(self.bn1(self.conv1(dummy_input))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            flattened_size = x.reshape(1, -1).shape[-1]
        
        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
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
