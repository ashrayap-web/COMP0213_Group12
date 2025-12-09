"""MLP neural network classifier implementation"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from .base_classifier import BaseClassifier


class MLP(nn.Module):
    """Multi-Layer Perceptron for grasp classification"""
    def __init__(self, input_dim):
        super().__init__()
        dropout = 0.15
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


class MLPClassifier(BaseClassifier):
    """MLP Neural Network classifier"""
    
    def __init__(self, csvfile="data/pr2_gripper_cylinder.csv", epochs=300, batch_size=64, lr=0.001):
        super().__init__(csvfile)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.scaler = None
        self.nn_model = None
        
    def train(self):
        """Train MLP classifier"""
        X, y = self.load_and_balance_data()
        
        # Create train, val and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )
        
        # Feature normalization
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to PyTorch tensors
        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
        y_val_t = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
        X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
        
        # Create data loaders
        train_data = TensorDataset(X_train_t, y_train_t)
        val_data = TensorDataset(X_val_t, y_val_t)
        
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)
        
        # Initialize model
        self.nn_model = MLP(X_train_t.shape[1])
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.nn_model.parameters(), lr=self.lr, weight_decay=1e-4)
        
        # Best model tracking
        best_val_loss = float('inf')
        best_model_epoch = 0
        
        # Training loop
        print("\nTraining MLP classifier...")
        
        for epoch in range(self.epochs):
            self.nn_model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.nn_model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * y_batch.size(0)
            
            # Validation
            self.nn_model.eval()
            correct_preds = 0
            val_loss = 0.0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = self.nn_model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * y_batch.size(0)
                    preds_cls = (outputs > 0.5).float()
                    correct_preds += preds_cls.eq(y_batch).sum().item()
            
            avg_val_loss = val_loss / len(val_loader.dataset)
            
            if (epoch + 1) % 50 == 0:
                print(f"Epoch: {(epoch+1)}/{self.epochs}   "
                      f"Train Loss: {train_loss/len(train_loader.dataset):.4f}   "
                      f"Val Accuracy: {100*correct_preds/len(val_loader.dataset):.2f}% -> {correct_preds}/{len(val_loader.dataset)}   "
                      f"Val Loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.nn_model.state_dict(), 'best_model.pth')
                best_model_epoch = epoch
        
        # Load best model for testing
        print(f"Loading best model from epoch {best_model_epoch + 1}")
        self.nn_model.load_state_dict(torch.load('best_model.pth'))
        self.nn_model.eval()
        with torch.no_grad():
            preds = self.nn_model(X_test_t)
            preds_class = (preds > 0.5).float()
            accuracy = (preds_class.eq(y_test_t).sum() / y_test_t.shape[0]).item()
        
        print(f"\nTest Accuracy: {accuracy:.4f}")
        
        # Convert to numpy for metrics
        y_test_np = y_test_t.numpy()
        y_pred_np = preds_class.numpy()
        
        print("Confusion Matrix\n", confusion_matrix(y_test_np, y_pred_np), "\n")
        print("Classification Report\n", classification_report(y_test_np, y_pred_np), "\n")
        
        return self
    
    def predict(self, X):
        """Predict class labels"""
        if self.nn_model is None or self.scaler is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.nn_model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            outputs = self.nn_model(X_tensor)
            predictions = (outputs > 0.5).float().numpy().flatten()
        
        return predictions
