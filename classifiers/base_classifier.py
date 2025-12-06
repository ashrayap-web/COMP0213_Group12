"""Base classifier interface"""

import pandas as pd
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class BaseClassifier(ABC):
    """Base classifier class for grasp success prediction"""
    
    def __init__(self, csvfile="data/pr2_gripper_cylinder.csv"):
        self.csvfile = csvfile
        self.model = None
        
    def load_and_balance_data(self):
        """Load CSV and balance the dataset"""
        df = pd.read_csv(self.csvfile)
        min_n = df["Result"].value_counts().min()
        print(f"Balancing dataset: {min_n} samples per class")
        balanced_df = df.groupby("Result", group_keys=False).sample(n=min_n, random_state=42)
        
        X = balanced_df.drop(columns=["Result"])
        y = balanced_df["Result"]
        
        return X, y
    
    @abstractmethod
    def train(self):
        """Train the classifier model"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Predict class labels for samples in X"""
        pass
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data"""
        y_pred = self.predict(X_test)
        
        print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("Confusion Matrix\n", confusion_matrix(y_test, y_pred), "\n")
        print("Classification Report\n", classification_report(y_test, y_pred), "\n")
