"""Logistic regression classifier implementation"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from .base_classifier import BaseClassifier


class LogisticRegressionClassifier(BaseClassifier):
    """Logistic Regression classifier with polynomial features"""
    
    def __init__(self, csvfile="data/pr2_gripper_cylinder.csv", degree=6, C=10.0):
        super().__init__(csvfile)
        self.degree = degree
        self.C = C
        
    def train(self):
        """Train logistic regression model"""
        X, y = self.load_and_balance_data()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Create pipeline with polynomial features and logistic regression
        self.model = make_pipeline(
            PolynomialFeatures(degree=self.degree),
            StandardScaler(),
            LogisticRegression(penalty="l2", C=self.C, max_iter=10000)
        )
        
        print("\nTraining Logistic Regression classifier...")
        self.model.fit(X_train, y_train)
        
        # Evaluate on train and test
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
        print("Test Accuracy:", accuracy_score(y_test, y_test_pred), "\n")
        
        print("Confusion Matrix\n", confusion_matrix(y_test, y_test_pred), "\n")
        print("Classification Report\n", classification_report(y_test, y_test_pred), "\n")
        
        return self.model
    
    def predict(self, X):
        """Predict class labels"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
