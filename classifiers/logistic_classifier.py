"""Logistic regression classifier implementation"""

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from .base_classifier import BaseClassifier


class LogisticRegressionClassifier(BaseClassifier):
    """Logistic Regression classifier with polynomial features and grid search"""
    
    def __init__(self, csvfile="data/pr2_gripper_cylinder.csv", use_grid_search=True):
        super().__init__(csvfile)
        self.use_grid_search = use_grid_search
        self.best_params = None
        
    def train(self):
        """Train logistic regression model with grid search for optimal hyperparameters"""
        X, y = self.load_and_balance_data()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Create pipeline with polynomial features and logistic regression
        model = make_pipeline(
            PolynomialFeatures(),
            StandardScaler(),
            LogisticRegression(penalty="l2", max_iter=10000)
        )
        
        # Define parameter grid for grid search
        param_grid = {
            'polynomialfeatures__degree': [4, 5, 6, 7, 8],
            'logisticregression__C': [1.0, 10.0, 100.0, 1000.0]
        }
        
        print("\nTraining Logistic Regression classifier with Grid Search...")
        print(f"Testing {len(param_grid['polynomialfeatures__degree']) * len(param_grid['logisticregression__C'])} parameter combinations...")
        
        # Perform grid search with 5-fold cross-validation
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=1)
        grid_search.fit(X_train, y_train)
        
        # Store best parameters and model
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        print(f"\nBest parameters found: {self.best_params}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Evaluate on train and test
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        print("\nTrain Accuracy:", accuracy_score(y_train, y_train_pred))
        print("Test Accuracy:", accuracy_score(y_test, y_test_pred), "\n")
        
        print("Confusion Matrix\n", confusion_matrix(y_test, y_test_pred), "\n")
        print("Classification Report\n", classification_report(y_test, y_test_pred), "\n")
        
        return self.model
    
    def predict(self, X):
        """Predict class labels"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
