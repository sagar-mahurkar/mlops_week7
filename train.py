# train.py

# import important modules
import os
import sys
import pandas as pd
import requests

# Imports for the new model and data source
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier # Changed from DecisionTree
import mlflow

# --- Configuration ---
MLFLOW_TRACKING_URI = "http://34.123.22.110:5000/"
MODEL_NAME = "iris-random-forest" # Updated model name
RUN_NAME = "Random Forest Hyperparameter Search" # Updated run name

def prepare_data():
    """Loads the Iris dataset into memory and performs the train/test split."""
    print("Preparing data...")
    # load the data
    data = pd.read_csv('./data.csv')
    data = pd.DataFrame(data, columns=['sepal_length','sepal_width','petal_length','petal_width', 'species'])

    # Split the data
    train, test = train_test_split(
        data, test_size=0.2, stratify=data['species'], random_state=42
    )
    
    # Define features and target
    feature_cols = ['sepal_length','sepal_width','petal_length','petal_width']
    X_train, y_train = train[feature_cols], train['species']
    X_test, y_test = test[feature_cols], test['species']
    
    print("Data split complete.")
    return X_train, y_train, X_test, y_test
    
def tune_random_forest(X_train, y_train, X_test, y_test):
    """
    Sets up MLflow, runs GridSearchCV for RandomForestClassifier,
    and manually logs the best results and the model.
    """
    print(f"Starting MLflow logging to: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # NOTE: Autologging is disabled, parameters, metrics, and model are logged manually.
    # mlflow.sklearn.autolog(max_tuning_runs=10, registered_model_name=MODEL_NAME)
    
    # Updated parameter grid for Random Forest (includes n_estimators)
    rf_param_grid = {
        'n_estimators': [50, 100, 200], # Added RF-specific parameter
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [3, 5, 10], # Slightly different list than original
        'class_weight': [None, 'balanced']
    }
    
    with mlflow.start_run(run_name=RUN_NAME):
        # Initialize RandomForestClassifier
        rf_model = RandomForestClassifier(random_state=42)
        
        # Setup Grid Search
        rf_grid_search = GridSearchCV(
            rf_model, rf_param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=2
        )

        print("Executing hyperparameter search...")
        rf_grid_search.fit(X_train, y_train)
        
        # Get results
        best_score_cv = rf_grid_search.best_score_
        test_score = rf_grid_search.score(X_test, y_test)
        
        print("\n--- Tuning Results ---")
        print(f"Best parameters: {rf_grid_search.best_params_}")
        print(f"Best cross-validation score: {best_score_cv:.4f}")
        print(f"Test set accuracy: {test_score:.4f}")
        
        # --- Manual MLflow Logging ---
        
        # 1. Log the best parameters found by GridSearchCV
        mlflow.log_params(rf_grid_search.best_params_)
        
        # 2. Log the final metrics
        mlflow.log_metric("best_cv_accuracy", best_score_cv)
        mlflow.log_metric("final_test_accuracy", test_score)
        
        # 3. Log the best model estimator
        mlflow.sklearn.log_model(
            rf_grid_search.best_estimator_, 
            "random_forest_model", 
            registered_model_name=MODEL_NAME
        )
        # -----------------------------
    print("MLflow run finished.")
    
    
if __name__ == "__main__":
    try:
        X_train, y_train, X_test, y_test = prepare_data()
        tune_random_forest(X_train, y_train, X_test, y_test)
    except Exception as e:
        print(f"An unexpected error occurred during execution: {e}")
        sys.exit(1)
        
