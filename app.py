# ==============================================================
# FastAPI Application: Train, Fetch, and Predict using MLflow
# Saves trained model locally under 'artifacts/' as well
# ==============================================================

import os
import sys
import mlflow
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------------------
# Configuration
# --------------------------------------------------------------
MLFLOW_TRACKING_URI = "http://136.112.40.182:5000/"
MODEL_NAME = "iris-random-forest"
RUN_NAME = "Random Forest Hyperparameter Search"

MODEL_DOWNLOAD_PATH = "downloaded_models"
MODEL_ARTIFACT_PATH = os.path.join(MODEL_DOWNLOAD_PATH, "random_forest_model")

LOCAL_ARTIFACT_DIR = "artifacts"
LOCAL_MODEL_PATH = os.path.join(LOCAL_ARTIFACT_DIR, "random_forest_model.pkl")

# Ensure local directories exist
os.makedirs(MODEL_DOWNLOAD_PATH, exist_ok=True)
os.makedirs(LOCAL_ARTIFACT_DIR, exist_ok=True)

# --------------------------------------------------------------
# FastAPI App
# --------------------------------------------------------------
app = FastAPI(
    title="MLflow Model API",
    description="Train, fetch, and predict using MLflow-managed models.",
    version="1.2.0",
)

# --------------------------------------------------------------
# Request Schema for Prediction
# --------------------------------------------------------------
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# --------------------------------------------------------------
# Utility: Prepare Dataset
# --------------------------------------------------------------
def prepare_data():
    """Load and split the Iris dataset."""
    try:
        print("Preparing data...")
        data = pd.read_csv("./data.csv")
        data = pd.DataFrame(
            data, columns=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
        )

        train, test = train_test_split(
            data, test_size=0.2, stratify=data["species"], random_state=42
        )

        feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        X_train, y_train = train[feature_cols], train["species"]
        X_test, y_test = test[feature_cols], test["species"]
        print("Data split complete.")
        return X_train, y_train, X_test, y_test
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing data: {e}")

# --------------------------------------------------------------
# Utility: Train and Log RandomForest Model
# --------------------------------------------------------------
def tune_random_forest(X_train, y_train, X_test, y_test):
    """Train a Random Forest model with GridSearchCV, log to MLflow, and save locally."""
    try:
        print(f"Starting MLflow logging to: {MLFLOW_TRACKING_URI}")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        rf_param_grid = {
            "n_estimators": [50, 100, 200],
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [3, 5, 10],
            "class_weight": [None, "balanced"],
        }

        with mlflow.start_run(run_name=RUN_NAME):
            rf_model = RandomForestClassifier(random_state=42)
            rf_grid_search = GridSearchCV(
                rf_model, rf_param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=2
            )

            print("Executing hyperparameter search...")
            rf_grid_search.fit(X_train, y_train)

            best_model = rf_grid_search.best_estimator_
            best_score_cv = rf_grid_search.best_score_
            test_score = rf_grid_search.score(X_test, y_test)

            print("\n--- Tuning Results ---")
            print(f"Best parameters: {rf_grid_search.best_params_}")
            print(f"Best CV score: {best_score_cv:.4f}")
            print(f"Test set accuracy: {test_score:.4f}")

            # --- Log to MLflow ---
            mlflow.log_params(rf_grid_search.best_params_)
            mlflow.log_metric("best_cv_accuracy", best_score_cv)
            mlflow.log_metric("final_test_accuracy", test_score)
            mlflow.sklearn.log_model(
                best_model,
                "random_forest_model",
                registered_model_name=MODEL_NAME,
            )

            # --- Save locally under artifacts/ ---
            joblib.dump(best_model, LOCAL_MODEL_PATH)
            print(f"✅ Model saved locally at: {LOCAL_MODEL_PATH}")

        print("MLflow run finished.")
        return {
            "best_params": rf_grid_search.best_params_,
            "cv_accuracy": best_score_cv,
            "test_accuracy": test_score,
            "local_model_path": LOCAL_MODEL_PATH,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model training: {e}")

# --------------------------------------------------------------
# Utility: Fetch and Download Latest Model from MLflow
# --------------------------------------------------------------
def fetch_latest_model():
    """Fetch and download the latest registered model version."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

        print(f"Searching for latest version of model: {MODEL_NAME}")
        versions = client.search_model_versions(
            filter_string=f"name='{MODEL_NAME}'",
            order_by=["version_number DESC"],
            max_results=1,
        )

        if not versions:
            raise HTTPException(status_code=404, detail="No registered versions found.")

        latest_version = versions[0]
        run_id = latest_version.run_id
        print(f"Found model v{latest_version.version}, Run ID: {run_id}")

        print(f"Downloading model artifact from run {run_id}...")
        downloaded_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="random_forest_model",
            dst_path=MODEL_DOWNLOAD_PATH,
        )

        print(f"Model downloaded successfully to: {downloaded_path}")
        return {
            "model_name": MODEL_NAME,
            "version": latest_version.version,
            "run_id": run_id,
            "download_path": downloaded_path,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching model: {e}")

# --------------------------------------------------------------
# Utility: Load Latest Model (from local dump or MLflow)
# --------------------------------------------------------------
def load_latest_model():
    """Load model from local 'artifacts' folder or from MLflow if missing."""
    try:
        if os.path.exists(LOCAL_MODEL_PATH):
            print(f"Loading local model from {LOCAL_MODEL_PATH}")
            return joblib.load(LOCAL_MODEL_PATH)

        print("Local model not found. Fetching latest from MLflow...")
        fetch_latest_model()
        model = mlflow.pyfunc.load_model(MODEL_ARTIFACT_PATH)
        return model

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

# --------------------------------------------------------------
# FastAPI Routes
# --------------------------------------------------------------

@app.post("/train", summary="Train and log a Random Forest model to MLflow")
def train_model():
    """Train model, log to MLflow, and dump locally."""
    X_train, y_train, X_test, y_test = prepare_data()
    result = tune_random_forest(X_train, y_train, X_test, y_test)
    return {"status": "success", "details": result}


@app.get("/fetch", summary="Fetch latest model from MLflow")
def fetch_model():
    """Fetch latest model version and download"""
    result = fetch_latest_model()
    return {"status": "success", "details": result}


@app.post("/predict", summary="Predict class using latest trained model")
def predict(input_data: IrisInput):
    """Predict species using the latest trained model"""
    model = load_latest_model()

    data = pd.DataFrame(
        [[input_data.sepal_length, input_data.sepal_width, input_data.petal_length, input_data.petal_width]],
        columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    )

    try:
        preds = model.predict(data)
        return {"status": "success", "prediction": preds.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

# --------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting FastAPI app on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
