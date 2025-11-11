# ==============================================================
# FastAPI Application: Train, Fetch, and Predict using MLflow
# --------------------------------------------------------------
# This app allows:
#   ✅ Training a RandomForest model and logging it to MLflow
#   ✅ Fetching the latest model version from MLflow
#   ✅ Making predictions using the latest local or remote model
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
# Configuration: Define MLflow setup and local storage paths
# --------------------------------------------------------------
MLFLOW_TRACKING_URI = "http://34.46.247.19:5000/"  # MLflow tracking server
MODEL_NAME = "iris-random-forest"                     # Registered model name
RUN_NAME = "Random Forest Hyperparameter Search"      # MLflow run name

# Local directories for saving models
MODEL_DOWNLOAD_PATH = "downloaded_models"             # Directory to download model artifacts
MODEL_ARTIFACT_PATH = os.path.join(MODEL_DOWNLOAD_PATH, "random_forest_model")

LOCAL_ARTIFACT_DIR = "artifacts"                      # Directory to save trained models locally
LOCAL_MODEL_PATH = os.path.join(LOCAL_ARTIFACT_DIR, "random_forest_model.pkl")

# Ensure required directories exist
os.makedirs(MODEL_DOWNLOAD_PATH, exist_ok=True)
os.makedirs(LOCAL_ARTIFACT_DIR, exist_ok=True)

# --------------------------------------------------------------
# FastAPI Application Setup
# --------------------------------------------------------------
app = FastAPI(
    title="MLflow Model API",
    description="Train, fetch, and predict using MLflow-managed models.",
    version="1.3.0",
)

# --------------------------------------------------------------
# Request Schema for Prediction Input
# --------------------------------------------------------------
class IrisInput(BaseModel):
    """Input schema for prediction requests"""
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# --------------------------------------------------------------
# Utility: Load and Split Dataset
# --------------------------------------------------------------
def prepare_data():
    """
    Load the Iris dataset from a CSV file, split it into
    training and testing sets, and return features/labels.
    """
    try:
        print("Preparing data...")
        data = pd.read_csv("./data.csv")

        # Ensure consistent column names
        data = pd.DataFrame(
            data, columns=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
        )

        # Split data into train/test sets (80/20)
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
# Utility: Train and Log RandomForest Model to MLflow
# --------------------------------------------------------------
def tune_random_forest(X_train, y_train, X_test, y_test):
    """
    Train a RandomForest model using GridSearchCV for hyperparameter tuning.
    Logs best model parameters and metrics to MLflow and saves the model locally.
    """
    try:
        print(f"Starting MLflow logging to: {MLFLOW_TRACKING_URI}")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Define hyperparameter grid for tuning
        rf_param_grid = {
            "n_estimators": [50, 100, 200],
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [3, 5, 10],
            "class_weight": [None, "balanced"],
        }

        # Start a new MLflow run
        with mlflow.start_run(run_name=RUN_NAME):
            rf_model = RandomForestClassifier(random_state=42)
            rf_grid_search = GridSearchCV(
                rf_model, rf_param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=2
            )

            print("Executing hyperparameter search...")
            rf_grid_search.fit(X_train, y_train)

            # Extract best model and metrics
            best_model = rf_grid_search.best_estimator_
            best_score_cv = rf_grid_search.best_score_
            test_score = rf_grid_search.score(X_test, y_test)

            print("\n--- Tuning Results ---")
            print(f"Best parameters: {rf_grid_search.best_params_}")
            print(f"Best CV score: {best_score_cv:.4f}")
            print(f"Test set accuracy: {test_score:.4f}")

            # --- Log experiment results to MLflow ---
            mlflow.log_params(rf_grid_search.best_params_)
            mlflow.log_metric("best_cv_accuracy", best_score_cv)
            mlflow.log_metric("final_test_accuracy", test_score)

            # Save and register model to MLflow
            mlflow.sklearn.log_model(
                best_model,
                "random_forest_model",
                registered_model_name=MODEL_NAME,
            )

            # Save model locally for quick inference
            joblib.dump(best_model, LOCAL_MODEL_PATH)
            print(f"✅ Model saved locally at: {LOCAL_MODEL_PATH}")

        print("MLflow run finished successfully.")
        return {
            "best_params": rf_grid_search.best_params_,
            "cv_accuracy": best_score_cv,
            "test_accuracy": test_score,
            "local_model_path": LOCAL_MODEL_PATH,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model training: {e}")

# --------------------------------------------------------------
# Utility: Fetch and Download Latest Registered Model from MLflow
# --------------------------------------------------------------
def fetch_latest_model():
    """
    Fetch the latest registered model version from MLflow,
    download its artifacts, and return metadata.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

        print(f"Searching for latest version of model: {MODEL_NAME}")

        # Retrieve most recent model version
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

        # Download model artifact from MLflow
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
# Utility: Load Model from Local or Remote Sources
# --------------------------------------------------------------
def load_latest_model():
    """
    Load the model in order of preference:
    1️⃣ Load from local artifacts ('artifacts/random_forest_model.pkl')
    2️⃣ Load from already downloaded MLflow artifacts
    3️⃣ Fetch from MLflow if missing locally
    """
    try:
        # Try local cached model first
        if os.path.exists(LOCAL_MODEL_PATH):
            print(f"✅ Loading model from local artifacts: {LOCAL_MODEL_PATH}")
            return joblib.load(LOCAL_MODEL_PATH)

        # Try downloaded MLflow artifact
        elif os.path.exists(MODEL_ARTIFACT_PATH):
            print(f"✅ Loading model from downloaded MLflow artifacts: {MODEL_ARTIFACT_PATH}")
            model = mlflow.sklearn.load_model(MODEL_ARTIFACT_PATH)
            joblib.dump(model, LOCAL_MODEL_PATH)
            print(f"💾 Cached model locally at: {LOCAL_MODEL_PATH}")
            return model

        # Fetch from MLflow if not found
        else:
            print("⚠️ Model not found locally. Fetching latest from MLflow...")
            fetch_latest_model()

            if os.path.exists(MODEL_ARTIFACT_PATH):
                model = mlflow.sklearn.load_model(MODEL_ARTIFACT_PATH)
                joblib.dump(model, LOCAL_MODEL_PATH)
                print(f"💾 Cached fetched model locally at: {LOCAL_MODEL_PATH}")
                return model
            else:
                raise HTTPException(status_code=404, detail="Model fetch failed — artifact not found after download.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

# --------------------------------------------------------------
# FastAPI Endpoints
# --------------------------------------------------------------

@app.get("/", summary="Welcome to Iris Classifier API")
def read_root():
    """Read the default route"""
    return {"message": "Welcome to the Iris Classifier API"}


@app.post("/train", summary="Train and log a Random Forest model to MLflow")
def train_model():
    """Train model, log to MLflow, and dump locally."""
    X_train, y_train, X_test, y_test = prepare_data()
    result = tune_random_forest(X_train, y_train, X_test, y_test)
    return {"status": "success", "details": result}


@app.get("/fetch", summary="Fetch latest model from MLflow")
def fetch_model():
    """Fetch the latest version of the model from MLflow."""
    result = fetch_latest_model()
    return {"status": "success", "details": result}


@app.post("/predict", summary="Predict class using latest trained model")
def predict(input_data: IrisInput):
    """
    Predict species class based on input Iris flower measurements.
    Automatically loads latest available model.
    """
    model = load_latest_model()

    # Convert input to DataFrame
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
# Entry Point: Run App with Uvicorn
# --------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI app on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
