# ==============================================================
# FastAPI Application: Train, Fetch, and Predict using MLflow
# --------------------------------------------------------------
# ✅ Trains a RandomForest model, logs to MLflow
# ✅ Fetches latest model version from MLflow
# ✅ Makes predictions using latest model
# ✅ Includes structured logging + OpenTelemetry tracing
# ==============================================================

import os
import time
import json
import joblib
import mlflow
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------------------
# OpenTelemetry setup
# --------------------------------------------------------------
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# --------------------------------------------------------------
# Logging Setup (Structured JSON logs)
# --------------------------------------------------------------
logger = logging.getLogger("iris-ml-service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "timestamp": self.formatTime(record, "%Y-%m-%d %H:%M:%S"),
        }
        return json.dumps(log_entry)

handler.setFormatter(JsonFormatter())
logger.addHandler(handler)

# --------------------------------------------------------------
# MLflow Configuration
# --------------------------------------------------------------
MLFLOW_TRACKING_URI = "http://34.123.22.110:5000/"
MODEL_NAME = "iris-random-forest"
RUN_NAME = "Random Forest Hyperparameter Search"

MODEL_DOWNLOAD_PATH = "downloaded_models"
MODEL_ARTIFACT_PATH = os.path.join(MODEL_DOWNLOAD_PATH, "random_forest_model")
LOCAL_ARTIFACT_DIR = "artifacts"
LOCAL_MODEL_PATH = os.path.join(LOCAL_ARTIFACT_DIR, "random_forest_model.pkl")

os.makedirs(MODEL_DOWNLOAD_PATH, exist_ok=True)
os.makedirs(LOCAL_ARTIFACT_DIR, exist_ok=True)

# --------------------------------------------------------------
# FastAPI Setup
# --------------------------------------------------------------
app = FastAPI(
    title="MLflow Model API",
    description="Train, fetch, and predict using MLflow-managed models.",
    version="1.3.0",
)

# --------------------------------------------------------------
# Input Schema
# --------------------------------------------------------------
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# --------------------------------------------------------------
# Data Preparation
# --------------------------------------------------------------
def prepare_data():
    try:
        data = pd.read_csv("./data.csv")
        data.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

        train, test = train_test_split(
            data, test_size=0.2, stratify=data["species"], random_state=42
        )

        X_train = train[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
        y_train = train["species"]
        X_test = test[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
        y_test = test["species"]

        return X_train, y_train, X_test, y_test
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data preparation failed: {e}")

# --------------------------------------------------------------
# Train and Log Model
# --------------------------------------------------------------
def tune_random_forest(X_train, y_train, X_test, y_test):
    try:
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
            grid = GridSearchCV(
                rf_model, rf_param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
            )
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            best_score_cv = grid.best_score_
            test_score = grid.score(X_test, y_test)

            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics({
                "best_cv_accuracy": best_score_cv,
                "final_test_accuracy": test_score
            })

            mlflow.sklearn.log_model(best_model, "random_forest_model", registered_model_name=MODEL_NAME)

            joblib.dump(best_model, LOCAL_MODEL_PATH)
            logger.info(f"Model trained and saved locally at {LOCAL_MODEL_PATH}")

        return {
            "best_params": grid.best_params_,
            "cv_accuracy": best_score_cv,
            "test_accuracy": test_score,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")

# --------------------------------------------------------------
# Fetch Latest Registered Model from MLflow
# --------------------------------------------------------------
def fetch_latest_model():
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if not versions:
            raise HTTPException(status_code=404, detail="No registered model found.")

        latest_version = max(versions, key=lambda v: int(v.version))
        run_id = latest_version.run_id

        downloaded_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="random_forest_model", dst_path=MODEL_DOWNLOAD_PATH
        )

        return {"version": latest_version.version, "download_path": downloaded_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fetch failed: {e}")

# --------------------------------------------------------------
# Load Latest Model
# --------------------------------------------------------------
def load_latest_model():
    try:
        if os.path.exists(LOCAL_MODEL_PATH):
            return joblib.load(LOCAL_MODEL_PATH)
        elif os.path.exists(MODEL_ARTIFACT_PATH):
            return mlflow.sklearn.load_model(MODEL_ARTIFACT_PATH)
        else:
            fetch_latest_model()
            return mlflow.sklearn.load_model(MODEL_ARTIFACT_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load failed: {e}")

# --------------------------------------------------------------
# Probes & Middleware
# --------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    app.state.is_alive = True
    app.state.is_ready = False
    time.sleep(2)
    app.state.is_ready = True

@app.get("/live_check", tags=["Probe"])
async def liveness_probe():
    if getattr(app.state, "is_alive", False):
        return {"status": "alive"}
    return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/ready_check", tags=["Probe"])
async def readiness_probe():
    if getattr(app.state, "is_ready", False):
        return {"status": "ready"}
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time-ms"] = str(round((time.time() - start) * 1000, 2))
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")
    logger.exception(json.dumps({
        "event": "unhandled_exception",
        "trace_id": trace_id,
        "path": str(request.url),
        "error": str(exc)
    }))
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error", "trace_id": trace_id})

# --------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Welcome to the Iris Classifier API"}

@app.post("/train")
def train_model():
    X_train, y_train, X_test, y_test = prepare_data()
    return {"status": "success", "details": tune_random_forest(X_train, y_train, X_test, y_test)}

@app.get("/fetch")
def fetch_model():
    return {"status": "success", "details": fetch_latest_model()}

@app.post("/predict")
async def predict(input_data: IrisInput, request: Request):
    with tracer.start_as_current_span("model_inference") as span:
        trace_id = format(span.get_span_context().trace_id, "032x")
        model = load_latest_model()

        data = pd.DataFrame([input_data.dict().values()], columns=input_data.dict().keys())
        preds = model.predict(data)
        latency = round((time.time() - request.scope.get("start_time", time.time())) * 1000, 2)

        logger.info(json.dumps({
            "event": "prediction",
            "trace_id": trace_id,
            "input": input_data.dict(),
            "prediction": preds.tolist(),
            "latency_ms": latency
        }))
        return {"status": "success", "prediction": preds.tolist()}

# --------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI app on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
