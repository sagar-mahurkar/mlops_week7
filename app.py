import os
import time
import json
import joblib
import mlflow
import logging
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Response, status, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------------------
# OpenTelemetry setup
# --------------------------------------------------------------
# NOTE: The opentelemetry-exporter-cloud-trace package is deprecated.
# We'll stick to the original for now, but in production, use opentelemetry-exporter-gcp-trace.
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
    """Formats log records as JSON strings."""
    def format(self, record):
        # Includes trace_id if available on the log record
        log_entry = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "timestamp": self.formatTime(record, "%Y-%m-%d %H:%M:%S"),
            "trace_id": getattr(record, "trace_id", "N/A"),
        }
        return json.dumps(log_entry)

handler.setFormatter(JsonFormatter())
# Clear existing handlers to avoid duplicate logs (common in Jupyter/CLI contexts)
if logger.hasHandlers():
    logger.handlers.clear() 
logger.addHandler(handler)

# --------------------------------------------------------------
# MLflow Configuration
# --------------------------------------------------------------
# Use environment variables for MLOps best practice
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://34.123.22.110:5000/")
MODEL_NAME = os.getenv("MODEL_NAME", "iris-random-forest")
# Use a specific stage (e.g., Production) to fetch the model from the registry
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production") 
RUN_NAME = "Random Forest Hyperparameter Search"

MODEL_DOWNLOAD_PATH = "downloaded_models"
# We will rely on MLflow to load directly from URI, so local paths are less critical for serving.
# LOCAL_ARTIFACT_DIR = "artifacts" 
# LOCAL_MODEL_PATH = os.path.join(LOCAL_ARTIFACT_DIR, "random_forest_model.pkl")

os.makedirs(MODEL_DOWNLOAD_PATH, exist_ok=True)
# os.makedirs(LOCAL_ARTIFACT_DIR, exist_ok=True)


# --------------------------------------------------------------
# Model Loading Utility (Refined for Startup)
# --------------------------------------------------------------

# Global variable to hold the model instance
LOADED_MODEL = None

def load_mlflow_model():
    """Fetches the latest model from the MLflow Model Registry (Production stage)."""
    global LOADED_MODEL
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    
    logger.info(f"Attempting to load model from MLflow: {model_uri}")
    try:
        # Load the model directly using the model URI (Production stage)
        model = mlflow.pyfunc.load_model(model_uri)
        LOADED_MODEL = model
        logger.info("MLflow model loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to load MLflow model from {model_uri}: {e}")
        LOADED_MODEL = None
        return False

def get_model():
    """FastAPI Dependency Injection function to access the model."""
    if LOADED_MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not yet loaded or failed to load. Try the /train endpoint first."
        )
    return LOADED_MODEL

# --------------------------------------------------------------
# FastAPI Setup using Lifespan (Startup/Shutdown)
# --------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup (model loading, readiness check)
    and shutdown.
    """
    logger.info("Application starting up...")
    
    # 1. Model Loading - Executed BEFORE the server accepts traffic
    if load_mlflow_model():
        app.state.is_ready = True
        logger.info("Application is ready to serve predictions.")
    else:
        app.state.is_ready = False
        logger.warning("Application started but model failed to load. Prediction endpoint will fail.")

    # 2. Yield control to the application to handle requests
    yield
    
    # 3. Shutdown - Executed AFTER the server stops accepting traffic
    logger.info("Application shutting down...")
    
    
# Initialize FastAPI app with the lifespan context
app = FastAPI(
    title="MLflow Model API",
    description="Train, fetch, and predict using MLflow-managed models.",
    version="1.3.1",
    lifespan=lifespan # Use lifespan instead of @app.on_event("startup")
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
# Data Preparation (No change needed)
# --------------------------------------------------------------
def prepare_data():
    # ... (same as original)
    try:
        # NOTE: data.csv must be available in the container's working directory!
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
        # NOTE: Changed from HTTPException to a standard exception to be caught 
        # by the global handler for better tracing in the /train endpoint.
        raise RuntimeError(f"Data preparation failed: {e}") from e

# --------------------------------------------------------------
# Train and Log Model (Minor cleanup)
# --------------------------------------------------------------
def tune_random_forest(X_train, y_train, X_test, y_test):
    # ... (same as original, removed redundant try/except around the whole function)
    with tracer.start_as_current_span("model_training"):
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

            # Logs model as MLflow's pyfunc flavor for generalized loading
            mlflow.sklearn.log_model(best_model, "random_forest_model", registered_model_name=MODEL_NAME)

            # NOTE: Removed joblib.dump and related paths. 
            # In production serving, we load via MLflow URI, not local files.
            logger.info("Model trained and registered with MLflow.")

        return {
            "best_params": grid.best_params_,
            "cv_accuracy": best_score_cv,
            "test_accuracy": test_score,
            "run_id": mlflow.last_active_run().info.run_id
        }


# --------------------------------------------------------------
# Probes & Middleware (Simplified)
# --------------------------------------------------------------
# NOTE: Removed the boilerplate @app.on_event("startup") and the manual 
# time.sleep(2) which is handled much better by the lifespan context.

@app.get("/live_check", tags=["Probe"])
async def liveness_probe():
    """Kubernetes Liveness Probe: Checks if the application process is running."""
    # Liveness check only confirms the app process is up and can respond.
    return {"status": "alive"}

@app.get("/ready_check", tags=["Probe"])
async def readiness_probe():
    """Kubernetes Readiness Probe: Checks if the application is ready to serve traffic."""
    # Readiness check confirms the app is up AND the model is loaded (app.state.is_ready).
    if getattr(app.state, "is_ready", False):
        return {"status": "ready"}
    
    # Use JSONResponse for consistent error format with other endpoints
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
        content={"status": "not_ready", "detail": "Model not loaded."}
    )

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Middleware for logging request duration and adding trace ID to logs.
    """
    # Start timer for process time header
    start = time.time()
    
    # Get current span/trace_id
    current_span = trace.get_current_span()
    trace_id = format(current_span.get_span_context().trace_id, "032x")
    
    # Add trace ID to the request state so it can be accessed in endpoints/exception handler
    request.scope["trace_id"] = trace_id 
    
    response = await call_next(request)
    
    # Calculate latency and add header
    response.headers["X-Process-Time-ms"] = str(round((time.time() - start) * 1000, 2))
    
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global handler to catch all unhandled exceptions and log them with trace ID."""
    # Retrieve trace_id from request scope (set in middleware)
    trace_id = request.scope.get("trace_id", "N/A")

    # Add trace_id to the log record for structured logging
    log_extra = {"trace_id": trace_id}
    logger.exception(
        "unhandled_exception", 
        exc_info=exc, 
        extra=log_extra # Inject trace_id into JsonFormatter
    )
    
    return JSONResponse(
        status_code=500, 
        content={"detail": "Internal Server Error", "trace_id": trace_id}
    )

# --------------------------------------------------------------
# Endpoints (Refined for Model Dependency)
# --------------------------------------------------------------

@app.get("/")
def root():
    return {"message": "Welcome to the Iris Classifier API"}

@app.post("/train", status_code=status.HTTP_201_CREATED)
async def train_and_register_model():
    """
    Trains a new model, registers it to MLflow, and re-loads the new model 
    into the application state.
    """
    with tracer.start_as_current_span("train_endpoint"):
        try:
            X_train, y_train, X_test, y_test = prepare_data()
            train_results = tune_random_forest(X_train, y_train, X_test, y_test)
            
            # CRITICAL: Re-load the model after successful training to serve the new version
            if load_mlflow_model():
                app.state.is_ready = True
                logger.info("New model loaded into serving state.")
            
            return {"status": "success", "details": train_results}
        except Exception as e:
            # Raise an HTTPException to be handled by FastAPI's default handler 
            # (or the global handler if it catches it)
            logger.error(f"Train endpoint failed: {e}")
            raise HTTPException(status_code=500, detail=f"Training failed: {e}")


@app.get("/fetch")
def fetch_model():
    """
    Simply triggers the model loading function without changing app state.
    Used for ad-hoc reloading.
    """
    if load_mlflow_model():
        app.state.is_ready = True
        return {"status": "success", "detail": f"Model '{MODEL_NAME}' successfully loaded/reloaded."}
    else:
        app.state.is_ready = False
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model fetch and load failed."
        )


@app.post("/predict")
async def predict(input_data: IrisInput, request: Request, model: object = Depends(get_model)):
    """
    Makes predictions using the model loaded at startup.
    Uses Dependency Injection (Depends(get_model)) to access the model.
    """
    with tracer.start_as_current_span("model_inference") as span:
        trace_id = request.scope.get("trace_id", "N/A")
        
        try:
            # Convert the Pydantic model into a Pandas DataFrame
            data = pd.DataFrame([input_data.model_dump().values()], columns=input_data.model_dump().keys())
            
            # Predict using the loaded model
            preds = model.predict(data)
            
            # Calculate latency (using start time from middleware)
            latency = round((time.time() - request.scope.get("start_time", time.time())) * 1000, 2)

            # Log the prediction event
            logger.info(
                "prediction", 
                extra={
                    "trace_id": trace_id,
                    "input": input_data.model_dump(),
                    "prediction": preds.tolist(),
                    "latency_ms": latency
                }
            )
            return {"status": "success", "prediction": preds.tolist()}
            
        except Exception as e:
            # Errors here are usually due to prediction logic or data format
            logger.error("Prediction error", extra={"trace_id": trace_id})
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# --------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    # NOTE: Using environment variable port 8200 for consistency with deployment.yaml
    port = int(os.getenv("PORT", 8200)) 
    print(f"🚀 Starting FastAPI app on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)