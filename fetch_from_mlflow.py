import os
import sys
import mlflow
from mlflow.tracking import MlflowClient

URI = "http://35.202.91.253:5000/"
NAME = "iris-random-forest"
PATH = "downloaded_models"

def fetch(client, name):
    print(f"Searching for latest version of model: {name}")
    try:
        versions = client.search_model_versions(
            filter_string=f"name='{name}'",
            order_by=["version_number DESC"],
            max_results=1
        )
        
        if versions:
            latest_version = versions[0]
            print(f"Found model version: v{latest_version.version}, Run ID: {latest_version.run_id}")
            return latest_version
        else:
            print(f"Error: No versions found for model '{name}'.")
            return None
            
    except Exception as e:
        print(f"An error occurred while fetching model info: {e}")
        return None
    
def download(version, save_path):
    if not version:
        print("Cannot download: Model version object is missing.")
        return

    run_id = version.run_id
    # Updated message to reflect the change
    print(f"Downloading model artifact ('random_forest_model') from run '{run_id}'...")
    
    try:
        downloaded_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            # CHANGE: Match this path to the subdirectory name used in log_model
            artifact_path="random_forest_model", 
            dst_path=save_path
        )
        print(f"Successfully downloaded model artifact to: {downloaded_path}")
        
    except Exception as e:
        print(f"Error downloading artifacts: {e}")

        
def main():
    mlflow.set_tracking_uri(URI)
    print(f"MLflow tracking URI set to: {URI}")
    
    client = MlflowClient(tracking_uri=URI)
    
    version = fetch(client, NAME)
    
    if version:
        download(version, PATH)

        
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"A critical error occurred in the workflow: {e}")
        sys.exit(1)
