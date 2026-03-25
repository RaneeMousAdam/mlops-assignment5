import os
import sys
import mlflow

# Read the Run ID written by train.py to identify the specific run to check
with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

print(f"Checking Run ID: {run_id}")

# Connect to the same MLflow Tracking Server used during training
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(tracking_uri)
print(f"MLflow Tracking URI: {tracking_uri}")

# Fetch the run and read the logged accuracy metric 
try:
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
except Exception as e:
    print(f"Could not retrieve run from MLflow: {e}")
    sys.exit(1)

accuracy = run.data.metrics.get("accuracy")

if accuracy is None:
    print("ERROR: 'accuracy' metric not found in the run!")
    sys.exit(1)

print(f"Model Accuracy: {accuracy}")

# Threshold check
THRESHOLD = 0.85

if accuracy < THRESHOLD:
    print(f"FAILED: Accuracy {accuracy:.4f} is below threshold {THRESHOLD}. Halting deployment.")
    sys.exit(1)
else:
    print(f"PASSED: Accuracy {accuracy:.4f} meets threshold {THRESHOLD}. Ready for deployment.")