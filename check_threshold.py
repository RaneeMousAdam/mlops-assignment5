import mlflow
import sys

THRESHOLD = 0.85

# Read the Run ID written by train.py
with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

print(f"Checking accuracy for Run ID: {run_id}")

# Fetch the run from MLflow
client = mlflow.MlflowClient()
run = client.get_run(run_id)
accuracy = run.data.metrics.get("accuracy")

if accuracy is None:
    print("ERROR: No accuracy metric found for this run.")
    sys.exit(1)

print(f"Accuracy: {accuracy:.4f} | Threshold: {THRESHOLD}")

if accuracy < THRESHOLD:
    print(f"FAILED: Accuracy {accuracy:.4f} is below threshold {THRESHOLD}.")
    sys.exit(1)

print(f"PASSED: Accuracy {accuracy:.4f} meets the threshold. Proceeding to deploy.")