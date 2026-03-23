import mlflow
import os
import sys

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

with open("model_info.txt") as f:
    run_id = f.read().strip()

run = mlflow.get_run(run_id)
acc = run.data.metrics["accuracy"]

print(f"Run ID: {run_id}")
print(f"Accuracy: {acc}")

if acc < 0.50:
    print("FAILED: Accuracy below threshold (0.50)")
    sys.exit(1)

print("PASSED: Accuracy meets threshold")