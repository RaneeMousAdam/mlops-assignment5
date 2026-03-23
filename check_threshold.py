import mlflow
import os
import sys

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

with open("model_info.txt") as f:
    run_id = f.read().strip()

run = mlflow.get_run(run_id)
acc = run.data.metrics["accuracy"]
print(f"Accuracy: {acc}")

if acc < 0.85:
    print("Failed: below threshold")
    sys.exit(1)

print("Passed!")