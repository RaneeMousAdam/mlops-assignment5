import sys
import mlflow

run_file = sys.argv[1]
threshold = float(sys.argv[2])

# Read Run ID
with open(run_file, "r") as f:
    run_id = f.read().strip()

client = mlflow.MlflowClient()
run = client.get_run(run_id)
accuracy = run.data.metrics.get("accuracy", 0)

print(f"Run ID: {run_id}, Accuracy: {accuracy}")

if accuracy < threshold:
    print(f"Accuracy {accuracy} below threshold {threshold}. Failing...")
    sys.exit(1)
else:
    print("Threshold passed!")