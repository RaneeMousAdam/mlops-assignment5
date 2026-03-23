import sys

with open("model_info.txt", "r") as f:
    accuracy = float(f.read().strip())

print(f"Checked Accuracy: {accuracy}")

if accuracy < 0.85:
    print("*_* Model failed")
    sys.exit(1)
else:
    print("*~* Model passed")