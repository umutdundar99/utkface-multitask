import sys
from utkface_multitask.train import train

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m utkface_multitask <task>")
        print("Available tasks: contrastive, classification")
        sys.exit(1)

    task = sys.argv[1].lower()
    if task not in ["contrastive", "classification"]:
        print(f"Unknown task: {task}")
        sys.exit(1)

    train(task)

if __name__ == "__main__":
    main()