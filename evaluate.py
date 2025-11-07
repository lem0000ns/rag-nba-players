from huggingface import load_dataset
import argparse
from dotenv import load_dotenv
import os

load_dotenv()

if __name__ == "__main__":
    argparse.add_argument("--model", type=str, required=True)
    args = argparse.parse_args()
    model = args.model

    print(f"Evaluating {model}...")
    print(f"Model: {model}")
    print(f"Model: {model}")