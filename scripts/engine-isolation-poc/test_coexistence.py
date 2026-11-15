"""Verify conflicting transformers versions coexist in separate interpreters."""

import argparse
import json
import subprocess

import torch
import transformers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-python", required=True)
    parser.add_argument("--runner")
    args = parser.parse_args()
    assert transformers.__version__ == "4.57.3"
    assert torch.__version__ == "2.6.0+cpu"
    command = ([args.runner, "-cpu", "IvyBridge"] if args.runner else []) + [
        args.worker_python,
        "-c",
        (
            "import json, os, torch, transformers; "
            "print(json.dumps({'pid': os.getpid(), 'torch': torch.__version__, "
            "'transformers': transformers.__version__}))"
        ),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180, check=True)
    worker = json.loads(result.stdout)
    assert worker["transformers"] == "5.2.0"
    assert worker["torch"] == "2.6.0+cpu"
    print(f"Host transformers={transformers.__version__}; worker={worker}", flush=True)


if __name__ == "__main__":
    main()
