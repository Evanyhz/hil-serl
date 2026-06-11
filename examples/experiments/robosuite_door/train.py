#!/usr/bin/env python3
"""Convenience launcher for the robosuite_door RLPD training entrypoint."""

import os
import subprocess
import sys
from pathlib import Path


def main():
    this_dir = Path(__file__).resolve().parent
    examples_dir = this_dir.parents[1]
    checkpoint_path = this_dir / "debug"

    cmd = [
        sys.executable,
        str(examples_dir / "train_rlpd.py"),
        "--exp_name=robosuite_door",
        f"--checkpoint_path={checkpoint_path}",
        *sys.argv[1:],
    ]

    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".2")


    print("[robosuite_door train.py] running:")
    print(" ".join(cmd), flush=True)    
    raise SystemExit(subprocess.call(cmd, cwd=str(examples_dir), env=env))


if __name__ == "__main__":
    main()

