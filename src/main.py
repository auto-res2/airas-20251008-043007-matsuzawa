"""src/main.py
Master orchestrator that sequentially runs all experiment variations defined
in a YAML file and then triggers evaluation.

Usage:
  python -m src.main --smoke-test --results-dir <path>
  python -m src.main --full-experiment --results-dir <path>
"""
from __future__ import annotations
import argparse, os, sys, subprocess, pathlib, shutil, json, yaml, time, select
from typing import List, Dict, Any

# --------------------------------- helpers ----------------------------------#

def read_yaml(path: os.PathLike):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def tee_subprocess(cmd: List[str], stdout_path: str, stderr_path: str):
    """Run *cmd* while streaming stdout/stderr live AND writing them to files."""

    with open(stdout_path, "w") as out_f, open(stderr_path, "w") as err_f:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

        # Use select for non-blocking read – works on POSIX. On Windows we fall back to blocking.
        stdout_lines, stderr_lines = [], []
        while True:
            reads = [process.stdout.fileno(), process.stderr.fileno()]
            ret = select.select(reads, [], [], 0.1)[0]
            for fd in ret:
                if fd == process.stdout.fileno():
                    line = process.stdout.readline()
                    if line:
                        print(line, end="")
                        out_f.write(line)
                        stdout_lines.append(line)
                elif fd == process.stderr.fileno():
                    line = process.stderr.readline()
                    if line:
                        print(line, end="", file=sys.stderr)
                        err_f.write(line)
                        stderr_lines.append(line)
            if process.poll() is not None:  # process finished
                # drain remaining
                for line in process.stdout:
                    print(line, end="")
                    out_f.write(line)
                    stdout_lines.append(line)
                for line in process.stderr:
                    print(line, end="", file=sys.stderr)
                    err_f.write(line)
                    stderr_lines.append(line)
                break
        return process.returncode


# --------------------------------- main logic -------------------------------#

def run_all(cfg_path: str, results_dir: str):
    cfg = read_yaml(cfg_path)
    experiments: List[Dict[str, Any]] = cfg["experiments"]

    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)

    print("===================== Experiment description =====================")
    print(json.dumps(cfg.get("description", {}), indent=2))
    print("==================================================================")

    for exp in experiments:
        run_id = exp["run_id"]
        run_dir = pathlib.Path(results_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save variation config
        run_cfg_path = run_dir / "run_config.yaml"
        with open(run_cfg_path, "w") as f:
            yaml.safe_dump(exp, f)

        # Construct command
        cmd = [sys.executable, "-m", "src.train", "--config", str(run_cfg_path), "--results-dir", str(results_dir)]
        print(f"\n===== Launching run: {run_id} =====")
        rc = tee_subprocess(cmd, stdout_path=str(run_dir / "stdout.log"), stderr_path=str(run_dir / "stderr.log"))
        if rc != 0:
            raise RuntimeError(f"Run {run_id} failed with return code {rc}")

    # After all runs – aggregate & evaluate
    eval_cmd = [sys.executable, "-m", "src.evaluate", "--results-dir", str(results_dir)]
    print("\n===== Running evaluation across all variations =====")
    rc = tee_subprocess(eval_cmd, stdout_path=str(pathlib.Path(results_dir) / "evaluate_stdout.log"),
                        stderr_path=str(pathlib.Path(results_dir) / "evaluate_stderr.log"))
    if rc != 0:
        raise RuntimeError(f"Evaluation script failed with return code {rc}")


# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke-test", action="store_true", help="Use config/smoke_test.yaml")
    g.add_argument("--full-experiment", action="store_true", help="Use config/full_experiment.yaml")
    p.add_argument("--results-dir", required=True, help="Directory to store results, figures, logs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = pathlib.Path(__file__).resolve().parent.parent

    if args.smoke_test:
        cfg_path = root / "config" / "smoke_test.yaml"
    else:
        cfg_path = root / "config" / "full_experiment.yaml"

    run_all(str(cfg_path), args.results_dir)
