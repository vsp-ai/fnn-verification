"""Settings (can be overridden from the environment) and helpers used by all scripts."""

import csv
import os
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = "fnn-benchmarks"

PLAJA_IMAGE = os.environ.get("PLAJA_IMAGE", "faisaarlab/fnn2026-deps:v1")
RRL_IMAGE = os.environ.get("RRL_IMAGE", "faisaarlab/fnn2026-rrl:v1")
TIMEOUT = int(os.environ.get("TIMEOUT", "3600"))  # seconds per verification run
EPISODES = int(os.environ.get("EPISODES", "50000"))
GUROBI_LICENSE = Path(os.environ.get("GRB_LICENSE_FILE", Path.home() / "gurobi.lic"))
DOCKER_ARGS = os.environ.get("DOCKER_ARGS", "").split()

os.chdir(ROOT)
sys.stdout.reconfigure(line_buffering=True)


def has_license():
    return GUROBI_LICENSE.is_file()


def run_plaja(arguments, log_file):
    """Run a command in the PlaJA image with the repository mounted, and return its exit code."""
    user = f"{os.getuid()}:{os.getgid()}"
    command = ["docker", "run", "--rm", "-u", user, "-v", f"{ROOT}:/fnn", "-w", "/fnn"]
    command += ["-v", f"{ROOT}/.cache/gurobi:/workspace/dependencies/gurobi:ro"]
    if has_license():
        command += ["-v", f"{GUROBI_LICENSE}:/gurobi.lic:ro", "-e", "GRB_LICENSE_FILE=/gurobi.lic"]
    roundingsat_file = Path("fnn-plaja/build/roundingsat.path")
    if roundingsat_file.exists():
        roundingsat = roundingsat_file.read_text().strip()
        command += ["-v", f"{roundingsat}:/workspace/dependencies/roundingsat:ro"]
    command += DOCKER_ARGS
    command += [PLAJA_IMAGE]
    command += arguments
    with open(log_file, "w") as log:
        return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode


def read_index(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def find_policy(policy):
    """The index row of a policy: from fnn-benchmarks/index.csv, or trained/index.csv for trained ones."""
    if policy.startswith("trained/"):
        index = "trained/index.csv"
    else:
        index = f"{BENCH}/index.csv"
    if os.path.exists(index):
        for row in read_index(index):
            if row["policy"] == policy:
                return row
    raise SystemExit(f"Unknown policy: {policy} (see {index})")


def policy_file(policy):
    """Path of a policy file, relative to the repository root."""
    if policy.startswith("trained/"):
        return policy
    return f"{BENCH}/{policy}"


def read_stats(path):
    """The last row of a PlaJA statistics file, as a dict. Empty if the file does not exist."""
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return {}
    header = rows[0]
    last = rows[-1]
    stats = {}
    for name, value in zip(header, last):
        if name not in stats:  # a few columns appear twice; the first is the overall value
            stats[name] = value
    return stats


_results_lock = threading.Lock()


def save_result(policy, check, verdict="", runtime="", goal_rate="", fail_rate=""):
    """Append one line to results/results.csv."""
    os.makedirs("results/runs", exist_ok=True)
    with _results_lock:
        new_file = not os.path.exists("results/results.csv")
        with open("results/results.csv", "a", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            if new_file:
                writer.writerow(["policy", "check", "verdict", "runtime_s", "goal_rate", "fail_rate"])
            writer.writerow([policy, check, verdict, runtime, goal_rate, fail_rate])
