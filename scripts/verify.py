#!/usr/bin/env python3
"""Verify one policy and append the result to results/results.csv.

Examples:
  scripts/verify.py blocksworld_ci_4_3/fnn/fnn-nn-64_64-mse.json z3
  scripts/verify.py blocksworld_ci_4_3/nn-32_32.nnet marabou
  scripts/verify.py trained/blocksworld_ci_4_3/demo.json z3
"""

import argparse
import os
import re

from common import BENCH, GUROBI_LICENSE, TIMEOUT, find_policy, has_license, policy_file, read_stats, run_plaja, save_result

PAPER_OPTIONS = [
    "--engine", "PA_CEGAR",
    "--prop", "1",
    "--applicability-filtering", "1",
    "--heuristic-search", "hamming",
    "--check-for-pa-terminal-states", "true",
    "--set-pa-goal-objective-terminal", "false",
    "--pa-state-aware", "true",
    "--split-point", "bs",
    "--selection-refinement", "all",
    "--entailment-mode", "entailment-only",
    "--check-policy-spuriousness", "false",
    "--incremental-search", "true",
    "--seed", "42",
]


def verify(policy, check, timeout=TIMEOUT):
    row = find_policy(policy)
    file = policy_file(policy)
    model = f"{BENCH}/{row['model']}"
    prop = f"{BENCH}/{row['property']}"
    interface = f"{BENCH}/{row['interface']}"

    name = policy.replace("/", "_") + "." + check
    log = f"results/runs/{name}.log"
    stats = f"results/runs/{name}.csv"
    os.makedirs("results/runs", exist_ok=True)
    if os.path.exists(stats):
        os.remove(stats)
    print(f"Verifying {policy} with {check} ...")

    # PlaJA stops itself after the time limit; the outer timeout only catches runs that do not.
    plaja = ["timeout", "-k", "60", str(timeout + 300), "fnn-plaja/build/PlaJA"]
    common_options = PAPER_OPTIONS + ["--max-time", str(timeout), "--model-file", model, "--additional-properties", prop]
    stats_options = ["--print-stats", "--stats-file", stats]

    if check == "z3":
        arguments = plaja + common_options + ["--fnn-sat", "z3", "--fnn-interface", interface, "--fnn", file] + stats_options

    elif check == "milp":
        if not has_license():
            print(f"Skipping: MILP needs a Gurobi license ({GUROBI_LICENSE} not found).")
            return
        arguments = plaja + common_options + ["--fnn-sat", "milp", "--fnn-interface", interface, "--fnn", file] + stats_options

    elif check == "pb-flat":
        arguments = plaja + common_options + ["--fnn-sat", "pb-constraint", "--use-incremental-pb", "false",
                                              "--fnn-interface", interface, "--fnn", file] + stats_options

    elif check == "pb-inc":
        arguments = plaja + common_options + ["--fnn-sat", "pb-constraint", "--use-incremental-pb", "true",
                                              "--incremental-pb-granularity", "label",
                                              "--fnn-interface", interface, "--fnn", file] + stats_options

    elif check == "marabou":
        if has_license():
            arguments = plaja + common_options + ["--nn-interface", interface, "--nn", file,
                                                  "--marabou-options", f"--timeout {timeout}"] + stats_options
        else:
            print("No Gurobi license: Marabou runs Reluplex with its native LP solver (the paper used Gurobi).")
            arguments = plaja + common_options + ["--nn-interface", interface, "--nn", file,
                                                  "--marabou-solver", "reluplex",
                                                  "--marabou-options", f"--timeout {timeout} --lp-solver native"] + stats_options

    exit_code = run_plaja(arguments, log)

    # PlaJA reports HasGoalPath=1 if an unsafe path exists and 0 if the policy is safe.
    result = read_stats(stats)
    goal_path = result.get("HasGoalPath", "")
    runtime = result.get("SearchTime", "")
    if goal_path == "1":
        verdict = "unsafe"
    elif goal_path == "0":
        verdict = "safe"
    elif exit_code in (0, 124, 137):
        verdict = "timeout"
    else:
        verdict = "error"

    save_result(policy, check, verdict, runtime)
    print(f"  {verdict} {runtime}  (log: {log})")
    if verdict == "error":
        with open(log) as f:
            for line in f:
                if re.search("FATAL|rror", line):
                    print("  " + line.strip())
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("policy", help="a policy from fnn-benchmarks/index.csv, or trained/<instance>/<name>.json")
    parser.add_argument("check", choices=["z3", "milp", "pb-flat", "pb-inc", "marabou"],
                        help="z3, milp, pb-flat or pb-inc for FNN policies (.json); marabou for NN policies (.nnet)")
    args = parser.parse_args()
    verify(args.policy, args.check)
