#!/usr/bin/env python3
"""Evaluate policies by simulation (EPISODES episodes, default 50000) and append the results to results/results.csv.

Examples:
  scripts/evaluate.py blocksworld_ci_4_3/fnn/fnn-nn-64_64-mse.json blocksworld_ci_4_3/nn-32_32.nnet
  scripts/evaluate.py all
"""

import argparse
import os

from common import BENCH, EPISODES, find_policy, policy_file, read_index, read_stats, run_plaja, save_result


def evaluate(policy):
    row = find_policy(policy)
    file = policy_file(policy)
    model = f"{BENCH}/{row['model']}"
    prop = f"{BENCH}/{row['property']}"
    interface = f"{BENCH}/{row['interface']}"

    name = policy.replace("/", "_") + ".eval"
    log = f"results/runs/{name}.log"
    stats = f"results/runs/{name}.csv"
    os.makedirs("results/runs", exist_ok=True)
    if os.path.exists(stats):
        os.remove(stats)
    print(f"Evaluating {policy} ({EPISODES} episodes) ...")

    arguments = ["fnn-plaja/build/PlaJA", "--engine", "POLICY_EVALUATION", "--prop", "1",
                 "--model-file", model, "--additional-properties", prop,
                 "--initial-state-enum", "sample", "--applicability-filtering", "1",
                 "--num-episodes", str(EPISODES), "--seed", "42",
                 "--print-stats", "--stats-file", stats]
    if row["kind"] == "fnn":
        arguments += ["--fnn-interface", interface, "--fnn", file]
    else:
        arguments += ["--nn-interface", interface, "--nn", file]
    run_plaja(arguments, log)

    result = read_stats(stats)
    if not result.get("Episodes"):
        save_result(policy, "eval", "error")
        print(f"  error (log: {log})")
        return
    episodes = float(result["Episodes"])
    goal_rate = float(result["Goal"]) / episodes
    fail_rate = float(result["Failure"]) / episodes
    save_result(policy, "eval", "", result["SearchTime"], f"{goal_rate:.5f}", f"{fail_rate:.5f}")
    print(f"  goal {goal_rate:.5f}  failure {fail_rate:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("policies", nargs="+", help="policies (FNN .json or NN .nnet), or 'all'")
    args = parser.parse_args()

    if args.policies == ["all"]:
        policies = [row["policy"] for row in read_index(f"{BENCH}/index.csv")]
    else:
        policies = args.policies
    for policy in policies:
        evaluate(policy)
