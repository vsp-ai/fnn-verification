#!/usr/bin/env python3
"""Train an FNN from a teacher policy and write it to trained/<instance>/<name>.json.

Examples:
  scripts/train.py blocksworld_ci_4_3 --name demo
  scripts/train.py transport_8_8 --teacher Transport --loss ce
"""

import argparse
import csv
import os
import subprocess

from common import BENCH, DOCKER_ARGS, RRL_IMAGE, ROOT, read_index, run_plaja


def train(args):
    teacher_row = None
    for row in read_index(f"{BENCH}/index.csv"):
        if row["instance"] == args.instance and row["kind"] == "fnn" and row["teacher"] == args.teacher:
            teacher_row = row
            break
    if teacher_row is None:
        raise SystemExit(f"No teacher {args.teacher} for instance {args.instance} (see {BENCH}/index.csv)")
    interface = f"{BENCH}/{teacher_row['interface']}"
    model = f"{BENCH}/{teacher_row['model']}"
    prop = f"{BENCH}/{teacher_row['property']}"

    episodes = args.episodes
    if episodes is None:
        if args.instance in ("beluga_5_2_2", "beluga_6_3_3", "transport_15_10"):
            episodes = 20000
        else:
            episodes = 10000

    name = args.name
    if name is None:
        name = f"fnn-{args.teacher}-{args.loss}-{args.structure.replace('@', '_')}-s{args.seed}"
        if args.use_not:
            name = name + "-not"

    if args.loss == "ce":
        task = "classification"
    else:
        task = "regression"

    work = f"results/train/{args.instance}/{name}"
    policy = f"trained/{args.instance}/{name}.json"
    os.makedirs(work, exist_ok=True)
    if os.path.exists(f"{work}/trace.json"):
        os.remove(f"{work}/trace.json")

    print(f"Collecting traces of {args.teacher} ({episodes} episodes) ...")
    arguments = ["fnn-plaja/build/PlaJA", "--engine", "POLICY_EVALUATION", "--prop", "1",
                 "--model-file", model, "--additional-properties", prop, "--nn-interface", interface,
                 "--initial-state-enum", "sample", "--applicability-filtering", "1", "--terminate-cycles",
                 "--num-episodes", str(episodes), "--max-len-episode", "1250", "--seed", str(args.seed),
                 "--store-traces", f"{work}/trace.json", "--store-traces-mode", task]
    if args.teacher.startswith("nn-"):
        arguments += ["--nn", f"{BENCH}/{args.instance}/{args.teacher}.nnet"]
    else:
        arguments += ["--expert-policy", args.teacher]
    run_plaja(arguments, f"{work}/plaja.log")
    if not os.path.exists(f"{work}/trace.json") or os.path.getsize(f"{work}/trace.json") == 0:
        raise SystemExit(f"PlaJA failed, see {work}/plaja.log")

    docker = ["docker", "run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}", "-e", "OMP_NUM_THREADS=1",
              "-v", f"{ROOT}:/fnn", "-w", "/fnn"] + DOCKER_ARGS + [RRL_IMAGE, "python"]

    print("Building the dataset ...")
    subprocess.run(docker + ["rrl/make_dataset.py", f"{work}/trace.json", model, interface, f"{work}/dataset"], check=True)

    print(f"Training ({args.loss}, {args.mode}, {args.structure}, seed {args.seed}, {args.epochs} epochs; "
          f"log: {work}/train.log) ...")
    command = docker + ["rrl/train.py", f"{work}/dataset", "--jani", model, "--iface", interface, "--out", policy,
                        "--mode", args.mode, "--loss", args.loss, "--kl-temp", f"{args.kl_temp:g}",
                        "--structure", args.structure, "--seed", str(args.seed), "--epochs", str(args.epochs)]
    if args.use_not:
        command += ["--use-not"]
    with open(f"{work}/train.log", "w") as log:
        finished = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    with open(f"{work}/train.log") as log:
        lines = log.read().splitlines()
    if finished.returncode != 0:
        print("\n".join(lines[-5:]))
        raise SystemExit("Training failed.")
    print(lines[-1])

    os.makedirs("trained", exist_ok=True)
    columns = ["instance", "kind", "policy", "interface", "model", "property", "teacher", "loss", "verified", "run_id"]
    rows = []
    if os.path.exists("trained/index.csv"):
        rows = [row for row in read_index("trained/index.csv") if row["policy"] != policy]
    rows.append({"instance": args.instance, "kind": "fnn", "policy": policy,
                 "interface": teacher_row["interface"], "model": teacher_row["model"], "property": teacher_row["property"],
                 "teacher": args.teacher, "loss": args.loss, "verified": "", "run_id": name})
    with open("trained/index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Verify it with: scripts/verify.py {policy} z3")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("instance", help="an instance folder in fnn-benchmarks/, e.g. blocksworld_ci_4_3")
    parser.add_argument("--teacher", default="nn-32_32",
                        help="nn-32_32 (default), nn-64_64, or the instance's hand-designed policy "
                             "(Transport, Transport_Feat, OneWayLinePark, TwoWayLinePark)")
    parser.add_argument("--loss", default="mse", choices=["mse", "kl", "ce"])
    parser.add_argument("--kl-temp", type=float, default=10, help="teacher temperature for kl (default 10)")
    parser.add_argument("--mode", default="predicates", choices=["predicates", "onehot"])
    parser.add_argument("--structure", default="0@64", help="0@<width>[@<width>...] (default 0@64)")
    parser.add_argument("--use-not", action="store_true", help="also learn negations (needs two or more layers)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, help="teacher episodes (default 10000, 20000 for the largest instances)")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--name", help="default fnn-<teacher>-<loss>-<structure>-s<seed>")
    train(parser.parse_args())
