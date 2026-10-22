#!/usr/bin/env python3
"""Verify all 263 runs of the paper (12 hours each, about 2,200 CPU-hours in total)."""

import argparse
from concurrent.futures import ThreadPoolExecutor

from common import BENCH, read_index
from verify import verify

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=int, default=1, help="verifications to run in parallel (default 1)")
    parser.add_argument("--timeout", type=int, default=43200, help="seconds per run (default 43200, i.e. 12 hours)")
    args = parser.parse_args()

    runs = []
    for row in read_index(f"{BENCH}/index.csv"):
        if row["verified"] != "yes":
            continue
        if row["kind"] == "fnn":
            runs.append((row["policy"], "z3"))
            runs.append((row["policy"], "milp"))
        else:
            runs.append((row["policy"], "marabou"))

    print(f"Running {len(runs)} verifications, {args.timeout} s each at most, with {args.jobs} job(s) ...")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for policy, check in runs:
            pool.submit(verify, policy, check, args.timeout)
