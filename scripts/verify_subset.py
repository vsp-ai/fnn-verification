#!/usr/bin/env python3
"""Verify ten of the paper's runs (about an hour).

Without a Gurobi license the four milp runs are skipped.
"""

import argparse

from verify import verify

SUBSET = [
    ("blocksworld_ci_4_3/fnn/fnn-nn-64_64-mse.json", "z3"),
    ("blocksworld_ci_4_3/fnn/fnn-nn-64_64-mse.json", "milp"),
    ("blocksworld_ci_4_3/nn-32_32.nnet", "marabou"),
    ("transport_8_8/fnn/fnn-Transport-mse.json", "z3"),
    ("transport_8_8/fnn/fnn-Transport-mse.json", "milp"),
    ("beluga_4_2_2/fnn/fnn-nn-32_32-mse.json", "z3"),
    ("beluga_4_2_2/fnn/fnn-nn-32_32-mse.json", "milp"),
    ("beluga_4_2_2/nn-32_32.nnet", "marabou"),
    ("owl_det_8_8/fnn/fnn-nn-64_64-mse.json", "z3"),
    ("owl_det_8_8/fnn/fnn-nn-64_64-mse.json", "milp"),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    for policy, check in SUBSET:
        verify(policy, check)
