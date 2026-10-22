"""Convert a PlaJA policy-evaluation trace (--store-traces) into an RRL dataset.

Usage: python rrl/make_dataset.py <trace.json> <model.jani> <interface.jani2nnet> <dataset>
Writes <dataset>.data and <dataset>.info. Classification traces store the teacher's
action, regression traces its action scores. Values of the (bounded integer) state
variables that never occur in the trace get one synthetic row each (and, for
classification, actions that never occur), as in the paper.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 0


def load_bounds(jani_path, features):
    j = json.loads(Path(jani_path).read_text())
    consts = {c["name"]: c["value"] for c in j["constants"]}
    types = {v["name"]: v["type"] for v in j["variables"]}
    bound = lambda b: int(consts[b]) if isinstance(b, str) else int(b)
    return {f: (bound(types[f]["lower-bound"]), bound(types[f]["upper-bound"])) for f in features}


def augment(df, features, bounds, targets, n_actions, classification, rng):
    values = df[features].to_numpy(dtype=float)
    rows = []
    for idx, f in enumerate(features):
        lo, hi = bounds[f]
        seen = set(int(round(v)) for v in df[f].unique())
        for value in (v for v in range(lo, hi + 1) if v not in seen):
            base = values[rng.integers(0, len(values))].copy()
            base[idx] = value
            row = {g: int(round(base[j])) for j, g in enumerate(features)}
            if classification:
                row["class"] = int(rng.choice(df["class"].unique().tolist()))
            else:
                src = int(rng.integers(0, len(df)))
                row.update({t: float(df.iloc[src][t]) for t in targets})
            rows.append(row)
    if rows:
        df = pd.concat([df, pd.DataFrame(rows, columns=features + targets)], ignore_index=True)
    if classification:
        missing = [a for a in range(n_actions) if a not in set(int(v) for v in df["class"].unique())]
        if missing:
            pool = df[features].to_numpy(dtype=float)
            label_rows = []
            for a in missing:
                base = pool[rng.integers(0, len(pool))].copy()
                label_rows.append({**{g: int(round(base[j])) for j, g in enumerate(features)}, "class": a})
            df = pd.concat([df, pd.DataFrame(label_rows, columns=features + ["class"])], ignore_index=True)
    return df


def main():
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    trace_path, jani_path, iface_path, dataset = sys.argv[1:]
    rng = np.random.default_rng(SEED)
    trace = json.loads(Path(trace_path).read_text())
    iface = json.loads(Path(iface_path).read_text())
    features, actions = [e["name"] for e in iface["input"]], iface["output"]
    bounds = load_bounds(jani_path, features)
    for i, row in enumerate(trace):
        if len(row["state"]) != len(features):
            raise ValueError(f"Sample {i}: state width {len(row['state'])} != interface width {len(features)}")
    df = pd.DataFrame([[int(round(float(v))) for v in row["state"]] for row in trace], columns=features)
    classification = not isinstance(trace[0]["action"], list)
    if classification:
        labels = [int(row["action"]) for row in trace]
        if any(a < 0 or a >= len(actions) for a in labels):
            raise ValueError("Action labels out of the interface's range")
        df["class"] = labels
        targets = ["class"]
    else:
        if any(len(row["action"]) != len(actions) for row in trace):
            raise ValueError("Action-score vector length does not match the interface")
        targets = actions
        for i, t in enumerate(targets):
            df[t] = [float(row["action"][i]) for row in trace]

    n = len(df)
    df = augment(df, features, bounds, targets, len(actions), classification, rng)
    Path(dataset).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{dataset}.data", header=False, index=False)
    with open(f"{dataset}.info", "w") as info:
        info.write("".join(f"{f} discrete feature\n" for f in features))
        if classification:
            info.write("class discrete target\nTASK classification\n")
        else:
            info.write("".join(f"{t} continuous target\n" for t in targets) + "TASK regression\n")
    print(f"{n} samples + {len(df) - n} augmented -> {dataset}.data")


if __name__ == "__main__":
    main()
