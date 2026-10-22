"""Train an FNN on a dataset written by make_dataset.py and export it for PlaJA.

Usage: python rrl/train.py <dataset> --jani <model.jani> --iface <interface.jani2nnet> --out <fnn.json>
                           [--mode predicates|onehot] [--loss mse|kl|ce] [--kl-temp 10]
                           [--structure 0@64] [--seed 42] [--epochs 25] [--use-not]
<dataset> is the path without extension of <dataset>.data and <dataset>.info.
The remaining settings are the paper's and fixed below.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split

from export import export_fnn
from model import Net, fidelity, train
from predicates import evaluate_predicates, parse_problem_predicates, predicate_to_str

BATCH_SIZE, LR, LR_DECAY_RATE, LR_DECAY_EPOCH = 128, 0.01, 0.75, 10
ALPHA, BETA, GAMMA, TEMPERATURE = 0.999, 8, 1, 1.0  # NLAF and softmax temperature (ce)


def read_info(path):
    columns, task = [], None
    for line in Path(path).read_text().splitlines():
        tokens = line.split()
        if len(tokens) == 2 and tokens[0].upper() == "TASK":
            task = tokens[1].lower()
        elif len(tokens) == 3:
            columns.append(tokens)  # name, type, role
    return columns, task


def load_dataset(dataset, mode, jani, iface):
    """Atoms X, teacher targets y (one-hot labels or action scores), atom and action names, task."""
    columns, task = read_info(f"{dataset}.info")
    D = pd.read_csv(f"{dataset}.data", header=None)
    D = D.sample(frac=1, random_state=0).reset_index(drop=True)
    D.columns = [name for name, _, _ in columns]
    features = [name for name, _, role in columns if role == "feature"]
    targets = [name for name, _, role in columns if role == "target"]
    X_df, y_df = D[features], D[targets]

    if task == "classification":
        labels = y_df.values.reshape(-1)
        y = (labels[:, None] == np.unique(labels)[None, :]).astype(np.float32)
    else:
        y = y_df.values.astype(np.float32)
    actions = json.loads(Path(iface).read_text())["output"]

    if mode == "onehot":  # one atom "<variable> eq <value>" per value seen in the data
        atoms, blocks = [], []
        for f in features:
            values = X_df[f].values
            categories = np.unique(values)
            blocks.append(values[:, None] == categories[None, :])
            atoms += [f"{f} eq {c}" for c in categories]
        X = np.concatenate(blocks, axis=1).astype(np.float32)
    else:  # predicates of the guards and the property, over the interface variables
        var_index = {name: i for i, name in enumerate(features)}
        predicates, kinds = parse_problem_predicates(jani)
        predicates = [p for p in predicates if all(v in var_index for v in p.vars)]
        atoms = [predicate_to_str(p, "+".join(sorted(kinds[(p.op, p.vars, p.coeffs, p.const)])))
                 for p in predicates]
        X = evaluate_predicates(X_df.values, predicates, var_index)
    return X, y, atoms, actions, task


def split(X, y):
    """Fold 0 of a fixed 5-fold split; 80% of its training part for training, 20% for validation."""
    n = len(X)
    order = np.arange(n)
    np.random.RandomState(0).shuffle(order)
    test = np.zeros(n, dtype=bool)
    test[order[:n // 5 + (1 if n % 5 else 0)]] = True
    train_set = Subset(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), np.arange(n)[~test])
    train_len = int(len(train_set) * 0.80)
    return random_split(train_set, [train_len, len(train_set) - train_len])


def main():
    ap = argparse.ArgumentParser(description="Train an FNN and export it for PlaJA.")
    ap.add_argument("dataset")
    ap.add_argument("--jani", required=True)
    ap.add_argument("--iface", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["predicates", "onehot"], default="predicates")
    ap.add_argument("--loss", choices=["mse", "kl", "ce"], default="mse")
    ap.add_argument("--kl-temp", type=float, default=10.0, help="teacher temperature of the kl loss")
    ap.add_argument("--structure", default="0@64", help="0@<width>[@<width>...]: logical layer widths")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--use-not", action="store_true", help="also use negated atoms")
    args = ap.parse_args()
    parts = args.structure.split("@")
    widths = [int(w) for w in parts[1:]]
    if parts[0] != "0" or not widths or min(widths) <= 0:
        ap.error("--structure must be 0@<width>[@<width>...]")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    X, y, atoms, actions, task = load_dataset(args.dataset, args.mode, args.jani, args.iface)
    if (task == "classification") != (args.loss == "ce"):
        raise SystemExit(f"Loss {args.loss} does not fit a {task} dataset (ce needs classification).")
    train_set, valid_set = split(X, y)
    print(f"{len(X)} samples ({len(train_set)} train, {len(valid_set)} valid), {len(atoms)} atoms, "
          f"{len(actions)} actions, device {device}", flush=True)

    net = Net(len(atoms), widths, len(actions), args.use_not, ALPHA, BETA, GAMMA, TEMPERATURE).to(device)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)
    train(net, train_loader, valid_loader, device, args.loss, args.epochs, LR, LR_DECAY_RATE, LR_DECAY_EPOCH,
          args.kl_temp)

    fnn = export_fnn(net, atoms, actions, DataLoader(train_set, batch_size=BATCH_SIZE), device, args.mode)
    fnn["training_metadata"] = {"mode": args.mode, "loss": args.loss,
                                "distillation_temperature": args.kl_temp if args.loss == "kl" else None,
                                "structure": args.structure, "seed": args.seed, "epochs": args.epochs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(fnn, indent=2, ensure_ascii=False) + "\n")
    print(f"valid fidelity {fidelity(net, valid_loader, device):.4f}, {len(fnn['rules'])} rules -> {args.out}")


if __name__ == "__main__":
    main()
