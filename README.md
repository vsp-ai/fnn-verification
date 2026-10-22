# FNN Verification (fnn2026)

Verify and evaluate the FNN and NN policies of the paper with [PlaJA](fnn-plaja) on 33 JANI
benchmarks, and train new FNNs.

## Setup

```bash
git clone --recurse-submodules https://github.com/vsp-ai/fnn-verification.git
cd fnn-verification
scripts/setup.py
```

MILP verification needs a Gurobi license (`~/gurobi.lic` or `GRB_LICENSE_FILE`).
Tested on Ubuntu 24.04 and 25.10.

## Benchmarks

`fnn-benchmarks/` has one folder per instance with the JANI model, the property, the NN
policies (`*.nnet`, `*.jani2nnet`) and the FNN policies (`fnn/fnn-<teacher>-<loss>.json`).
`fnn-benchmarks/index.csv` lists all policies (`verified=yes` for the 164 verified in the
paper) and `fnn-benchmarks/reference.csv` has the paper's results.

## Verify and evaluate

```bash
scripts/verify_subset.py                                            # 10 runs, about an hour
scripts/verify.py blocksworld_ci_4_3/fnn/fnn-nn-64_64-mse.json z3   # FNN: z3 or milp
scripts/verify.py blocksworld_ci_4_3/nn-32_32.nnet marabou          # NN: marabou
scripts/evaluate.py blocksworld_ci_4_3/fnn/fnn-nn-64_64-mse.json
scripts/evaluate.py all                                             # all 260 policies
scripts/verify_all.py --jobs 16                                     # all 263 runs of the paper
```

Results are written to `results/results.csv`.

`verify.py` and `verify_subset.py` stop a run after 1 hour (`TIMEOUT=<seconds>`),
`verify_all.py` after 12 hours as in the paper. The full run takes about 2,200 CPU-hours.

Without a Gurobi license, MILP runs are skipped and Marabou uses its native LP solver instead
of Gurobi.

## Train an FNN

```bash
scripts/train.py blocksworld_ci_4_3 --name demo
scripts/verify.py trained/blocksworld_ci_4_3/demo.json z3
```

See `scripts/train.py -h` for the options (teacher, loss, structure, ...).

## PB verification

Needs RoundingSat with the IPASIR-PB interface, which is not public yet. With it:
`scripts/setup.py --roundingsat <dir>`, then verify with `pb-flat` or `pb-inc`.

## License

PlaJA: see `fnn-plaja/`, a reduced copy of the
[original PlaJA code base](https://gitlab.cs.uni-saarland.de/fai/code/plaja).
The FNN learner in `rrl/` is a reduced copy of the
[original RRL code base](https://github.com/12wang3/rrl) (Wang et al., 2021, MIT).

## Citation

```bibtex
@article{scherer-putrich-et-al-socs2026,
  title   = {Formula Neural Networks as Action Policies in Safety-Critical Planning: An Empirical Evaluation},
  author  = {Scherer Putrich, Victor and Bauerschmidt, Tim Nico and Jain, Chaahat and Lin, Songtuan and Hoffmann, Joerg and Valera, Isabel},
  journal = {Proceedings of the International Symposium on Combinatorial Search},
  volume  = {19},
  number  = {1},
  pages   = {240--245},
  year    = {2026},
  doi     = {10.1609/socs.v19i1.43095}
}
```
