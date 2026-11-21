#!/usr/bin/env bash
# The whole pipeline. Pass extra arguments through to the training step, e.g.
#   ./run_all.sh --n-repeats 3            quick pass while you are iterating
#   ./run_all.sh                          the full run
set -e

python scripts/01_audit.py
python scripts/02_equations.py
python scripts/03_reclassification.py

for et in raw legacy neutral ownfit_race ownfit_norace; do
    python scripts/04_train.py --equation-type "$et" "$@"
done

python scripts/05_compare.py
python scripts/06_simulate.py
python scripts/07_construct.py
