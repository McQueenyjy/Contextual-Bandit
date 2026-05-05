# Contextual Bandits for E-Commerce Recommendation

This project studies contextual bandit policies and off-policy evaluation (OPE)
on the Open Bandit Dataset collected from ZOZOTOWN.

The main question is:

> Can we estimate how a new recommendation policy would perform using only
> logged historical bandit feedback?

## Project Structure

```text
.
├── Project Proposal.pdf
├── load_zozotown.py
├── ips_baseline.py
├── run_experiments.py
├── requirements.txt
└── src
    ├── __init__.py
    ├── data.py
    ├── features.py
    ├── ope.py
    └── policies.py
```

## Setup

Use a clean Python 3.10 environment if possible.

```bash
conda create -n RL_final python=3.10 -y
conda activate RL_final
python -m pip install -r requirements.txt
```

## Run

```bash
python load_zozotown.py
python ips_baseline.py
python run_experiments.py
```

The full experiment saves a result table to `experiment_results.csv`.
