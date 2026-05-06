# Contextual Bandits for E-Commerce Recommendation

This project studies contextual bandit policies and off-policy evaluation (OPE)
on the Open Bandit Dataset collected from ZOZOTOWN.

The main question is:

> Can we estimate how a new recommendation policy would perform using only
> logged historical bandit feedback?

## Project Structure

```text
.
|-- Project Proposal.pdf
|-- load_zozotown.py
|-- ips_baseline.py
|-- run_experiments.py
|-- run_policy_sweep.py
|-- train_and_save_results.ipynb
|-- visualize_model_performance.ipynb
|-- requirements.txt
`-- src
    |-- __init__.py
    |-- data.py
    |-- evaluation.py
    |-- features.py
    |-- obp_adapter.py
    |-- ope.py
    `-- policies.py
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
python run_policy_sweep.py
```

Open `train_and_save_results.ipynb` to train a reward model, use CUDA when a
CUDA-enabled PyTorch install is available, and save fresh CSV files for
visualization. The notebooks are configured to use the `Python (RL_Final)`
kernel when that conda environment is registered.

The core experiment saves a result table to `experiment_results.csv`. It includes
hand-written IPS, SNIPS, DM, and DR estimates, bootstrap confidence intervals,
importance-weight diagnostics, and parity columns from OBP's official
`OffPolicyEvaluation` API when OBP is installed. The evaluated policies include
uniform random, epsilon-greedy popularity, LinUCB, Logistic UCB, and Linear
Thompson Sampling.

The policy sweep saves a wider parameter-search table to
`policy_sweep_results.csv`.

Open `visualize_model_performance.ipynb` to compare policy values, confidence
intervals, OPE stability diagnostics, parameter sweeps, and OBP parity checks.

Useful options:

```bash
python run_experiments.py --bootstrap-samples 500
python run_experiments.py --skip-obp
python run_policy_sweep.py --bootstrap-samples 100 --output sweep_with_ci.csv
```
