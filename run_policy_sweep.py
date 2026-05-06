from __future__ import annotations

import argparse

import pandas as pd

from src.data import load_open_bandit_feedback, train_eval_split
from src.evaluation import evaluate_policy
from src.ope import fit_reward_model, predict_expected_rewards
from src.policies import (
    epsilon_greedy_popularity_policy,
    linear_thompson_sampling_policy,
    linucb_policy,
    logistic_ucb_policy,
    uniform_random_policy,
)


def main() -> None:
    args = parse_args()
    feedback = load_open_bandit_feedback(
        behavior_policy=args.behavior_policy,
        campaign=args.campaign,
        data_path=args.data_path,
    )
    train_feedback, eval_feedback = train_eval_split(feedback, eval_size=args.eval_size, seed=args.seed)

    reward_model = fit_reward_model(train_feedback)
    q_hat = predict_expected_rewards(reward_model, eval_feedback, batch_size=args.predict_batch_size)

    rows = []
    for name, family, parameter, value, action_dist in build_policy_grid(train_feedback, eval_feedback, args.seed):
        row = evaluate_policy(
            name=name,
            action_dist=action_dist,
            eval_feedback=eval_feedback,
            q_hat=q_hat,
            bootstrap_samples=args.bootstrap_samples,
            alpha=args.alpha,
            seed=args.seed,
            include_obp=not args.skip_obp,
        )
        row["policy_family"] = family
        row["sweep_parameter"] = parameter
        row["sweep_value"] = value
        rows.append(row)
        del action_dist

    results = pd.DataFrame(rows)
    first_columns = ["policy_family", "sweep_parameter", "sweep_value", "policy"]
    results = results[first_columns + [column for column in results.columns if column not in first_columns]]

    print("\nPolicy sweep results")
    print(results.to_string(index=False))
    results.to_csv(args.output, index=False)
    print(f"\nSaved sweep results to {args.output}")


def build_policy_grid(train_feedback: dict, eval_feedback: dict, seed: int):
    yield ("uniform_random", "uniform_random", "none", 0.0, uniform_random_policy(eval_feedback))

    for epsilon in [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0]:
        action_dist, best_action = epsilon_greedy_popularity_policy(
            train_feedback,
            eval_feedback,
            epsilon=epsilon,
        )
        yield (
            f"epsilon_greedy_popularity_epsilon_{epsilon:g}_best_action_{best_action}",
            "epsilon_greedy_popularity",
            "epsilon",
            epsilon,
            action_dist,
        )

    for alpha in [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]:
        yield (
            f"linucb_alpha_{alpha:g}",
            "linucb",
            "alpha",
            alpha,
            linucb_policy(train_feedback, eval_feedback, alpha=alpha),
        )

    for alpha in [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]:
        yield (
            f"logistic_ucb_alpha_{alpha:g}",
            "logistic_ucb",
            "alpha",
            alpha,
            logistic_ucb_policy(train_feedback, eval_feedback, alpha=alpha),
        )

    for sample_scale in [0.05, 0.1, 0.2, 0.5, 1.0]:
        yield (
            f"linear_thompson_sampling_scale_{sample_scale:g}",
            "linear_thompson_sampling",
            "sample_scale",
            sample_scale,
            linear_thompson_sampling_policy(
                train_feedback,
                eval_feedback,
                sample_scale=sample_scale,
                seed=seed,
            ),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep target-policy parameters and evaluate them with OPE.")
    parser.add_argument("--behavior-policy", default="random", help="Open Bandit Dataset behavior policy.")
    parser.add_argument("--campaign", default="all", help="Open Bandit Dataset campaign.")
    parser.add_argument("--data-path", default=None, help="Optional local Open Bandit Dataset path.")
    parser.add_argument("--eval-size", type=float, default=0.5, help="Fraction of rounds used for OPE.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for splitting, policies, and bootstrap.")
    parser.add_argument("--bootstrap-samples", type=int, default=0, help="Bootstrap samples for confidence intervals.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Confidence interval alpha.")
    parser.add_argument("--skip-obp", action="store_true", help="Skip official OBP estimator parity columns.")
    parser.add_argument("--output", default="policy_sweep_results.csv", help="Output CSV path.")
    parser.add_argument("--predict-batch-size", type=int, default=8192, help="Rows per reward-prediction batch.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
