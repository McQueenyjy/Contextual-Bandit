from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

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
    for name, action_dist in build_core_policies(train_feedback, eval_feedback):
        print(f"evaluating {name}")
        rows.append(
            evaluate_policy(
                name,
                action_dist,
                eval_feedback,
                q_hat,
                bootstrap_samples=args.bootstrap_samples,
                alpha=args.alpha,
                seed=args.seed,
                include_obp=not args.skip_obp,
            )
        )
        del action_dist
        gc.collect()

    results = pd.DataFrame(rows)

    print("\nDataset summary")
    print(f"train rounds: {train_feedback['n_rounds']}")
    print(f"eval rounds:  {eval_feedback['n_rounds']}")
    print(f"n_actions:    {feedback['n_actions']}")
    print(f"context dim:  {feedback['context'].shape[1]}")

    print("\nOPE results")
    print(results.to_string(index=False))
    results.to_csv(args.output, index=False)
    print(f"\nSaved results to {args.output}")
    write_metadata(args, feedback, train_feedback, eval_feedback)


def build_core_policies(train_feedback: dict, eval_feedback: dict):
    yield ("uniform_random", uniform_random_policy(eval_feedback))

    eps_policy, best_action = epsilon_greedy_popularity_policy(
        train_feedback,
        eval_feedback,
        epsilon=0.2,
    )
    yield (f"epsilon_greedy_popularity_best_action_{best_action}", eps_policy)
    yield ("linucb_alpha_0.5", linucb_policy(train_feedback, eval_feedback, alpha=0.5))
    yield ("logistic_ucb_alpha_0.5", logistic_ucb_policy(train_feedback, eval_feedback, alpha=0.5))
    yield (
        "linear_thompson_sampling",
        linear_thompson_sampling_policy(train_feedback, eval_feedback, sample_scale=0.2),
    )


def write_metadata(args: argparse.Namespace, feedback: dict, train_feedback: dict, eval_feedback: dict) -> None:
    if not args.metadata_output:
        return
    metadata_path = Path(args.metadata_output)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "seed": args.seed,
        "behavior_policy": args.behavior_policy,
        "campaign": args.campaign,
        "eval_size": args.eval_size,
        "data_path": args.data_path,
        "dataset_summary": {
            "n_rounds": int(feedback["n_rounds"]),
            "train_rounds": int(train_feedback["n_rounds"]),
            "eval_rounds": int(eval_feedback["n_rounds"]),
            "n_actions": int(feedback["n_actions"]),
            "context_dim": int(feedback["context"].shape[1]),
            "action_context_dim": int(feedback["action_context"].shape[1]),
            "behavior_mean_reward_eval": float(eval_feedback["reward"].mean()),
            "positive_rewards_eval": int(eval_feedback["reward"].sum()),
        },
        "reward_model_backend": "sklearn_logistic_regression",
        "device_used": "cpu",
        "core_bootstrap_samples": args.bootstrap_samples,
        "include_obp_parity": not args.skip_obp,
        "outputs": {
            "experiment_results": args.output,
            "artifact_dir": str(metadata_path.parent),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved metadata to {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the core contextual bandit OPE experiment.")
    parser.add_argument("--behavior-policy", default="random", help="Open Bandit Dataset behavior policy.")
    parser.add_argument("--campaign", default="all", help="Open Bandit Dataset campaign.")
    parser.add_argument("--data-path", default=None, help="Optional local Open Bandit Dataset path.")
    parser.add_argument("--eval-size", type=float, default=0.5, help="Fraction of rounds used for OPE.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for splitting and bootstrap.")
    parser.add_argument("--bootstrap-samples", type=int, default=200, help="Bootstrap samples for confidence intervals.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Confidence interval alpha.")
    parser.add_argument("--skip-obp", action="store_true", help="Skip official OBP estimator parity columns.")
    parser.add_argument("--output", default="experiment_results.csv", help="Output CSV path.")
    parser.add_argument("--predict-batch-size", type=int, default=8192, help="Rows per reward-prediction batch.")
    parser.add_argument(
        "--metadata-output",
        default="training_artifacts/training_metadata.json",
        help="Optional metadata JSON output path. Use an empty string to skip.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
