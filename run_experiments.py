from __future__ import annotations

import pandas as pd

from src.data import load_open_bandit_feedback, train_eval_split
from src.ope import (
    estimate_dm,
    estimate_dr,
    estimate_ips,
    estimate_snips,
    fit_reward_model,
    get_logged_action_policy_prob,
    predict_expected_rewards,
)
from src.policies import (
    epsilon_greedy_popularity_policy,
    linear_thompson_sampling_policy,
    linucb_policy,
    uniform_random_policy,
)


def evaluate_policy(name: str, action_dist, eval_feedback, q_hat) -> dict:
    reward = eval_feedback["reward"]
    pscore = eval_feedback["pscore"]
    action = eval_feedback["action"]
    pi_logged = get_logged_action_policy_prob(action_dist, action)

    return {
        "policy": name,
        "behavior_mean_reward": reward.mean(),
        "ips": estimate_ips(reward, pscore, pi_logged),
        "snips": estimate_snips(reward, pscore, pi_logged),
        "dm": estimate_dm(action_dist, q_hat),
        "dr": estimate_dr(reward, pscore, action, action_dist, q_hat),
        "logged_action_match_rate": (pi_logged > 0).mean(),
        "mean_importance_weight": (pi_logged / pscore).mean(),
    }


def main() -> None:
    feedback = load_open_bandit_feedback(behavior_policy="random", campaign="all")
    train_feedback, eval_feedback = train_eval_split(feedback, eval_size=0.5, seed=12345)

    reward_model = fit_reward_model(train_feedback)
    q_hat = predict_expected_rewards(reward_model, eval_feedback)

    policies = [("uniform_random", uniform_random_policy(eval_feedback))]

    eps_policy, best_action = epsilon_greedy_popularity_policy(
        train_feedback,
        eval_feedback,
        epsilon=0.2,
    )
    policies.append((f"epsilon_greedy_popularity_best_action_{best_action}", eps_policy))
    policies.append(("linucb_alpha_0.5", linucb_policy(train_feedback, eval_feedback, alpha=0.5)))
    policies.append(
        (
            "linear_thompson_sampling",
            linear_thompson_sampling_policy(train_feedback, eval_feedback, sample_scale=0.2),
        )
    )

    rows = [evaluate_policy(name, action_dist, eval_feedback, q_hat) for name, action_dist in policies]
    results = pd.DataFrame(rows)

    print("\nDataset summary")
    print(f"train rounds: {train_feedback['n_rounds']}")
    print(f"eval rounds:  {eval_feedback['n_rounds']}")
    print(f"n_actions:    {feedback['n_actions']}")
    print(f"context dim:  {feedback['context'].shape[1]}")

    print("\nOPE results")
    print(results.to_string(index=False))
    results.to_csv("experiment_results.csv", index=False)
    print("\nSaved results to experiment_results.csv")


if __name__ == "__main__":
    main()
