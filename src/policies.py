from __future__ import annotations

import numpy as np


def uniform_random_policy(feedback: dict) -> np.ndarray:
    """Uniform random target policy."""
    return np.ones((feedback["n_rounds"], feedback["n_actions"]), dtype=float) / feedback["n_actions"]


def epsilon_greedy_popularity_policy(
    train_feedback: dict,
    eval_feedback: dict,
    epsilon: float = 0.2,
) -> tuple[np.ndarray, int]:
    """Epsilon-greedy policy using historical mean reward per action."""
    n_actions = train_feedback["n_actions"]
    action = train_feedback["action"]
    reward = train_feedback["reward"]
    global_mean = float(np.mean(reward))
    mean_rewards = np.zeros(n_actions, dtype=float)

    for a in range(n_actions):
        mask = action == a
        mean_rewards[a] = float(np.mean(reward[mask])) if np.any(mask) else global_mean

    best_action = int(np.argmax(mean_rewards))
    action_dist = np.ones((eval_feedback["n_rounds"], n_actions), dtype=float) * epsilon / n_actions
    action_dist[:, best_action] += 1.0 - epsilon
    return action_dist, best_action


def linucb_policy(
    train_feedback: dict,
    eval_feedback: dict,
    alpha: float = 0.5,
    ridge_lambda: float = 1.0,
) -> np.ndarray:
    """Learn one linear model per action and choose actions by an optimistic score."""
    betas, inv_designs = _fit_per_action_linear_models(train_feedback, ridge_lambda)
    context = eval_feedback["context"]
    n_actions = eval_feedback["n_actions"]
    scores = np.zeros((eval_feedback["n_rounds"], n_actions), dtype=float)

    for a in range(n_actions):
        mean = context @ betas[a]
        uncertainty = np.sqrt(np.sum((context @ inv_designs[a]) * context, axis=1))
        scores[:, a] = mean + alpha * uncertainty

    return _one_hot(np.argmax(scores, axis=1), n_actions)


def linear_thompson_sampling_policy(
    train_feedback: dict,
    eval_feedback: dict,
    sample_scale: float = 0.2,
    ridge_lambda: float = 1.0,
    seed: int = 12345,
) -> np.ndarray:
    """Linear Thompson Sampling with one sampled parameter vector per action."""
    rng = np.random.default_rng(seed)
    betas, inv_designs = _fit_per_action_linear_models(train_feedback, ridge_lambda)
    sampled_betas = []

    for a in range(eval_feedback["n_actions"]):
        covariance = sample_scale**2 * inv_designs[a]
        sampled_betas.append(rng.multivariate_normal(betas[a], covariance))

    scores = eval_feedback["context"] @ np.vstack(sampled_betas).T
    return _one_hot(np.argmax(scores, axis=1), eval_feedback["n_actions"])


def _fit_per_action_linear_models(feedback: dict, ridge_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    context = feedback["context"]
    action = feedback["action"]
    reward = feedback["reward"]
    n_actions = feedback["n_actions"]
    n_features = context.shape[1]
    identity = np.eye(n_features)
    betas = np.zeros((n_actions, n_features), dtype=float)
    inv_designs = np.zeros((n_actions, n_features, n_features), dtype=float)

    for a in range(n_actions):
        x_a = context[action == a]
        y_a = reward[action == a]
        design = ridge_lambda * identity + x_a.T @ x_a
        inv_design = np.linalg.pinv(design)
        betas[a] = inv_design @ x_a.T @ y_a if x_a.shape[0] else np.zeros(n_features)
        inv_designs[a] = inv_design

    return betas, inv_designs


def _one_hot(actions: np.ndarray, n_actions: int) -> np.ndarray:
    action_dist = np.zeros((actions.shape[0], n_actions), dtype=float)
    action_dist[np.arange(actions.shape[0]), actions] = 1.0
    return action_dist
