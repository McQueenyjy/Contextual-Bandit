from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features import make_all_action_features, make_logged_action_features


def uniform_random_policy(feedback: dict) -> np.ndarray:
    """Uniform random target policy."""
    return np.ones((feedback["n_rounds"], feedback["n_actions"]), dtype=np.float32) / feedback["n_actions"]


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
    action_dist = np.ones((eval_feedback["n_rounds"], n_actions), dtype=np.float32) * epsilon / n_actions
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
        variance = np.sum((context @ inv_designs[a]) * context, axis=1)
        uncertainty = np.sqrt(np.clip(variance, 0.0, None))
        scores[:, a] = mean + alpha * uncertainty

    return _one_hot(np.argmax(scores, axis=1), n_actions)


def logistic_ucb_policy(
    train_feedback: dict,
    eval_feedback: dict,
    alpha: float = 0.5,
    ridge_lambda: float = 1.0,
    max_iter: int = 1000,
    use_ipw: bool = False,
    batch_size: int = 8192,
) -> np.ndarray:
    """Logistic UCB using P(click | user context, action context) plus an uncertainty bonus."""
    model, scaler, inv_design = _fit_logistic_click_model(
        train_feedback=train_feedback,
        ridge_lambda=ridge_lambda,
        max_iter=max_iter,
        use_ipw=use_ipw,
    )
    context = eval_feedback["context"]
    action_context = eval_feedback["action_context"]
    n_rounds = eval_feedback["n_rounds"]
    n_actions = eval_feedback["n_actions"]
    chosen_actions = np.empty(n_rounds, dtype=np.int64)

    for start in range(0, n_rounds, batch_size):
        end = min(start + batch_size, n_rounds)
        all_features = make_all_action_features(context[start:end], action_context)
        _, _, n_features = all_features.shape
        flat_features = all_features.reshape((end - start) * n_actions, n_features)
        flat_features = _standardize_with_fallback(scaler, flat_features)
        flat_features_aug = _add_intercept(flat_features)

        click_prob = model.predict_proba(flat_features)[:, 1]
        uncertainty = np.sqrt(np.sum((flat_features_aug @ inv_design) * flat_features_aug, axis=1))
        scores = (click_prob + alpha * uncertainty).reshape(end - start, n_actions)
        chosen_actions[start:end] = np.argmax(scores, axis=1)

    return _one_hot(chosen_actions, n_actions)


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
        covariance = sample_scale**2 * _make_positive_semidefinite(inv_designs[a])
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


def _fit_logistic_click_model(
    train_feedback: dict,
    ridge_lambda: float,
    max_iter: int,
    use_ipw: bool,
) -> tuple[LogisticRegression, StandardScaler, np.ndarray]:
    x_train = make_logged_action_features(train_feedback)
    y_train = train_feedback["reward"]
    scaler = StandardScaler()
    x_train = _standardize_with_fallback(scaler, x_train, fit=True)

    sample_weight = None
    if use_ipw:
        sample_weight = 1.0 / np.clip(train_feedback["pscore"], 1e-12, None)

    model = LogisticRegression(max_iter=max_iter, solver="lbfgs", C=1.0 / max(ridge_lambda, 1e-12))
    model.fit(x_train, y_train, sample_weight=sample_weight)

    click_prob = model.predict_proba(x_train)[:, 1]
    curvature = np.clip(click_prob * (1.0 - click_prob), 1e-6, None)
    if sample_weight is not None:
        curvature = curvature * sample_weight

    x_aug = _add_intercept(x_train)
    design = ridge_lambda * np.eye(x_aug.shape[1])
    design += x_aug.T @ (curvature[:, None] * x_aug)
    inv_design = np.linalg.pinv(design)
    return model, scaler, inv_design


def _standardize_with_fallback(
    scaler: StandardScaler,
    features: np.ndarray,
    fit: bool = False,
) -> np.ndarray:
    if fit:
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)
    return np.nan_to_num(features, copy=False)


def _add_intercept(features: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((features.shape[0], 1), dtype=features.dtype), features])


def _make_positive_semidefinite(matrix: np.ndarray, jitter: float = 1e-10) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    min_eigenvalue = float(np.min(np.linalg.eigvalsh(symmetric)))
    if min_eigenvalue < 0:
        symmetric += (-min_eigenvalue + jitter) * np.eye(symmetric.shape[0])
    else:
        symmetric += jitter * np.eye(symmetric.shape[0])
    return symmetric


def _one_hot(actions: np.ndarray, n_actions: int) -> np.ndarray:
    action_dist = np.zeros((actions.shape[0], n_actions), dtype=np.float32)
    action_dist[np.arange(actions.shape[0]), actions] = 1.0
    return action_dist
