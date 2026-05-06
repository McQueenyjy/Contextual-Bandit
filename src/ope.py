from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import make_all_action_features, make_logged_action_features


def get_logged_action_policy_prob(action_dist: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Return pi(a_i | x_i) for each logged action a_i."""
    return action_dist[np.arange(action.shape[0]), action]


def estimate_ips(reward: np.ndarray, pscore: np.ndarray, pi_logged: np.ndarray) -> float:
    """Inverse Propensity Score estimator."""
    return float(np.mean(reward * pi_logged / pscore))


def estimate_snips(reward: np.ndarray, pscore: np.ndarray, pi_logged: np.ndarray) -> float:
    """Self-Normalized IPS estimator."""
    weights = pi_logged / pscore
    denominator = np.sum(weights)
    if denominator <= 0:
        return float("nan")
    return float(np.sum(weights * reward) / denominator)


def fit_reward_model(train_feedback: dict) -> object:
    """Fit q(x, a) = E[r | x, a] with a simple logistic model."""
    x_train = make_logged_action_features(train_feedback)
    y_train = train_feedback["reward"]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    model.fit(x_train, y_train)
    return model


def predict_expected_rewards(model: object, feedback: dict, batch_size: int = 8192) -> np.ndarray:
    """Predict q_hat(x, a) for every round and every action."""
    context = feedback["context"]
    action_context = feedback["action_context"]
    n_rounds = feedback["n_rounds"]
    n_actions = feedback["n_actions"]
    q_hat = np.empty((n_rounds, n_actions), dtype=np.float32)

    for start in range(0, n_rounds, batch_size):
        end = min(start + batch_size, n_rounds)
        all_features = make_all_action_features(context[start:end], action_context)
        _, _, n_features = all_features.shape
        flat_features = all_features.reshape((end - start) * n_actions, n_features)
        q_hat[start:end] = model.predict_proba(flat_features)[:, 1].reshape(end - start, n_actions)

    return q_hat


def estimate_dm(action_dist: np.ndarray, q_hat: np.ndarray) -> float:
    """Direct Method estimator."""
    return float(np.mean(np.sum(action_dist * q_hat, axis=1)))


def estimate_dr(
    reward: np.ndarray,
    pscore: np.ndarray,
    action: np.ndarray,
    action_dist: np.ndarray,
    q_hat: np.ndarray,
) -> float:
    """Doubly Robust estimator."""
    dm_round = np.sum(action_dist * q_hat, axis=1)
    pi_logged = get_logged_action_policy_prob(action_dist, action)
    q_logged = q_hat[np.arange(action.shape[0]), action]
    correction = pi_logged / pscore * (reward - q_logged)
    return float(np.mean(dm_round + correction))
