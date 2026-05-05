from __future__ import annotations

import numpy as np


def make_logged_action_features(feedback: dict) -> np.ndarray:
    """Build features for the logged action in each round."""
    context = feedback["context"]
    action_context = feedback["action_context"][feedback["action"]]
    return np.hstack([context, action_context])


def make_all_action_features(context: np.ndarray, action_context: np.ndarray) -> np.ndarray:
    """Build a feature tensor with shape (n_rounds, n_actions, n_features)."""
    n_rounds = context.shape[0]
    n_actions = action_context.shape[0]
    repeated_context = np.repeat(context[:, None, :], n_actions, axis=1)
    repeated_action_context = np.repeat(action_context[None, :, :], n_rounds, axis=0)
    return np.concatenate([repeated_context, repeated_action_context], axis=2)
