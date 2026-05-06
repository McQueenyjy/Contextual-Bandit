from __future__ import annotations

import numpy as np


def infer_len_list(position: np.ndarray | None) -> int:
    """Infer OBP's list length from logged positions."""
    if position is None or position.size == 0:
        return 1
    return int(np.max(position)) + 1


def to_obp_action_dist(action_dist: np.ndarray, position: np.ndarray | None = None) -> np.ndarray:
    """Convert a 2D policy distribution to OBP's 3D action_dist format."""
    if action_dist.ndim == 3:
        return action_dist
    if action_dist.ndim != 2:
        raise ValueError("action_dist must have shape (n_rounds, n_actions) or (n_rounds, n_actions, len_list)")
    len_list = infer_len_list(position)
    return np.repeat(action_dist[:, :, None], len_list, axis=2)


def to_obp_estimated_rewards(q_hat: np.ndarray, position: np.ndarray | None = None) -> np.ndarray:
    """Convert 2D expected rewards to OBP's 3D estimated_rewards_by_reg_model format."""
    if q_hat.ndim == 3:
        return q_hat
    if q_hat.ndim != 2:
        raise ValueError("q_hat must have shape (n_rounds, n_actions) or (n_rounds, n_actions, len_list)")
    len_list = infer_len_list(position)
    return np.repeat(q_hat[:, :, None], len_list, axis=2)


def estimate_with_obp(eval_feedback: dict, action_dist: np.ndarray, q_hat: np.ndarray | None = None) -> dict:
    """Estimate policy values with OBP's official OffPolicyEvaluation class."""
    try:
        from obp.ope import (  # type: ignore
            DirectMethod,
            DoublyRobust,
            InverseProbabilityWeighting,
            OffPolicyEvaluation,
            SelfNormalizedInverseProbabilityWeighting,
        )
    except ImportError as exc:
        try:
            from obp.ope.estimators import (  # type: ignore
                DirectMethod,
                DoublyRobust,
                InverseProbabilityWeighting,
                SelfNormalizedInverseProbabilityWeighting,
            )
            from obp.ope.meta import OffPolicyEvaluation  # type: ignore
        except ImportError:
            raise ImportError(
                "OBP is not installed or its OPE API is incompatible. "
                "Install project dependencies with `python -m pip install -r requirements.txt`."
            ) from exc

    estimators = [
        InverseProbabilityWeighting(),
        SelfNormalizedInverseProbabilityWeighting(),
    ]
    estimated_rewards_by_reg_model = None
    if q_hat is not None:
        estimators.extend([DirectMethod(), DoublyRobust()])
        estimated_rewards_by_reg_model = to_obp_estimated_rewards(q_hat, eval_feedback.get("position"))

    ope = OffPolicyEvaluation(
        bandit_feedback=eval_feedback,
        ope_estimators=estimators,
    )
    return ope.estimate_policy_values(
        action_dist=to_obp_action_dist(action_dist, eval_feedback.get("position")),
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    )
