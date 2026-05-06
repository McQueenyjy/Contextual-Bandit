from __future__ import annotations

import math

import numpy as np

from src.obp_adapter import estimate_with_obp
from src.ope import (
    estimate_dm,
    estimate_dr,
    estimate_ips,
    estimate_snips,
    get_logged_action_policy_prob,
)


ESTIMATORS = ("ips", "snips", "dm", "dr")


def effective_sample_size(weights: np.ndarray) -> float:
    """Return the importance-weight effective sample size."""
    denominator = float(np.sum(weights**2))
    if denominator <= 0:
        return 0.0
    return float(np.sum(weights) ** 2 / denominator)


def estimate_policy_values(eval_feedback: dict, action_dist: np.ndarray, q_hat: np.ndarray) -> dict:
    """Estimate a target policy with the project's hand-written OPE estimators."""
    reward = eval_feedback["reward"]
    pscore = eval_feedback["pscore"]
    action = eval_feedback["action"]
    pi_logged = get_logged_action_policy_prob(action_dist, action)

    return {
        "ips": estimate_ips(reward, pscore, pi_logged),
        "snips": estimate_snips(reward, pscore, pi_logged),
        "dm": estimate_dm(action_dist, q_hat),
        "dr": estimate_dr(reward, pscore, action, action_dist, q_hat),
    }


def bootstrap_intervals(
    eval_feedback: dict,
    action_dist: np.ndarray,
    q_hat: np.ndarray,
    alpha: float = 0.05,
    n_bootstrap_samples: int = 200,
    seed: int = 12345,
) -> dict[str, dict[str, float]]:
    """Estimate nonparametric bootstrap confidence intervals for each OPE estimator."""
    if n_bootstrap_samples <= 0:
        return {}
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    round_terms = _precompute_round_terms(eval_feedback, action_dist, q_hat)
    rng = np.random.default_rng(seed)
    n_rounds = eval_feedback["n_rounds"]
    estimates = {name: np.empty(n_bootstrap_samples, dtype=float) for name in ESTIMATORS}

    for bootstrap_idx in range(n_bootstrap_samples):
        indices = rng.integers(0, n_rounds, size=n_rounds)
        estimates["ips"][bootstrap_idx] = float(np.mean(round_terms["ips_round"][indices]))
        estimates["snips"][bootstrap_idx] = _estimate_snips_from_terms(
            round_terms["weighted_reward"][indices],
            round_terms["weight"][indices],
        )
        estimates["dm"][bootstrap_idx] = float(np.mean(round_terms["dm_round"][indices]))
        estimates["dr"][bootstrap_idx] = float(np.mean(round_terms["dr_round"][indices]))

    lower_q = 100 * alpha / 2
    upper_q = 100 * (1 - alpha / 2)
    return {
        name: {
            "bootstrap_mean": float(np.mean(values)),
            "bootstrap_std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "ci_lower": float(np.percentile(values, lower_q)),
            "ci_upper": float(np.percentile(values, upper_q)),
        }
        for name, values in estimates.items()
    }


def evaluate_policy(
    name: str,
    action_dist: np.ndarray,
    eval_feedback: dict,
    q_hat: np.ndarray,
    bootstrap_samples: int = 0,
    alpha: float = 0.05,
    seed: int = 12345,
    include_obp: bool = True,
) -> dict:
    """Build a result row with hand-written estimates, diagnostics, optional CIs, and OBP parity checks."""
    reward = eval_feedback["reward"]
    pscore = eval_feedback["pscore"]
    action = eval_feedback["action"]
    pi_logged = get_logged_action_policy_prob(action_dist, action)
    weights = pi_logged / pscore
    estimates = estimate_policy_values(eval_feedback, action_dist, q_hat)

    row = {
        "policy": name,
        "behavior_mean_reward": float(reward.mean()),
        **estimates,
        "logged_action_match_rate": float((pi_logged > 0).mean()),
        "mean_importance_weight": float(weights.mean()),
        "effective_sample_size": effective_sample_size(weights),
        "effective_sample_size_ratio": effective_sample_size(weights) / eval_feedback["n_rounds"],
    }

    for estimator, interval in bootstrap_intervals(
        eval_feedback=eval_feedback,
        action_dist=action_dist,
        q_hat=q_hat,
        alpha=alpha,
        n_bootstrap_samples=bootstrap_samples,
        seed=seed,
    ).items():
        row[f"{estimator}_ci_lower"] = interval["ci_lower"]
        row[f"{estimator}_ci_upper"] = interval["ci_upper"]
        row[f"{estimator}_bootstrap_std"] = interval["bootstrap_std"]

    if include_obp:
        row.update(_obp_comparison_columns(eval_feedback, action_dist, q_hat, estimates))

    return row


def _obp_comparison_columns(
    eval_feedback: dict,
    action_dist: np.ndarray,
    q_hat: np.ndarray,
    manual_estimates: dict,
) -> dict:
    try:
        obp_estimates = estimate_with_obp(eval_feedback, action_dist, q_hat)
    except ImportError:
        return {"obp_status": "not_installed"}
    except Exception as exc:  # pragma: no cover - protects long-running experiments from optional parity failures.
        return {"obp_status": f"error: {type(exc).__name__}: {exc}"}

    row = {"obp_status": "ok"}
    name_map = {
        "ipw": "ips",
        "snipw": "snips",
        "dm": "dm",
        "dr": "dr",
    }
    for obp_name, manual_name in name_map.items():
        if obp_name in obp_estimates and manual_name in manual_estimates:
            obp_value = float(obp_estimates[obp_name])
            row[f"obp_{obp_name}"] = obp_value
            row[f"obp_{obp_name}_abs_diff"] = math.fabs(obp_value - manual_estimates[manual_name])
    return row


def _precompute_round_terms(eval_feedback: dict, action_dist: np.ndarray, q_hat: np.ndarray) -> dict[str, np.ndarray]:
    reward = eval_feedback["reward"]
    pscore = eval_feedback["pscore"]
    action = eval_feedback["action"]
    pi_logged = get_logged_action_policy_prob(action_dist, action)
    weight = pi_logged / pscore
    q_logged = q_hat[np.arange(action.shape[0]), action]
    dm_round = np.sum(action_dist * q_hat, axis=1)
    return {
        "weight": weight,
        "weighted_reward": weight * reward,
        "ips_round": reward * weight,
        "dm_round": dm_round,
        "dr_round": dm_round + weight * (reward - q_logged),
    }


def _estimate_snips_from_terms(weighted_reward: np.ndarray, weight: np.ndarray) -> float:
    denominator = float(np.sum(weight))
    if denominator <= 0:
        return float("nan")
    return float(np.sum(weighted_reward) / denominator)
