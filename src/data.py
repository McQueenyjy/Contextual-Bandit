from __future__ import annotations

import numpy as np


def load_open_bandit_feedback(
    behavior_policy: str = "random",
    campaign: str = "all",
    data_path: str | None = None,
) -> dict:
    """Load logged bandit feedback from the Open Bandit Dataset."""
    try:
        from obp.dataset import OpenBanditDataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "OBP is not installed. Install project dependencies with `python -m pip install -r requirements.txt`."
        ) from exc

    dataset = OpenBanditDataset(
        behavior_policy=behavior_policy,
        campaign=campaign,
        data_path=data_path,
    )
    return dataset.obtain_batch_bandit_feedback()


def train_eval_split(feedback: dict, eval_size: float = 0.5, seed: int = 12345) -> tuple[dict, dict]:
    """Split logged feedback into a policy/model training part and an OPE part."""
    n_rounds = feedback["n_rounds"]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_rounds)
    n_eval = int(n_rounds * eval_size)
    eval_idx = indices[:n_eval]
    train_idx = indices[n_eval:]
    return _subset_feedback(feedback, train_idx), _subset_feedback(feedback, eval_idx)


def _subset_feedback(feedback: dict, indices: np.ndarray) -> dict:
    return {
        "n_rounds": len(indices),
        "n_actions": feedback["n_actions"],
        "action": feedback["action"][indices],
        "position": feedback["position"][indices],
        "reward": feedback["reward"][indices],
        "pscore": feedback["pscore"][indices],
        "context": feedback["context"][indices],
        "action_context": feedback["action_context"],
    }
