from src.data import load_open_bandit_feedback


bandit_feedback = load_open_bandit_feedback(
    behavior_policy="random",
    campaign="all",
)

print(bandit_feedback.keys())
print("n_rounds:", bandit_feedback["n_rounds"])
print("n_actions:", bandit_feedback["n_actions"])
print("action shape:", bandit_feedback["action"].shape)
print("reward shape:", bandit_feedback["reward"].shape)
print("pscore shape:", bandit_feedback["pscore"].shape)
print("context shape:", bandit_feedback["context"].shape)
print("action_context shape:", bandit_feedback["action_context"].shape)
