import numpy as np

from src.data import load_open_bandit_feedback


bandit_feedback = load_open_bandit_feedback(
    behavior_policy="random",
    campaign="all",
)

n_actions = bandit_feedback["n_actions"]
reward = bandit_feedback["reward"]
pscore = bandit_feedback["pscore"]

# Target policy: uniform random.
target_policy_prob = np.ones_like(reward) / n_actions

ips_policy_value = np.mean(reward * target_policy_prob / pscore)

print("n_rounds:", bandit_feedback["n_rounds"])
print("n_actions:", n_actions)
print("mean reward of behavior policy:", reward.mean())
print("IPS estimated value of random policy:", ips_policy_value)
