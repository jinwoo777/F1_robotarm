"""연속 행동공간용 Gaussian Actor-Critic (state-independent log_std)."""
import torch
import torch.nn as nn
from torch.distributions import Normal


def _mlp(in_dim, out_dim, hidden=64):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.actor_mean = _mlp(obs_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)
        self.critic = _mlp(obs_dim, 1)

    def dist(self, obs: torch.Tensor) -> Normal:
        mean = torch.tanh(self.actor_mean(obs))
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def act(self, obs: torch.Tensor):
        d = self.dist(obs)
        action = d.sample()
        logp = d.log_prob(action).sum(-1)
        return action, logp, self.value(obs)

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        d = self.dist(obs)
        logp = d.log_prob(action).sum(-1)
        entropy = d.entropy().sum(-1)
        return logp, entropy, self.value(obs)
