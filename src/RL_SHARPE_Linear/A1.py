from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256, l2_lambda: float = 0.0):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 3*obs_dim),
            nn.Linear(3*obs_dim, hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)  # logits
        self.value_head = nn.Linear(hidden, 1)
        self.l2_lambda = l2_lambda

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def l2_reg(self) -> torch.Tensor:
        if self.l2_lambda <= 0:
            return torch.zeros((), device=next(self.parameters()).device)
        l2 = torch.tensor(0.0, device=next(self.parameters()).device)
        for p in self.parameters():
            l2 = l2 + torch.sum(p * p)
        return self.l2_lambda * l2


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-4
    epochs: int = 10
    batch_size: int = 2048
    minibatch_size: int = 256
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    l2_lambda: float = 1e-4


class PPOAgent:
    def __init__(self, obs_dim: int, n_actions: int, device: str = "cuda", cfg: PPOConfig = PPOConfig()):
        self.device = (device if torch.cuda.is_available() else "cpu") if device == "cuda" else "cpu"
        self.cfg = cfg
        self.model = ActorCritic(obs_dim, n_actions, hidden=128, l2_lambda=cfg.l2_lambda).to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=cfg.lr)

    def act(self, obs: np.ndarray) -> Tuple[int, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.model(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)
        return int(action.item()), float(logp.item()), float(value.item())

    def evaluate_actions(self, obs_batch: torch.Tensor, act_batch: torch.Tensor):
        logits, values = self.model(obs_batch)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(act_batch)
        entropy = dist.entropy()
        return logp, entropy, values

    def update(self, buffer: Dict[str, np.ndarray]) -> Dict[str, float]:
        cfg = self.cfg
        obs = torch.as_tensor(buffer["obs"], dtype=torch.float32, device=self.device)
        acts = torch.as_tensor(buffer["acts"], dtype=torch.long, device=self.device)
        old_logp = torch.as_tensor(buffer["logp"], dtype=torch.float32, device=self.device)
        adv = torch.as_tensor(buffer["adv"], dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(buffer["ret"], dtype=torch.float32, device=self.device)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = obs.size(0)
        idxs = np.arange(n)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        for _ in range(cfg.epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, cfg.minibatch_size):
                batch = idxs[start:start + cfg.minibatch_size]
                ob_b = obs[batch]
                act_b = acts[batch]
                old_logp_b = old_logp[batch]
                adv_b = adv[batch]
                ret_b = ret[batch]

                new_logp, entropy, values = self.evaluate_actions(ob_b, act_b)
                ratio = torch.exp(new_logp - old_logp_b)

                unclipped = ratio * adv_b
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_b
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_loss = 0.5 * (ret_b - values).pow(2).mean()

                entropy_mean = entropy.mean()

                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_mean
                loss = loss + self.model.l2_reg()

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.opt.step()

                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy_mean.item())

        denom = (cfg.epochs * max(1, n // cfg.minibatch_size))
        for k in stats:
            stats[k] /= max(1, denom)
        return stats
