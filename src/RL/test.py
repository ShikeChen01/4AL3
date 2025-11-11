from environment import Environment
from agent import PPOAgent
from typing import Dict, Tuple, List
import numpy as np
import torch


# =========================
# Evaluation (deterministic)
# =========================
def evaluate_mse(env: Environment, agent: PPOAgent) -> Tuple[float, float]:
    """
    Run once over env.df in order. Deterministic policy (argmax).
    Returns (mse, accuracy) where predictions are in {-1,0,1} vs raw 'target'.
    """
    device = agent.device
    y_true: List[float] = []
    y_pred: List[int] = []

    obs = env.reset()
    done = False
    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = agent.model(obs_t)
            a_idx = int(torch.argmax(logits, dim=-1).item())
        idx_to_action = {0: -1, 1: 0, 2: 1}
        a_env = idx_to_action[a_idx]

        t = env.t
        y_true.append(float(env.y[t]))
        y_pred.append(int(a_env))

        obs, _, done, _ = env.step(a_env)

    y_true_arr = np.asarray(y_true, dtype=np.float32)
    y_pred_arr = np.asarray(y_pred, dtype=np.float32)
    mse = float(np.mean((y_pred_arr - y_true_arr) ** 2))
    acc = float(np.mean(np.sign(y_pred_arr) == np.sign(y_true_arr)))
    return mse, acc