from environment import Environment
from A1 import PPOAgent
from typing import Dict, Tuple, List
import numpy as np
import torch

def evaluate_mse(env: Environment, agent: PPOAgent) -> Tuple[float, float]:
    """
    Run once over env.df in order. Deterministic policy (argmax).
    Returns (mse, accuracy) where predictions are Leverage Amounts vs 'target'.
    """
    device = agent.device
    y_true: List[float] = []
    y_pred: List[float] = []

    obs = env.reset()
    done = False
    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = agent.model(obs_t)
            # Get Argmax Index
            a_idx = int(torch.argmax(logits, dim=-1).item())
            
        # Map index to Leverage (0.0 to 2.0)
        a_lev = float(env.action_map[a_idx])

        t = env.t
        y_true.append(float(env.y[t]))
        y_pred.append(a_lev)

        obs, _, done, _ = env.step(a_idx)

    y_true_arr = np.asarray(y_true, dtype=np.float32)
    y_pred_arr = np.asarray(y_pred, dtype=np.float32)
    
    # MSE between leverage (0.0-2.0) and Return (0.0-0.05) is just a stability metric
    mse = float(np.mean((y_pred_arr - y_true_arr) ** 2))
    
    # Accuracy: Did we hold leverage (>0.1) when return > 0?
    pos_ret = y_true_arr > 0
    pos_pos = y_pred_arr > 0.1
    acc = float(np.mean(pos_ret == pos_pos))
    
    return mse, acc