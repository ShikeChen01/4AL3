from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256, l2_lambda: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden
        self.l2_lambda = l2_lambda
        
        # 1. Feature Extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU()
        )
        
        # 2. LSTM (Batch First)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        
        # 3. Heads
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

        # 4. Internal State Storage
        self.lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def reset_hidden_state(self, batch_size: int = 1, device: str = "cpu"):
        """Resets the internal state. Call at start of episode."""
        h = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        self.lstm_state = (h, c)

    def forward(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (Batch, Obs) or (Batch, Seq, Obs)
            state: Optional override for hidden state (used during training).
                   If None, uses self.lstm_state (used during inference).
        """
        # Ensure input is 3D: (Batch, Seq, Feat)
        if x.dim() == 2: 
            x = x.unsqueeze(1) # Add sequence dim (Batch, 1, Feat)

        # 1. Embed
        x = self.feature_extractor(x)
        
        # 2. Select State (Internal vs External)
        if state is None:
            # Inference Mode: Use internal state
            if self.lstm_state is None:
                self.reset_hidden_state(x.size(0), x.device)
            state = self.lstm_state
        
        # 3. Run LSTM
        # output shape: (Batch, Seq, Hidden)
        x, (h_n, c_n) = self.lstm(x, state)
        
        # 4. Update Internal State (Only if using internal state)
        # We DETACH to stop gradients flowing back endlessly across episodes
        if self.lstm_state is not None and state is self.lstm_state:
            self.lstm_state = (h_n.detach(), c_n.detach())

        # 5. Heads (Flatten sequence dim if it's 1, or keep it for block training)
        # For PPO block training, we process (1, 256, Hidden) -> (256, Hidden)
        x = x.reshape(-1, self.hidden_dim)
        
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        
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
    lr: float = 1e-4
    epochs: int = 10
    batch_size: int = 2048
    minibatch_size: int = 256 # This acts as the "Block Size" for shuffling
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
        self.model.reset_hidden_state(batch_size=1, device=self.device)

    def reset_hidden_state(self):
        self.model.reset_hidden_state(batch_size=1, device=self.device)

    def act(self, obs: np.ndarray) -> Tuple[int, float, float, Tuple[np.ndarray, np.ndarray]]:
        """
        Simpler Act: No longer takes hidden state as input.
        Returns hidden state TUPLE at the end for the buffer.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            # Model uses its internal self.lstm_state automatically
            logits, value = self.model(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)
        
        # Extract the state from the model to save in replay buffer
        # Shape: (1, Batch, Hidden) -> (Hidden,)
        h = self.model.lstm_state[0].squeeze(0).squeeze(0).detach()
        c = self.model.lstm_state[1].squeeze(0).squeeze(0).detach()

        return int(action.item()), float(logp.item()), float(value.item()), (h, c)

    def evaluate_actions(self, obs_batch: torch.Tensor, act_batch: torch.Tensor, hidden_state: Tuple[torch.Tensor, torch.Tensor]):
        """
        For training: We pass the explicit hidden state of the block start.
        """
        # obs_batch shape is (Block_Size, Obs_Dim)
        # We unsqueeze to (1, Block_Size, Obs_Dim) so LSTM processes it as one sequence
        obs_seq = obs_batch.unsqueeze(0)
        
        logits, values = self.model(obs_seq, hidden_state)
        
        # Logits come out as (Block_Size, Actions), matching act_batch
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
        
        # Retrieve stored states (N, Hidden)
        h_buf = buffer["hidden_h"] 
        c_buf = buffer["hidden_c"]
        
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # --- BLOCK SHUFFLING ---
        n = obs.size(0)
        block_size = cfg.minibatch_size
        num_blocks = n // block_size
        
        # Create indices for blocks: [[0..255], [256..511], ...]
        indices = np.arange(num_blocks * block_size).reshape(num_blocks, block_size)
        
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        for _ in range(cfg.epochs):
            # Shuffle blocks, but keep time contiguous INSIDE the block
            np.random.shuffle(indices)
            
            for i in range(num_blocks):
                batch_idxs = indices[i]
                
                ob_b = obs[batch_idxs]
                act_b = acts[batch_idxs]
                old_logp_b = old_logp[batch_idxs]
                adv_b = adv[batch_idxs]
                ret_b = ret[batch_idxs]
                
                # IMPORTANT: Get the hidden state from the FIRST step of the block
                # shape (1, 1, Hidden)
                start_idx = batch_idxs[0]
                h_start = h_buf[start_idx].unsqueeze(0).unsqueeze(0)
                c_start = c_buf[start_idx].unsqueeze(0).unsqueeze(0)
                hidden_start = (h_start, c_start)

                # Pass the whole sequence + start state
                new_logp, entropy, values = self.evaluate_actions(ob_b, act_b, hidden_start)
                
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

        denom = (cfg.epochs * num_blocks)
        for k in stats:
            stats[k] /= max(1, denom)
        return stats