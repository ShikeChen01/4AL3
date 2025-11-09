from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Hashable, Iterable, Dict, Tuple, List, Callable, DefaultDict
from collections import defaultdict
import random

# ---------- Core types ----------
State = Hashable
Action = Hashable
Reward = float
Gamma = float

# Optional reward shaping: adds to the environment reward
RewardShaping = Callable[[State, Action, State], float]

# ---------- Environment interface ----------
class Env(Protocol):
    def reset(self) -> State: ...
    def actions(self, s: State) -> Iterable[Action]: ...
    def step(self, a: Action) -> Tuple[State, Reward, bool, Dict]: ...

# ---------- Policy interface ----------
class Policy(Protocol):
    def pi(self, s: State) -> Action: ...
    def probs(self, s: State) -> Dict[Action, float]: ...  # optional for analysis

# Epsilon-greedy policy over Q(s,a)
class EpsilonGreedyPolicy:
    def __init__(self, Q: Dict[Tuple[State, Action], float], env: Env, epsilon: float):
        self.Q = Q
        self.env = env
        self.epsilon = epsilon

    def pi(self, s: State) -> Action:
        acts = list(self.env.actions(s))
        if not acts:
            raise ValueError("No available actions.")
        if random.random() < self.epsilon:
            return random.choice(acts)
        # greedy
        q_vals = [self.Q.get((s, a), 0.0) for a in acts]
        return acts[max(range(len(acts)), key=lambda i: q_vals[i])]

    def probs(self, s: State) -> Dict[Action, float]:
        acts = list(self.env.actions(s))
        if not acts:
            return {}
        n = len(acts)
        # identify greedy actions
        q_vals = {a: self.Q.get((s, a), 0.0) for a in acts}
        max_q = max(q_vals.values(), default=0.0)
        greedy = [a for a in acts if q_vals[a] == max_q]
        p = {a: self.epsilon / n for a in acts}
        for a in greedy:
            p[a] += (1.0 - self.epsilon) / len(greedy)
        return p

# ---------- Episode container ----------
@dataclass(frozen=True)
class Step:
    s: State
    a: Action
    r: Reward

Episode = List[Step]  # terminal state is implied by env.done at the end

# ---------- Monte Carlo Agent ----------
class MCAgent:
    def __init__(self, gamma: Gamma = 0.99, first_visit: bool = True, shaping: RewardShaping | None = None):
        self.gamma = gamma
        self.first_visit = first_visit
        self.shaping = shaping

    def generate_episode(self, env: Env, policy: Policy, max_len: int = 10_000) -> Episode:
        ep: Episode = []
        s = env.reset()
        for _ in range(max_len):
            a = policy.pi(s)
            s_next, r, done, _ = env.step(a)
            if self.shaping:
                r = r + self.shaping(s, a, s_next)
            ep.append(Step(s, a, r))
            if done:
                break
            s = s_next
        return ep

    # Policy evaluation -> returns V(s)
    def evaluate(self, env: Env, policy: Policy, episodes: int) -> Dict[State, float]:
        returns: DefaultDict[State, List[float]] = defaultdict(list)
        V: Dict[State, float] = {}
        for _ in range(episodes):
            ep = self.generate_episode(env, policy)
            G = 0.0
            # work backwards to accumulate returns
            visited: set[State] = set()
            for t in reversed(range(len(ep))):
                G = self.gamma * G + ep[t].r
                s_t = ep[t].s
                if self.first_visit and s_t in visited:
                    continue
                returns[s_t].append(G)
                visited.add(s_t)
        for s, R in returns.items():
            V[s] = sum(R) / len(R)
        return V

    # MC control with epsilon-greedy improvement -> returns Q(s,a) and improved policy
    def control(self, env: Env, episodes: int, epsilon: float) -> Tuple[Dict[Tuple[State, Action], float], EpsilonGreedyPolicy]:
        Q: DefaultDict[Tuple[State, Action], float] = defaultdict(float)
        returns: DefaultDict[Tuple[State, Action], List[float]] = defaultdict(list)
        policy = EpsilonGreedyPolicy(Q, env, epsilon)

        for _ in range(episodes):
            ep = self.generate_episode(env, policy)
            # Compute returns G_t for each (s,a)
            G = 0.0
            visited_sa: set[Tuple[State, Action]] = set()
            for t in reversed(range(len(ep))):
                s_t, a_t, r_t = ep[t].s, ep[t].a, ep[t].r
                G = self.gamma * G + r_t
                key = (s_t, a_t)
                if self.first_visit and key in visited_sa:
                    continue
                returns[key].append(G)
                Q[key] = sum(returns[key]) / len(returns[key])  # incremental avg (simple but clear)
                visited_sa.add(key)
            # policy is implicitly improved via Q in EpsilonGreedyPolicy
        return Q, policy

'''
# ---------- Minimal example environment ----------
# 1D LineWorld: states {0,1,2,3,4}, start at 2, terminal at 0 (reward 0) and 4 (reward +1)
# Actions: 'L' or 'R'
class LineWorld:
    def __init__(self, length: int = 5):
        assert length >= 2
        self.N = length
        self.s: int = 2

    def reset(self) -> State:
        self.s = self.N // 2
        return self.s

    def actions(self, s: State) -> Iterable[Action]:
        if s == 0 or s == self.N - 1:
            return []  # terminal
        return ("L", "R")

    def step(self, a: Action) -> Tuple[State, Reward, bool, Dict]:
        if a == "L":
            self.s -= 1
        elif a == "R":
            self.s += 1
        else:
            raise ValueError("Unknown action")
        done = (self.s == 0) or (self.s == self.N - 1)
        r: Reward = 1.0 if self.s == self.N - 1 else 0.0
        return self.s, r, done, {}

# ---------- Example usage ----------
if __name__ == "__main__":
    random.seed(0)

    env = LineWorld(length=5)

    # Start with a uniform random policy (epsilon=1.0 over Q=0)
    agent = MCAgent(gamma=1.0, first_visit=True)

    # Control: learn Q and an epsilon-greedy policy
    Q, pi_star = agent.control(env, episodes=10_000, epsilon=0.1)

    # Evaluate learned policy
    V = agent.evaluate(env, pi_star, episodes=1_000)

    # Pretty-print a few values
    print("V(s):")
    for s in range(env.N):
        print(s, round(V.get(s, 0.0), 3))
    print("\nSample action probs at s=2:", pi_star.probs(2))
'''