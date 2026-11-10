"""Evolution-strategy adapter for recovering quality after quantization.

Inspired by Agentic ESOpt (gradient-free, full-parameter updates using
inference-level memory). This is a lightweight interface for adapting a
quantized model to a target task using scalar rewards.
"""

import random


class EvolutionStrategyAdapter:
    def __init__(self, population_size: int = 8, sigma: float = 0.01, learning_rate: float = 0.05):
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate

    def sample_perturbations(self, params):
        """Return G perturbed parameter copies (conceptual)."""
        return [params for _ in range(self.population_size)]

    def update(self, rewards):
        """Reward-weighted update given scalar rewards."""
        if not rewards:
            return 0.0
        mean = sum(rewards) / len(rewards)
        std = (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5 or 1.0
        z = [(r - mean) / std for r in rewards]
        return self.learning_rate * sum(z) / len(z)

    def step(self, eval_fn):
        """Run one ES generation using a scalar reward eval function."""
        rewards = [eval_fn() for _ in range(self.population_size)]
        return self.update(rewards), rewards
