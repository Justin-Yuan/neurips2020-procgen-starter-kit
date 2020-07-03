""" references
- paper: https://arxiv.org/pdf/1907.00953.pdf
- repo: https://github.com/ku2482/slac.pytorch
"""
from algorithms.simple.ppo_trainer import PPO_CONFIG, SimplePPOTrainer
from algorithms.simple.ppo_policy import SimplePPOTorchPolicy

__all__ = [
    "PPO_CONFIG",
    "SimplePPOTrainer",
    "SimplePPOTorchPolicy"
]