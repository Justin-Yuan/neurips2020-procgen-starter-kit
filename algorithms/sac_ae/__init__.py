"""
references:
    - paper: https://arxiv.org/pdf/1910.01741.pdf
    - repo: https://github.com/denisyarats/pytorch_sac_ae
"""

from algorithms.sac_ae.sac_ae_trainer import SAC_CONFIG, SACAETrainer
from algorithms.sac_ae.sac_ae_policy import SACAETorchPolicy

__all__ = [
    "SAC_CONFIG",
    "SACAETrainer",
    "SACAETorchPolicy"
]