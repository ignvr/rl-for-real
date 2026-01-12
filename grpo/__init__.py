"""GRPO Training Components for Reasoning Gym.

This module provides components for training language models using
Group Relative Policy Optimization (GRPO) on reasoning-gym tasks.
"""

from .config import DatasetConfig, DatasetConfigItem
from .dataset import ReasoningGymDataset, prepare_datasets
from .trainer import CustomGRPOTrainer
from .eval_callback import FixedEvalCallback
from .utils import build_wandb_config, log_trainable_params

__all__ = [
    "DatasetConfig",
    "DatasetConfigItem",
    "ReasoningGymDataset",
    "prepare_datasets",
    "CustomGRPOTrainer",
    "FixedEvalCallback",
    "build_wandb_config",
    "log_trainable_params",
]
