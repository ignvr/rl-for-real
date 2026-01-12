"""Configuration dataclasses for GRPO training."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatasetConfigItem:
    """Configuration for a single dataset within the composite."""
    weight: Optional[float] = field(default=1.0)
    config: Optional[dict] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    """Configuration for datasets and evaluation."""
    dataset_size: int = field(default=10000)
    developer_prompt: str = field(default="DeepSeekZero")
    developer_role: str = field(default="system")
    datasets: dict[str, DatasetConfigItem] = field(default=None)
    
    # Evaluation datasets (optional - if None, uses same config as training)
    eval_datasets: dict[str, DatasetConfigItem] = field(default=None)
    
    # Fixed evaluation settings
    eval_num_samples: int = field(default=30)
    eval_num_examples_to_print: int = field(default=3)
    fixed_eval_steps: int = field(default=50)
    
    # Variance logging
    log_reward_variance: bool = field(default=True)
    verbose_variance_logging: bool = field(default=False)
    
    # Resume control
    auto_resume: bool = field(default=False)  # If True, auto-resume from last checkpoint
    
    # Eval-only mode
    eval_only: bool = field(default=False)  # If True, only run evaluation (no training)
    checkpoint_path: str = field(default=None)  # Path to checkpoint for eval-only mode

    def __post_init__(self):
        # Convert training datasets
        if self.datasets:
            converted_datasets = {}
            for name, config_item in self.datasets.items():
                if isinstance(config_item, dict):
                    converted_datasets[name] = DatasetConfigItem(**config_item)
                else:
                    converted_datasets[name] = config_item
            self.datasets = converted_datasets
        
        # Convert eval datasets (if provided)
        if self.eval_datasets:
            converted_eval_datasets = {}
            for name, config_item in self.eval_datasets.items():
                if isinstance(config_item, dict):
                    converted_eval_datasets[name] = DatasetConfigItem(**config_item)
                else:
                    converted_eval_datasets[name] = config_item
            self.eval_datasets = converted_eval_datasets
