"""Dataset wrapper and preparation for reasoning-gym."""

from typing import Optional

from torch.utils.data import Dataset

import reasoning_gym
from reasoning_gym.composite import DatasetSpec
from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.utils import SYSTEM_PROMPTS

from .config import DatasetConfig


class ReasoningGymDataset(Dataset):
    """
    Dataset wrapper for reasoning-gym procedural datasets.
    
    Formats prompts with chat templates for use with GRPOTrainer.
    """
    
    def __init__(
        self,
        tokenizer,
        procedural_dataset: ProceduralDataset,
        developer_prompt: Optional[str] = None,
        developer_role: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.data = procedural_dataset
        self.developer_prompt = developer_prompt
        self.developer_role = developer_role

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        question = item["question"]

        chat = []
        if self.developer_role is not None:
            chat.append({"role": self.developer_role, "content": self.developer_prompt})
        chat.append({"role": "user", "content": question})

        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "item": item}


def prepare_datasets(
    config: DatasetConfig,
    tokenizer,
) -> tuple[ReasoningGymDataset, ReasoningGymDataset]:
    """Prepare training and evaluation datasets from config."""
    developer_prompt = SYSTEM_PROMPTS[config.developer_prompt]

    # Training dataset specs
    train_dataset_specs = [
        DatasetSpec(
            name=name,
            weight=ds_config.weight,
            config=ds_config.config,
        )
        for name, ds_config in config.datasets.items()
    ]
    
    # Eval dataset specs: use eval_datasets if provided, otherwise use same as training
    if config.eval_datasets:
        eval_dataset_specs = [
            DatasetSpec(
                name=name,
                weight=ds_config.weight,
                config=ds_config.config,
            )
            for name, ds_config in config.eval_datasets.items()
        ]
    else:
        eval_dataset_specs = train_dataset_specs
    
    train_data = reasoning_gym.create_dataset(
        "composite", seed=1, size=config.dataset_size, datasets=train_dataset_specs
    )
    eval_data = reasoning_gym.create_dataset(
        "composite", seed=2, size=config.eval_num_samples, datasets=eval_dataset_specs
    )
    
    train_dataset = ReasoningGymDataset(
        tokenizer=tokenizer,
        procedural_dataset=train_data,
        developer_prompt=developer_prompt,
        developer_role=config.developer_role,
    )
    eval_dataset = ReasoningGymDataset(
        tokenizer=tokenizer,
        procedural_dataset=eval_data,
        developer_prompt=developer_prompt,
        developer_role=config.developer_role,
    )
    
    return train_dataset, eval_dataset
