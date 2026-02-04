"""Custom GRPO Trainer with reasoning-gym reward functions."""

import re

import torch
from trl import GRPOConfig, GRPOTrainer

from reasoning_gym.utils import extract_answer

from .dataset import ReasoningGymDataset

# Try to import wandb for logging (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class CustomGRPOTrainer(GRPOTrainer):
    """
    Extended GRPOTrainer with reasoning-gym reward functions.
    
    Includes:
        - Accuracy reward based on reasoning-gym's scoring
        - Format reward for proper <think>/<answer> tags
        - Optional reward variance logging
    """
    
    def __init__(
        self,
        model,
        args: GRPOConfig,
        tokenizer,
        train_dataset: ReasoningGymDataset,
        eval_dataset: ReasoningGymDataset,
        log_reward_variance: bool = True,
        verbose_variance_logging: bool = False,
        batch_counts_as_step: bool = False,
    ):
        super().__init__(
            model=model,
            reward_funcs=[
                self._accuracy_reward,
                self._format_reward,
            ],
            args=args,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        self.log_reward_variance = log_reward_variance
        self.verbose_variance_logging = verbose_variance_logging
        self.batch_counts_as_step = batch_counts_as_step
        self._step_count = 0
    
    def log(self, logs: dict, start_time: float = None) -> None:
        """Override log to manually log to wandb with correct x-axis."""
        if self.state is not None and WANDB_AVAILABLE and wandb.run is not None:
            # Determine which step metric to use
            if self.batch_counts_as_step:
                step_value = self.state.global_step * self.args.gradient_accumulation_steps
            else:
                step_value = self.state.global_step
            
            try:
                wandb_logs = {}
                
                # Add all train metrics
                for key, value in logs.items():
                    if isinstance(value, (int, float)):
                        # Prefix with train/ if not already prefixed
                        if not key.startswith("train/") and not key.startswith("eval/"):
                            wandb_logs[f"train/{key}"] = value
                        else:
                            wandb_logs[key] = value
                
                # Use step parameter to force wandb to use our step value as the x-axis
                if wandb_logs:
                    wandb.log(wandb_logs, step=step_value)
            except Exception as e:
                # Log error but don't crash training
                print(f"Warning: Failed to log to wandb at step {step_value}: {e}")
        
        # Still call parent to keep HuggingFace's internal logging
        super().log(logs, start_time)

    def _accuracy_reward(self, completions: list[str], **kwargs) -> list[float]:
        """Compute accuracy reward using reasoning-gym's scoring function."""
        assert "item" in kwargs, "The 'item' argument must be provided."
        assert len(kwargs["item"]) == len(completions)
        
        answers = [extract_answer(c) for c in completions]
        rewards = [
            self.train_dataset.data.score_answer(answer, item) 
            for answer, item in zip(answers, kwargs["item"])
        ]
        
        if self.log_reward_variance:
            self._log_variance(rewards, answers, completions)
        
        return rewards
    
    def _log_variance(self, rewards: list[float], answers: list[str], completions: list[str]):
        """Log the percentage of prompts with reward variance across rollouts."""
        num_gens = self.args.num_generations
        num_prompts = len(rewards) // num_gens
        
        if num_prompts == 0:
            return
        
        prompts_with_reward_variance = 0
        prompts_with_answer_variance = 0
        
        for i in range(num_prompts):
            start_idx = i * num_gens
            end_idx = (i + 1) * num_gens
            group_rewards = rewards[start_idx:end_idx]
            group_answers = answers[start_idx:end_idx]
            
            has_reward_variance = len(set(group_rewards)) > 1
            has_answer_variance = len(set(group_answers)) > 1
            
            if has_reward_variance:
                prompts_with_reward_variance += 1
            if has_answer_variance:
                prompts_with_answer_variance += 1
            
            # Verbose: print full completions when there's variance
            if self.verbose_variance_logging and has_answer_variance:
                group_completions = completions[start_idx:end_idx]
                print(f"\n  {'─'*60}")
                print(f"  📊 Prompt {i}: rewards={group_rewards}")
                for gen_idx, (comp, rew, ans) in enumerate(zip(group_completions, group_rewards, group_answers)):
                    comp_preview = comp[:300].replace('\n', ' ↵ ')
                    if len(comp) > 300:
                        comp_preview += "..."
                    print(f"    Gen {gen_idx} [reward={rew}]: {comp_preview}")
                    print(f"           extracted: {ans}")
                print(f"  {'─'*60}")
        
        reward_variance_pct = (prompts_with_reward_variance / num_prompts * 100)
        answer_variance_pct = (prompts_with_answer_variance / num_prompts * 100)
        
        self._step_count += 1
        
        if self.verbose_variance_logging:
            print(f"🎲 Step {self._step_count}: {reward_variance_pct:.0f}% prompts have reward variance, "
                  f"{answer_variance_pct:.0f}% have answer variance ({num_prompts} prompts × {num_gens} gens)")
        
        # Log to wandb
        if WANDB_AVAILABLE and wandb.run is not None:
            try:
                wandb.log({
                    "train/reward_variance_pct": reward_variance_pct,
                    "train/answer_variance_pct": answer_variance_pct,
                }, commit=False)
            except:
                pass
        
        # Periodic GPU cleanup
        if self._step_count % 10 == 0:
            torch.cuda.empty_cache()

    def _format_reward(self, completions: list[str], **kwargs) -> list[float]:
        """Reward proper formatting with <think> and <answer> tags."""
        def count_tags(text: str) -> float:
            count = 0.0
            if re.search(r"\s*<think>\s*", text):
                count += 0.25
            if re.search(r"\s*</think>\s*", text):
                count += 0.25
            if re.search(r"\s*<answer>\s*", text):
                count += 0.25
            if re.search(r"\s*</answer>\s*", text):
                count += 0.25
            return count

        return [count_tags(c) for c in completions]
