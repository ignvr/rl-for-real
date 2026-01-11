#!/usr/bin/env python3
"""
RL for Real - GRPO Training on Reasoning Gym Tasks

This script trains language models using Group Relative Policy Optimization (GRPO)
on procedurally generated reasoning tasks from reasoning-gym.

Usage:
    python train_grpo.py --config-path config --config-name grpo

Features:
    - Fixed evaluation on consistent samples for reliable progress tracking
    - Reward variance logging to diagnose exploration
    - Baseline evaluation at step 0
    - Automatic wandb integration
    - JSON export of evaluation results
"""

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import torch
import transformers
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser

import reasoning_gym
from reasoning_gym.composite import DatasetSpec
from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.utils import SYSTEM_PROMPTS, extract_answer

# Try to import wandb for logging (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Try to import peft for LoRA (optional)
try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


# =============================================================================
# Configuration Dataclasses
# =============================================================================

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


# =============================================================================
# Dataset Wrapper
# =============================================================================

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


# =============================================================================
# Custom GRPO Trainer
# =============================================================================

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
        self._step_count = 0

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


# =============================================================================
# Fixed Evaluation Callback
# =============================================================================

class FixedEvalCallback(TrainerCallback):
    """
    Callback for reliable evaluation on a fixed set of samples.
    
    - Evaluates the same samples every time for comparable metrics
    - Logs to wandb
    - Saves results to JSON
    - Prints example outputs
    """
    
    def __init__(
        self,
        eval_dataset: ReasoningGymDataset,
        tokenizer,
        num_samples: int = 30,
        num_examples_to_print: int = 3,
        max_new_tokens: int = 512,
        output_dir: str = None,
        eval_steps: int = 50,
    ):
        self.tokenizer = tokenizer
        self.num_samples = min(num_samples, len(eval_dataset))
        self.num_examples_to_print = num_examples_to_print
        self.max_new_tokens = max_new_tokens
        self.output_dir = output_dir
        self.eval_steps = eval_steps
        self.last_eval_step = -1
        
        # Create fixed eval samples
        self.fixed_samples = []
        for i in range(self.num_samples):
            item = eval_dataset[i]
            # Get task name from composite dataset metadata (key is "source_dataset")
            task_name = item["item"].get("metadata", {}).get("source_dataset", "unknown")
            self.fixed_samples.append({
                "prompt": item["prompt"],
                "item": item["item"],
                "question": item["item"]["question"],
                "answer": item["item"]["answer"],
                "task": task_name,
            })
        
        # Count samples per task for logging
        task_counts = {}
        for sample in self.fixed_samples:
            task_counts[sample["task"]] = task_counts.get(sample["task"], 0) + 1
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"FixedEvalCallback: {self.num_samples} samples, eval every {eval_steps} steps")
        self.logger.info(f"  Tasks: {task_counts}")
    
    def on_log(self, args, state, control, model=None, **kwargs):
        """Run evaluation at fixed intervals."""
        if model is None:
            return
        
        step = state.global_step
        if step > 0 and step % self.eval_steps == 0 and step != self.last_eval_step:
            self._run_eval(model, step)
            self.last_eval_step = step
    
    def on_save(self, args, state, control, model=None, **kwargs):
        """Also run evaluation when checkpoints are saved."""
        if model is None:
            return
        
        step = state.global_step
        if step != self.last_eval_step:
            self._run_eval(model, step)
            self.last_eval_step = step
    
    def _run_eval(self, model, step: int):
        """Run evaluation and log/save results."""
        print(f"\n{'='*60}")
        print(f"Running fixed evaluation at step {step}")
        print(f"{'='*60}")

        results = self._evaluate(model, step)
        self._log_to_wandb(results, step)
        self._save_results(results, step)
        self._print_results(results, step)
        torch.cuda.empty_cache()
    
    def _evaluate(self, model, step: int) -> dict:
        """Evaluate model on fixed samples."""
        model.eval()
        device = next(model.parameters()).device
        
        num_correct = 0
        format_scores = []
        examples = []
        
        # Per-task tracking
        task_stats = {}  # task_name -> {"correct": int, "total": int, "format_scores": list}
        
        for sample in self.fixed_samples:
            task_name = sample["task"]
            
            # Initialize task stats if needed
            if task_name not in task_stats:
                task_stats[task_name] = {"correct": 0, "total": 0, "format_scores": []}
            
            inputs = self.tokenizer(
                sample["prompt"],
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            extracted = extract_answer(generated_text)

            is_correct = False
            if isinstance(extracted, str):
                is_correct = extracted.strip().lower() == str(sample["answer"]).strip().lower()

            # PERMISSIVE = True
            # if PERMISSIVE:
            #     is_correct |= sample["answer"].strip().lower() in generated_text[-20:].strip().lower()

            if is_correct:
                num_correct += 1
                task_stats[task_name]["correct"] += 1
            
            task_stats[task_name]["total"] += 1
            
            format_score = self._score_format(generated_text)
            format_scores.append(format_score)
            task_stats[task_name]["format_scores"].append(format_score)
            
            examples.append({
                "question": sample["question"],
                "ground_truth": sample["answer"],
                "generated": generated_text,
                "extracted": extracted,
                "correct": is_correct,
                "format_score": format_score,
                "task": task_name,
            })
        
        torch.cuda.empty_cache()
        model.train()
        
        accuracy = (num_correct / len(self.fixed_samples)) * 100 if self.fixed_samples else 0
        avg_format = (sum(format_scores) / len(format_scores)) * 100 if format_scores else 0
        
        # Compute per-task metrics
        per_task_results = {}
        for task_name, stats in task_stats.items():
            task_accuracy = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
            task_format = (sum(stats["format_scores"]) / len(stats["format_scores"]) * 100) if stats["format_scores"] else 0
            per_task_results[task_name] = {
                "accuracy": task_accuracy,
                "format_score": task_format,
                "num_correct": stats["correct"],
                "num_samples": stats["total"],
            }
        
        return {
            "accuracy": accuracy,
            "format_score": avg_format,
            "num_correct": num_correct,
            "num_samples": len(self.fixed_samples),
            "examples": examples,
            "per_task": per_task_results,
        }
    
    def _score_format(self, text: str) -> float:
        """Score format based on presence of tags."""
        score = 0.0
        if re.search(r"<think>", text, re.IGNORECASE):
            score += 0.25
        if re.search(r"</think>", text, re.IGNORECASE):
            score += 0.25
        if re.search(r"<answer>", text, re.IGNORECASE):
            score += 0.25
        if re.search(r"</answer>", text, re.IGNORECASE):
            score += 0.25
        return score
    
    def _log_to_wandb(self, results: dict, step: int):
        """Log evaluation results to wandb."""
        if not WANDB_AVAILABLE or wandb.run is None:
            return
        try:
            log_dict = {
                "eval/accuracy": results["accuracy"],
                "eval/format_score": results["format_score"],
                "eval/num_correct": results["num_correct"],
                "eval/step": step,
            }
            
            # Add per-task metrics
            for task_name, task_results in results.get("per_task", {}).items():
                log_dict[f"eval/{task_name}/accuracy"] = task_results["accuracy"]
                log_dict[f"eval/{task_name}/format_score"] = task_results["format_score"]
            
            wandb.log(log_dict, commit=False)
            self.logger.info(f"📊 Logged to wandb: accuracy={results['accuracy']:.1f}%")
        except Exception as e:
            self.logger.warning(f"Failed to log to wandb: {e}")
    
    def _save_results(self, results: dict, step: int):
        """Save evaluation results to JSON file."""
        if self.output_dir is None:
            return
        
        eval_dir = os.path.join(self.output_dir, "eval_results")
        os.makedirs(eval_dir, exist_ok=True)
        
        save_data = {
            "step": step,
            "accuracy": results["accuracy"],
            "format_score": results["format_score"],
            "num_correct": results["num_correct"],
            "num_samples": results["num_samples"],
            "per_task": results.get("per_task", {}),
            "examples": results["examples"],
        }
        
        filepath = os.path.join(eval_dir, f"eval_step_{step}.json")
        try:
            with open(filepath, "w") as f:
                json.dump(save_data, f, indent=2, default=str)
            self.logger.info(f"📁 Saved eval results to {filepath}")
        except Exception as e:
            self.logger.warning(f"Failed to save eval results: {e}")
    
    def _print_results(self, results: dict, step: int):
        """Print evaluation results summary and examples."""
        print(f"\n{'='*70}")
        print(f"📊 Evaluation Results (Step {step})")
        print(f"{'='*70}")
        print(f"   Overall Accuracy:     {results['accuracy']:.1f}% ({results['num_correct']}/{results['num_samples']})")
        print(f"   Overall Format Score: {results['format_score']:.1f}%")
        
        # Print per-task breakdown
        per_task = results.get("per_task", {})
        if per_task:
            print(f"\n   {'─'*50}")
            print(f"   Per-Task Breakdown:")
            print(f"   {'─'*50}")
            for task_name in sorted(per_task.keys()):
                task_results = per_task[task_name]
                print(f"   {task_name:25s} {task_results['accuracy']:5.1f}% "
                      f"({task_results['num_correct']}/{task_results['num_samples']}) "
                      f"| format: {task_results['format_score']:.0f}%")
        
        print(f"{'='*70}")
        
        examples = results["examples"]
        correct_examples = [e for e in examples if e["correct"]]
        incorrect_examples = [e for e in examples if not e["correct"]]
        
        to_print = []
        if correct_examples:
            to_print.append(correct_examples[0])
        to_print.extend(incorrect_examples[:self.num_examples_to_print - len(to_print)])
        if len(to_print) < self.num_examples_to_print:
            to_print.extend(correct_examples[1:self.num_examples_to_print - len(to_print) + 1])
        
        for i, ex in enumerate(to_print[:self.num_examples_to_print]):
            status = "✓ CORRECT" if ex["correct"] else "✗ WRONG"
            task_name = ex.get("task", "unknown")
            print(f"\n{'─'*70}")
            print(f"Example {i+1} [{status}] | Task: {task_name} | Format: {ex['format_score']*100:.0f}%")
            print(f"{'─'*70}")
            
            q = ex['question']
            if len(q) > 300:
                q = q[:300] + "..."
            print(f"QUESTION:\n{q}\n")
            print(f"EXPECTED: {ex['ground_truth']}")
            print(f"EXTRACTED: {ex['extracted']}")
            
            print(f"\nMODEL OUTPUT:")
            output = ex['generated']
            if len(output) > 1000:
                output = output[:1000] + "\n... [truncated]"
            print(output)
        
        print("=" * 70 + "\n")


# =============================================================================
# Dataset Preparation
# =============================================================================

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


# =============================================================================
# Eval-Only Mode
# =============================================================================

def _run_eval_only(reasoning_gym_args, training_args, model_args, logger):
    """Run evaluation only (no training)."""
    print("\n" + "=" * 70)
    print("🔍 EVAL-ONLY MODE")
    print("=" * 70)
    
    # Determine model path: use checkpoint_path if provided, else base model
    model_path = reasoning_gym_args.checkpoint_path or model_args.model_name_or_path
    logger.info(f"Loading model from: {model_path}")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Try to load tokenizer from checkpoint, fall back to base model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except:
        logger.info(f"Tokenizer not found at {model_path}, using base model tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    
    logger.info("✓ Model loaded")
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n   Model: {model_path}")
    print(f"   Parameters: {total_params:,}")
    
    # Prepare eval dataset only
    logger.info("Preparing evaluation dataset...")
    _, eval_dataset = prepare_datasets(reasoning_gym_args, tokenizer)
    logger.info(f"✓ Eval dataset: {len(eval_dataset)} samples available, using {reasoning_gym_args.eval_num_samples}")
    
    # Create eval callback (reuse existing infrastructure)
    fixed_eval_callback = FixedEvalCallback(
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        num_samples=reasoning_gym_args.eval_num_samples,
        num_examples_to_print=reasoning_gym_args.eval_num_examples_to_print,
        output_dir=training_args.output_dir,
        eval_steps=reasoning_gym_args.fixed_eval_steps,
    )
    
    # Initialize wandb if configured
    if WANDB_AVAILABLE and "wandb" in (training_args.report_to or []):
        try:
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "rl-for-real"),
                name=f"{training_args.run_name}_eval" if training_args.run_name else "eval",
                config={
                    "mode": "eval_only",
                    "model": model_path,
                    "eval_num_samples": reasoning_gym_args.eval_num_samples,
                },
            )
            logger.info("✓ Initialized wandb")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
    
    # Run evaluation
    print("\n" + "=" * 70)
    print("Running evaluation...")
    print("=" * 70)
    
    results = fixed_eval_callback._evaluate(model, step=0)
    fixed_eval_callback._print_results(results, step=0)
    fixed_eval_callback._log_to_wandb(results, step=0)
    fixed_eval_callback._save_results(results, step=0)
    
    # Clean up
    print("\n✓ Evaluation complete!")
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()
    del model
    torch.cuda.empty_cache()


# =============================================================================
# Utility Functions
# =============================================================================

def _log_trainable_params(model):
    """Print trainable vs total parameters to check if LoRA is active."""
    total_params = 0
    trainable_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    pct = (trainable_params / total_params * 100) if total_params > 0 else 0
    
    # Check if PEFT/LoRA is active
    is_peft = hasattr(model, 'peft_config') or 'PeftModel' in type(model).__name__
    
    print("\n" + "=" * 60)
    print("📊 Model Parameter Summary")
    print("=" * 60)
    print(f"   Total parameters:     {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Trainable %:          {pct:.2f}%")
    print(f"   PEFT/LoRA active:     {'✓ YES' if is_peft else '✗ NO (full fine-tuning)'}")
    
    if is_peft and hasattr(model, 'peft_config'):
        for adapter_name, config in model.peft_config.items():
            print(f"   LoRA rank (r):        {getattr(config, 'r', 'N/A')}")
            print(f"   LoRA alpha:           {getattr(config, 'lora_alpha', 'N/A')}")
            print(f"   Target modules:       {getattr(config, 'target_modules', 'N/A')}")
    
    print("=" * 60 + "\n")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    # Parse arguments from config
    parser = TrlParser((DatasetConfig, GRPOConfig, ModelConfig))
    reasoning_gym_args, training_args, model_args = parser.parse_args_and_config()
    set_seed(training_args.seed)

    # Set up logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    # Create output directory as {output_dir}/{run_name}/
    if training_args.run_name:
        training_args.output_dir = os.path.join(training_args.output_dir, training_args.run_name)
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    logger.info(f"Model: {model_args}")
    logger.info(f"Training: {training_args}")
    logger.info(f"Output directory: {training_args.output_dir}")

    # =========================================================================
    # EVAL-ONLY MODE
    # =========================================================================
    if reasoning_gym_args.eval_only:
        _run_eval_only(reasoning_gym_args, training_args, model_args, logger)
        return

    # =========================================================================
    # TRAINING MODE
    # =========================================================================
    
    # Load model
    logger.info(f"Loading model: {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=model_args.attn_implementation,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    logger.info("✓ Model loaded")
    
    # Apply PEFT/LoRA if configured
    if getattr(model_args, 'use_peft', False):
        if not PEFT_AVAILABLE:
            raise ImportError("use_peft=True but peft is not installed. Run: pip install peft")
        
        lora_r = getattr(model_args, 'lora_r', 16)
        lora_alpha = getattr(model_args, 'lora_alpha', lora_r)  # default alpha = r
        lora_dropout = getattr(model_args, 'lora_dropout', 0.05)
        target_modules = getattr(model_args, 'lora_target_modules', None)
        
        # Handle "all-linear" shortcut
        if target_modules == "all-linear":
            target_modules = None  # PEFT will auto-detect all linear layers
            
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, lora_config)
        logger.info(f"✓ Applied LoRA: r={lora_r}, alpha={lora_alpha}, target_modules={target_modules or 'auto'}")

    # Prepare datasets
    logger.info("Preparing datasets...")
    training_args.reasoning_gym = reasoning_gym_args
    train_dataset, eval_dataset = prepare_datasets(reasoning_gym_args, tokenizer)
    logger.info(f"✓ Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

    # Initialize trainer
    trainer = CustomGRPOTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        log_reward_variance=reasoning_gym_args.log_reward_variance,
        verbose_variance_logging=reasoning_gym_args.verbose_variance_logging,
    )
    
    # Log parameter info (check if LoRA is active)
    _log_trainable_params(trainer.model)
    
    # Add fixed evaluation callback
    fixed_eval_callback = FixedEvalCallback(
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        num_samples=reasoning_gym_args.eval_num_samples,
        num_examples_to_print=reasoning_gym_args.eval_num_examples_to_print,
        output_dir=training_args.output_dir,
        eval_steps=reasoning_gym_args.fixed_eval_steps,
    )
    trainer.add_callback(fixed_eval_callback)
    logger.info(f"✓ Fixed eval callback: {reasoning_gym_args.eval_num_samples} samples every {reasoning_gym_args.fixed_eval_steps} steps")

    # Check for resume checkpoint
    # Priority: explicit resume_from_checkpoint > auto_resume > start fresh
    ckpt = None
    if training_args.resume_from_checkpoint is not None:
        # Explicit checkpoint path provided
        ckpt = training_args.resume_from_checkpoint
        logger.info(f"Resuming from explicit checkpoint: {ckpt}")
    elif reasoning_gym_args.auto_resume and os.path.isdir(training_args.output_dir):
        # Auto-resume enabled: look for last checkpoint
        ckpt = get_last_checkpoint(training_args.output_dir)
        if ckpt:
            logger.info(f"Auto-resuming from checkpoint: {ckpt}")
        else:
            logger.info("Auto-resume enabled but no checkpoint found. Starting fresh.")
    else:
        logger.info("Starting training from scratch (auto_resume=False)")

    # Run baseline evaluation (Step 0)
    if WANDB_AVAILABLE and "wandb" in (training_args.report_to or []):
        try:
            if wandb.run is None:
                wandb.init(
                    project=os.environ.get("WANDB_PROJECT", "rl-for-real"),
                    name=training_args.run_name,
                    config={
                        "model": model_args.model_name_or_path,
                        "dataset_size": reasoning_gym_args.dataset_size,
                        "max_steps": training_args.max_steps,
                    },
                )
                logger.info("✓ Initialized wandb")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
    
    print("\n" + "="*60)
    print("Running baseline evaluation (Step 0)...")
    print("="*60)
    baseline_results = fixed_eval_callback._evaluate(model, step=0)
    fixed_eval_callback._print_results(baseline_results, step=0)
    fixed_eval_callback._log_to_wandb(baseline_results, step=0)
    fixed_eval_callback._save_results(baseline_results, step=0)

    # Train!
    logger.info("\n" + "="*60)
    logger.info("Starting training...")
    logger.info("="*60 + "\n")
    
    train_result = trainer.train(resume_from_checkpoint=ckpt)
    train_metrics = train_result.metrics
    
    # Save final model and training state
    logger.info("\n" + "="*60)
    logger.info("Saving final model and training state...")
    logger.info("="*60)
    
    # Save metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    
    # Save trainer state (trainer_state.json)
    trainer.save_state()
    
    # Save the final model
    final_model_path = os.path.join(training_args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"✓ Final model saved to: {final_model_path}")
    
    # Also save to output_dir root for convenience
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"✓ Model also saved to: {training_args.output_dir}")

    # Clean up
    logger.info("\n✓ Training complete!")
    del trainer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

