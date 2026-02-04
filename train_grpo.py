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

import logging
import os
import sys
import math
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, ModelConfig, TrlParser

import grpo

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
# Setup Functions
# =============================================================================

def _setup_logging(training_args):
    """Configure logging for the training run."""
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
    return logger


def _load_model(model_args, training_args, logger):
    """Load model and tokenizer, optionally applying LoRA."""
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
        model = _apply_lora(model, model_args, logger)
    
    return model, tokenizer


def _apply_lora(model, model_args, logger):
    """Apply LoRA adapter to the model."""
    if not PEFT_AVAILABLE:
        raise ImportError("use_peft=True but peft is not installed. Run: pip install peft")
    
    lora_r = getattr(model_args, 'lora_r', 16)
    lora_alpha = getattr(model_args, 'lora_alpha', lora_r)
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
    return model


def _find_checkpoint(reasoning_gym_args, training_args, logger):
    """Determine if we should resume from a checkpoint."""
    if training_args.resume_from_checkpoint is not None:
        logger.info(f"Resuming from explicit checkpoint: {training_args.resume_from_checkpoint}")
        return training_args.resume_from_checkpoint
    
    if reasoning_gym_args.auto_resume and os.path.isdir(training_args.output_dir):
        ckpt = get_last_checkpoint(training_args.output_dir)
        if ckpt:
            logger.info(f"Auto-resuming from checkpoint: {ckpt}")
            return ckpt
        logger.info("Auto-resume enabled but no checkpoint found. Starting fresh.")
    else:
        logger.info("Starting training from scratch (auto_resume=False)")
    
    return None


def _init_wandb(reasoning_gym_args, training_args, model_args, logger, mode="training"):
    """Initialize wandb if configured."""
    if not WANDB_AVAILABLE or "wandb" not in (training_args.report_to or []):
        return
    
    try:
        # Disable HuggingFace's automatic wandb integration - we'll log manually
        training_args.report_to = []
        
        if wandb.run is None:
            full_config = grpo.build_wandb_config(reasoning_gym_args, training_args, model_args)
            full_config["mode"] = mode
            
            name = training_args.run_name
            if mode == "eval_only":
                name = f"{name}_eval" if name else "eval"
            
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "rl-for-real"),
                name=name,
                config=full_config,
            )
            
            logger.info(f"✓ Initialized wandb manually (disabled HuggingFace's automatic logging)")
            logger.info(f"  Using step parameter in wandb.log() for explicit x-axis control")
    except Exception as e:
        logger.warning(f"Failed to initialize wandb: {e}")


def _save_model(trainer, tokenizer, training_args, logger):
    """Save the final model and training state."""
    logger.info("\n" + "="*60)
    logger.info("Saving final model and training state...")
    logger.info("="*60)
    
    # Save trainer state
    trainer.save_state()
    
    # Save the final model to dedicated directory
    final_model_path = os.path.join(training_args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"✓ Final model saved to: {final_model_path}")
    
    # Also save to output_dir root for convenience
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"✓ Model also saved to: {training_args.output_dir}")


# =============================================================================
# Eval-Only Mode
# =============================================================================

def _run_eval_only(reasoning_gym_args, training_args, model_args, logger):
    """Run evaluation only (no training)."""
    print("\n" + "=" * 70)
    print("🔍 EVAL-ONLY MODE")
    print("=" * 70)
    
    # Determine model path
    model_path = reasoning_gym_args.checkpoint_path or model_args.model_name_or_path
    logger.info(f"Loading model from: {model_path}")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except:
        logger.info(f"Tokenizer not found at {model_path}, using base model tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    
    logger.info("✓ Model loaded")
    print(f"\n   Model: {model_path}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Prepare eval dataset
    logger.info("Preparing evaluation dataset...")
    _, eval_dataset = grpo.prepare_datasets(reasoning_gym_args, tokenizer)
    logger.info(f"✓ Eval dataset: {len(eval_dataset)} samples")
    
    # Create eval callback
    eval_callback = grpo.FixedEvalCallback(
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        num_samples=reasoning_gym_args.eval_num_samples,
        num_examples_to_print=reasoning_gym_args.eval_num_examples_to_print,
        output_dir=training_args.output_dir,
        eval_steps=reasoning_gym_args.fixed_eval_steps,
        temperature=reasoning_gym_args.eval_temperature,
    )
    
    # Initialize wandb
    _init_wandb(reasoning_gym_args, training_args, model_args, logger, mode="eval_only")
    
    # Run evaluation
    print("\n" + "=" * 70)
    print("Running evaluation...")
    print("=" * 70)
    
    results = eval_callback._evaluate(model, step=0)
    eval_callback._print_results(results, step=0)
    eval_callback._log_to_wandb(results, step=0)
    eval_callback._save_results(results, step=0)
    
    # Clean up
    print("\n✓ Evaluation complete!")
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()
    del model
    torch.cuda.empty_cache()


# =============================================================================
# Training Mode
# =============================================================================

def _run_training(reasoning_gym_args, training_args, model_args, logger):
    """Run the full training pipeline."""
    # Load model
    model, tokenizer = _load_model(model_args, training_args, logger)
    
    # Prepare datasets
    logger.info("Preparing datasets...")
    training_args.reasoning_gym = reasoning_gym_args
    train_dataset, eval_dataset = grpo.prepare_datasets(reasoning_gym_args, tokenizer)
    logger.info(f"✓ Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

    # Initialize wandb BEFORE creating trainer so we can set define_metric before HuggingFace's WandbCallback
    _init_wandb(reasoning_gym_args, training_args, model_args, logger, mode="training")

    # Initialize trainer
    trainer = grpo.CustomGRPOTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        log_reward_variance=reasoning_gym_args.log_reward_variance,
        verbose_variance_logging=reasoning_gym_args.verbose_variance_logging,
        batch_counts_as_step=reasoning_gym_args.batch_counts_as_step,
    )
    
    # Log parameter info
    grpo.log_trainable_params(trainer.model)
    
    # Add evaluation callback
    eval_callback = grpo.FixedEvalCallback(
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        num_samples=reasoning_gym_args.eval_num_samples,
        num_examples_to_print=reasoning_gym_args.eval_num_examples_to_print,
        output_dir=training_args.output_dir,
        eval_steps=reasoning_gym_args.fixed_eval_steps,
        temperature=reasoning_gym_args.eval_temperature,
    )
    trainer.add_callback(eval_callback)
    logger.info(f"✓ Eval callback: {reasoning_gym_args.eval_num_samples} samples every {reasoning_gym_args.fixed_eval_steps} steps")

    # Find checkpoint for resuming
    ckpt = _find_checkpoint(reasoning_gym_args, training_args, logger)
    
    # Run baseline evaluation (Step 0)
    print("\n" + "="*60)
    print("Running baseline evaluation (Step 0)...")
    print("="*60)
    baseline_results = eval_callback._evaluate(model, step=0)
    eval_callback._print_results(baseline_results, step=0)
    eval_callback._log_to_wandb(baseline_results, step=0)
    eval_callback._save_results(baseline_results, step=0)

    # Train!
    logger.info("\n" + "="*60)
    logger.info("Starting training...")
    logger.info("="*60 + "\n")
    
    train_result = trainer.train(resume_from_checkpoint=ckpt)
    
    # Log and save metrics
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    
    # Save model
    _save_model(trainer, tokenizer, training_args, logger)

    # Clean up
    logger.info("\n✓ Training complete!")
    del trainer
    torch.cuda.empty_cache()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    # Parse arguments from config
    parser = TrlParser((grpo.DatasetConfig, GRPOConfig, ModelConfig))
    reasoning_gym_args, training_args, model_args = parser.parse_args_and_config()
    set_seed(training_args.seed)

    # Adjust max_steps and fixed_eval_steps if batch_counts_as_step mode
    # Each step = gradient_accumulation_steps batches, so divide to keep same total batches
    if reasoning_gym_args.batch_counts_as_step:
        original_max_steps = training_args.max_steps
        original_eval_steps = reasoning_gym_args.fixed_eval_steps
        training_args.max_steps = int(math.ceil(original_max_steps / training_args.gradient_accumulation_steps))
        reasoning_gym_args.fixed_eval_steps = int(math.ceil(original_eval_steps / training_args.gradient_accumulation_steps))

    # Set up logging
    logger = _setup_logging(training_args)
    
    # Log adjustment info
    if reasoning_gym_args.batch_counts_as_step:
        logger.info(f"batch_counts_as_step: max_steps {original_max_steps} / {training_args.gradient_accumulation_steps} = {training_args.max_steps}")
        logger.info(f"batch_counts_as_step: fixed_eval_steps {original_eval_steps} / {training_args.gradient_accumulation_steps} = {reasoning_gym_args.fixed_eval_steps}")
    
    # Create output directory
    if training_args.run_name:
        training_args.output_dir = os.path.join(training_args.output_dir, training_args.run_name)
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    logger.info(f"Model: {model_args}")
    logger.info(f"Training: {training_args}")
    logger.info(f"Output directory: {training_args.output_dir}")

    # Dispatch to appropriate mode
    if reasoning_gym_args.eval_only:
        _run_eval_only(reasoning_gym_args, training_args, model_args, logger)
    else:
        _run_training(reasoning_gym_args, training_args, model_args, logger)


if __name__ == "__main__":
    main()
