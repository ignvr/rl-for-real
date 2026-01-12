"""Utility functions for GRPO training."""

from dataclasses import asdict


def build_wandb_config(reasoning_gym_args, training_args, model_args) -> dict:
    """Build a comprehensive config dict for wandb logging."""
    config = {}
    
    # Add reasoning_gym (DatasetConfig) args
    try:
        rg_dict = asdict(reasoning_gym_args)
        # Convert nested DatasetConfigItem objects to dicts
        if "datasets" in rg_dict and rg_dict["datasets"]:
            rg_dict["datasets"] = {
                name: asdict(item) if hasattr(item, '__dataclass_fields__') else item
                for name, item in reasoning_gym_args.datasets.items()
            }
        if "eval_datasets" in rg_dict and rg_dict["eval_datasets"]:
            rg_dict["eval_datasets"] = {
                name: asdict(item) if hasattr(item, '__dataclass_fields__') else item
                for name, item in reasoning_gym_args.eval_datasets.items()
            }
        config["reasoning_gym"] = rg_dict
    except Exception:
        config["reasoning_gym"] = str(reasoning_gym_args)
    
    # Add training args (GRPOConfig)
    try:
        if hasattr(training_args, 'to_dict'):
            config["training"] = training_args.to_dict()
        else:
            config["training"] = {k: v for k, v in vars(training_args).items() 
                                   if not k.startswith('_') and not callable(v)}
    except Exception:
        config["training"] = str(training_args)
    
    # Add model args
    try:
        if hasattr(model_args, 'to_dict'):
            config["model"] = model_args.to_dict()
        elif hasattr(model_args, '__dataclass_fields__'):
            config["model"] = asdict(model_args)
        else:
            config["model"] = {k: v for k, v in vars(model_args).items() 
                               if not k.startswith('_') and not callable(v)}
    except Exception:
        config["model"] = str(model_args)
    
    return config


def log_trainable_params(model):
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
