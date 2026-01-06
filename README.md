# RL for Real

Minimal RL training on reasoning tasks using [reasoning-gym](https://github.com/open-thought/reasoning-gym).

## Quick Start

### 1. Create Environment

```bash
# Create and activate venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Login to Services (Optional)

```bash
# For model access (if using gated models)
huggingface-cli login

# For experiment tracking
wandb login
```

### 3. Run Training

```bash
# Basic run with default config
python train_grpo.py --config-path config --config-name grpo

# Override settings via command line
python train_grpo.py --config-path config --config-name grpo \
    max_steps=100 \
    model_name_or_path=Qwen/Qwen2.5-1.5B-Instruct

# Disable wandb
python train_grpo.py --config-path config --config-name grpo \
    report_to=none
```

## Configuration

Edit `config/grpo.yaml` to customize:

| Section | Key Settings |
|---------|-------------|
| **Datasets** | `datasets` - which reasoning-gym tasks to train on |
| **Model** | `model_name_or_path` - base model to train |
| **Training** | `max_steps`, `learning_rate`, `per_device_train_batch_size` |
| **GRPO** | `num_generations` - rollouts per prompt |
| **Evaluation** | `fixed_eval_steps` - how often to evaluate |

### Example: Change Training Tasks

```yaml
datasets:
  # Simple arithmetic
  basic_arithmetic:
    weight: 1.0
    config:
      min_digits: 1
      max_digits: 2
      operators: ["+", "-"]
  
  # Logic puzzles
  syllogism:
    weight: 1.0
    config:
      num_premises: 2
```

### Available Tasks

See [reasoning-gym datasets](https://github.com/open-thought/reasoning-gym/tree/main/reasoning_gym) for full list. Some examples:

| Category | Tasks |
|----------|-------|
| **Arithmetic** | `basic_arithmetic`, `chain_sum`, `complex_arithmetic` |
| **Logic** | `syllogism`, `propositional_logic`, `knights_knaves` |
| **Algorithmic** | `spell_backward`, `word_sorting`, `letter_counting` |
| **Games** | `sudoku`, `countdown`, `game_of_24` |

## Outputs

Training creates:

```
outputs/grpo_trained/
├── checkpoint-100/          # Model checkpoints
├── checkpoint-200/
├── eval_results/            # JSON files with evaluation metrics
│   ├── eval_step_0.json     # Baseline (pre-training)
│   ├── eval_step_50.json
│   └── ...
└── train_results.json       # Final training metrics
```

## Features

- **Fixed Evaluation**: Same problems evaluated every time for reliable comparisons
- **Baseline at Step 0**: See starting performance before any training
- **Variance Logging**: Track exploration (% of prompts with diverse rollouts)
- **WandB Integration**: Automatic logging of train and eval metrics

## GPU Requirements

| Model | VRAM Needed |
|-------|-------------|
| Qwen2.5-1.5B-Instruct | ~16GB |
| Qwen2.5-3B-Instruct | ~24GB |

Reduce batch size or enable gradient checkpointing if running low on memory.

## Troubleshooting

**Flash Attention not available:**
```yaml
attn_implementation: eager  # instead of flash_attention_2
```

**Out of memory:**
```yaml
per_device_train_batch_size: 4  # reduce from 8
gradient_accumulation_steps: 8  # increase to maintain effective batch
```

**Slow training:**
```yaml
fixed_eval_steps: 100  # evaluate less often
eval_num_samples: 20   # fewer eval samples
```

