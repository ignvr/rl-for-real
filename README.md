# RL for Real

RL training on reasoning tasks using [reasoning-gym](https://github.com/open-thought/reasoning-gym).

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
# Run with default config
python train_grpo.py --config config/grpo.yaml

# Toy run with Qwen-1.5B
python train_grpo.py --config config/qwen_1.5B_syllogism.yaml
```

### Available Tasks

See [reasoning-gym datasets](https://github.com/open-thought/reasoning-gym/tree/main/reasoning_gym) for full list. Some examples:

| Category | Tasks |
|----------|-------|
| **Arithmetic** | `basic_arithmetic`, `chain_sum`, `complex_arithmetic` |
| **Logic** | `syllogism`, `propositional_logic`, `knights_knaves` |
| **Algorithmic** | `spell_backward`, `word_sorting`, `letter_counting` |
| **Games** | `sudoku`, `countdown`, `game_of_24` |

## GPU Requirements

| Model | VRAM Needed |
|-------|-------------|
| Qwen2.5-1.5B-Instruct | ~16GB |
| Qwen2.5-3B-Instruct | ~24GB |

Reduce batch size (`per_device_train_batch_size`), enable `lora` or enable gradient checkpointing if running low on memory.
