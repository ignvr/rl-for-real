# RL for Real

RL training on reasoning tasks using [reasoning-gym](https://github.com/open-thought/reasoning-gym).

## Quick Start

### 1. Create Environment

```bash
# Create and activate venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

### 2. Install PyTorch with CUDA

Install PyTorch **before** other dependencies. Use [pytorch.org/get-started](https://pytorch.org/get-started/locally/) to get the right command for your system. Example:

```bash
# Example for CUDA 12.8 (check pytorch.org for current options)
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Verify (`Available` should be `True`):
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, Available: {torch.cuda.is_available()}')"
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Login to Services (Optional)

```bash
# For model access (if using gated models)
huggingface-cli login

# For experiment tracking
wandb login
```

### 5. Run Training

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

## Troubleshooting

### CUDA error: no kernel image is available for execution on the device

PyTorch doesn't have CUDA kernels for your GPU. Run this diagnostic:

```bash
python -c "import torch; c=torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0,0); print(f'CUDA: {torch.version.cuda}, Available: {torch.cuda.is_available()}'); print(f'GPU compute: sm_{c[0]}{c[1]}, Supported: {torch.cuda.get_arch_list()}') if torch.cuda.is_available() else print('No CUDA')"
```

If your GPU's `sm_XX` is **not** in the `Supported` list, that's the problem. Reinstall PyTorch from [pytorch.org/get-started](https://pytorch.org/get-started/locally/) with a CUDA version that supports your GPU, or your GPU may be too old.
