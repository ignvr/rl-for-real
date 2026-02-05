# RL for Real

RL training on reasoning tasks using [TRL](https://github.com/huggingface/trl) framework and [reasoning-gym](https://github.com/open-thought/reasoning-gym) benchmark.

This repo was written by Ido Greenberg with advisement by Oran Lang and Gal Chechik for the course [RL-for-Real](https://docs.google.com/document/d/1fmfYp7EH9fqcB7CWWBvrZ40MtCN89Sr_o3o3EG9hWyE/edit?usp=sharing).
The course is organized by NVIDIA Research in collaboration with Google Research, Mentee Robotics, Tel-Aviv University, Bar-Ilan University, and the Technion.

## Quick Start

### 1. Create Environment

```bash
# Create and activate conda environment with Python 3.10+
conda create -n rl-for-real python=3.10 -y
conda activate rl-for-real
```

### 2. Install PyTorch with CUDA

Install PyTorch **before** other dependencies. Use [pytorch.org/get-started](https://pytorch.org/get-started/locally/) to get the right command for your system. Example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Verify (`Available` should be `True`):
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, Available: {torch.cuda.is_available()}')"
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Login to Wandb (Optional)

```bash
wandb login
```

## Exercise

### Warmup

Run
```bash
python train_grpo.py --config config/syllogism_toy.yaml
```
This will train Llama-3.2-1B-Instruct for several minutes to solve the `syllogism` task.

Observe the training and evaluation curves at wandb, as well as examples for questions and answers in the stdout logs. What changes in the model behavior?

### Train Qwen for better reasoning

#### Intro

You may run
```bash
python train_grpo.py --config config/multi_tasks.yaml
```
to train Qwen2.5-3B-Instruct to solve several reasoning tasks (`number_filtering`, `string_insertion`, `family_relationships`, `maze`) and test on the same tasks as well as the additional task of `base_conversion`.

Using `config/multi_tasks.yaml` as provided should result in ~40min training.

#### Task

Try to improve Qwen's training and achieve the best evaluation scores you can.

#### Potential directions

You may take whichever approach you like to improve the results.
Below are several ideas.
1. Consider hyper-parameters in `config/multi_tasks.yaml`, specifically under the section titles `Training steps`, `Batch sizes`, `Learning rate`, `GRPO-specific`, `Model behavior`, `LORA`. Can you tell what each hyper-parameter does? Which ones do you expect to have the largest effects on training?
2. What is the GPU utilization under the current training? Which hyper-parameters can be used to increase it, if needed?
3. The function `_format_reward()` in `trainer.py` defines the reward for answer format correctness, and is summed with `_accuracy_reward()`. Currently, each of the two functions returns a reward component of the same scale, in `[0,1]`. Is this the right balance? Consider the logic and scale of `_format_reward()` in light of the training curves.
4. Extended coding:
    * Try using TRL's KL-regularization. Notice that it restricts the model updates with respect to the *original* model weights ("global" regularization). Modify the regularization to be "local": every update, restrict the new model with respect to the *current* model, not the original one.
    * TRL uses [`get_off_policy_mask()`](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py#L1929) to remove training examples that are both non-beneficial (advantage<0) and off-policy (KL>threshold). Consider changing the threshold (`off_policy_threshold`) to extend or focus the scope of the train data.
    * Implement an actor-critic training framework.

## GPU Requirements

| Model | VRAM Needed |
|-------|-------------|
| Qwen2.5-1.5B-Instruct | 16GB |
| Qwen2.5-3B-Instruct | Minimal: 24GB; Recommended: 40GB |

Reduce batch size (`per_device_train_batch_size`), enable `lora` or enable gradient checkpointing if running low on memory.

## Troubleshooting

### CUDA error: no kernel image is available for execution on the device

PyTorch doesn't have CUDA kernels for your GPU. Run this diagnostic:

```bash
python -c "import torch; c=torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0,0); print(f'CUDA: {torch.version.cuda}, Available: {torch.cuda.is_available()}'); print(f'GPU compute: sm_{c[0]}{c[1]}, Supported: {torch.cuda.get_arch_list()}') if torch.cuda.is_available() else print('No CUDA')"
```

If your GPU's `sm_XX` is **not** in the `Supported` list, that's the problem. Reinstall PyTorch from [pytorch.org/get-started](https://pytorch.org/get-started/locally/) with a CUDA version that supports your GPU, or your GPU may be too old.
