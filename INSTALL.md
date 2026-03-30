# Installation Guide: OpenSora + OpenVLA-OFT (H100)

## Prerequisites

- NVIDIA H100 or A100 GPU (sm_90 / sm_80)
- CUDA 12.4+ driver installed (bare-metal CUDA 12.8 is also supported)
- Git, wget, and root access (or `--no-root` if system deps are pre-installed)
- ~30 GB disk for the venv + cloned repos
- Model checkpoints downloaded:
  - `RLinf-OpenSora-LIBERO-Spatial/` (world model)
  - `Openvla-oft-SFT-libero-spatial-traj1/` (policy)

## Quick Start

```bash
# From the repo root:
bash install_opensora_h100.sh

# Activate the environment:
source .venv/bin/activate

# Run training:
bash examples/embodiment/run_embodiment.sh opensora_libero_spatial_grpo_openvlaoft LIBERO \
  runner.max_steps=3 \
  algorithm.group_size=2 \
  algorithm.rollout_epoch=2 \
  env.train.total_num_envs=2
```

## What the Install Script Does (8 Steps)

| Step | What | Why |
|------|------|-----|
| 1 | PyTorch 2.6.0+cu124 | `uv sync` installs CPU-only torch; this overrides with the CUDA build |
| 2 | LIBERO + ManiSkill | Evaluation environments (editable install, `--no-deps`) |
| 3 | OpenSora world model | Video diffusion model; deps installed with `--no-deps` to avoid torch conflicts |
| 4 | Apex (CUDA extensions) | `FusedLayerNorm` required by OpenSora; prebuilt wheel or source build |
| 5 | Flash Attention 2.7.4 | Prebuilt wheel matched to torch version and CUDA major version |
| 6 | OpenVLA-OFT + dlimp | Policy model (`--no-deps` to avoid torch 2.2 downgrade) |
| 7 | Unified pin file | Restores numpy, timm, jsonlines, tensorflow, and other cross-cutting deps |
| 8 | Ray restart | Workers must inherit the new environment |

## Running Training

The `run_embodiment.sh` script accepts:
- `$1`: config name (e.g., `opensora_libero_spatial_grpo_openvlaoft`)
- `$2`: robot platform (e.g., `LIBERO`, `ALOHA`, `BRIDGE`)
- `$3+`: Hydra overrides forwarded to the training script

### Single-GPU Quick Test

Use small values to verify the setup runs end-to-end before committing to a full run:

```bash
bash examples/embodiment/run_embodiment.sh opensora_libero_spatial_grpo_openvlaoft LIBERO \
  runner.max_steps=3 \
  runner.val_check_interval=2 \
  runner.save_interval=3 \
  algorithm.group_size=2 \
  algorithm.rollout_epoch=2 \
  algorithm.filter_rewards=False \
  env.train.total_num_envs=2 \
  env.train.max_episode_steps=16 \
  env.train.max_steps_per_rollout_epoch=16 \
  env.eval.total_num_envs=2 \
  env.eval.max_episode_steps=16 \
  env.eval.max_steps_per_rollout_epoch=16 \
  actor.micro_batch_size=4 \
  actor.global_batch_size=4
```

Note: `global_batch_size` must divide evenly into
`total_num_envs * group_size * rollout_epoch`. With the values above:
2 * 2 * 2 = 8, so `global_batch_size` of 4 or 8 works.

### Config File

The YAML config is at:
```
examples/embodiment/config/opensora_libero_spatial_grpo_openvlaoft.yaml
```

Update `opensora_wm_hf_ckpt_path` and `model_path` to point to your local
checkpoint directories before running.

## Known Failure Patterns and Fixes

### 1. `ModuleNotFoundError: No module named 'prismatic'`

**Cause:** `openvla-oft` was not installed, or was installed without the
`prismatic` package being available.

**Fix:**
```bash
pip install git+https://github.com/moojink/openvla-oft.git --no-deps --force-reinstall
pip install git+https://github.com/moojink/dlimp_openvla.git --no-deps
```

### 2. `ModuleNotFoundError: No module named 'jsonlines'`

**Cause:** `openvla-oft` is installed with `--no-deps`, so its transitive
dependency `jsonlines` (used by `prismatic.training.metrics`) is missing.

**Fix:**
```bash
pip install jsonlines
```

### 3. `TIMM Version must be >= 0.9.10 and < 1.0.0`

**Cause:** `uv sync` or another install step pulls in `timm>=1.0`, but
`prismatic` has a hard version check.

**Fix:**
```bash
pip install "timm>=0.9.10,<1.0.0"
```

### 4. `ModuleNotFoundError: No module named 'opensora'`

**Cause:** OpenSora was never cloned/installed, or its editable install was
clobbered by a later `pip install` step.

**Fix:**
```bash
# Re-clone if needed
git clone https://github.com/RLinf/opensora.git .venv/opensora

# Re-install editable
pip install -e .venv/opensora --no-deps

# Ensure PYTHONPATH includes it (for Ray workers)
export PYTHONPATH=$(realpath .venv/opensora):$PYTHONPATH
```

### 5. `ModuleNotFoundError: No module named 'colossalai'` / `'mmengine'` / `'xformers'` / `'tensornvme'`

**Cause:** OpenSora deps were not installed, or `uv pip` targeted the system
Python instead of the venv.

**Fix:** Use `pip` (not `uv pip`) to install into the activated venv:
```bash
pip install colossalai==0.5.0 --no-deps
pip install xformers==0.0.29.post2 --no-deps
pip install mmengine addict rotary_embedding_torch decord termcolor
pip install git+https://github.com/fangqi-Zhu/TensorNVMe.git --no-build-isolation
```

### 6. `RuntimeError: FusedLayerNorm not available. Please install apex.`

**Cause:** Apex was installed as pure Python (without CUDA extensions), or not
installed at all.

**Fix:** If the prebuilt wheel is not available, build from source with the
CUDA version check patched:
```bash
APEX_DIR=$(mktemp -d)
git clone https://github.com/RLinf/apex.git "$APEX_DIR"

# Patch the CUDA version check (12.8 bare-metal vs 12.4 torch binary)
python -c "
import re, pathlib
p = pathlib.Path('$APEX_DIR/setup.py')
t = p.read_text()
t = re.sub(r'raise RuntimeError\(', 'import warnings; warnings.warn(', t, count=1)
p.write_text(t)
"

APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -e "$APEX_DIR" --no-build-isolation
```

### 7. `ValueError: signal only works in main thread of the main interpreter`

**Cause:** `CollectEpisode.__init__` calls `signal.signal(SIGTERM, ...)` but
Ray workers run in non-main threads.

**Fix:** Already patched in `rlinf/envs/wrappers/collect_episode.py`. The
signal handler is only registered when running in the main thread.

### 8. `flash_attn_2_cuda.cpython-311: undefined symbol` / `ImportError`

**Cause:** `flash-attn` was built from source against a different torch ABI
than the installed torch. `pip install flash-attn` builds the latest version
which may not match torch 2.6.0+cu124.

**Fix:** Install the prebuilt wheel matched to your exact torch version:
```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

### 9. `torch.OutOfMemoryError: CUDA out of memory`

**Cause:** Actor (~47 GB), rollout (~15 GB), and env (~13 GB) workers all
share a single 80 GB GPU. The world model VAE decoder allocation pushes
past the limit.

**Fix options:**
- Reduce `env.train.total_num_envs` and `algorithm.group_size`
- Ensure `enable_offload: True` is set for actor, rollout, and env
- Use multiple GPUs (set `cluster.num_nodes` or adjust placement)

### 10. `AssertionError: N is not divisible by M`

**Cause:** `rollout_size` (= `total_num_envs * group_size * rollout_epoch`)
is not divisible by `actor.global_batch_size`.

**Fix:** Adjust parameters so the division is exact. For example, with
`total_num_envs=2, group_size=2, rollout_epoch=2` the rollout_size is 8,
so `global_batch_size` must be 1, 2, 4, or 8.

### 11. `ResolutionImpossible` when installing opensora.txt

**Cause:** `colossalai==0.5.0` requires `torch<=2.5.1` but
`xformers==0.0.29.post2` requires `torch==2.6.0`. pip cannot resolve both
in a single install command.

**Fix:** Install them separately with `--no-deps`:
```bash
pip install colossalai==0.5.0 --no-deps
pip install xformers==0.0.29.post2 --no-deps
pip install -r requirements/embodied/models/opensora.txt --no-deps
```

### 12. numpy upgraded to 2.x

**Cause:** Many packages pull in `numpy>=2.0` as a transitive dependency, but
OpenSora and TensorFlow 2.15 require `numpy<2.0`.

**Fix:** Re-pin after any install step that might upgrade it:
```bash
pip install numpy==1.26.4
```

## Verifying the Installation

After install, the script runs a verification check. You can re-run it manually:

```bash
source .venv/bin/activate
python -c "
import torch
print(f'PyTorch:    {torch.__version__}')
print(f'CUDA:       {torch.version.cuda}')
print(f'GPU:        {torch.cuda.get_device_name(0)}')

from opensora.registry import MODELS
print('opensora:   OK')

import prismatic
print('prismatic:  OK')

from libero.libero.envs import OffScreenRenderEnv
print('libero:     OK')

from apex.normalization.fused_layer_norm import FusedLayerNorm
print('apex:       OK')

import flash_attn
print(f'flash_attn: {flash_attn.__version__}')

print()
print('All checks passed!')
"
```

## Mirror Support (China)

Pass `--use-mirror` to use ghfast.top for GitHub and Aliyun for PyPI:

```bash
bash install_opensora_h100.sh --use-mirror
```
