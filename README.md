# Physical-State-Guided Diffusion Sampling for Full-Waveform Inversion

Implementation accompanying **Physical-State-Guided Diffusion Sampling for
Full-Waveform Inversion**, by Chen Min, Haowen Jiang, Zheng Ma, and Xiong-Bin Yan.

This repository provides the currently released implementation components of
our work. The current release includes:

- **Acoustic modeling and gradients:** a CuPy forward solver with a discrete
  adjoint, and a native PyTorch solver whose gradients are computed entirely
  by automatic differentiation.
- **Diffusion-prior training:** the EDM network, training script, and dataset
  configurations for FlatVel-B, FlatFault-B, CurveVel-B, and CurveFault-B.
- **Pretrained priors:** one 400,000-step EMA model per dataset.
- **Unconditional generation:** 18-step EDM Heun sampling and 1000-step DDIM
  sampling on a VP noise schedule, both using the same pretrained EDM priors.
- **Example data:** one velocity model and its seismic observation per dataset.

The full PSG inversion implementation and experiment reproduction workflow
will be released after publication. The modules available here can be used
independently for acoustic modeling and diffusion-prior research.

## Installation

Use Python 3.10 or newer. Install a PyTorch build compatible with your GPU,
then clone the repository and install the toolkit:

```bash
git clone https://github.com/minchen073/OpenFWI-Prior-Toolkit.git
cd OpenFWI-Prior-Toolkit
python -m pip install -e .
python -m pip install 'cupy-cuda12x>=13,<14'
python scripts/download_assets.py
python scripts/verify_assets.py
```

The four pretrained models and four example bundles are available in the
[v0.1.0 release](https://github.com/minchen073/OpenFWI-Prior-Toolkit/releases/tag/v0.1.0).
The download script installs them under `assets/`, checks file sizes and SHA-256
checksums, and skips existing files that already match. The total download is
approximately 394 MiB. Model and example files are distributed through Releases
and are not stored in Git history.

The **CuPy backend** requires an NVIDIA CUDA GPU and CuPy; its acoustic calls
use logical CUDA device 0. Select another physical GPU with
`CUDA_VISIBLE_DEVICES` before starting Python. Use one CuPy acoustic worker per
process because its CUDA graphs and buffers are shared within the process.

The **PyTorch autograd backend** has no CuPy dependency and supports CPU and
CUDA. GPU execution is recommended for the full acoustic grid. Training and
unconditional generation also use PyTorch; generation supports `--device cpu`.
If using only PyTorch, omit the CuPy installation command above.

## Included resources

| Family | Training fields | Monitoring fields | Example index | Pretrained model |
|---|---:|---:|---:|---|
| FlatVel-B | 24,000 | 6,000 | 454 | `FlatVel-B_EDM_ema_400k.pt` |
| FlatFault-B | 48,000 | 6,000 | 434 | `FlatFault-B_EDM_ema_400k.pt` |
| CurveVel-B | 24,000 | 6,000 | 475 | `CurveVel-B_EDM_ema_400k.pt` |
| CurveFault-B | 48,000 | 6,000 | 449 | `CurveFault-B_EDM_ema_400k.pt` |

Weights are in `assets/checkpoints/`. Each is the EMA state at 400,000 training
steps and includes its network configuration and normalization. The supplied
weights contain inference state. Training checkpoints created by this toolkit
also contain the non-EMA network, optimizer, and gradient-scaler state.

Each `assets/examples/<Family>.npz` contains exactly one velocity model
(`velocity_m_s`, shape `(70, 70)`) and its observation (`observation`, shape
`(5, 1000, 70)`). Example indices are zero-based within the source file recorded
in `assets/manifest.json`. These small example bundles are for operator usage;
training requires the full velocity files described below.

## Unconditional generation

Both samplers load the same EDM checkpoint and draw their initial Gaussian
noise from the specified seed.

| Sampler | Command option | Default steps | Denoiser evaluations | Noise schedule |
|---|---|---:|---:|---|
| EDM Heun | `--sampler edm-heun` | 18 | 35 | EDM, sigma 80 → 0.002 → 0, rho = 7 |
| VP DDIM | `--sampler vp-ddim` | 1000 | 1000 | Continuous VP, time 1 → 0.001 → 0, eta = 0 |

**18-step EDM Heun:**

```bash
python scripts/sample.py \
  --checkpoint assets/checkpoints/FlatVel-B_EDM_ema_400k.pt \
  --sampler edm-heun --steps 18 --num-samples 4 --seed 42 \
  --output outputs/flatvel_edm_heun
```

**1000-step VP DDIM:**

```bash
python scripts/sample.py \
  --checkpoint assets/checkpoints/FlatVel-B_EDM_ema_400k.pt \
  --sampler vp-ddim --steps 1000 --num-samples 4 --seed 42 \
  --output outputs/flatvel_vp_ddim
```

Use the corresponding checkpoint filename for each of the other families.
When `--steps` is omitted, the selected sampler uses the default shown above.
The VP sampler uses `beta(t) = 0.1 + 19.9 t`, with
`alpha_bar(t) = exp(-0.1 t - 19.9 t² / 2)`. At each step it converts the VP
state to EDM coordinates, evaluates the clean-velocity estimate, and applies
a deterministic DDIM update. Its equivalent EDM noise at time 1 is about
152.17; this is determined by the VP schedule and is not capped at the EDM
Heun endpoint of 80. See [sampling details](docs/SAMPLING.md) for the equations.

Each run writes `samples.npz`, `samples.png`, and `settings.json` to a new
output directory. Arrays contain normalized fields and velocities in m/s.
Samples are not clipped; the plot uses a fixed 1500–4500 m/s color range.
These are unconditional prior samples and do not use seismic observations.

## Acoustic forward and gradient implementations

Both backends use the same grid, wavelet, absorbing boundary, and observation
layout. They return float32 observations and use float64 internally.

| Backend | Gradient calculation | Devices | Additional dependency |
|---|---|---|---|
| `cupy` | Discrete adjoint through a custom PyTorch backward | CUDA device 0 | CuPy |
| `torch` | Native PyTorch autograd through the time-stepping operations | CPU or CUDA | None beyond PyTorch |

**CuPy forward and adjoint:**

```bash
python scripts/acoustic_example.py --backend cupy --device cuda:0 \
  --example assets/examples/FlatVel-B.npz --output outputs/acoustic_cupy
```

**Native PyTorch forward and automatic differentiation:**

```bash
python scripts/acoustic_example.py --backend torch --device cuda:0 \
  --example assets/examples/FlatVel-B.npz --output outputs/acoustic_torch
```

For the torch backend, `--device cpu` enables CPU execution. Its
`--block-size` option (default 20 time steps) controls activation checkpointing,
which reduces the memory required by automatic differentiation. No hand-written
adjoint or custom backward is used by this backend.

The example smooths the input velocity, predicts seismograms, and computes
the velocity gradient of half the mean-squared data residual. It saves the
input velocity, predictions, and gradient, and checks the explicit
vector-Jacobian interface against differentiation of the scalar objective.

```python
import torch
from openfwi_prior.acoustic import forward, adjoint

# v: (70, 70), m/s; observed: (5, 1000, 70), on the same device.
backend = "torch"  # Use "cupy" for the CUDA discrete-adjoint implementation.
v = v.detach().requires_grad_(True)
predicted = forward(v, backend=backend)
loss = 0.5 * (predicted - observed).square().mean()
gradient = torch.autograd.grad(loss, v)[0]

cotangent = (predicted.detach() - observed) / observed.numel()
gradient_explicit = adjoint(v.detach(), cotangent, backend=backend)
```

See [the operator specification](docs/OPERATORS.md) for discretization, units,
and interface limits.

## Training

Prepare the OpenFWI velocity files in a directory for each family. Seismic
files are not needed for unconditional prior training. Each velocity file
must have shape `(500, 1, 70, 70)` in m/s.

| Families | Training filenames | Monitoring filenames |
|---|---|---|
| FlatVel-B, CurveVel-B | `model1.npy`–`model48.npy` | `model49.npy`–`model60.npy` |
| FlatFault-B, CurveFault-B | `vel{p}_1_{i}.npy`, `p = 6, 7, 8`, `i = 0…31` | Same prefixes, `i = 32…35` |

Each YAML configuration enumerates the exact filenames. Point `--data-root`
at the directory containing those files directly:

```bash
python scripts/train.py --config configs/FlatVel-B.yaml \
  --data-root /path/to/FlatVel-B --dry-run

python scripts/train.py --config configs/FlatVel-B.yaml \
  --data-root /path/to/FlatVel-B --output outputs/training
```

Replace the configuration and data directory to train another family. Relative
data, output, and resume paths are resolved from the current working directory.
The full training dataset is not bundled. The defaults use batch size 32,
400,000 iterations, Adam, and an EMA prior. See [the training protocol](docs/TRAINING.md)
for the architecture, objective, and update conventions.

For a short installation check, add `--steps 2 --batch-size 2 --num-workers 0`.
This only checks the training pipeline; it does not train a useful prior.
Each run creates a new timestamped output directory with the effective
configuration, dataset description, and training checkpoints.

To continue training, supply a checkpoint produced by this trainer:

```bash
python scripts/train.py --config configs/FlatVel-B.yaml \
  --data-root /path/to/FlatVel-B \
  --resume /path/to/run/latest.pt --steps 400000
```

`--steps` is the total target step count. Resume restores model, EMA, optimizer,
and gradient-scaler states; the data-shuffle and random-number streams restart.
It is not a bitwise continuation of an interrupted run. To sample a newly
trained model, pass its `final.pt` to `scripts/sample.py`.

## Tests

```bash
python -m pip install -e '.[test]'
python -m pytest -q
OPENFWI_TEST_CUDA=1 python -m pytest tests/test_physics_cuda.py -q
```

The optional CUDA tests compare full-grid forward outputs and first-order
gradients with an independent native PyTorch implementation.

## Attribution and licenses

Code is provided under the [MIT license](LICENSE). Example data and pretrained
weights are provided under [CC BY-NC-SA 4.0](assets/LICENSE.md). See
[third-party attribution](THIRD_PARTY_NOTICES.md) for the network implementation
and OpenFWI dataset credits.

A [Chinese quick-start guide](docs/README_zh.md) is also included.
