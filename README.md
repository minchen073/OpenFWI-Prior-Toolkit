# OpenFWI Prior Toolkit

Acoustic forward and adjoint operators, unconditional diffusion-prior training,
and pretrained EDM models for four OpenFWI velocity families.

The toolkit contains a standalone CUDA acoustic solver, a PyTorch gradient
interface, training configurations, one pretrained EMA model per family, and
unconditional velocity-model generation. It supports research on physical
models and generative priors. PSG-conditioned inversion and its reproduction
workflow are outside the scope of this package.

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

CuPy and an NVIDIA CUDA GPU are required for acoustic propagation. Training
and unconditional sampling use PyTorch; sampling also supports `--device cpu`.
The acoustic example uses logical CUDA device 0. To select another physical
GPU, set `CUDA_VISIBLE_DEVICES` before starting Python. Use one acoustic
worker per process; its CUDA graphs and buffers are shared within the process.

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

```bash
python scripts/sample.py \
  --checkpoint assets/checkpoints/FlatVel-B_EDM_ema_400k.pt \
  --num-samples 4 --steps 40 --seed 42 \
  --output outputs/flatvel_samples
```

Use the corresponding checkpoint filename for each of the other families.
The script uses deterministic Heun integration of the EDM probability-flow ODE,
with a noise range of 0.002–80 and schedule exponent 7. The random seed controls
the initial Gaussian noise. It writes `samples.npz`, `samples.png`, and
`settings.json` to a new output directory.

The saved arrays contain normalized fields and velocities in m/s. Samples are
not clipped; the plot uses a fixed 1500–4500 m/s color range. These are draws
from an unconditional velocity prior and do not use seismic observations.

## Forward and adjoint example

```bash
python scripts/acoustic_example.py \
  --example assets/examples/FlatVel-B.npz \
  --output outputs/acoustic
```

The example smooths the input velocity, predicts seismograms, and evaluates
the velocity gradient of half the mean-squared data residual. It saves the
input velocity, predicted observations, and gradient. It also checks that the
explicit adjoint interface agrees with PyTorch's gradient interface.

```python
import torch
from openfwi_prior.acoustic import forward, adjoint

# v: (70, 70), m/s; observed: (5, 1000, 70), both on CUDA device 0.
v = v.detach().requires_grad_(True)
predicted = forward(v)
loss = 0.5 * (predicted - observed).square().mean()
gradient = torch.autograd.grad(loss, v)[0]

# An arbitrary data-space cotangent can also be applied explicitly.
cotangent = (predicted.detach() - observed) / observed.numel()
gradient_explicit = adjoint(v.detach(), cotangent)
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
