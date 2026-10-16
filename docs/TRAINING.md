# Diffusion-prior training protocol

All four families use a one-channel DhariwalUNet with EDM preconditioning.
Fields are normalized by `(velocity_m_s - 3000) / 1500`, without clipping.
The external grid is 70 × 70. One replicated cell is added on every side
before the 72 × 72 network; its output is cropped back to 70 × 70.

The base channel count is 64, channel multipliers are `[1, 2, 3, 4]`, and
there are two blocks per resolution. Attention operates at resolutions
36, 18, and 9. Dropout and class conditioning are disabled. The EDM
preconditioning parameter `sigma_data` is 0.5.

For clean normalized fields `x`, training draws `log(sigma)` from a normal
distribution with mean −1.2 and standard deviation 1.2, and independent
unit Gaussian noise `epsilon`. The objective is

```text
mean[ ((sigma² + sigma_data²) / (sigma * sigma_data)²)
      * (D(x + sigma * epsilon, sigma) - x)² ].
```

The mean is over batch and spatial dimensions. All supplied configurations
use seed 42, batch size 32, Adam with learning rate 1e-4, betas `(0.9, 0.999)`,
zero weight decay, gradient-norm clipping at 1, and FP16 mixed precision.
The optimizer starts at the configured learning rate. After update `k`,
its rate for the next update is set to `lr * min(1, k / 500)`.

After each update, the EMA half-life in images is
`min(204800, max(k * batch_size, 1) * 0.05)`, and its decay is
`0.5 ** (batch_size / max(half_life, 1))`.

Training and monitoring filenames are listed explicitly in each configuration.
Monitoring files are used only for loss evaluation, without gradient updates
or checkpoint selection. Monitoring runs every 5,000 steps for at most 20
batches. The released EMA states are from the fixed 400,000-step endpoint.
The toolkit preserves the training objective, architecture, data split,
normalization, and numerical update conventions of these models. A new run
can differ numerically across hardware and software environments.

The distributed model files contain `ema`, `model_config`, `normalization`,
`family`, `step`, and `format_version`. They do not contain optimizer state.
Training checkpoints include `net`, `ema`, `optimizer`, `scaler`, `step`,
and the effective `config`. Both formats are accepted by the sampling script.
