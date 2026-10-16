# Unconditional sampling

Both samplers operate on normalized velocities `(v - 3000) / 1500` using the
same pretrained EDM denoiser `D(y, sigma)`. They save the final clean estimate
without clipping. No seismic observations or physical guidance enter sampling.

## EDM Heun: 18 steps

For `N = 18`, define `N` positive noise levels by

```text
sigma_i = [80^(1/7) + i/(N-1) * (0.002^(1/7) - 80^(1/7))]^7,
i = 0, ..., N-1,
sigma_N = 0.
```

Initialize `y_0 = sigma_0 * z`, where `z` is standard Gaussian noise.
The probability-flow ODE in EDM coordinates is
`dy/dsigma = (y - D(y, sigma)) / sigma`. Each interval uses a Heun predictor
and corrector, except the final interval to zero, which uses one Euler update.
The 18 intervals therefore require 35 denoiser evaluations. There is no
stochastic churn during the trajectory.

## VP DDIM: 1000 steps

The continuous VP schedule is

```text
beta(t) = 0.1 + 19.9 t
alpha_bar(t) = exp(-0.1 t - 19.9 t²/2)
s(t) = sqrt(alpha_bar(t))
n(t) = sqrt(1 - alpha_bar(t))
sigma(t) = n(t) / s(t).
```

Use 1000 evaluation times spaced uniformly from 1 to 0.001, then append a
terminal time of 0. Initialize the VP state as `x_1 = n(1) * z`. For each
transition from evaluation time `t` to the next smaller time `r`, compute

```text
x0_hat = D(x_t / s(t), sigma(t))
epsilon_hat = (x_t - s(t) * x0_hat) / n(t)
x_r = s(r) * x0_hat + n(r) * epsilon_hat.
```

This is deterministic DDIM with `eta = 0`. At `r = 0`, `s(r) = 1` and
`n(r) = 0`, so the final output is the last denoised estimate. The model is
never evaluated at sigma zero. The trajectory makes exactly 1000 denoiser
calls; only its initial noise is random.

The equivalent EDM noise is approximately 152.17 at `t = 1` and 0.01049 at
the last evaluation time. The VP schedule is not clipped to the Heun noise
range. The network accepts positive sigma values, and its log-normal training
noise distribution has no upper cutoff; the value 80 in the model metadata
sets the default EDM sampling endpoint.

In unscaled coordinates `y_t = x_t / s(t)`, a DDIM step is equivalently
`y_r = x0_hat + sigma(r) * (y_t - x0_hat) / sigma(t)`. This provides a direct
check of the conversion between VP and EDM coordinates.

The seed controls the initial Gaussian tensor. A fixed seed is repeatable on
the same numerical environment; the two samplers use different trajectories
and are not expected to generate identical fields.
