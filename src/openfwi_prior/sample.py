"""Generate velocity models from an unconditional EDM prior."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .network import build_edm_model


def load_prior(path, device="cuda:0"):
    """Load a distributed EMA model or a checkpoint produced by the trainer."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "model_config" in checkpoint:
        config = dict(checkpoint["model_config"])
        normalization = checkpoint["normalization"]
        family = checkpoint["family"]
    else:
        config = dict(checkpoint["config"]["model"])
        normalization = {
            key: checkpoint["config"]["data"][key]
            for key in ("center_m_s", "scale_m_s")
        }
        family = checkpoint["config"]["family"]
    device = torch.device(device)
    config["use_fp16"] = device.type == "cuda" and bool(config.get("use_fp16", True))
    model = build_edm_model(config).to(device)
    model.load_state_dict(checkpoint["ema"], strict=True)
    model.eval().requires_grad_(False)
    metadata = {
        "family": family,
        "step": int(checkpoint["step"]),
        "normalization": normalization,
        "model_config": config,
    }
    return model, metadata


@torch.no_grad()
def sample_edm(
    model, count=4, steps=40, seed=42, sigma_min=0.002, sigma_max=80.0, rho=7.0
):
    """Integrate the unconditional probability-flow ODE with Heun's method."""
    if count < 1 or steps < 2 or not 0 < sigma_min < sigma_max or rho <= 0:
        raise ValueError(
            "Require count >= 1, steps >= 2, 0 < sigma_min < sigma_max, rho > 0"
        )
    if sigma_min < model.sigma_min or sigma_max > model.sigma_max:
        raise ValueError("Sampling noise range exceeds the model's configured range")
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    indices = torch.arange(steps, device=device, dtype=torch.float32)
    sigmas = (
        sigma_max ** (1 / rho)
        + indices / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    sigmas = torch.cat([model.round_sigma(sigmas), sigmas.new_zeros(1)])
    x = torch.randn(count, 1, 70, 70, generator=generator, device=device) * sigmas[0]
    for current, following in zip(sigmas[:-1], sigmas[1:]):
        denoised = model(x, current.expand(count)).float()
        derivative = (x - denoised) / current
        trial = x + (following - current) * derivative
        if following > 0:
            next_denoised = model(trial, following.expand(count)).float()
            next_derivative = (trial - next_denoised) / following
            x = x + (following - current) * (derivative + next_derivative) / 2
        else:
            x = trial
    if not torch.isfinite(x).all():
        raise FloatingPointError("Sampling produced non-finite values")
    return x


def save_grid(path, velocities):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = len(velocities)
    cols = min(count, 4)
    rows = (count + cols - 1) // cols
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.2 * cols, 3.1 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for i, ax in enumerate(axes.flat):
        if i >= count:
            ax.set_visible(False)
            continue
        image = ax.imshow(
            velocities[i], cmap="jet", vmin=1500, vmax=4500, extent=(0, 0.7, 0.7, 0)
        )
        ax.set(title=f"Sample {i + 1}", xlabel="Distance (km)", ylabel="Depth (km)")
    fig.colorbar(image, ax=list(axes.flat), label="Velocity (m/s)", shrink=0.85)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    model, metadata = load_prior(args.checkpoint, args.device)
    samples = sample_edm(
        model,
        count=args.num_samples,
        steps=args.steps,
        seed=args.seed,
        sigma_min=model.sigma_min,
        sigma_max=model.sigma_max,
    )
    normalized = samples.cpu().numpy()[:, 0]
    norm = metadata["normalization"]
    velocity = normalized * norm["scale_m_s"] + norm["center_m_s"]
    args.output.mkdir(parents=True)
    np.savez_compressed(
        args.output / "samples.npz", normalized=normalized, velocity_m_s=velocity
    )
    save_grid(args.output / "samples.png", velocity)
    metadata.update(
        {
            "seed": args.seed,
            "num_samples": args.num_samples,
            "steps": args.steps,
            "sampler": "EDM probability-flow ODE, Heun",
            "rho": 7.0,
            "sigma_min": model.sigma_min,
            "sigma_max": model.sigma_max,
            "saved_values_clipped": False,
        }
    )
    (args.output / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {args.num_samples} unconditional samples to {args.output}")


if __name__ == "__main__":
    main()
