"""Compute predicted observations and the gradient of a data-misfit objective."""

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from openfwi_prior.acoustic import forward, adjoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--example", type=Path, default=Path("assets/examples/FlatVel-B.npz")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/acoustic"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    torch.cuda.set_device(0)
    with np.load(args.example, allow_pickle=False) as bundle:
        velocity = gaussian_filter(bundle["velocity_m_s"], sigma=3).astype(np.float32)
        observed = torch.as_tensor(bundle["observation"], device="cuda:0")
    v = torch.tensor(velocity, device="cuda:0", requires_grad=True)
    predicted = forward(v)
    loss = 0.5 * (predicted - observed).square().mean()
    gradient = torch.autograd.grad(loss, v)[0]
    cotangent = (predicted.detach() - observed) / observed.numel()
    explicit = adjoint(v.detach(), cotangent)
    torch.testing.assert_close(gradient, explicit, rtol=1e-5, atol=1e-10)
    args.output.mkdir(parents=True)
    np.savez_compressed(
        args.output / "acoustic.npz",
        velocity_m_s=velocity,
        predicted=predicted.detach().cpu().numpy(),
        gradient=gradient.cpu().numpy(),
    )
    print(f"Half mean-squared data misfit: {loss.item():.6e}")
    print(f"Saved observations and velocity gradient to {args.output}")


if __name__ == "__main__":
    main()
