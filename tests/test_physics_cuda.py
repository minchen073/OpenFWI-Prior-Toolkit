"""Optional full-grid forward/gradient check against independent native autograd."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

pytestmark = pytest.mark.skipif(
    os.environ.get("OPENFWI_TEST_CUDA") != "1",
    reason="Set OPENFWI_TEST_CUDA=1 with CUDA, CuPy and example assets installed",
)


@pytest.mark.parametrize(
    "family", ["FlatVel-B", "FlatFault-B", "CurveVel-B", "CurveFault-B"]
)
def test_full_discrete_forward_and_gradient(family):
    from native_acoustic import native_forward
    from openfwi_prior.physics import seismic_master_forward_modeling

    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    with np.load(root / f"assets/examples/{family}.npz") as data:
        v = (
            gaussian_filter((data["velocity_m_s"] - 3000) / 1500, sigma=10) * 1500
            + 3000
        ).astype(np.float64)
        target = torch.tensor(data["observation"], device="cuda")
    x = torch.tensor(v, device="cuda", requires_grad=True)
    native, _ = native_forward(x)
    native = native.float()
    for squared in [True, False]:
        prediction = seismic_master_forward_modeling(x)
        torch.testing.assert_close(prediction, native, rtol=1e-6, atol=1e-10)
        residual = prediction - target
        reference_residual = native - target
        loss = 0.5 * residual.square().mean() if squared else residual.abs().mean()
        reference_loss = (
            0.5 * reference_residual.square().mean()
            if squared
            else reference_residual.abs().mean()
        )
        actual = torch.autograd.grad(loss, x)[0]
        expected = torch.autograd.grad(reference_loss, x, retain_graph=squared)[0]
        error = (
            torch.linalg.vector_norm(actual - expected)
            / torch.linalg.vector_norm(expected)
        ).item()
        assert error < 1e-8, f"{family}: relative gradient L2 error {error}"
