from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from openfwi_prior.data import VelocityDataset, make_datasets
from openfwi_prior.sample import sample_edm, sample_vp_ddim, vp_schedule


def test_dataset_normalization_and_disjoint_splits(tmp_path):
    raw = np.linspace(1500, 4500, 2 * 70 * 70, dtype=np.float32).reshape(2, 1, 70, 70)
    np.save(tmp_path / "a.npy", raw)
    data = VelocityDataset(tmp_path, ["a.npy"], samples_per_file=2)
    assert len(data) == 2
    np.testing.assert_array_equal(data[1].numpy(), (raw[1] - 3000) / 1500)
    with pytest.raises(ValueError, match="disjoint"):
        make_datasets(
            {
                "root": tmp_path,
                "train_files": ["a.npy"],
                "validation_files": ["./a.npy"],
                "samples_per_file": 2,
                "center_m_s": 3000,
                "scale_m_s": 1500,
            }
        )
    with pytest.raises(ValueError, match="unique"):
        VelocityDataset(tmp_path, ["a.npy", "a.npy"], samples_per_file=2)


def test_dataset_rejects_wrong_shape(tmp_path):
    np.save(tmp_path / "a.npy", np.zeros((2, 70, 70), dtype=np.float32))
    with pytest.raises(ValueError, match="Expected"):
        VelocityDataset(tmp_path, ["a.npy"], samples_per_file=2)


@pytest.mark.parametrize(
    "family,train_count",
    [("FlatVel-B", 48), ("CurveVel-B", 48), ("FlatFault-B", 96), ("CurveFault-B", 96)],
)
def test_config_splits(family, train_count):
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / f"configs/{family}.yaml").read_text())
    data = config["data"]
    assert len(data["train_files"]) == train_count
    assert len(data["validation_files"]) == 12
    assert not set(data["train_files"]) & set(data["validation_files"])
    assert config["training"]["total_steps"] == 400000


class GaussianDenoiser(torch.nn.Module):
    """Analytic MMSE denoiser for a unit Gaussian prior."""

    sigma_min = 0.002
    sigma_max = 80.0

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x, sigma):
        return x / (1 + sigma.reshape(-1, 1, 1, 1).square())

    def round_sigma(self, sigma):
        return sigma


def test_sampler_gaussian_prior_and_seed():
    model = GaussianDenoiser()
    torch.set_num_threads(2)
    a = sample_edm(model, count=4, steps=80, seed=12)
    b = sample_edm(model, count=4, steps=80, seed=12)
    c = sample_edm(model, count=4, steps=80, seed=13)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.equal(a, c)
    assert abs(a.mean().item()) < 0.04
    assert abs(a.std().item() - 1) < 0.04
    with pytest.raises(ValueError):
        sample_edm(model, steps=1)


def test_vp_ddim_gaussian_prior_and_edm_coordinate_equivalence():
    torch.set_num_threads(2)
    model = GaussianDenoiser()
    actual = sample_vp_ddim(model, count=4, seed=12)
    assert abs(actual.mean().item()) < 0.04
    assert abs(actual.std().item() - 1) < 0.04
    torch.testing.assert_close(
        actual, sample_vp_ddim(model, count=4, seed=12), rtol=0, atol=0
    )
    assert not torch.equal(actual, sample_vp_ddim(model, count=4, seed=13))

    # Integrate independently in unscaled EDM coordinates.
    times, alpha_bar = vp_schedule()
    assert len(times) == 1001 and times[-1] == 0 and alpha_bar[-1] == 1
    assert torch.all(times[:-1] > times[1:])
    torch.testing.assert_close(
        alpha_bar[0], torch.tensor(np.exp(-10.05), dtype=torch.float64)
    )
    sigmas = ((1 - alpha_bar) / alpha_bar).sqrt().float()
    assert sigmas[0] > 150
    expected = (
        torch.randn(4, 1, 70, 70, generator=torch.Generator().manual_seed(12))
        * sigmas[0]
    )
    for sigma, following in zip(sigmas[:-1], sigmas[1:]):
        clean = model(expected, sigma.expand(4))
        expected = clean + following / sigma * (expected - clean)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_acoustic_backend_validation_without_cupy():
    from openfwi_prior.acoustic import forward

    with pytest.raises(ValueError, match="backend"):
        forward(torch.ones(70, 70), backend="unknown")
    with pytest.raises(ValueError, match="70, 70"):
        forward(torch.ones(2, 2), backend="torch")
    with pytest.raises(ValueError, match="positive"):
        forward(torch.zeros(70, 70), backend="torch")
    with pytest.raises(ValueError, match="block_size"):
        forward(torch.full((70, 70), 2000.0), backend="torch", block_size=0)
