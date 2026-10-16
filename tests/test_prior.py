from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from openfwi_prior.data import VelocityDataset, make_datasets
from openfwi_prior.sample import sample_edm


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
