"""Native PyTorch reference for the discrete acoustic forward operator."""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def native_forward(v, block_size=20):
    vp = F.pad(v[None, None], (120, 120, 120, 120), mode="replicate")[0, 0]
    c = v.amin()
    # Corner cells use horizontal damping, exactly as in the production scheme.
    ramp = 3 * np.log(1e7) / (2 * 1190) * (np.arange(120) / 119) ** 2
    d = np.zeros((310, 310), dtype=np.float64)
    d[:, :120] = ramp[::-1]
    d[:, 190:] = ramp
    d[:120, 120:190] = ramp[::-1, None]
    d[190:, 120:190] = ramp[:, None]
    damping = torch.tensor(d, device=v.device)
    alpha = (vp * 0.0001).square()
    kappa = c * damping * 0.001
    a = 2 - 5 * alpha - kappa
    b = 1 - kappa
    nw = 2 * (int(2.2 / 15 / 0.001) // 2) + 1
    beta = ((nw // 2 + 1 - np.arange(1, nw + 1)) * 15 * 0.001 * np.pi) ** 2
    wave = np.zeros(1000)
    wave[:nw] = (1 - 2 * beta) * np.exp(-beta)
    wave = torch.tensor(wave, device=v.device)
    source_x = torch.tensor([120, 137, 154, 172, 189], device=v.device)
    source_mask = torch.zeros((5, 310, 310), dtype=v.dtype, device=v.device)
    source_mask[torch.arange(5, device=v.device), 121, source_x] = 1
    injection = (vp[121, source_x] * 0.001).square()[:, None, None] * source_mask

    def block(p0, p1, a, b, alpha, injection, w):
        recordings = []
        for s in w.unbind():
            t1 = (
                torch.roll(p1, -1, -1)
                + torch.roll(p1, 1, -1)
                + torch.roll(p1, -1, -2)
                + torch.roll(p1, 1, -2)
            )
            t2 = (
                torch.roll(p1, -2, -1)
                + torch.roll(p1, 2, -1)
                + torch.roll(p1, -2, -2)
                + torch.roll(p1, 2, -2)
            )
            lap = (4 / 3) * t1 + (-1 / 12) * t2
            p2 = a * p1 - b * p0 + alpha * lap + injection * s
            recordings.append(p2[:, 121, 120:190])
            p0, p1 = p1, p2
        return p0, p1, torch.stack(recordings, dim=1)

    p0 = torch.zeros_like(source_mask)
    p1 = torch.zeros_like(source_mask)
    records = []
    for start in range(0, 1000, block_size):
        p0, p1, record = checkpoint(
            block,
            p0,
            p1,
            a,
            b,
            alpha,
            injection,
            wave[start : start + block_size],
            use_reentrant=False,
        )
        records.append(record)
    return torch.cat(records, dim=1), c
