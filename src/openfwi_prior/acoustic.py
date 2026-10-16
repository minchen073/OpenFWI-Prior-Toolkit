"""First-order acoustic forward and adjoint interfaces on CUDA device 0."""

import torch


def forward(velocity_m_s):
    """Map a (70, 70) velocity field to (5, 1000, 70) pressure observations.

    Inputs must be finite positive floating-point tensors on CUDA device 0.
    The result is float32; wave propagation and adjoint accumulation use float64.
    """
    if velocity_m_s.shape != (70, 70) or not velocity_m_s.is_floating_point():
        raise ValueError("Expected a floating-point (70, 70) velocity tensor")
    if velocity_m_s.device != torch.device("cuda:0"):
        raise ValueError(
            "Use CUDA device 0; select a physical GPU with CUDA_VISIBLE_DEVICES"
        )
    if not torch.isfinite(velocity_m_s).all() or not (velocity_m_s > 0).all():
        raise ValueError("Velocity must be finite and positive")
    from .physics import seismic_master_forward_modeling

    return seismic_master_forward_modeling(velocity_m_s)


def adjoint(velocity_m_s, data_cotangent):
    """Return J(velocity)^T data_cotangent, in the input velocity units.

    The data cotangent is the derivative of a scalar objective with respect to
    predicted observations. This function does not differentiate its own result.
    """
    if (
        data_cotangent.shape != (5, 1000, 70)
        or data_cotangent.device != velocity_m_s.device
    ):
        raise ValueError("Expected a (5, 1000, 70) cotangent on the velocity device")
    if not torch.isfinite(data_cotangent).all():
        raise ValueError("Data cotangent must be finite")
    with torch.enable_grad():
        velocity = velocity_m_s.detach().requires_grad_(True)
        predicted = forward(velocity)
        return torch.autograd.grad(predicted, velocity, data_cotangent.detach())[0]
