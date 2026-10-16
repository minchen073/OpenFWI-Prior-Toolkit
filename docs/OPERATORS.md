# Acoustic forward and adjoint operators

The solver implements a 2D constant-density acoustic wave equation, with
second-order time stepping and a fourth-order spatial finite-difference
stencil. The acquisition geometry is fixed in this distribution:

| Parameter | Value |
|---|---|
| Velocity grid | 70 × 70, ordered `(depth, horizontal)` |
| Grid spacing | 10 m |
| Time sampling | 1 ms, 1,000 samples |
| Source wavelet | 15 Hz Ricker |
| Source columns | 0, 17, 34, 52, 69, zero-based |
| Source and receiver row | 1, zero-based |
| Receivers | All 70 horizontal positions |
| Absorbing layer | 120 cells on each side |
| Internal padded grid | 310 × 310 |
| Observation ordering | `(source, time, receiver)` |

`forward(v, backend="cupy")` accepts positive finite velocities in m/s on logical
CUDA device 0 and differentiates using the discrete adjoint.
`forward(v, backend="torch")` evaluates the same time stepping with native
PyTorch operations on CPU or CUDA. Its gradients are computed by automatic
differentiation, without CuPy or a custom backward. Both use float64 internally;
predicted observations
are returned as float32. Gradients are returned in the input velocity dtype.
The geometry and wavelet match the included OpenFWI examples. The solver is
intended for the 1500–4500 m/s velocity range used by these datasets.

`adjoint(v, q, backend=...)` evaluates the vector-Jacobian product `J(v)^T q` with
the selected implementation. For
`L = 0.5 * mean((forward(v) - y)^2)`, use
`q = (forward(v) - y) / y.numel()`. For other objectives, supply their
derivative with respect to the predicted observations. Differentiating a
scalar loss through `forward` applies the same adjoint automatically.

The discrete adjoint includes velocity dependence in propagation, source
injection, replicated padding, and absorbing-layer damping. Damping uses
the minimum interior velocity; its sensitivity is shared equally among tied
minimum cells, consistent with `torch.amin`.

The shared interface is intended for first-order derivatives of a single
velocity field. Runtime geometry changes and batched velocity inputs are not
supported. The CuPy backend reuses CUDA graph and wavefield buffers and requires
sequential calls within a process; its backward recomputes forward wavefields.
Select a physical GPU with `CUDA_VISIBLE_DEVICES` before importing that backend.

The PyTorch backend uses activation checkpointing in blocks of 20 time steps
by default. Change `block_size` in `forward`/`adjoint`, or `--block-size` in the
example script, to adjust the recomputation/memory tradeoff. The checkpoint
boundaries do not change the discretization. Its internal float64 conversion
remains part of the autograd graph, so gradients reach float32 or float64 input
tensors in their original dtype. CPU execution is supported but slower for
the full 5-shot, 1000-step acquisition.
