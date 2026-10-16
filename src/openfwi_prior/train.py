#!/usr/bin/env python3
"""Train an unconditional EDM prior on OpenFWI velocity fields."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


from .data import make_datasets
from .network import build_edm_model, edm_parameter_count


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")
    return config


def validate_config(config: dict[str, Any]) -> None:
    model_cfg = config["model"]
    if int(model_cfg["img_resolution"]) != 70 or int(model_cfg["img_channels"]) != 1:
        raise ValueError("OpenFWI EDM must use one-channel 70x70 fields")
    training_cfg = config["training"]
    if int(training_cfg["total_steps"]) <= 0:
        raise ValueError("training.total_steps must be positive")
    if int(training_cfg["batch_size"]) <= 0:
        raise ValueError("training.batch_size must be positive")
    if str(training_cfg.get("mixed_precision", "no")).lower() not in (
        "no",
        "fp16",
    ):
        raise ValueError("training.mixed_precision must be no or fp16")
    if int(training_cfg["num_workers"]) < 0:
        raise ValueError("training.num_workers must be nonnegative")


def make_autocast(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision == "fp16"
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)


def edm_loss(
    model: torch.nn.Module,
    clean: torch.Tensor,
    p_mean: float,
    p_std: float,
    sigma_data: float,
    device: torch.device,
    precision: str,
) -> torch.Tensor:
    noise_log_sigma = torch.randn(clean.shape[0], 1, 1, 1, device=device)
    sigma = (noise_log_sigma * p_std + p_mean).exp()
    noise = torch.randn_like(clean) * sigma
    noisy = clean + noise
    weight = (sigma.square() + sigma_data**2) / (sigma * sigma_data).square()
    with make_autocast(device, precision):
        denoised = model(noisy, sigma).float()
    return (weight * (denoised - clean).square()).mean()


@torch.no_grad()
def validation_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_cfg: dict[str, Any],
    device: torch.device,
    precision: str,
    max_batches: int,
) -> float:
    was_training = model.training
    model.eval()
    values: list[float] = []
    for batch_index, clean in enumerate(loader):
        if batch_index >= max_batches:
            break
        clean = clean.to(device, non_blocking=True)
        value = edm_loss(
            model,
            clean,
            float(loss_cfg["p_mean"]),
            float(loss_cfg["p_std"]),
            float(loss_cfg["sigma_data"]),
            device,
            precision,
        )
        values.append(float(value.item()))
    model.train(was_training)
    return float(np.mean(values)) if values else float("nan")


def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_parameter, parameter in zip(ema.parameters(), model.parameters()):
            ema_parameter.lerp_(parameter, 1.0 - decay)
        for ema_buffer, buffer in zip(ema.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    ema: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict[str, Any],
    scaler,
) -> None:
    torch.save(
        {
            "net": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": config,
            "scaler": scaler.state_dict(),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, help="Total target step count")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device")
    parser.add_argument(
        "--resume", type=Path, help="Training checkpoint including optimizer state"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check data and configuration without training",
    )
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    if args.data_root is not None:
        config["data"]["root"] = str(args.data_root)
    if args.output is not None:
        config["output"]["root"] = str(args.output)
    if args.device is not None:
        config["device"] = args.device
    for arg, key in [
        (args.steps, "total_steps"),
        (args.batch_size, "batch_size"),
        (args.num_workers, "num_workers"),
    ]:
        if arg is not None:
            config["training"][key] = arg
    if args.resume is not None:
        config["training"]["resume"] = str(args.resume)
    validate_config(config)
    train_set, validation_set = make_datasets(config["data"])
    if (
        len(train_set) < config["training"]["batch_size"]
        or len(validation_set) < config["training"]["batch_size"]
    ):
        raise ValueError("Each split must contain at least one complete batch")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "family": config["family"],
                    "train": train_set.describe(),
                    "monitoring": validation_set.describe(),
                },
                indent=2,
            )
        )
        return

    seed = int(config.get("seed", 42))
    set_seed(seed)
    requested_device = str(config.get("device", "cuda:0"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is requested but torch.cuda.is_available() is false")
    device = torch.device(requested_device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    training_cfg = config["training"]
    batch_size = int(training_cfg["batch_size"])
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": int(training_cfg["num_workers"]),
        "pin_memory": device.type == "cuda",
        "drop_last": True,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_kwargs)

    model = build_edm_model(config["model"]).to(device)
    ema = build_edm_model(config["model"]).to(device)
    ema.load_state_dict(model.state_dict())
    ema.eval().requires_grad_(False)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg["lr"]),
        betas=tuple(float(value) for value in training_cfg["betas"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    precision = str(training_cfg.get("mixed_precision", "no")).lower()
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and precision == "fp16",
    )
    total_steps = int(training_cfg["total_steps"])
    warmup_steps = int(training_cfg.get("warmup_steps", 0))
    ema_halflife_nimg = float(training_cfg.get("ema_halflife_nimg", 204800))
    ema_rampup_ratio = float(training_cfg.get("ema_rampup_ratio", 0.05))
    grad_clip = float(training_cfg.get("grad_clip", 0.0))
    log_every = max(1, int(training_cfg.get("log_every", 100)))
    validation_every = max(1, int(training_cfg.get("validation_every", 5000)))
    validation_max_batches = max(1, int(training_cfg.get("validation_max_batches", 20)))
    save_every = max(1, int(training_cfg.get("save_every", 50000)))

    output_cfg = config["output"]
    run_root = resolve_path(output_cfg["root"], Path.cwd())
    run_name = str(output_cfg.get("run_name", "edm"))
    run_dir = run_root / f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config_used.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    with (run_dir / "dataset.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "train": train_set.describe(),
                "validation": validation_set.describe(),
                "validation_note": (
                    "No validation gradient or checkpoint selection; final.pt is the fixed "
                    "training.total_steps endpoint."
                ),
            },
            handle,
            indent=2,
        )

    start_step = 0
    resume = training_cfg.get("resume")
    if resume not in (None, "", "null", "None"):
        checkpoint = torch.load(
            resolve_path(str(resume), Path.cwd()),
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint["net"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_step = int(checkpoint["step"])
        if start_step >= total_steps:
            raise ValueError("Target steps must exceed the resumed checkpoint step")
        print(f"[train] resumed from step {start_step}", flush=True)

    print(
        f"[train] device={device} parameters={edm_parameter_count(model):,}",
        flush=True,
    )
    print(
        f"[train] train={len(train_set)} validation={len(validation_set)} "
        f"root={train_set.root}",
        flush=True,
    )
    print(f"[train] output={run_dir} total_steps={total_steps}", flush=True)

    loss_cfg = config["edm_loss"]
    model.train()
    step = start_step
    start_time = time.perf_counter()
    while step < total_steps:
        for clean in train_loader:
            if step >= total_steps:
                break
            clean = clean.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = edm_loss(
                model,
                clean,
                float(loss_cfg["p_mean"]),
                float(loss_cfg["p_std"]),
                float(loss_cfg["sigma_data"]),
                device,
                precision,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at step {step + 1}")
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            step += 1
            if warmup_steps > 0:
                learning_rate_scale = min(1.0, step / warmup_steps)
                for group in optimizer.param_groups:
                    group["lr"] = float(training_cfg["lr"]) * learning_rate_scale
            current_images = max(step * batch_size, 1)
            halflife = min(ema_halflife_nimg, current_images * ema_rampup_ratio)
            decay = 0.5 ** (batch_size / max(halflife, 1.0))
            update_ema(ema, model, decay)

            if step == 1 or step % log_every == 0:
                elapsed_minutes = (time.perf_counter() - start_time) / 60.0
                print(
                    f"[train] step {step:7d}/{total_steps} "
                    f"loss={loss.item():.6e} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e} "
                    f"elapsed={elapsed_minutes:.1f}m",
                    flush=True,
                )
            if step % validation_every == 0:
                validation = validation_loss(
                    ema,
                    validation_loader,
                    loss_cfg,
                    device,
                    precision,
                    max_batches=validation_max_batches,
                )
                print(
                    f"[train] validation step={step} " f"edm_loss={validation:.6e}",
                    flush=True,
                )
            if step % save_every == 0 or step == total_steps:
                save_checkpoint(
                    run_dir / f"ckpt_{step:07d}.pt",
                    model,
                    ema,
                    optimizer,
                    step,
                    config,
                    scaler,
                )
                save_checkpoint(
                    run_dir / "latest.pt", model, ema, optimizer, step, config, scaler
                )
                print(f"[train] checkpoint saved at step {step}", flush=True)

    save_checkpoint(run_dir / "final.pt", model, ema, optimizer, step, config, scaler)
    print(f"[train] finished: {run_dir / 'final.pt'}", flush=True)


if __name__ == "__main__":
    main()
