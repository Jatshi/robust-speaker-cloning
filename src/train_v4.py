"""Train only an identity-initialised adapter on frozen raw CAMPPlus pairs."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.cached_embedding_data import CachedCampPlusPairs, clean_statistics
from src.losses_v4 import campplus_calibration_loss
from src.models.campplus_residual_adapter import CampPlusResidualAdapter
from src.quality import perturb_quality
from src.splits import split_speakers


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _epoch(
    loader: DataLoader,
    model: CampPlusResidualAdapter,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    quality_jitter: tuple[float, float],
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums: dict[str, float] = {}
    baseline_cosine = 0.0
    predicted_cosine = 0.0
    count = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            degraded, clean, prototype, quality = (
                batch[key].to(device, non_blocking=True) for key in ("degraded", "clean", "prototype", "quality")
            )
            quality = perturb_quality(quality, quality_jitter, training)
            predicted, diagnostics = model(degraded, quality, alpha=1.0)
            clean_output, _ = model(clean, torch.ones_like(quality), alpha=1.0)
            losses = campplus_calibration_loss(
                predicted,
                clean,
                prototype,
                clean_output,
                clean,
                model.standardise(clean),
                diagnostics,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            batch_size = degraded.shape[0]
            for key, value in losses.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach()) * batch_size
            baseline_cosine += float(torch.nn.functional.cosine_similarity(degraded, clean).sum())
            predicted_cosine += float(torch.nn.functional.cosine_similarity(predicted, clean).sum())
            count += batch_size
    result = {key: value / count for key, value in sums.items()}
    result.update(
        {
            "baseline_target_cosine": baseline_cosine / count,
            "predicted_target_cosine": predicted_cosine / count,
            "embedding_cosine_gain": (predicted_cosine - baseline_cosine) / count,
        }
    )
    return result


def _save(
    path: Path,
    model: CampPlusResidualAdapter,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best: float,
    mean: torch.Tensor,
    std: torch.Tensor,
    row: dict,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "V4 frozen CAMPPlus + identity-initialised residual calibration",
            "embedding_space": "raw_cosyvoice_campplus_no_l2_normalization",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_validation_total": best,
            "embedding_mean": mean.cpu(),
            "embedding_std": std.cpu(),
            "metrics": row,
            "model_config": {
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "max_residual_z": args.max_residual_z,
            },
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max-residual-z", type=float, default=3.0)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--allow-cpu-smoke", action="store_true")
    args = parser.parse_args()
    _seed(args.seed)
    requested = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    device = torch.device(requested)
    if device.type != "cuda" and not args.allow_cpu_smoke:
        raise RuntimeError("formal V4 training requires CUDA; use --allow-cpu-smoke only for tests")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    splits = split_speakers(list(manifest), args.seed)
    mean, std = clean_statistics(args.embedding_cache, splits["train"])
    train_data = CachedCampPlusPairs(args.embedding_cache, splits["train"])
    validation_data = CachedCampPlusPairs(args.embedding_cache, splits["selection"])
    generator = torch.Generator().manual_seed(args.seed)
    options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **options)
    validation_loader = DataLoader(validation_data, shuffle=False, **options)
    model = CampPlusResidualAdapter(
        mean, std, hidden_dim=args.hidden_dim, dropout=args.dropout, max_residual_z=args.max_residual_z
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    start_epoch, best, stale, history = 1, float("inf"), 0, []
    if args.resume and args.resume.exists():
        state = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best = int(state["epoch"]) + 1, float(state["best_validation_total"])
    args.output.mkdir(parents=True, exist_ok=True)
    history_path = args.output / "history.json"
    if history_path.exists() and args.resume:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "event": "setup_v4",
                "device": str(device),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "train_pairs": len(train_data),
                "validation_pairs": len(validation_data),
            }
        ),
        flush=True,
    )
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        train_metrics = _epoch(train_loader, model, device, optimizer, (0.08, 0.04))
        validation_metrics = _epoch(validation_loader, model, device, None, (0.0, 0.0))
        scheduler.step()
        row = {
            "epoch": epoch,
            "seconds": time.time() - started,
            "learning_rate": scheduler.get_last_lr()[0],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        improved = validation_metrics["total"] < best
        if improved:
            best, stale = validation_metrics["total"], 0
        else:
            stale += 1
        _save(args.output / "last.pt", model, optimizer, scheduler, epoch, best, mean, std, row, args)
        if improved:
            _save(args.output / "best.pt", model, optimizer, scheduler, epoch, best, mean, std, row, args)
        history.append(row)
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
        if stale >= args.patience:
            print(json.dumps({"event": "early_stop", "epoch": epoch, "patience": args.patience}), flush=True)
            break


if __name__ == "__main__":
    main()
