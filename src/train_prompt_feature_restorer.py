"""Train the deployable V4 prompt-feature restorer."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.cached_prompt_feature_data import CachedPromptFeatures, prompt_feature_statistics
from src.models.prompt_feature_restorer import PromptFeatureRestorer
from src.prompt_feature_loss import restoration_loss
from src.quality import perturb_quality
from src.splits import split_speakers


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _scaler(device: torch.device):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    return torch.cuda.amp.GradScaler(enabled=device.type == "cuda")


def _epoch(loader, model, device, optimizer=None, scaler=None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums: dict[str, float] = {}
    baseline_l1 = predicted_l1 = 0.0
    count = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            degraded, clean, quality = (batch[key].to(device, non_blocking=True) for key in ("degraded", "clean", "quality"))
            quality = perturb_quality(quality, (0.08, 0.04), training)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                predicted, diagnostics = model(degraded, quality)
                clean_output, _ = model(clean, torch.ones_like(quality))
                losses = restoration_loss(predicted, clean, model.standardise(clean), clean_output, clean, diagnostics)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            batch_size = degraded.shape[0]
            for key, value in losses.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach()) * batch_size
            baseline_l1 += float((degraded - clean).abs().mean(dim=(1, 2)).sum())
            predicted_l1 += float((predicted - clean).abs().mean(dim=(1, 2)).sum())
            count += batch_size
    result = {key: value / count for key, value in sums.items()}
    result.update(
        {
            "baseline_feature_l1": baseline_l1 / count,
            "predicted_feature_l1": predicted_l1 / count,
            "feature_l1_gain": (baseline_l1 - predicted_l1) / count,
        }
    )
    return result


def _save(path, model, optimizer, scheduler, epoch, best, mean, std, row, args) -> None:
    torch.save(
        {
            "architecture": "V4 frozen CosyVoice + identity-initialised prompt-feature restorer",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_validation_total": best,
            "feature_mean": mean.cpu(),
            "feature_std": std.cpu(),
            "metrics": row,
            "model_config": {
                "channels": args.channels,
                "blocks": args.blocks,
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
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max-residual-z", type=float, default=4.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--allow-cpu-smoke", action="store_true")
    args = parser.parse_args()
    _seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu_smoke:
        raise RuntimeError("formal training requires CUDA")
    speakers = list(json.loads(args.manifest.read_text(encoding="utf-8")))
    splits = split_speakers(speakers, args.seed)
    mean, std = prompt_feature_statistics(args.feature_cache, splits["train"])
    train_data = CachedPromptFeatures(args.feature_cache, splits["train"])
    validation_data = CachedPromptFeatures(args.feature_cache, splits["selection"])
    options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_data, shuffle=True, generator=torch.Generator().manual_seed(args.seed), **options)
    validation_loader = DataLoader(validation_data, shuffle=False, **options)
    model = PromptFeatureRestorer(mean, std, args.channels, args.blocks, args.dropout, args.max_residual_z).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = _scaler(device)
    start_epoch, best, stale, history = 1, float("inf"), 0, []
    args.output.mkdir(parents=True, exist_ok=True)
    if args.resume and args.resume.exists():
        state = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best = int(state["epoch"]) + 1, float(state["best_validation_total"])
        history_path = args.output / "history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
    print(json.dumps({"event": "feature_restorer_setup", "device": str(device), "parameters": sum(p.numel() for p in model.parameters()), "train_pairs": len(train_data), "validation_pairs": len(validation_data)}), flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        train_metrics = _epoch(train_loader, model, device, optimizer, scaler)
        validation_metrics = _epoch(validation_loader, model, device)
        scheduler.step()
        row = {"epoch": epoch, "seconds": time.time() - started, "learning_rate": scheduler.get_last_lr()[0], "train": train_metrics, "validation": validation_metrics}
        improved = validation_metrics["total"] < best
        if improved:
            best, stale = validation_metrics["total"], 0
        else:
            stale += 1
        _save(args.output / "last.pt", model, optimizer, scheduler, epoch, best, mean, std, row, args)
        if improved:
            _save(args.output / "best.pt", model, optimizer, scheduler, epoch, best, mean, std, row, args)
        history.append(row)
        (args.output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
        if stale >= args.patience:
            print(json.dumps({"event": "early_stop", "epoch": epoch}), flush=True)
            break


if __name__ == "__main__":
    main()
