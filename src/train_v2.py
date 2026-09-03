"""V2：Transformer RobustSpeakerEncoder 与 BWE 的联合 GPU 训练。"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.cached_data import CachedRobustDataset, CachedUniqueSpeakerSampler, collate
from src.models.losses import combined_loss
from src.models.robust_speaker_encoder import LightweightBWENet, RobustSpeakerEncoder, apply_soft_bwe
from src.quality import perturb_quality
from src.splits import split_speakers


class EmbeddingQueue:
    """FIFO memory bank that increases contrastive negatives beyond one batch."""

    def __init__(self, capacity: int, dimension: int = 192) -> None:
        self.capacity = capacity
        self.values = torch.empty(0, dimension)

    def negatives(self, device: torch.device) -> torch.Tensor | None:
        return self.values.to(device, non_blocking=True) if self.values.numel() else None

    def update(self, values: torch.Tensor) -> None:
        if self.capacity <= 0:
            return
        combined = torch.cat([self.values, values.detach().float().cpu()], dim=0)
        self.values = combined[-self.capacity :]

    def state_dict(self) -> dict:
        return {"capacity": self.capacity, "values": self.values}

    def load_state_dict(self, state: dict) -> None:
        self.capacity = int(state["capacity"])
        self.values = state["values"].float().cpu()


def _make_grad_scaler(device: torch.device):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    return torch.cuda.amp.GradScaler(enabled=device.type == "cuda")


def _initialise_worker(_: int) -> None:
    """在线退化含卷积；防止 8 个 worker 各自占满全部 CPU 线程。"""
    torch.set_num_threads(1)


def _split_speakers(manifest: Path, seed: int = 42) -> tuple[set[str], set[str]]:
    splits = split_speakers(list(json.loads(manifest.read_text(encoding="utf-8"))), seed)
    return splits["train"], splits["selection"]


def _save(path: Path, encoder: RobustSpeakerEncoder, bwe: LightweightBWENet, optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler.LRScheduler, queue: EmbeddingQueue, epoch: int, step: int,
          best: float, metrics: dict, arguments: argparse.Namespace) -> None:
    serialized_arguments = {key: str(value) if isinstance(value, Path) else value for key, value in vars(arguments).items()}
    torch.save({"encoder": encoder.state_dict(), "bwe": bwe.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "queue": queue.state_dict(), "epoch": epoch, "step": step,
                "best_validation_total": best, "metrics": metrics, "arguments": serialized_arguments,
                "architecture": "V3: quality-conditioned 6-layer Transformer + soft-routed mel U-Net BWE"}, path)


def run_epoch(loader: DataLoader, encoder: RobustSpeakerEncoder, bwe: LightweightBWENet, optimizer, scaler,
              scheduler, device: torch.device, training: bool, max_batches: int = 0,
              phase: str = "train", queue: EmbeddingQueue | None = None,
              quality_jitter: tuple[float, float] = (0.0, 0.0)) -> dict[str, float]:
    encoder.train(training); bwe.train(training); sums = {key: 0.0 for key in ("total", "info_nce", "distill", "bwe_l1", "bwe_mse")}; count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader, 1):
            degraded, clean, teacher, quality = (batch[key].to(device, non_blocking=True) for key in ("degraded_mel", "clean_mel", "teacher", "quality"))
            quality = perturb_quality(quality, quality_jitter, training)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                predicted = bwe(degraded)
                enhanced, gate = apply_soft_bwe(degraded, predicted, quality)
                anchor = encoder(enhanced, quality)
                clean_embedding = encoder(clean, torch.ones_like(quality))
                losses = combined_loss(
                    anchor, clean_embedding, teacher, torch.arange(anchor.size(0), device=device), gate,
                    predicted, clean, negatives=queue.negatives(device) if training and queue else None,
                )
            if training:
                optimizer.zero_grad(set_to_none=True); scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(bwe.parameters()), 1.0)
                scaler.step(optimizer); scaler.update(); scheduler.step()
                if queue:
                    queue.update(clean_embedding)
            for key, value in losses.items(): sums[key] += value.detach().float().item()
            count += 1
            if training and batch_index % 100 == 0:
                print(json.dumps({"event": "batch", "phase": phase, "batch": batch_index,
                                  "batches": len(loader), "mean_total": sums["total"] / count,
                                  "lr": scheduler.get_last_lr()[0]}), flush=True)
            if max_batches and batch_index >= max_batches:
                break
    return {key: value / max(count, 1) for key, value in sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--teacher-cache", type=Path, required=True); parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16); parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4); parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality-snr-jitter", type=float, default=0.08, help="SNR/30 condition noise std")
    parser.add_argument("--quality-bandwidth-jitter", type=float, default=0.04, help="bandwidth/8000 condition noise std")
    parser.add_argument("--memory-bank-size", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--allow-cpu-smoke", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0, help="仅用于安全冒烟；0 表示完整 epoch")
    parser.add_argument("--max-validation-batches", type=int, default=0, help="仅用于安全冒烟；0 表示完整 epoch")
    arguments = parser.parse_args(); torch.manual_seed(arguments.seed)
    requested_device = "cuda" if arguments.device == "auto" and torch.cuda.is_available() else "cpu" if arguments.device == "auto" else arguments.device
    device = torch.device(requested_device)
    if device.type != "cuda" and not arguments.allow_cpu_smoke: raise RuntimeError("Formal training requires CUDA; use --allow-cpu-smoke only for a bounded dry-run")
    train_speakers, validation_speakers = _split_speakers(arguments.manifest, arguments.seed)
    train_data = CachedRobustDataset(arguments.feature_cache, arguments.teacher_cache, arguments.manifest, train_speakers)
    validation_data = CachedRobustDataset(arguments.feature_cache, arguments.teacher_cache, arguments.manifest, validation_speakers)
    train_sampler = CachedUniqueSpeakerSampler(train_data, arguments.batch_size); validation_sampler = CachedUniqueSpeakerSampler(validation_data, arguments.batch_size)
    loader_options = {"num_workers": arguments.workers, "pin_memory": device.type == "cuda", "persistent_workers": arguments.workers > 0, "collate_fn": collate, "worker_init_fn": _initialise_worker if arguments.workers else None}
    train_loader = DataLoader(train_data, batch_sampler=train_sampler, **loader_options)
    validation_loader = DataLoader(validation_data, batch_sampler=validation_sampler, **loader_options)
    encoder, bwe = RobustSpeakerEncoder().to(device), LightweightBWENet().to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(bwe.parameters()), lr=arguments.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, len(train_loader) * arguments.epochs), eta_min=1e-6); scaler = _make_grad_scaler(device)
    queue = EmbeddingQueue(arguments.memory_bank_size)
    print(json.dumps({"event": "setup", "device": str(device), "train_batches_per_epoch": len(train_loader),
                      "validation_batches_per_epoch": len(validation_loader), "epochs": arguments.epochs,
                      "total_train_steps": len(train_loader) * arguments.epochs,
                      "encoder_parameters": sum(parameter.numel() for parameter in encoder.parameters()),
                      "bwe_parameters": sum(parameter.numel() for parameter in bwe.parameters())}), flush=True)
    start_epoch = 1; step = 0; best = float("inf"); history: list[dict] = []; arguments.output.mkdir(parents=True, exist_ok=True)
    if arguments.resume and arguments.resume.exists():
        resume = torch.load(arguments.resume, map_location=device, weights_only=True); encoder.load_state_dict(resume["encoder"]); bwe.load_state_dict(resume["bwe"]); optimizer.load_state_dict(resume["optimizer"]); scheduler.load_state_dict(resume["scheduler"]); start_epoch, step = resume["epoch"] + 1, resume["step"]
        best = float(resume.get("best_validation_total", float("inf")))
        if "queue" in resume: queue.load_state_dict(resume["queue"])
        history_path = arguments.output / "history.json"
        if history_path.exists(): history = json.loads(history_path.read_text(encoding="utf-8"))
    for epoch in range(start_epoch, arguments.epochs + 1):
        started = time.time(); train_sampler.set_epoch(epoch); validation_sampler.set_epoch(epoch)
        jitter = (arguments.quality_snr_jitter, arguments.quality_bandwidth_jitter)
        train_metrics = run_epoch(train_loader, encoder, bwe, optimizer, scaler, scheduler, device, True, arguments.max_train_batches, "train", queue, jitter); step += min(len(train_loader), arguments.max_train_batches or len(train_loader))
        validation_metrics = run_epoch(validation_loader, encoder, bwe, optimizer, scaler, scheduler, device, False, arguments.max_validation_batches, "validation")
        row = {"epoch": epoch, "step": step, "seconds": time.time() - started, "lr": scheduler.get_last_lr()[0], "train": train_metrics, "validation": validation_metrics}; history.append(row)
        improved = validation_metrics["total"] < best
        if improved: best = validation_metrics["total"]
        _save(arguments.output / "last.pt", encoder, bwe, optimizer, scheduler, queue, epoch, step, best, row, arguments)
        _save(arguments.output / f"epoch-{epoch:03d}.pt", encoder, bwe, optimizer, scheduler, queue, epoch, step, best, row, arguments)
        if improved: _save(arguments.output / "best.pt", encoder, bwe, optimizer, scheduler, queue, epoch, step, best, row, arguments)
        (arguments.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)


if __name__ == "__main__": main()
