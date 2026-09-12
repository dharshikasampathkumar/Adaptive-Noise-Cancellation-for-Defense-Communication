"""Tiny GRU-based DCCRN baseline with mandatory staged training gates.

The model is DCCRN-inspired, not a reproduction of the original LSTM DCCRN:
it uses real-valued tensors with paired real/imaginary channels, complex
convolution algebra, a lightweight causal GRU, and complex decoder blocks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    root: Path = Path(r"C:\SIH\synthetic_dataset")
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 128
    win_length: int = 512
    segment_seconds: float = 2.0
    batch_size: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 100
    short_epochs: int = 3
    gru_hidden_size: int = 128
    gru_layers: int = 1
    model_channels: Tuple[int, ...] = (16, 32, 64)
    si_snr_weight: float = 1.0
    spectral_loss_weight: float = 1.0
    random_seed: int = 42
    patience: int = 15
    output_dir: Path = Path(r"C:\SIH\dccrn_outputs")

    @property
    def segment_samples(self) -> int:
        return int(self.sample_rate * self.segment_seconds)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SpeechDataset(Dataset):
    def __init__(self, config: Config, split: str, training: bool):
        self.config = config
        self.training = training
        self.split = split
        split_path = config.root / "splits" / f"{split}.csv"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")
        with split_path.open(newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"Split is empty: {split_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_noisy(self, row: Dict[str, str]) -> Path:
        snr = int(float(row["target_snr_db"]))
        path = self.config.root / "noisy_speech" / f"snr_{snr}dB" / row["noisy_filename"]
        if not path.exists():
            raise FileNotFoundError(f"Noisy file from {self.split}.csv not found: {path}")
        return path

    def _resolve_clean(self, row: Dict[str, str]) -> Path:
        # The generated dataset stores split clean_filename as source metadata,
        # while clean_reference contains one generated file per sample ID.
        generated = self.config.root / "clean_reference" / f"{row['sample_id']}_clean_ref.wav"
        source = self.config.root / "clean_reference" / row["clean_filename"]
        if generated.exists():
            return generated
        if source.exists():
            return source
        raise FileNotFoundError(
            f"Clean file not found for {row['sample_id']}: tried {generated} and {source}"
        )

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Dict[str, object]]:
        row = self.rows[index]
        noisy_path, clean_path = self._resolve_noisy(row), self._resolve_clean(row)
        for path in (noisy_path, clean_path):
            info = sf.info(path)
            if info.format != "WAV" or info.subtype != "PCM_16":
                raise ValueError(
                    f"Expected WAV/PCM_16, got {info.format}/{info.subtype}: {path}"
                )
        noisy, noisy_sr = sf.read(noisy_path, dtype="float32", always_2d=True)
        clean, clean_sr = sf.read(clean_path, dtype="float32", always_2d=True)
        if noisy_sr != self.config.sample_rate or clean_sr != self.config.sample_rate:
            raise ValueError(f"Expected {self.config.sample_rate} Hz: {noisy_path}, {clean_path}")
        noisy = noisy.mean(axis=1)
        clean = clean.mean(axis=1)
        length = min(len(noisy), len(clean))
        noisy, clean = noisy[:length], clean[:length]
        target = self.config.segment_samples
        if length >= target:
            start = random.randint(0, length - target) if self.training else 0
            noisy, clean = noisy[start : start + target], clean[start : start + target]
        else:
            pad = target - length
            noisy = np.pad(noisy, (0, pad))
            clean = np.pad(clean, (0, pad))
        metadata = dict(row)
        return torch.from_numpy(noisy.copy()), torch.from_numpy(clean.copy()), metadata


# ---------------------------------------------------------------------------
# Model architecture (preserved exactly)
# ---------------------------------------------------------------------------

class ComplexConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: Tuple[int, int]):
        super().__init__()
        self.real = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.imag = nn.Conv2d(in_channels, out_channels, 3, stride, 1)

    def forward(self, x: Tensor) -> Tensor:
        real, imag = x[:, 0::2], x[:, 1::2]
        return torch.stack(
            (self.real(real) - self.imag(imag), self.real(imag) + self.imag(real)), dim=2
        ).flatten(1, 2)


class ComplexConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Tuple[int, int],
        output_padding: Tuple[int, int],
    ):
        super().__init__()
        self.real = nn.ConvTranspose2d(
            in_channels, out_channels, 3, stride, 1, output_padding=output_padding
        )
        self.imag = nn.ConvTranspose2d(
            in_channels, out_channels, 3, stride, 1, output_padding=output_padding
        )

    def forward(self, x: Tensor) -> Tensor:
        real, imag = x[:, 0::2], x[:, 1::2]
        return torch.stack(
            (self.real(real) - self.imag(imag), self.real(imag) + self.imag(real)), dim=2
        ).flatten(1, 2)


class TinyDCCRN(nn.Module):
    """Lightweight GRU-based DCCRN variant operating on paired complex channels."""

    def __init__(
        self,
        channels: Sequence[int],
        gru_hidden: int,
        gru_layers: int,
        input_shape: Tuple[int, int],
    ):
        super().__init__()
        if len(channels) != 3:
            raise ValueError("TinyDCCRN currently requires exactly three model channel stages")
        self.channels = tuple(channels)
        enc: List[nn.Module] = []
        in_channels = 1
        strides = [(2, 2), (2, 2), (1, 2)]
        for index, out_channels in enumerate(channels):
            enc += [
                ComplexConv2d(in_channels, out_channels, strides[index]),
                nn.BatchNorm2d(out_channels * 2),
                nn.PReLU(),
            ]
            in_channels = out_channels
        self.encoder = nn.ModuleList(enc)
        with torch.no_grad():
            encoded = torch.zeros(1, 2, *input_shape)
            for layer in self.encoder:
                encoded = layer(encoded)
        _, encoded_channels, encoded_frequency, _ = encoded.shape
        self.gru_input_size = encoded_channels * encoded_frequency
        self.gru = nn.GRU(self.gru_input_size, gru_hidden, gru_layers, batch_first=True)
        self.gru_projection = nn.Linear(gru_hidden, self.gru_input_size)
        dec: List[nn.Module] = []
        reversed_channels = list(reversed(channels))
        decoder_strides = [(1, 2), (2, 2), (2, 2)]
        decoder_output_padding = [(0, 0), (0, 1), (0, 0)]
        for i, current in enumerate(reversed_channels):
            out_channels = reversed_channels[i + 1] if i + 1 < len(reversed_channels) else 1
            dec += [
                ComplexConvTranspose2d(
                    current, out_channels, decoder_strides[i], decoder_output_padding[i]
                )
            ]
            if i + 1 < len(reversed_channels):
                dec += [nn.BatchNorm2d(out_channels * 2), nn.PReLU()]
        self.decoder = nn.ModuleList(dec)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError(f"Expected [B, 2, F, T] complex input, got {tuple(x.shape)}")
        skips: List[Tensor] = []
        y = x
        for layer in self.encoder:
            y = layer(y)
            if isinstance(layer, ComplexConv2d):
                skips.append(y)
        b, c, f, t = y.shape
        sequence = y.permute(0, 3, 1, 2).reshape(b, t, c * f)
        sequence, _ = self.gru(sequence)
        y = self.gru_projection(sequence).reshape(b, t, c, f).permute(0, 2, 3, 1)
        decoder_index = 0
        # The first decoder stage returns the encoder's second resolution;
        # the second returns the encoder's first resolution.
        skip_indices = (1, 0)
        for layer in self.decoder:
            y = layer(y)
            if isinstance(layer, ComplexConvTranspose2d) and skips:
                if decoder_index < len(skip_indices):
                    skip = skips[skip_indices[decoder_index]]
                    if y.shape != skip.shape:
                        raise RuntimeError(
                            "Deterministic skip connection mismatch: "
                            f"decoder={tuple(y.shape)}, encoder={tuple(skip.shape)}"
                        )
                    y = y + skip
                decoder_index += 1
        if y.shape != x.shape:
            raise RuntimeError(
                f"Decoder output must exactly match input: output={tuple(y.shape)}, input={tuple(x.shape)}"
            )
        return y


# ---------------------------------------------------------------------------
# Model factory – always use this to ensure a fresh, uncontaminated instance
# ---------------------------------------------------------------------------

def build_model(config: Config, device: torch.device) -> TinyDCCRN:
    """Construct a freshly initialised TinyDCCRN from the current configuration.

    Call this function every time a new training stage begins so that no
    previous optimizer updates or gradient state can leak into a new run.
    """
    example_spec = stft(torch.zeros(1, config.segment_samples), config)
    model = TinyDCCRN(
        config.model_channels,
        config.gru_hidden_size,
        config.gru_layers,
        (example_spec.shape[-2], example_spec.shape[-1]),
    ).to(device)
    return model


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def stft(waveform: Tensor, config: Config) -> Tensor:
    window = torch.hann_window(config.win_length, device=waveform.device)
    return torch.stft(
        waveform, config.n_fft, config.hop_length, config.win_length, window,
        return_complex=True, center=True,
    )


def istft(spectrogram: Tensor, config: Config, length: int) -> Tensor:
    window = torch.hann_window(config.win_length, device=spectrogram.device)
    return torch.istft(
        spectrogram, config.n_fft, config.hop_length, config.win_length, window,
        length=length, center=True,
    )


# ---------------------------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------------------------

def si_snr_loss(estimate: Tensor, target: Tensor) -> Tensor:
    """SI-SNR loss (lower is better; used for gradient optimisation).

    This is the *negative* SI-SNR.  Its numeric value is NOT a dB SNR reading.
    """
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    projection = (estimate * target).sum(-1, keepdim=True) * target
    projection = projection / (target.pow(2).sum(-1, keepdim=True) + 1e-8)
    noise = estimate - projection
    score = 10 * torch.log10(
        (projection.pow(2).sum(-1) + 1e-8) / (noise.pow(2).sum(-1) + 1e-8)
    )
    return -score.mean()


def compute_si_snr_metric(estimate: Tensor, target: Tensor) -> Tensor:
    """Return SI-SNR in dB; unlike si_snr_loss, higher values are better."""
    return -si_snr_loss(estimate, target)


def calculate_snr(clean: np.ndarray, estimate: np.ndarray) -> float:
    """Reconstruction SNR in dB: 10*log10(clean_power / error_power).

    Measures how well *estimate* reconstructs *clean*.
    NOT the environmental/dataset input SNR – use metadata measured_snr_db for that.
    """
    clean = clean.astype(np.float64)
    estimate = estimate.astype(np.float64)
    noise = estimate - clean
    return float(10.0 * np.log10((np.mean(clean ** 2) + 1e-12) / (np.mean(noise ** 2) + 1e-12)))


def calculate_stoi(estimate: np.ndarray, clean: np.ndarray, sample_rate: int) -> float:
    """Calculate STOI; raises RuntimeError with pip hint when unavailable."""
    try:
        from pystoi import stoi  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("STOI requires `pip install pystoi`.") from error
    return float(stoi(clean, estimate, sample_rate, extended=False))


def calculate_pesq(estimate: np.ndarray, clean: np.ndarray, sample_rate: int) -> float:
    """Calculate wideband PESQ for 16 kHz audio."""
    if sample_rate != 16000:
        raise ValueError("Wideband PESQ requires 16000 Hz sample rate.")
    try:
        from pesq import pesq  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("PESQ requires `pip install pesq`.") from error
    return float(pesq(sample_rate, clean, estimate, "wb"))


def losses(
    enhanced: Tensor,
    clean_wave: Tensor,
    predicted: Tensor,
    clean_spec: Tensor,
    config: Config,
) -> Tuple[Tensor, Tensor, Tensor]:
    si = si_snr_loss(enhanced, clean_wave)
    spectral = nn.functional.l1_loss(predicted.real, clean_spec.real) + nn.functional.l1_loss(
        predicted.imag, clean_spec.imag
    )
    return si, spectral, config.si_snr_weight * si + config.spectral_loss_weight * spectral


def finite(tensors: Iterable[Tensor]) -> bool:
    return all(bool(torch.isfinite(tensor).all()) for tensor in tensors)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def make_loaders(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    common = dict(
        batch_size=config.batch_size, num_workers=0, pin_memory=torch.cuda.is_available()
    )
    return (
        DataLoader(SpeechDataset(config, "train", True), shuffle=True, **common),
        DataLoader(SpeechDataset(config, "validation", False), shuffle=False, **common),
        DataLoader(SpeechDataset(config, "test", False), shuffle=False, **common),
    )


# ---------------------------------------------------------------------------
# Sanity check – self-contained, creates and discards its own temporary model
# ---------------------------------------------------------------------------

def sanity_check(config: Config, device: torch.device, loader: DataLoader) -> None:
    """Run all pipeline checks on a temporary model.

    The temporary model and optimizer are local to this function and are
    deleted before it returns.  They have NO effect on any training model.
    """
    print("\n" + "=" * 56)
    print("SANITY CHECK  (temporary model – will be discarded)")
    print("=" * 56)

    temp_model = build_model(config, device)
    temp_optimizer = torch.optim.AdamW(temp_model.parameters(), lr=config.learning_rate)

    stage = "dataset loading"
    passed: List[str] = []
    try:
        noisy, clean, _ = next(iter(loader))
        noisy, clean = noisy.to(device), clean.to(device)
        print(f"  Noisy waveform shape : {tuple(noisy.shape)}")
        print(f"  Clean waveform shape : {tuple(clean.shape)}")
        print(f"  Sample rate          : {config.sample_rate} Hz")
        print(f"  Batch size           : {noisy.shape[0]}")
        print(f"  Segment samples      : {config.segment_samples}")
        passed.append("Dataset loading")

        stage = "STFT"
        noisy_spec = stft(noisy, config)
        print(
            f"  STFT shape           : {tuple(noisy_spec.shape)}  "
            f"(freq bins={noisy_spec.shape[-2]}, time frames={noisy_spec.shape[-1]})"
        )
        print(f"  Real component shape : {tuple(noisy_spec.real.shape)}")
        print(f"  Imag component shape : {tuple(noisy_spec.imag.shape)}")
        passed.append("STFT")

        stage = "model forward pass"
        model_input = torch.stack((noisy_spec.real, noisy_spec.imag), dim=1)
        output = temp_model(model_input)
        print(f"  Model input shape    : {tuple(model_input.shape)}")
        print(f"  Model output shape   : {tuple(output.shape)}")
        passed.append("Model forward pass")

        stage = "output dimensions"
        if output.shape != model_input.shape:
            raise RuntimeError(
                f"Model output not a valid complex spectrogram shape: {tuple(output.shape)}"
            )
        passed.append("Output dimensions")

        stage = "iSTFT"
        enhanced_spec = torch.complex(output[:, 0], output[:, 1])
        enhanced = istft(enhanced_spec, config, noisy.shape[-1])
        clean_spec = stft(clean, config)
        if enhanced.shape != clean.shape:
            raise RuntimeError(
                f"Enhanced shape {tuple(enhanced.shape)} != clean shape {tuple(clean.shape)}"
            )
        print(
            f"  Enhanced shape       : {tuple(enhanced.shape)}  "
            f"duration={enhanced.shape[-1] / config.sample_rate:.3f}s"
        )
        passed.append("iSTFT")

        stage = "loss calculation"
        si, spectral, total = losses(enhanced, clean, enhanced_spec, clean_spec, config)
        # SI-SNR loss is the negative SI-SNR scalar; do NOT label it as dB.
        print(f"  SI-SNR Loss          : {si.item():.6f}  (optimizer objective, lower is better)")
        print(f"  Spectral L1 Loss     : {spectral.item():.6f}")
        print(f"  Total Loss           : {total.item():.6f}")
        passed.append("Loss calculation")

        stage = "NaN/Inf check"
        if not finite((noisy, clean, noisy_spec.real, output, enhanced, si, spectral, total)):
            raise RuntimeError("At least one sanity-check tensor contains NaN or Inf")
        passed.append("NaN/Inf check")

        stage = "backward propagation"
        temp_optimizer.zero_grad(set_to_none=True)
        total.backward()
        passed.append("Backward propagation")

        stage = "gradient check"
        has_grad = any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in temp_model.parameters()
            if p.requires_grad
        )
        if not has_grad:
            raise RuntimeError("No finite gradients were produced")
        passed.append("Finite gradients")

        stage = "optimizer update"
        temp_optimizer.step()
        passed.append("Optimizer update")

        print("\n  SANITY CHECK RESULTS")
        print("  " + "-" * 40)
        for name in passed:
            print(f"  {name:<30}: PASS")
        print("  ALL SANITY CHECKS PASSED")

    except (OSError, RuntimeError, ValueError, FloatingPointError) as error:
        print(f"\n  SANITY CHECK FAILED during [{stage}]: {error}")
        raise
    finally:
        del temp_model, temp_optimizer

    print("\nSANITY CHECK MODEL IS DISCARDED.")
    print("A FRESH MODEL WILL BE USED FOR TRAINING.")
    print("=" * 56)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    config: Config,
    device: torch.device,
) -> Dict[str, float]:
    model.train(optimizer is not None)
    totals: Dict[str, List[float]] = {
        "si_snr_loss": [], "spectral_loss": [], "total_loss": []
    }
    for noisy, clean, _ in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        noisy_spec, clean_spec = stft(noisy, config), stft(clean, config)
        predicted = model(torch.stack((noisy_spec.real, noisy_spec.imag), dim=1))
        enhanced = istft(
            torch.complex(predicted[:, 0], predicted[:, 1]), config, noisy.shape[-1]
        )
        si, spectral, total = losses(
            enhanced, clean,
            torch.complex(predicted[:, 0], predicted[:, 1]),
            clean_spec, config,
        )
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite loss encountered")
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        totals["si_snr_loss"].append(float(si.detach().cpu()))
        totals["spectral_loss"].append(float(spectral.detach().cpu()))
        totals["total_loss"].append(float(total.detach().cpu()))
    return {name: float(np.mean(values)) for name, values in totals.items()}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    epoch: int,
    loss: float,
    best_loss: float,
    config: Config,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "validation_loss": loss,
            "best_validation_loss": best_loss,
            "config": {
                k: str(v) if isinstance(v, Path) else v
                for k, v in asdict(config).items()
            },
            "model_config": {
                "channels": config.model_channels,
                "gru_hidden_size": config.gru_hidden_size,
                "gru_layers": config.gru_layers,
            },
            "stft_config": {
                "sample_rate": config.sample_rate,
                "n_fft": config.n_fft,
                "hop_length": config.hop_length,
                "win_length": config.win_length,
            },
            "random_seed": config.random_seed,
        },
        path,
    )


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
    device: torch.device,
    epochs: int,
    label: str = "FULL TRAINING",
) -> List[Dict[str, object]]:
    """Train for *epochs* epochs and return the complete history list."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    best = math.inf
    best_epoch = 0
    stale = 0
    history: List[Dict[str, object]] = []
    cumulative_time = 0.0
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 56}")
    print(f"{label}  ({epochs} epochs)")
    print("=" * 56)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_metrics = run_epoch(model, train_loader, optimizer, config, device)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, None, config, device)
        scheduler.step(val_metrics["total_loss"])
        learning_rate = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start
        cumulative_time += epoch_time

        if val_metrics["total_loss"] < best:
            best = val_metrics["total_loss"]
            best_epoch = epoch
            stale = 0
            save_checkpoint(
                config.output_dir / "best_dccrn.pth",
                model, optimizer, scheduler,
                epoch, val_metrics["total_loss"], best, config,
            )
        else:
            stale += 1

        save_checkpoint(
            config.output_dir / "latest_dccrn.pth",
            model, optimizer, scheduler,
            epoch, val_metrics["total_loss"],
            min(best, val_metrics["total_loss"]), config,
        )

        record: Dict[str, object] = {
            "epoch": epoch,
            "train_si_snr_loss": train_metrics["si_snr_loss"],
            "train_spectral_loss": train_metrics["spectral_loss"],
            "train_total_loss": train_metrics["total_loss"],
            "val_si_snr_loss": val_metrics["si_snr_loss"],
            "val_spectral_loss": val_metrics["spectral_loss"],
            "val_total_loss": val_metrics["total_loss"],
            "learning_rate": learning_rate,
            "epoch_time_sec": round(epoch_time, 2),
            "cumulative_time_sec": round(cumulative_time, 2),
            "best_val_loss": best,
            "best_epoch": best_epoch,
            "patience_counter": stale,
        }
        history.append(record)

        # Structured per-epoch console output
        print(f"\nEpoch {epoch}/{epochs}")
        print("  TRAIN:")
        print(f"    SI-SNR Loss  : {train_metrics['si_snr_loss']:.6f}"
              "  (optimizer objective)")
        print(f"    Spectral L1  : {train_metrics['spectral_loss']:.6f}")
        print(f"    Total Loss   : {train_metrics['total_loss']:.6f}")
        print("  VALIDATION:")
        print(f"    SI-SNR Loss  : {val_metrics['si_snr_loss']:.6f}"
              "  (optimizer objective)")
        print(f"    Spectral L1  : {val_metrics['spectral_loss']:.6f}")
        print(f"    Total Loss   : {val_metrics['total_loss']:.6f}")
        print(f"  Learning Rate  : {learning_rate:.2e}")
        print(f"  Epoch Time     : {epoch_time:.1f}s"
              f"  (cumulative: {cumulative_time:.1f}s)")
        print(f"  Best Val Loss  : {best:.6f}  (epoch {best_epoch})")
        print(f"  Patience       : {stale}/{config.patience}")

        if stale >= config.patience:
            print(f"\n  Early stopping triggered (patience={config.patience}).")
            break

    # Persist history
    history_json = config.output_dir / "training_history.json"
    history_csv = config.output_dir / "training_history.csv"
    with history_json.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    if history:
        _write_csv(history_csv, history, list(history[0].keys()))  # type: ignore[arg-type]
    print(f"\nTraining history saved to:")
    print(f"  {history_json}")
    print(f"  {history_csv}")
    return history


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def enhance_waveform(
    model: nn.Module, waveform: Tensor, config: Config, device: torch.device
) -> Tensor:
    with torch.no_grad():
        spec = stft(waveform.to(device), config)
        output = model(torch.stack((spec.real, spec.imag), dim=1))
        return istft(
            torch.complex(output[:, 0], output[:, 1]), config, waveform.shape[-1]
        ).cpu()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _mean(values: List[float]) -> float:
    valid = [v for v in values if math.isfinite(v)]
    return float(statistics.mean(valid)) if valid else float("nan")


def _std(values: List[float]) -> float:
    valid = [v for v in values if math.isfinite(v)]
    return float(statistics.stdev(valid)) if len(valid) >= 2 else float("nan")


def _min(values: List[float]) -> float:
    valid = [v for v in values if math.isfinite(v)]
    return float(min(valid)) if valid else float("nan")


def _max(values: List[float]) -> float:
    valid = [v for v in values if math.isfinite(v)]
    return float(max(valid)) if valid else float("nan")


def _safe_diff(a: float, b: float) -> float:
    """Return a - b; propagate NaN if either operand is non-finite."""
    return (a - b) if (math.isfinite(a) and math.isfinite(b)) else float("nan")


def _write_csv(
    path: Path, rows: List[Dict[str, object]], columns: List[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {col: row.get(col, "") for col in columns} for row in rows
        )


# ---------------------------------------------------------------------------
# Test-set evaluation
# ---------------------------------------------------------------------------

def evaluate_test_set(
    model: nn.Module,
    dataset: SpeechDataset,
    config: Config,
    device: torch.device,
) -> List[Dict[str, object]]:
    """Evaluate the held-out test split once without gradients or model selection.

    Input SNR is taken from the dataset metadata (measured_snr_db), not
    recomputed from clean/noisy waveforms.
    Output (reconstruction) SNR is computed from the model output vs. clean.
    """
    model.eval()
    results: List[Dict[str, object]] = []
    stoi_available = True
    pesq_available = True
    stoi_error_msg = ""
    pesq_error_msg = ""

    print(f"\nEvaluating {len(dataset)} test samples...")
    for index in range(len(dataset)):
        noisy, clean, metadata = dataset[index]

        # Enhance; keep as float32 numpy for metric computation.
        enhanced_tensor = enhance_waveform(
            model, noisy.unsqueeze(0), config, device
        ).squeeze(0)
        enhanced_np = enhanced_tensor.numpy()

        # Audio quality checks before any writing.
        max_amp = float(np.max(np.abs(enhanced_np)))
        rms = float(np.sqrt(np.mean(enhanced_np ** 2)))
        has_nan = bool(np.any(np.isnan(enhanced_np)))
        has_inf = bool(np.any(np.isinf(enhanced_np)))
        clipped = False

        if has_nan or has_inf:
            print(
                f"  WARNING [{metadata['sample_id']}]: enhanced audio contains "
                "NaN/Inf – replacing with zeros for metric calculation."
            )
            enhanced_np = np.zeros_like(enhanced_np)
        elif max_amp > 1.0:
            print(
                f"  WARNING [{metadata['sample_id']}]: enhanced max amplitude = "
                f"{max_amp:.4f} > 1.0. "
                "Metrics computed from raw float; WAV output will be clipped."
            )
            clipped = True

        noisy_np = noisy.numpy()
        clean_np = clean.numpy()

        # SI-SNR metric in dB (higher is better).
        input_si = float(compute_si_snr_metric(noisy, clean).item())
        output_si = float(
            compute_si_snr_metric(torch.from_numpy(enhanced_np.copy()), clean).item()
        )

        # Reconstruction SNR: 10*log10(clean_power / error_power) where
        # error = enhanced - clean.  This is NOT the environmental mixture SNR.
        output_reconstruction_snr = calculate_snr(clean_np, enhanced_np)

        # Dataset input SNR: taken directly from metadata generated at dataset
        # creation time.  It is the measured mixture SNR of noisy vs. clean.
        # Do NOT mix this with output_reconstruction_snr_db below.
        try:
            measured_input_snr = float(metadata["measured_snr_db"])
        except (KeyError, ValueError):
            measured_input_snr = float("nan")

        try:
            target_snr = float(metadata["target_snr_db"])
        except (KeyError, ValueError):
            target_snr = float("nan")

        # SNR improvement = output_reconstruction_snr - measured_input_mixture_snr.
        # These use different definitions and that is explicitly documented here.
        snr_improvement = _safe_diff(output_reconstruction_snr, measured_input_snr)

        input_stoi = output_stoi = float("nan")
        if stoi_available:
            try:
                input_stoi = calculate_stoi(noisy_np, clean_np, config.sample_rate)
                output_stoi = calculate_stoi(enhanced_np, clean_np, config.sample_rate)
            except RuntimeError as error:
                stoi_available = False
                stoi_error_msg = str(error)
                print(f"  STOI unavailable: {error}")

        input_pesq = output_pesq = float("nan")
        if pesq_available:
            try:
                input_pesq = calculate_pesq(noisy_np, clean_np, config.sample_rate)
                output_pesq = calculate_pesq(enhanced_np, clean_np, config.sample_rate)
            except RuntimeError as error:
                pesq_available = False
                pesq_error_msg = str(error)
                print(f"  PESQ unavailable: {error}")

        results.append(
            {
                "sample_id": metadata["sample_id"],
                "target_snr_db": target_snr,
                # measured_input_snr_db: mixture SNR from dataset metadata.
                "measured_input_snr_db": measured_input_snr,
                # output_reconstruction_snr_db: 10*log10(clean_power/error_power).
                "output_reconstruction_snr_db": output_reconstruction_snr,
                # snr_improvement_db: output_reconstruction_snr - measured_input_mixture_snr.
                "snr_improvement_db": snr_improvement,
                "input_si_snr_db": input_si,
                "output_si_snr_db": output_si,
                "si_snr_improvement_db": _safe_diff(output_si, input_si),
                "input_stoi": input_stoi,
                "output_stoi": output_stoi,
                "stoi_improvement": _safe_diff(output_stoi, input_stoi),
                "input_pesq": input_pesq,
                "output_pesq": output_pesq,
                "pesq_improvement": _safe_diff(output_pesq, input_pesq),
                "noise_category": metadata.get("noise_category", ""),
                "max_amplitude_enhanced": max_amp,
                "rms_enhanced": rms,
                "clipped_for_wav": clipped,
                # Internal waveform arrays (excluded from CSV by prefix '_').
                "_noisy": noisy_np,
                "_clean": clean_np,
                "_enhanced": enhanced_np,
            }
        )

    if not stoi_available:
        print(f"\n  NOTE: STOI unavailable this run. {stoi_error_msg}")
        print("        STOI values will be NaN in output CSVs.")
    if not pesq_available:
        print(f"\n  NOTE: PESQ unavailable this run. {pesq_error_msg}")
        print("        PESQ values will be NaN in output CSVs.")

    return results


# ---------------------------------------------------------------------------
# Group analysis
# ---------------------------------------------------------------------------

def evaluate_by_group(
    results: List[Dict[str, object]],
    key: str,
    output_path: Path,
) -> List[Dict[str, object]]:
    """Aggregate per-sample results by *key* and write a summary CSV."""
    metric_names = [
        "measured_input_snr_db", "output_reconstruction_snr_db", "snr_improvement_db",
        "input_si_snr_db", "output_si_snr_db", "si_snr_improvement_db",
        "input_stoi", "output_stoi", "stoi_improvement",
        "input_pesq", "output_pesq", "pesq_improvement",
    ]
    groups: Dict[object, List[Dict[str, object]]] = {}
    for row in results:
        groups.setdefault(row[key], []).append(row)

    summary: List[Dict[str, object]] = []
    for group, items in sorted(groups.items(), key=lambda pair: str(pair[0])):
        summary_row: Dict[str, object] = {key: group, "sample_count": len(items)}
        for metric in metric_names:
            values: List[float] = []
            for item in items:
                try:
                    v = float(item.get(metric, float("nan")))  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    v = float("nan")
                values.append(v)
            summary_row[f"mean_{metric}"] = _mean(values)
        summary.append(summary_row)

    if summary:
        _write_csv(output_path, summary, list(summary[0].keys()))
    return summary


# ---------------------------------------------------------------------------
# Audio examples
# ---------------------------------------------------------------------------

def save_audio_examples(
    results: List[Dict[str, object]], config: Config
) -> None:
    """Save one deterministic representative audio example per target-SNR group.

    Selection rule (fully deterministic, same result every run):
      1. Sort the entire results list by sample_id (lexicographic) so
         tie-breaking is deterministic regardless of evaluation order.
      2. For each target-SNR level (sorted ascending), pick the first sample
         from the sorted list that has not yet been selected.  When multiple
         noise categories are available within an SNR group, prefer a category
         not yet represented; otherwise accept any remaining sample.
      3. After the SNR pass, add one sample per noise category that is still
         unrepresented, again picking the lowest sample_id.

    Metrics have already been computed from raw float32 data.
    WAV output for clipped samples is clipped to [-1, 1] before writing.
    """
    examples_dir = config.output_dir / "results" / "audio_examples"
    target_snr_levels = sorted({float(row["target_snr_db"]) for row in results})

    # Deterministic ordering by sample_id for repeatable selection.
    sorted_results = sorted(results, key=lambda r: str(r["sample_id"]))

    selected: List[Dict[str, object]] = []
    seen_ids: set = set()
    seen_category: set = set()

    for snr in target_snr_levels:
        # Within this SNR level, prefer a noise category not yet seen.
        candidates = [
            r for r in sorted_results
            if float(r["target_snr_db"]) == snr and r["sample_id"] not in seen_ids
        ]
        if not candidates:
            continue
        # Try to pick a sample from an unseen noise category first.
        novel = [
            r for r in candidates
            if str(r.get("noise_category", "")) not in seen_category
        ]
        chosen = novel[0] if novel else candidates[0]
        selected.append(chosen)
        seen_ids.add(chosen["sample_id"])
        cat = str(chosen.get("noise_category", ""))
        if cat:
            seen_category.add(cat)

    # Second pass: cover any noise categories still unrepresented.
    all_categories = {
        str(r.get("noise_category", "")) for r in results if r.get("noise_category", "")
    }
    for cat in sorted(all_categories):  # sorted for determinism
        if cat not in seen_category:
            candidates = [
                r for r in sorted_results
                if str(r.get("noise_category", "")) == cat
                and r["sample_id"] not in seen_ids
            ]
            if candidates:
                chosen = candidates[0]
                selected.append(chosen)
                seen_ids.add(chosen["sample_id"])
                seen_category.add(cat)

    for row in selected:
        sid = row["sample_id"]
        directory = examples_dir / f"sample_{sid}"
        directory.mkdir(parents=True, exist_ok=True)

        enhanced_np = row["_enhanced"]
        # Clip only for PCM WAV writing; metrics were computed from raw float.
        enhanced_wav = (
            np.clip(enhanced_np, -1.0, 1.0)
            if row.get("clipped_for_wav")
            else enhanced_np
        )

        sf.write(directory / "clean.wav", row["_clean"], config.sample_rate, subtype="PCM_16")
        sf.write(directory / "noisy.wav", row["_noisy"], config.sample_rate, subtype="PCM_16")
        sf.write(directory / "enhanced.wav", enhanced_wav, config.sample_rate, subtype="PCM_16")

        with (directory / "info.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "sample_id": sid,
                    "target_snr_db": row["target_snr_db"],
                    "measured_input_snr_db": row["measured_input_snr_db"],
                    # output_reconstruction_snr_db: reconstruction error vs. clean.
                    "output_reconstruction_snr_db": row["output_reconstruction_snr_db"],
                    "snr_improvement_db": row["snr_improvement_db"],
                    "noise_category": row.get("noise_category", ""),
                    "clipped_for_wav": row.get("clipped_for_wav", False),
                },
                handle,
                indent=2,
            )

    print(f"Audio examples saved ({len(selected)} samples) to {examples_dir}")


# ---------------------------------------------------------------------------
# Overfitting diagnostic
# ---------------------------------------------------------------------------

def detect_overfitting(history: List[Dict[str, object]]) -> None:
    """Simple heuristic diagnostic based on training and validation loss curves."""
    print("\n" + "=" * 56)
    print("OVERFITTING DIAGNOSTIC")
    print("=" * 56)
    if len(history) < 5:
        print("  Too few epochs for a reliable overfitting check.")
        return

    train_losses = [float(h["train_total_loss"]) for h in history]  # type: ignore[arg-type]
    val_losses = [float(h["val_total_loss"]) for h in history]  # type: ignore[arg-type]
    best_val_idx = int(min(range(len(val_losses)), key=lambda i: val_losses[i]))
    best_val_epoch = best_val_idx + 1
    final_val = val_losses[-1]
    best_val = val_losses[best_val_idx]

    # Heuristic: in the second half of training, did train keep falling while
    # val rose more than 5 % above its best?
    half = len(history) // 2
    train_still_falling = train_losses[-1] < train_losses[half]
    val_rose_from_best = (final_val - best_val) > 0.05 * abs(best_val)

    trend_train = "decreasing" if train_still_falling else "plateaued / rising"
    trend_val = "rose after best" if val_rose_from_best else "stable / still improving"

    print(f"  Training loss trend   : {trend_train}")
    print(f"  Validation loss trend : {trend_val}")
    print(f"  Best validation epoch : {best_val_epoch}")
    print(f"  Best validation loss  : {best_val:.6f}")
    print(f"  Final validation loss : {final_val:.6f}")

    if train_still_falling and val_rose_from_best:
        print("\n  Potential overfitting detected.")
        print("  (Training loss kept falling while validation rose from its best.)")
    else:
        print("\n  No strong overfitting pattern detected from loss curves.")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_plots(
    history: List[Dict[str, object]],
    results: List[Dict[str, object]],
    snr_summary: List[Dict[str, object]],
    config: Config,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        print(f"Plots unavailable (install matplotlib): {error}")
        return

    plot_dir = config.output_dir / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    epochs = [int(item["epoch"]) for item in history]  # type: ignore[arg-type]

    def _fig(title: str, ylabel: str):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)
        return fig, ax

    # 1. Training total loss
    fig, ax = _fig("Training Total Loss", "Total Loss")
    ax.plot(epochs, [h["train_total_loss"] for h in history], color="#2176AE", lw=1.8)
    fig.tight_layout()
    fig.savefig(plot_dir / "training_total_loss.png", dpi=150)
    plt.close(fig)

    # 2. Validation total loss
    fig, ax = _fig("Validation Total Loss", "Total Loss")
    ax.plot(epochs, [h["val_total_loss"] for h in history], color="#F7882F", lw=1.8)
    fig.tight_layout()
    fig.savefig(plot_dir / "validation_total_loss.png", dpi=150)
    plt.close(fig)

    # 3. Train vs. validation
    fig, ax = _fig("Training vs. Validation Total Loss", "Total Loss")
    ax.plot(epochs, [h["train_total_loss"] for h in history],
            label="Train", color="#2176AE", lw=1.8)
    ax.plot(epochs, [h["val_total_loss"] for h in history],
            label="Validation", color="#F7882F", lw=1.8)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(plot_dir / "training_vs_validation_loss.png", dpi=150)
    plt.close(fig)

    # 4. Training SI-SNR loss
    fig, ax = _fig("Training SI-SNR Loss (optimizer objective)",
                   "SI-SNR Loss")
    ax.plot(epochs, [h["train_si_snr_loss"] for h in history], color="#26A96C", lw=1.8)
    fig.tight_layout()
    fig.savefig(plot_dir / "training_si_snr_loss.png", dpi=150)
    plt.close(fig)

    # 5. Validation SI-SNR loss
    fig, ax = _fig("Validation SI-SNR Loss (optimizer objective)",
                   "SI-SNR Loss")
    ax.plot(epochs, [h["val_si_snr_loss"] for h in history], color="#E76F51", lw=1.8)
    fig.tight_layout()
    fig.savefig(plot_dir / "validation_si_snr_loss.png", dpi=150)
    plt.close(fig)

    # 6. Training spectral loss
    fig, ax = _fig("Training Spectral L1 Loss", "Spectral L1 Loss")
    ax.plot(epochs, [h["train_spectral_loss"] for h in history], color="#7B2D8B", lw=1.8)
    fig.tight_layout()
    fig.savefig(plot_dir / "training_spectral_loss.png", dpi=150)
    plt.close(fig)

    # 7. Validation spectral loss
    fig, ax = _fig("Validation Spectral L1 Loss", "Spectral L1 Loss")
    ax.plot(epochs, [h["val_spectral_loss"] for h in history], color="#C77DFF", lw=1.8)
    fig.tight_layout()
    fig.savefig(plot_dir / "validation_spectral_loss.png", dpi=150)
    plt.close(fig)

    # Grouped bar charts from snr_summary
    if snr_summary:
        snr_labels = [str(r.get("target_snr_db", "")) for r in snr_summary]
        x_pos = list(range(len(snr_labels)))
        width = 0.38

        # 8. SNR improvement by target SNR
        # snr_improvement = output_reconstruction_snr - measured_input_mixture_snr.
        snr_imp = [
            float(r.get("mean_snr_improvement_db", float("nan")))
            for r in snr_summary
        ]
        fig, ax = _fig(
            "SNR Improvement by Target SNR\n"
            "(output reconstruction SNR \u2212 measured input mixture SNR)",
            "Mean SNR Improvement (dB)",
        )
        colors = ["#2176AE" if v >= 0 else "#E63946" for v in snr_imp]
        ax.bar(snr_labels, snr_imp, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Target SNR (dB)", fontsize=11)
        fig.tight_layout()
        fig.savefig(plot_dir / "snr_improvement_by_target_snr.png", dpi=150)
        plt.close(fig)

        # 9. STOI by target SNR
        in_stoi = [float(r.get("mean_input_stoi", float("nan"))) for r in snr_summary]
        out_stoi = [float(r.get("mean_output_stoi", float("nan"))) for r in snr_summary]
        fig, ax = _fig("STOI by Target SNR (Input vs. Output)", "Mean STOI")
        ax.bar([p - width / 2 for p in x_pos], in_stoi, width,
               label="Input STOI", color="#2176AE", edgecolor="white")
        ax.bar([p + width / 2 for p in x_pos], out_stoi, width,
               label="Output STOI", color="#F7882F", edgecolor="white")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(snr_labels)
        ax.set_xlabel("Target SNR (dB)", fontsize=11)
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(plot_dir / "stoi_by_target_snr.png", dpi=150)
        plt.close(fig)

        # 10. PESQ by target SNR
        in_pesq = [float(r.get("mean_input_pesq", float("nan"))) for r in snr_summary]
        out_pesq = [float(r.get("mean_output_pesq", float("nan"))) for r in snr_summary]
        fig, ax = _fig("PESQ by Target SNR (Input vs. Output)", "Mean PESQ (WB)")
        ax.bar([p - width / 2 for p in x_pos], in_pesq, width,
               label="Input PESQ", color="#2176AE", edgecolor="white")
        ax.bar([p + width / 2 for p in x_pos], out_pesq, width,
               label="Output PESQ", color="#F7882F", edgecolor="white")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(snr_labels)
        ax.set_xlabel("Target SNR (dB)", fontsize=11)
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(plot_dir / "pesq_by_target_snr.png", dpi=150)
        plt.close(fig)

    print(f"Plots saved to {plot_dir}")


# ---------------------------------------------------------------------------
# Model report and run configuration
# ---------------------------------------------------------------------------

def save_model_report(
    model: nn.Module,
    config: Config,
    device: torch.device,
    history: List[Dict[str, object]],
    test_results: List[Dict[str, object]],
    dataset_counts: Dict[str, int],
) -> None:
    best = min(history, key=lambda h: float(h["val_total_loss"]))  # type: ignore[arg-type]
    total = sum(p.numel() for p in model.parameters())

    def _agg(field: str) -> object:
        val = _mean([float(r[field]) for r in test_results])
        return round(val, 6) if math.isfinite(val) else None

    best_path = config.output_dir / "best_dccrn.pth"
    ckpt_mb = (
        round(best_path.stat().st_size / (1024 ** 2), 4)
        if best_path.exists()
        else None
    )

    report = {
        "model_description": "Tiny GRU-based DCCRN-inspired speech enhancement model",
        "total_parameters": total,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "approximate_fp32_weight_memory_mb": round(total * 4 / (1024 ** 2), 4),
        "approximate_checkpoint_size_mb": ckpt_mb,
        "sample_rate": config.sample_rate,
        "n_fft": config.n_fft,
        "hop_length": config.hop_length,
        "win_length": config.win_length,
        "segment_duration_sec": config.segment_seconds,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau(factor=0.5, patience=3)",
        "gru_hidden_size": config.gru_hidden_size,
        "gru_layers": config.gru_layers,
        "model_channels": list(config.model_channels),
        "si_snr_loss_weight": config.si_snr_weight,
        "spectral_loss_weight": config.spectral_loss_weight,
        "random_seed": config.random_seed,
        "device": str(device),
        "number_of_epochs_completed": len(history),
        "best_epoch": int(best["epoch"]),  # type: ignore[arg-type]
        "best_validation_loss": float(best["val_total_loss"]),  # type: ignore[arg-type]
        "dataset_counts": dataset_counts,
        "test_set_metrics": {
            "mean_measured_input_snr_db": _agg("measured_input_snr_db"),
            "mean_output_reconstruction_snr_db": _agg("output_reconstruction_snr_db"),
            "mean_snr_improvement_db": _agg("snr_improvement_db"),
            "mean_input_si_snr_db": _agg("input_si_snr_db"),
            "mean_output_si_snr_db": _agg("output_si_snr_db"),
            "mean_si_snr_improvement_db": _agg("si_snr_improvement_db"),
            "mean_input_stoi": _agg("input_stoi"),
            "mean_output_stoi": _agg("output_stoi"),
            "mean_stoi_improvement": _agg("stoi_improvement"),
            "mean_input_pesq": _agg("input_pesq"),
            "mean_output_pesq": _agg("output_pesq"),
            "mean_pesq_improvement": _agg("pesq_improvement"),
        },
        "snr_improvement_note": (
            "snr_improvement_db = output_reconstruction_snr_db - measured_input_snr_db. "
            "These two quantities use different definitions: "
            "output_reconstruction_snr_db measures 10*log10(clean_power/error_power); "
            "measured_input_snr_db is the mixture SNR from dataset metadata."
        ),
    }
    report_path = config.output_dir / "model_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Model report saved to {report_path}")


def save_run_config(config: Config, device: torch.device) -> None:
    """Persist all configuration and environment details for reproducibility."""
    gpu_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    )
    run_cfg = {
        "config": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in asdict(config).items()
        },
        "random_seed": config.random_seed,
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "operating_system": platform.platform(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
    }
    path = config.output_dir / "run_config.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(run_cfg, handle, indent=2)
    print(f"Run configuration saved to {path}")


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

def print_final_summary(
    config: Config,
    model: nn.Module,
    history: List[Dict[str, object]],
    test_results: List[Dict[str, object]],
    best_epoch: int,
    best_val_loss: float,
) -> None:
    total = sum(p.numel() for p in model.parameters())
    size_mb = total * 4 / (1024 ** 2)

    def _vals(field: str) -> List[float]:
        return [float(r[field]) for r in test_results]

    def _fmt_stat(field: str, unit: str = "", signed: bool = False) -> str:
        """Return 'mean ± std  [min, max]' for a metric field."""
        vals = _vals(field)
        mn = _mean(vals)
        sd = _std(vals)
        lo = _min(vals)
        hi = _max(vals)
        sign = "+" if signed else ""
        if math.isfinite(mn):
            return (
                f"{mn:{sign}.2f}{unit}  "
                f"\u00b1{sd:.2f}{unit}  "
                f"[{lo:{sign}.2f}{unit}, {hi:{sign}.2f}{unit}]"
            )
        return "N/A"

    print("\n" + "=" * 64)
    print("FINAL BASELINE EXPERIMENT SUMMARY")
    print("=" * 64)
    print(f"Best Epoch           : {best_epoch}")
    print(f"Best Validation Loss : {best_val_loss:.6f}")
    print()
    print("TEST SET:")
    print(f"  Number of samples  : {len(test_results)}")
    print()
    print("  Format: mean \u00b1 std  [min, max]")
    print()
    print("  SNR  (note: improvement = output reconstruction SNR - input mixture SNR)")
    print(f"    Measured Input SNR (mixture)    : {_fmt_stat('measured_input_snr_db', ' dB', True)}")
    print(f"    Output Reconstruction SNR        : {_fmt_stat('output_reconstruction_snr_db', ' dB', True)}")
    print(f"    SNR Improvement                  : {_fmt_stat('snr_improvement_db', ' dB', True)}")
    print()
    print("  SI-SNR  (dB, higher is better)")
    print(f"    Input SI-SNR                     : {_fmt_stat('input_si_snr_db', ' dB', True)}")
    print(f"    Output SI-SNR                    : {_fmt_stat('output_si_snr_db', ' dB', True)}")
    print(f"    SI-SNR Improvement               : {_fmt_stat('si_snr_improvement_db', ' dB', True)}")
    print()
    print("  STOI  (0 to 1, higher is better)")
    print(f"    Input STOI                       : {_fmt_stat('input_stoi')}")
    print(f"    Output STOI                      : {_fmt_stat('output_stoi')}")
    print(f"    STOI Improvement                 : {_fmt_stat('stoi_improvement', signed=True)}")
    print()
    print("  PESQ  (wideband, higher is better)")
    print(f"    Input PESQ                       : {_fmt_stat('input_pesq')}")
    print(f"    Output PESQ                      : {_fmt_stat('output_pesq')}")
    print(f"    PESQ Improvement                 : {_fmt_stat('pesq_improvement', signed=True)}")
    print()

    # ------------------------------------------------------------------
    # Target benchmark comparison (for reference only).
    # The model is NOT declared invalid if it does not meet these targets.
    # Benchmarks are illustrative targets for a DCCRN-inspired baseline on
    # mixed-environment defence noise at -5 to +20 dB input SNR.
    # ------------------------------------------------------------------
    BENCHMARKS: Dict[str, Tuple[float, str]] = {
        "si_snr_improvement_db":   (2.0,  "SI-SNR improvement >= 2.0 dB"),
        "stoi_improvement":        (0.05, "STOI improvement >= 0.05"),
        "pesq_improvement":        (0.10, "PESQ improvement >= 0.10"),
    }
    print("  " + "-" * 60)
    print("  TARGET BENCHMARK COMPARISON  (reference only)")
    print("  " + "-" * 60)
    for field, (target, label) in BENCHMARKS.items():
        achieved = _mean(_vals(field))
        if math.isfinite(achieved):
            met = achieved >= target
            status = "MEET" if met else "BELOW"
            print(f"    {label}")
            print(f"      Achieved: {achieved:+.3f}  -> {status}")
        else:
            print(f"    {label}")
            print(f"      Achieved: N/A (metric unavailable)")
    print("  " + "-" * 60)
    print("  NOTE: Not meeting a benchmark does NOT invalidate this run.")
    print("        Benchmarks are reference targets, not pass/fail criteria.")
    print()

    print(f"Model Parameters     : {total:,}")
    print(f"Approx. FP32 Size    : {size_mb:.2f} MB")
    print()
    print(f"Output Directory     : {config.output_dir}")
    print("=" * 64)
    print()
    print("Baseline training and held-out test evaluation completed.")


# ---------------------------------------------------------------------------
# Dataset count helper
# ---------------------------------------------------------------------------

def _dataset_counts(config: Config) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for split in ("train", "validation", "test"):
        path = config.root / "splits" / f"{split}.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                counts[split] = sum(1 for _ in csv.DictReader(handle))
        else:
            counts[split] = 0
    return counts


# ---------------------------------------------------------------------------
# Full evaluation pipeline (called once after best checkpoint is loaded)
# ---------------------------------------------------------------------------

def finalize_evaluation(
    model: nn.Module,
    test_dataset: SpeechDataset,
    config: Config,
    device: torch.device,
    history: List[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], int, float]:
    """Run the complete held-out test evaluation and generate all outputs.

    Returns (results, best_epoch, best_val_loss).
    The test set is evaluated exactly once and is never used for model selection.
    """
    (config.output_dir / "results").mkdir(parents=True, exist_ok=True)

    results = evaluate_test_set(model, test_dataset, config, device)

    # Per-sample CSV – skip internal waveform arrays.
    csv_columns = [k for k in results[0] if not k.startswith("_")]
    _write_csv(
        config.output_dir / "results" / "evaluation_results.csv",
        results,
        csv_columns,
    )

    snr_summary = evaluate_by_group(
        results,
        "target_snr_db",
        config.output_dir / "results" / "snr_summary.csv",
    )
    category_summary = evaluate_by_group(
        results,
        "noise_category",
        config.output_dir / "results" / "noise_category_summary.csv",
    )

    save_audio_examples(results, config)
    save_plots(history, results, snr_summary, config)

    dataset_counts = _dataset_counts(config)
    save_model_report(model, config, device, history, results, dataset_counts)
    save_run_config(config, device)

    best_record = min(history, key=lambda h: float(h["val_total_loss"]))  # type: ignore[arg-type]
    best_epoch = int(best_record["epoch"])  # type: ignore[arg-type]
    best_val_loss = float(best_record["val_total_loss"])  # type: ignore[arg-type]

    detect_overfitting(history)

    print("\nSNR GROUP SUMMARY:")
    print("  (output reconstruction SNR - measured input mixture SNR = improvement)")
    for row in snr_summary:
        print(
            f"  Target {row.get('target_snr_db', '?'):>4} dB"
            f" | n={row['sample_count']:>3}"
            f" | Input SNR={row.get('mean_measured_input_snr_db', float('nan')):+.1f} dB"
            f" | Output Recon SNR={row.get('mean_output_reconstruction_snr_db', float('nan')):+.1f} dB"
            f" | SI-SNR Imp={row.get('mean_si_snr_improvement_db', float('nan')):+.2f} dB"
        )

    print("\nNOISE CATEGORY SUMMARY:")
    for row in category_summary:
        print(
            f"  {str(row.get('noise_category', '?')):<32}"
            f" | n={row['sample_count']:>3}"
            f" | Input SNR={row.get('mean_measured_input_snr_db', float('nan')):+.1f} dB"
            f" | Output Recon SNR={row.get('mean_output_reconstruction_snr_db', float('nan')):+.1f} dB"
            f" | SI-SNR Imp={row.get('mean_si_snr_improvement_db', float('nan')):+.2f} dB"
        )

    return results, best_epoch, best_val_loss


# ---------------------------------------------------------------------------
# Entry point – 30-step execution pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Tiny GRU-based DCCRN-inspired speech enhancement baseline."
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Full-run epochs (default: config.num_epochs = 100).",
    )
    parser.add_argument(
        "--skip-short-test", action="store_true",
        help="Skip short training test (requires prior short-test marker file).",
    )
    parser.add_argument(
        "--sanity-only", action="store_true",
        help="Run sanity checks and stop immediately after.",
    )
    args = parser.parse_args()

    # STEP 1: Load configuration
    config = Config()
    if args.epochs is not None:
        config.num_epochs = args.epochs

    # STEP 2: Set random seeds
    seed_everything(config.random_seed)

    # STEP 3: Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device             : {device}")
    print(f"CUDA available     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name           : {torch.cuda.get_device_name(0)}")

    # STEP 4: Load existing split CSVs (splits are never altered)
    train_loader, val_loader, _ = make_loaders(config)

    # STEP 5: Print dataset counts
    dataset_counts = _dataset_counts(config)
    print(f"\nDataset:")
    print(f"  Train      : {dataset_counts.get('train', 0):,} samples")
    print(f"  Validation : {dataset_counts.get('validation', 0):,} samples")
    print(f"  Test       : {dataset_counts.get('test', 0):,} samples")

    # Print model size using a probe (immediately discarded)
    probe = build_model(config, device)
    total_params = sum(p.numel() for p in probe.parameters())
    trainable_params = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    del probe
    print(f"\nModel: Tiny GRU-based DCCRN-inspired architecture")
    print(f"  Parameters         : {total_params:,} total / {trainable_params:,} trainable")
    print(f"  Approx. FP32 size  : {total_params * 4 / (1024 ** 2):.2f} MB")

    # STEP 6 & 7: Build temporary sanity-check model; run sanity checks.
    # sanity_check() creates and destroys its own model+optimizer internally.
    sanity_check(config, device, train_loader)

    # STEP 8: Sanity-check model is discarded inside sanity_check().
    if args.sanity_only:
        print("\n--sanity-only flag set. Stopping after sanity checks.")
        return

    # STEP 9 & 10: Build fresh short-test model; mandatory short training.
    if not args.skip_short_test:
        print(f"\nBuilding fresh model for short training test"
              f" ({config.short_epochs} epochs).")
        seed_everything(config.random_seed)
        short_model = build_model(config, device)
        train(
            short_model, train_loader, val_loader, config, device,
            config.short_epochs, label="SHORT TRAINING TEST",
        )
        # STEP 11: Verify and record that short training passed.
        config.output_dir.mkdir(parents=True, exist_ok=True)
        with (config.output_dir / "short_training_passed.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump({"epochs": config.short_epochs, "status": "passed"}, handle, indent=2)
        print("\nShort training test PASSED.")
        print("Run again with --skip-short-test for the full baseline training.")
        # STEP 12: Discard short-test model.
        del short_model
        return

    # Guard: full training is blocked until short test has passed.
    marker = config.output_dir / "short_training_passed.json"
    if not marker.exists():
        # Recovery: accept a training_history.json with enough valid epochs.
        history_path = config.output_dir / "training_history.json"
        if history_path.exists():
            with history_path.open(encoding="utf-8") as handle:
                prior_history = json.load(handle)
            if len(prior_history) >= config.short_epochs and all(
                math.isfinite(item["train_total_loss"])
                and math.isfinite(item["val_total_loss"])
                for item in prior_history[: config.short_epochs]
            ):
                with marker.open("w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "epochs": config.short_epochs,
                            "status": "passed",
                            "recovered": True,
                        },
                        handle,
                        indent=2,
                    )
    if not marker.exists():
        raise RuntimeError(
            "Full training is blocked: run the short training test first "
            "(invoke without --skip-short-test)."
        )

    # STEP 13: Build completely fresh full-training model.
    # This model has NEVER received an optimizer update.
    print("\nBuilding completely fresh model for full baseline training.")
    seed_everything(config.random_seed)
    full_model = build_model(config, device)

    # STEPS 14–19: Train on train split, monitor validation split,
    # save latest and best checkpoints, apply early stopping.
    history = train(
        full_model, train_loader, val_loader, config, device,
        config.num_epochs, label="FULL BASELINE TRAINING",
    )

    # STEP 20: Load best checkpoint and verify.
    best_path = config.output_dir / "best_dccrn.pth"
    checkpoint = torch.load(best_path, map_location=device)
    full_model.load_state_dict(checkpoint["model_state_dict"])
    best_ckpt_epoch = checkpoint["epoch"]
    best_ckpt_val_loss = checkpoint["best_validation_loss"]
    print(f"\nLoaded best checkpoint from epoch : {best_ckpt_epoch}")
    print(f"Best validation loss              : {best_ckpt_val_loss:.6f}")

    # STEPS 21–29: Evaluate held-out test split; generate all outputs.
    test_dataset = SpeechDataset(config, "test", False)
    test_results, best_epoch, best_val_loss = finalize_evaluation(
        full_model, test_dataset, config, device, history
    )

    # STEP 30: Print final experiment summary.
    print_final_summary(
        config, full_model, history, test_results, best_epoch, best_val_loss
    )


if __name__ == "__main__":
    main()
