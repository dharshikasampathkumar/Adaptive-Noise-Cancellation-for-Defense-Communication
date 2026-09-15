"""Tiny GRU-based DCCRN, EXPERIMENT 3: identity-initialised bounded CRM.

The model is DCCRN-inspired, not a reproduction of the original LSTM DCCRN:
it uses real-valued tensors with paired real/imaginary channels, complex
convolution algebra, a lightweight causal GRU, and complex decoder blocks.

EXPERIMENT 3 — SINGLE CHANGE FROM THE CORRECTED BASELINE (dccrn_train.py):
---------------------------------------------------------------------------
Diagnosis (from Experiment 1 and Experiment 2 test results): the network's
final layer previously predicted the enhanced complex spectrum directly
("spectral mapping" - see the old TinyDCCRN.forward, which returned the
decoder output y unchanged). A mapping architecture has no structural
notion of "leave the input alone" - at +15/+20 dB, where the noisy input is
already close to clean, the network still has to reconstruct the *entire*
spectrum from scratch through a compressed ~4.4M-parameter GRU bottleneck,
and the small reconstruction error this introduces is enough to move STOI
and PESQ in the wrong direction even though SI-SNR/waveform-L1 stay
favourable (both metrics are sensitive to fine short-time/phase structure
that scale-invariant, whole-segment losses do not penalise).

Experiment 2 tried to fix this at the LOSS level (an SNR-weighted
transparency term pulling the output toward the noisy input at high SNR).
That failed (see EXPERIMENT_02 test results: STOI/PESQ both got *worse*,
not better) because a loss-level penalty only nudges the *average* behaviour
of a mapping network over an SNR bucket - it cannot give the network a
sample-specific "this one is already fine" signal, and it still has to
fight the same from-scratch reconstruction burden every single step.

Experiment 3 instead changes the OUTPUT PARAMETERISATION of the existing
architecture (same encoder/decoder/GRU/channel sizes - nothing else is
touched): the final decoder output is reinterpreted as a bounded complex
ratio mask (CRM) that multiplies the noisy input spectrum, instead of being
used as the spectrum directly:

    enhanced_spec = (1 + bounded(decoder_output)) * noisy_spec

The final ComplexConvTranspose2d layer is zero-initialised, so at the start
of training bounded(decoder_output) == 0 and the network is an exact
identity function (enhanced == noisy) before a single gradient step. From
there, the same SI-SNR / spectral / amplitude / waveform-L1 losses used in
the corrected baseline (unchanged - no transparency loss, no new loss terms)
pull the mask away from 1 only where doing so actually reduces the loss:
strongly in noise-dominated low-SNR bins (mask can move arbitrarily far from
1, so low-SNR suppression capacity is not reduced - see the docstring above
TinyDCCRN.forward for why this reparameterisation loses no expressiveness),
negligibly in already-clean high-SNR bins, where "mask = 1" is both the
initial condition and the loss-minimising answer.

This requires no ground-truth SNR at inference: the mask is a function of
the noisy spectrogram alone (exactly what enhance_waveform() has access to
on the Raspberry Pi). It adds zero trainable parameters (same layer shapes)
and negligible extra compute (one elementwise complex multiply against a
tensor - noisy_spec - the model already consumes as input).
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# ---------------------------------------------------------------------------
# Experiment 3 identity constants
# ---------------------------------------------------------------------------

# Every output file for Experiment 3 must live inside this root.
# The _assert_exp3_path() guard enforces this before every write.
EXPERIMENT_NAME: str = "Experiment_03_Identity_Residual"
_EXP3_ROOT: Path = Path(r"C:\SIH") / EXPERIMENT_NAME


def _assert_exp3_path(path: Path) -> None:
    """Abort if *path* is outside the Experiment 3 root.

    Prevents any accidental write into Experiment 1 (dccrn_outputs/) or
    Experiment 2 (models/Experiment_02_SNR_Aware/) directories.
    Called before every torch.save() and file open() in this script.
    """
    try:
        Path(path).resolve().relative_to(_EXP3_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(
            f"SAFETY GUARD: attempted write OUTSIDE Experiment 3 root!\n"
            f"  Blocked path : {path}\n"
            f"  Expected root: {_EXP3_ROOT}\n"
            "Aborting to protect Experiment 1 and Experiment 2 outputs."
        ) from exc


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
    # EXPERIMENT 2 ADDITION: penalises global level/RMS mismatch between the
    # enhanced waveform and clean target, in dB. SI-SNR loss is *scale
    # invariant* by construction (see si_snr_loss docstring), so it supplies
    # zero gradient signal telling the network to match absolute output
    # level. The Experiment-1 checkpoint exploited exactly this gap: it
    # learned an enhanced waveform whose RMS was consistently ~20-30% of the
    # clean target's RMS (see DIAGNOSTIC_REPORT.md) while still lowering the
    # SI-SNR loss, because SI-SNR does not see that mismatch at all.
    # amplitude_consistency_loss is expressed in dB (like SI-SNR) so its
    # natural numeric range (roughly 0-15 for the errors seen in Experiment 1)
    # is directly comparable to the SI-SNR loss's range, which is why a
    # starting weight of 1.0 was used in Experiment 2. ROUND-2 AUDIT UPDATE:
    # now that waveform_loss_weight (below) also constrains amplitude - more
    # directly, since it's a literal waveform L1 distance - keeping this term
    # at its original weight of 1.0 as well would double-penalise the same
    # failure mode and could over-constrain amplitude at the expense of
    # spectral/SI-SNR quality. Reduced to 0.3: still contributes a
    # level-only signal (useful because it is insensitive to phase, so it
    # keeps working even when waveform L1 is noisy early in training), but
    # no longer the primary mechanism.
    amplitude_loss_weight: float = 0.3
    # EXPERIMENT 2 ADDITION: STOI/PESQ are the two metrics run_epoch()
    # deliberately does NOT compute every step (they are ~10-50x slower per
    # sample than the tensor-only diagnostics above and there is no
    # autograd need for them). Instead, full_validation_diagnostics() below
    # runs them on the validation split every N epochs, so validation-time
    # perceptual quality is still tracked over the course of training
    # (Step 5 of the diagnostic brief) without materially slowing down each
    # of the up-to-100 epochs on a CPU-only laptop.
    full_val_metrics_every: int = 5
    # ROUND-2 AUDIT ADDITION: literal waveform-domain L1 loss, exactly as
    # specified in the round-2 brief: L_waveform = mean(|enhanced - clean|).
    # This is a DIFFERENT (and more standard/direct) mechanism than
    # amplitude_consistency_loss above: the dB-scale term only penalises
    # whole-segment RMS mismatch (level), while this L1 term penalises the
    # full waveform shape (level + phase/timing together), so it also
    # constrains scale, is measurable in an interpretable linear-amplitude
    # unit, and is exactly the reconstruction-error quantity calculate_snr()
    # reports at evaluation time - optimising it directly targets the metric
    # this whole audit is about.
    #
    # Weight justification (measured on the Experiment-1 checkpoint against
    # the 7 saved example triples - see measure_wave_loss.py output in the
    # round-2 report): mean SI-SNR loss = -8.23, mean waveform L1 loss =
    # 0.0494. Matching magnitudes exactly would need a weight of ~167
    # (0.0494 * 167 ~= 8.2). We deliberately start lower, at 50, for two
    # reasons: (1) that measurement is from only 7 samples and should not be
    # treated as precise; (2) grad_norm clipping (clip_grad_norm_ to 5.0,
    # unchanged from Experiment 1) rescales the combined gradient's
    # magnitude every step regardless of weight, but does NOT fix
    # directional domination - an under-weighted term still gets its
    # direction drowned out even after clipping. 50 gives the waveform term
    # real influence on gradient direction without being the sole driver.
    # Watch `train/val_rms_ratio` (still logged every epoch) after the short
    # training test: increase this weight toward ~150-200 if RMS ratio has
    # not clearly moved toward 1.0; decrease it if SI-SNR/STOI/PESQ regress
    # once RMS ratio looks corrected.
    waveform_loss_weight: float = 50.0
    # ROUND-2 AUDIT ADDITION: per-target-SNR multiplier used to build a
    # WeightedRandomSampler for the TRAINING split only (see make_loaders).
    # Values are relative, not probabilities - they are renormalised
    # automatically. Defaults give the three hardest conditions (-5/0/+5 dB)
    # roughly 2-3x the sampling frequency of an unweighted epoch, while
    # still drawing from +10/+15/+20 dB every epoch for generalisation, per
    # the brief's explicit instruction not to remove the easier conditions.
    # Deliberately NOT applied to validation/test loaders (see make_loaders)
    # so those splits stay representative of the true dataset distribution.
    snr_sampling_weights: Optional[Dict[float, float]] = None  # populated in __post_init__
    use_snr_weighted_sampling: bool = True
    random_seed: int = 42
    patience: int = 15
    # EXPERIMENT 3: isolated output directory — must NOT point to dccrn_outputs
    # (Experiment 1) or models/Experiment_02_SNR_Aware (Experiment 2).
    output_dir: Path = _EXP3_ROOT
    # mask_bound: tanh saturation value for the CRM components.
    # Real and imaginary parts of M are independently bounded to (-mask_bound, +mask_bound).
    # mask_bound=2.0 → (1+M) real part in (-1, 3), giving the model enough
    # range to suppress noise-dominated bins without catastrophic amplification.
    mask_bound: float = 2.0

    def __post_init__(self) -> None:
        if self.snr_sampling_weights is None:
            self.snr_sampling_weights = {
                -5.0: 3.0, 0.0: 2.5, 5.0: 2.0, 10.0: 1.0, 15.0: 1.0, 20.0: 1.0,
            }

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

# ---------------------------------------------------------------------------
# Experiment 3 CRM helpers
# ---------------------------------------------------------------------------

def _zero_init_final_layer(model: TinyDCCRN) -> None:
    """Zero-initialise the final ComplexConvTranspose2d weights and biases.

    At initialisation this makes bounded(decoder_output) = tanh(0) * mask_bound = 0,
    so enhanced_spec = (1 + 0) * noisy_spec = noisy_spec — exact identity.
    Training then pulls the mask away from 0 only where doing so actually
    reduces the combined loss; at high SNR the identity is already a good
    answer, so the mask stays small.

    The final layer is model.decoder[-1] (the last ComplexConvTranspose2d).
    """
    final: ComplexConvTranspose2d = model.decoder[-1]  # type: ignore[assignment]
    nn.init.zeros_(final.real.weight)
    nn.init.zeros_(final.real.bias)
    nn.init.zeros_(final.imag.weight)
    nn.init.zeros_(final.imag.bias)


def apply_crm(
    model_output: torch.Tensor,
    noisy_spec: torch.Tensor,
    mask_bound: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply a bounded Complex Ratio Mask (CRM) to the noisy spectrogram.

    Enhanced spectrum formula:
        M_r = tanh(output_r / mask_bound) * mask_bound
        M_i = tanh(output_i / mask_bound) * mask_bound
        S_enh = (1 + M) * S_noisy   (complex multiplication)

    At zero initialisation:
        M = 0  →  S_enh = S_noisy   (exact identity)

    Bounding:
        |M_r|, |M_i| < mask_bound   (default 2.0)
        → (1 + M_r) ∈ (-1, 3)  — can suppress a bin to near-zero or
          amplify by up to 3× without saturation

    Returns:
        enhanced_spec : torch.complex tensor  [B, F, T]
        mask          : torch.complex tensor  [B, F, T]  (for statistics)
    """
    # Bound each CRM component independently with tanh.
    mask_r = torch.tanh(model_output[:, 0] / mask_bound) * mask_bound
    mask_i = torch.tanh(model_output[:, 1] / mask_bound) * mask_bound

    noisy_r = noisy_spec.real
    noisy_i = noisy_spec.imag

    # Complex multiply: (1 + M) * S_noisy
    #   real part = (1 + M_r)*S_r  -  M_i*S_i
    #   imag part = (1 + M_r)*S_i  +  M_i*S_r
    enh_r = (1.0 + mask_r) * noisy_r - mask_i * noisy_i
    enh_i = (1.0 + mask_r) * noisy_i + mask_i * noisy_r

    return torch.complex(enh_r, enh_i), torch.complex(mask_r, mask_i)


def build_model(config: Config, device: torch.device) -> TinyDCCRN:
    """Construct a freshly initialised TinyDCCRN from the current configuration.

    EXPERIMENT 3: the final layer is ZERO-INITIALISED so that at t=0 the
    CRM is identically zero and enhanced == noisy (identity solution).
    Call this function at every new training stage; it discards any previous
    optimizer/gradient state.
    """
    example_spec = stft(torch.zeros(1, config.segment_samples), config)
    model = TinyDCCRN(
        config.model_channels,
        config.gru_hidden_size,
        config.gru_layers,
        (example_spec.shape[-2], example_spec.shape[-1]),
    ).to(device)
    # Zero-init the final decoder layer → identity solution at t=0.
    _zero_init_final_layer(model)
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
    """SI-SNR loss with non-negative projection.

    Lower is better. Negative SI-SNR is returned for optimisation.
    The non-negative projection prevents a sign-inverted estimate
    from receiving the same SI-SNR score as a correctly signed estimate.
    """
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    dot = (estimate * target).sum(-1, keepdim=True)
    target_energy = target.pow(2).sum(-1, keepdim=True) + 1e-8

    alpha = dot / target_energy

    # Prevent sign-inverted estimates from being rewarded by SI-SNR.
    alpha = torch.clamp(alpha, min=0.0)

    projection = alpha * target

    noise = estimate - projection

    score = 10 * torch.log10(
        (projection.pow(2).sum(-1) + 1e-8)
        / (noise.pow(2).sum(-1) + 1e-8)
    )

    return -score.mean()


def compute_si_snr_metric(estimate: Tensor, target: Tensor) -> Tensor:
    """Return SI-SNR in dB; unlike si_snr_loss, higher values are better."""
    return -si_snr_loss(estimate, target)


def amplitude_consistency_loss(estimate: Tensor, target: Tensor) -> Tensor:
    """Penalise global energy/level mismatch between estimate and target, in dB.

    This is deliberately NOT a waveform-shape-matching loss (it ignores
    phase/temporal alignment entirely, only comparing whole-segment power),
    and it is computed against the CLEAN target, not the noisy mixture, so
    it cannot push the model toward copying noise back in. Its only job is
    to restore the output-level gradient that si_snr_loss structurally
    cannot provide (SI-SNR is invariant to a constant scalar multiplied
    onto the estimate; see si_snr_loss above). Lower is better; 0 means the
    enhanced and clean segments have identical RMS.
    """
    est_power = estimate.pow(2).mean(dim=-1) + 1e-8
    tgt_power = target.pow(2).mean(dim=-1) + 1e-8
    est_db = 10.0 * torch.log10(est_power)
    tgt_db = 10.0 * torch.log10(tgt_power)
    return (est_db - tgt_db).abs().mean()


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


def waveform_l1_loss(estimate: Tensor, target: Tensor) -> Tensor:
    """L_waveform = mean(|estimate - target|), as specified in the round-2 audit brief.

    Unlike si_snr_loss (scale-invariant) and unlike amplitude_consistency_loss
    (level-only, ignores shape/phase), this penalises the raw time-domain
    waveform difference directly - the same quantity calculate_snr() reports
    at evaluation time, just as an L1 distance instead of a log-power ratio.
    Computed against the CLEAN target only, so it cannot encourage the model
    to reproduce noise from the mixture.
    """
    return nn.functional.l1_loss(estimate, target)


def losses(
    enhanced: Tensor,
    clean_wave: Tensor,
    predicted: Tensor,
    clean_spec: Tensor,
    config: Config,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    si = si_snr_loss(enhanced, clean_wave)
    spectral = nn.functional.l1_loss(predicted.real, clean_spec.real) + nn.functional.l1_loss(
        predicted.imag, clean_spec.imag
    )
    amplitude = amplitude_consistency_loss(enhanced, clean_wave)
    waveform = waveform_l1_loss(enhanced, clean_wave)
    total = (
        config.si_snr_weight * si
        + config.spectral_loss_weight * spectral
        + config.amplitude_loss_weight * amplitude
        + config.waveform_loss_weight * waveform
    )
    return si, spectral, amplitude, waveform, total


def finite(tensors: Iterable[Tensor]) -> bool:
    return all(bool(torch.isfinite(tensor).all()) for tensor in tensors)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def make_loaders(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/validation/test DataLoaders.

    DATA LEAKAGE AUDIT (verified safe):
      Train/Validation clean_filename overlaps : 0
      Train/Test       clean_filename overlaps : 0
      Validation/Test  clean_filename overlaps : 0
    All splits use distinct clean utterances — no speaker/utterance leakage.

    WEIGHTED SAMPLING (training split only):
      If config.use_snr_weighted_sampling is True, a WeightedRandomSampler
      oversamples the hardest conditions (-5/0/+5 dB) while keeping the easier
      conditions in every epoch for generalisation. Validation and test loaders
      are always unweighted and representative of the true data distribution.
    """
    common = dict(
        batch_size=config.batch_size, num_workers=0, pin_memory=torch.cuda.is_available()
    )

    train_dataset = SpeechDataset(config, "train", True)

    if config.use_snr_weighted_sampling and config.snr_sampling_weights:
        weights_map = config.snr_sampling_weights  # e.g. {-5.0: 3.0, 0.0: 2.5, ...}
        sample_weights: List[float] = []
        for row in train_dataset.rows:
            snr_key = float(row["target_snr_db"])
            sample_weights.append(weights_map.get(snr_key, 1.0))
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        # Print the expected sampling distribution for one epoch.
        total_w = sum(sample_weights)
        snr_totals: Dict[float, float] = {}
        for row, w in zip(train_dataset.rows, sample_weights):
            k = float(row["target_snr_db"])
            snr_totals[k] = snr_totals.get(k, 0.0) + w
        print("\nTraining sampling distribution (weighted):")
        for snr_level in sorted(snr_totals):
            pct = 100.0 * snr_totals[snr_level] / total_w
            print(f"  {snr_level:+.0f} dB : {pct:.1f}%")
        train_loader = DataLoader(train_dataset, sampler=sampler, **common)
    else:
        print("\nTraining sampling: uniform (unweighted).")
        train_loader = DataLoader(train_dataset, shuffle=True, **common)

    return (
        train_loader,
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

        stage = "CRM / iSTFT"
        # EXPERIMENT 3: apply the bounded CRM before iSTFT.
        enhanced_spec, mask_sanity = apply_crm(output, noisy_spec, config.mask_bound)
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
        passed.append("CRM / iSTFT")

        stage = "identity at initialisation"
        # The final layer was zero-initialised → mask == 0 → enhanced ≈ noisy.
        # Verify with a very tight tolerance (should be numerically near-exact).
        identity_mse = float(torch.mean((enhanced - noisy.to(enhanced.device)) ** 2).detach())
        mask_mag_init = float(mask_sanity.abs().mean().detach())
        mask_max_init = float(mask_sanity.abs().max().detach())
        print(f"  Identity MSE (zero-init): {identity_mse:.2e}  (expect < 1e-10)")
        print(f"  Mean |M| at init        : {mask_mag_init:.2e}  (expect < 1e-6)")
        print(f"  Max  |M| at init        : {mask_max_init:.2e}  (expect < 1e-6)")
        if identity_mse > 1e-6:
            raise RuntimeError(
                f"Zero-init identity check failed: MSE={identity_mse:.2e} > 1e-6. "
                "Check that _zero_init_final_layer() was called after build_model()."
            )
        passed.append("Identity at initialisation")

        stage = "mask bound"
        # After one forward pass the mask should be bounded by config.mask_bound.
        # At init it will be near-zero; the bound check is structural.
        mask_r_max = float(torch.abs(mask_sanity.real).max())
        mask_i_max = float(torch.abs(mask_sanity.imag).max())
        print(f"  Max |M_r| : {mask_r_max:.4f}  (must be < {config.mask_bound})")
        print(f"  Max |M_i| : {mask_i_max:.4f}  (must be < {config.mask_bound})")
        if mask_r_max >= config.mask_bound or mask_i_max >= config.mask_bound:
            raise RuntimeError(
                f"Mask bound violated: max real={mask_r_max:.4f}, "
                f"max imag={mask_i_max:.4f} must be < {config.mask_bound}."
            )
        passed.append("Mask bound")

        # ROUND-2 AUDIT ADDITION (Section 11/12): STFT/iSTFT-only round-trip
        # check, with no model involved, confirming the transform itself is
        # not introducing amplitude error before the model is even blamed.
        stage = "STFT/iSTFT round-trip (no model)"
        clean_roundtrip = istft(stft(clean, config), config, clean.shape[-1])
        roundtrip_mse = float(torch.mean((clean_roundtrip - clean) ** 2))
        roundtrip_snr = calculate_snr(clean.cpu().numpy(), clean_roundtrip.cpu().numpy())
        print(f"  STFT/iSTFT round-trip MSE (clean) : {roundtrip_mse:.10f}")
        print(f"  STFT/iSTFT round-trip SNR (clean) : {roundtrip_snr:+.2f} dB  (expect >60 dB)")
        if roundtrip_snr < 60.0:
            raise RuntimeError(
                f"STFT/iSTFT round-trip SNR too low ({roundtrip_snr:.2f} dB) - "
                "check n_fft/hop_length/win_length/window/center configuration."
            )
        passed.append("STFT/iSTFT round-trip")

        stage = "loss calculation"
        si, spectral, amplitude, waveform, total = losses(
            enhanced, clean, enhanced_spec, clean_spec, config
        )
        # SI-SNR loss is the negative SI-SNR scalar; do NOT label it as dB.
        print(f"  SI-SNR Loss          : {si.item():.6f}  (optimizer objective, lower is better)")
        print(f"  Spectral L1 Loss     : {spectral.item():.6f}")
        print(f"  Amplitude (dB) Loss  : {amplitude.item():.6f}  (0 = matched RMS with clean)")
        print(f"  Waveform L1 Loss     : {waveform.item():.6f}  (mean|enhanced-clean|, 0 = perfect)")
        print(f"  Total Loss           : {total.item():.6f}")
        passed.append("Loss calculation")

        # ROUND-2 AUDIT ADDITION (Section 12): amplitude/SNR diagnostics the
        # brief explicitly asks the sanity check to cover, computed on this
        # same batch, before any training has happened.
        stage = "amplitude and SNR diagnostics"
        noisy_np, clean_np, enh_np = noisy.detach().cpu().numpy(), clean.detach().cpu().numpy(), enhanced.detach().cpu().numpy()

        def _rms(x): return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
        def _peak(x): return float(np.max(np.abs(x.astype(np.float64))))

        clip_pct = float((np.abs(enh_np) > 1.0).mean() * 100.0)
        input_snr = calculate_snr(clean_np, noisy_np)
        output_snr = calculate_snr(clean_np, enh_np)
        print(f"  Clean  RMS / Peak    : {_rms(clean_np):.6f} / {_peak(clean_np):.6f}")
        print(f"  Noisy  RMS / Peak    : {_rms(noisy_np):.6f} / {_peak(noisy_np):.6f}")
        print(f"  Enhanced RMS / Peak  : {_rms(enh_np):.6f} / {_peak(enh_np):.6f}"
              f"  (untrained model - not expected to be meaningful yet)")
        print(f"  RMS ratio (enh/clean): {_rms(enh_np) / (_rms(clean_np) + 1e-12):.4f}")
        print(f"  Clipping (enhanced)  : {clip_pct:.3f} %")
        print(f"  Input  SNR (waveform, clean vs noisy)    : {input_snr:+.2f} dB")
        print(f"  Output SNR (waveform, clean vs enhanced) : {output_snr:+.2f} dB"
              f"  (untrained - not meaningful yet)")
        if not math.isfinite(input_snr) or not math.isfinite(output_snr):
            raise RuntimeError("Non-finite SNR computed during sanity check.")
        passed.append("Amplitude/SNR diagnostics")

        stage = "NaN/Inf check"
        if not finite((noisy, clean, noisy_spec.real, output, enhanced, si, spectral, amplitude, waveform, total)):
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
        "si_snr_loss": [], "spectral_loss": [], "amplitude_loss": [],
        "waveform_loss": [], "total_loss": [],
        # Fast, always-on diagnostics (no external deps) so an amplitude/scale
        # regression shows up every epoch.
        "rms_ratio": [], "reconstruction_snr_db": [], "clipping_pct": [],
        # EXPERIMENT 3: mean |M| per batch — key indicator of whether the model
        # is learning input-dependent enhancement strength.
        "mask_magnitude_mean": [],
    }
    for noisy, clean, _ in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        noisy_spec, clean_spec = stft(noisy, config), stft(clean, config)
        model_input = torch.stack((noisy_spec.real, noisy_spec.imag), dim=1)
        raw_output = model(model_input)
        # EXPERIMENT 3: apply CRM — S_enh = (1 + M) * S_noisy.
        # mask is returned for statistics only; it does not enter the loss.
        enhanced_spec, mask = apply_crm(raw_output, noisy_spec, config.mask_bound)
        enhanced = istft(enhanced_spec, config, noisy.shape[-1])
        # FIX (Bug 1): losses() returns 5 values.  Previously only 4 were
        # unpacked, causing `total` to receive the raw waveform tensor instead
        # of the computed weighted total — the optimizer was effectively
        # minimising waveform_l1_loss only, ignoring all other terms.
        si, spectral, amplitude, waveform, total = losses(
            enhanced, clean,
            enhanced_spec,
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
        totals["amplitude_loss"].append(float(amplitude.detach().cpu()))
        totals["waveform_loss"].append(float(waveform.detach().cpu()))
        totals["total_loss"].append(float(total.detach().cpu()))
        # Mask magnitude — detached, no grad needed.
        with torch.no_grad():
            mask_mag = mask.abs().mean().item()
        totals["mask_magnitude_mean"].append(mask_mag)

        with torch.no_grad():
            enhanced_np = enhanced.detach().cpu().numpy()
            clean_np = clean.detach().cpu().numpy()
            clean_rms = np.sqrt(np.mean(clean_np ** 2, axis=-1)) + 1e-12
            enh_rms = np.sqrt(np.mean(enhanced_np ** 2, axis=-1))
            totals["rms_ratio"].extend((enh_rms / clean_rms).tolist())
            noise_err = enhanced_np - clean_np
            recon_snr = 10.0 * np.log10(
                (np.mean(clean_np ** 2, axis=-1) + 1e-12)
                / (np.mean(noise_err ** 2, axis=-1) + 1e-12)
            )
            totals["reconstruction_snr_db"].extend(recon_snr.tolist())
            clip_frac = (np.abs(enhanced_np) > 1.0).mean(axis=-1) * 100.0
            totals["clipping_pct"].extend(clip_frac.tolist())
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
            best_path = config.output_dir / "checkpoints" / "Experiment_03_best.pth"
            _assert_exp3_path(best_path)
            save_checkpoint(
                best_path,
                model, optimizer, scheduler,
                epoch, val_metrics["total_loss"], best, config,
            )
        else:
            stale += 1

        latest_path = config.output_dir / "checkpoints" / "Experiment_03_latest.pth"
        _assert_exp3_path(latest_path)
        save_checkpoint(
            latest_path,
            model, optimizer, scheduler,
            epoch, val_metrics["total_loss"],
            min(best, val_metrics["total_loss"]), config,
        )

        record: Dict[str, object] = {
            "epoch": epoch,
            "train_si_snr_loss": train_metrics["si_snr_loss"],
            "train_spectral_loss": train_metrics["spectral_loss"],
            "train_amplitude_loss_db": train_metrics["amplitude_loss"],
            "train_waveform_loss": train_metrics["waveform_loss"],
            "train_total_loss": train_metrics["total_loss"],
            "train_rms_ratio": train_metrics["rms_ratio"],
            "train_reconstruction_snr_db": train_metrics["reconstruction_snr_db"],
            "train_clipping_pct": train_metrics["clipping_pct"],
            # EXPERIMENT 3: mask magnitude — tracks how much the model modifies the input.
            "train_mask_magnitude_mean": train_metrics.get("mask_magnitude_mean", float("nan")),
            "val_si_snr_loss": val_metrics["si_snr_loss"],
            "val_spectral_loss": val_metrics["spectral_loss"],
            "val_amplitude_loss_db": val_metrics["amplitude_loss"],
            "val_waveform_loss": val_metrics["waveform_loss"],
            "val_total_loss": val_metrics["total_loss"],
            "val_rms_ratio": val_metrics["rms_ratio"],
            "val_reconstruction_snr_db": val_metrics["reconstruction_snr_db"],
            "val_clipping_pct": val_metrics["clipping_pct"],
            "val_mask_magnitude_mean": val_metrics.get("mask_magnitude_mean", float("nan")),
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
        print(f"    SI-SNR Loss        : {train_metrics['si_snr_loss']:.6f}"
              "  (optimizer objective, lower=better)")
        print(f"    Spectral L1        : {train_metrics['spectral_loss']:.6f}")
        print(f"    Amplitude Loss (dB): {train_metrics['amplitude_loss']:.6f}")
        print(f"    Waveform L1 Loss   : {train_metrics['waveform_loss']:.6f}")
        print(f"    Total Loss         : {train_metrics['total_loss']:.6f}")
        print(f"    RMS ratio (enh/clean): {train_metrics['rms_ratio']:.4f}"
              "  (target ~1.0)")
        print("  VALIDATION:")
        print(f"    SI-SNR Loss        : {val_metrics['si_snr_loss']:.6f}"
              "  (optimizer objective, lower=better)")
        print(f"    Spectral L1        : {val_metrics['spectral_loss']:.6f}")
        print(f"    Amplitude Loss (dB): {val_metrics['amplitude_loss']:.6f}")
        print(f"    Waveform L1 Loss   : {val_metrics['waveform_loss']:.6f}")
        print(f"    Total Loss         : {val_metrics['total_loss']:.6f}")
        print(f"    RMS ratio (enh/clean): {val_metrics['rms_ratio']:.4f}"
              "  (target ~1.0)")
        print(f"    Reconstruction SNR   : {val_metrics['reconstruction_snr_db']:+.2f} dB"
              "  (waveform output_snr, NOT input/environmental SNR)")
        print(f"    Clipping             : {val_metrics['clipping_pct']:.3f} %")
        print(f"  Learning Rate  : {learning_rate:.2e}")
        print(f"  Epoch Time     : {epoch_time:.1f}s"
              f"  (cumulative: {cumulative_time:.1f}s)")
        print(f"  Best Val Loss  : {best:.6f}  (epoch {best_epoch})")
        print(f"  Patience       : {stale}/{config.patience}")

        if epoch == 1 or epoch == epochs or epoch % config.full_val_metrics_every == 0:
            full_val = full_validation_diagnostics(model, val_loader, config, device)
            record.update(full_val)
            print("  Full Validation Diagnostics (STOI/PESQ, periodic):")
            print(
                f"    SI-SNR before/after : {full_val['val_si_snr_before_db_full']:.2f} / "
                f"{full_val['val_si_snr_after_db_full']:.2f} dB "
                f"(improvement {full_val['val_si_snr_improvement_db_full']:+.2f} dB)"
            )
            print(
                f"    Reconstruction SNR   : {full_val['val_reconstruction_snr_db_full']:+.2f} dB"
            )
            print(
                f"    STOI before/after    : {full_val['val_stoi_before']:.4f} / "
                f"{full_val['val_stoi_after']:.4f}"
            )
            print(
                f"    PESQ before/after    : {full_val['val_pesq_before']:.4f} / "
                f"{full_val['val_pesq_after']:.4f}"
            )

        if stale >= config.patience:
            print(f"\n  Early stopping triggered (patience={config.patience}).")
            break

    # Persist history to logs/ subdirectory.
    history_json = config.output_dir / "logs" / "training_history.json"
    history_csv = config.output_dir / "logs" / "training_history.csv"
    _assert_exp3_path(history_json)
    _assert_exp3_path(history_csv)
    history_json.parent.mkdir(parents=True, exist_ok=True)
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
    """Run inference through the CRM-based model and return enhanced waveform.

    EXPERIMENT 3: uses apply_crm() — S_enh = (1 + M) * S_noisy.
    This is the same path used during training.
    """
    with torch.no_grad():
        noisy_spec = stft(waveform.to(device), config)
        model_input = torch.stack((noisy_spec.real, noisy_spec.imag), dim=1)
        raw_output = model(model_input)
        enhanced_spec, _ = apply_crm(raw_output, noisy_spec, config.mask_bound)
        return istft(enhanced_spec, config, waveform.shape[-1]).cpu()


def enhance_waveform_with_mask(
    model: nn.Module, waveform: Tensor, config: Config, device: torch.device
) -> Tuple[Tensor, np.ndarray]:
    """Like enhance_waveform() but also returns the mask magnitude array.

    Returns:
        enhanced  : float32 Tensor  [1, T]
        mask_mag  : float32 ndarray [F, T]  — |M| for statistics reporting
    """
    with torch.no_grad():
        noisy_spec = stft(waveform.to(device), config)
        model_input = torch.stack((noisy_spec.real, noisy_spec.imag), dim=1)
        raw_output = model(model_input)
        enhanced_spec, mask = apply_crm(raw_output, noisy_spec, config.mask_bound)
        enhanced = istft(enhanced_spec, config, waveform.shape[-1]).cpu()
        # mask: [B, F, T] complex; take magnitude, squeeze batch dim
        mask_mag = mask[0].abs().cpu().numpy()  # [F, T]
    return enhanced, mask_mag


def full_validation_diagnostics(
    model: nn.Module, loader: DataLoader, config: Config, device: torch.device
) -> Dict[str, float]:
    """Perceptual/quality diagnostics on the validation split (Step 5).

    Distinct from run_epoch()'s per-batch tensor diagnostics: this adds
    SI-SNR-before/after, STOI, and PESQ on validation audio, labelled with
    the same "reconstruction SNR is not input SNR" convention used
    everywhere else in this file. Called periodically (see
    Config.full_val_metrics_every), not every epoch, to bound runtime.

    STOI/PESQ are only computed for segments that pass the
    _check_perceptual_metrics_validity() guard (sufficient duration, non-silent
    reference and estimate). Skipped samples are counted by reason and
    excluded from the aggregate mean so the reported metric is not
    contaminated by 1e-5 fallback values from pystoi.
    """
    model.eval()
    si_before: List[float] = []
    si_after: List[float] = []
    recon_snr: List[float] = []
    stoi_before: List[float] = []
    stoi_after: List[float] = []
    pesq_before: List[float] = []
    pesq_after: List[float] = []
    stoi_ok = True
    pesq_ok = True
    # Validity tracking (per-sample, then aggregated at the end).
    stoi_n_total: int = 0
    stoi_skip_counts: Dict[str, int] = {}
    pesq_n_total: int = 0
    pesq_skip_counts: Dict[str, int] = {}
    with torch.no_grad():
        for noisy, clean, _ in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            enhanced = enhance_waveform(model, noisy, config, device)
            for b in range(noisy.shape[0]):
                noisy_np = noisy[b].detach().cpu().numpy()
                clean_np = clean[b].detach().cpu().numpy()
                enhanced_np = enhanced[b].numpy()
                si_before.append(
                    float(compute_si_snr_metric(noisy[b : b + 1], clean[b : b + 1]).item())
                )
                si_after.append(
                    float(
                        compute_si_snr_metric(
                            enhanced[b : b + 1], clean[b : b + 1].detach().cpu()
                        ).item()
                    )
                )
                recon_snr.append(calculate_snr(clean_np, enhanced_np))

                # --- STOI ---
                if stoi_ok:
                    stoi_n_total += 1
                    # Check validity before calling pystoi.  An invalid pair
                    # (too short, silent, length mismatch) is skipped rather
                    # than scored; pystoi returns 1e-5 for such inputs which
                    # would silently bias the aggregate mean.
                    stoi_reason = _check_perceptual_metrics_validity(
                        clean_np, noisy_np, config.sample_rate
                    )
                    enh_stoi_reason = _check_perceptual_metrics_validity(
                        clean_np, enhanced_np, config.sample_rate
                    )
                    combined_reason = stoi_reason or enh_stoi_reason
                    if combined_reason:
                        stoi_skip_counts[combined_reason] = (
                            stoi_skip_counts.get(combined_reason, 0) + 1
                        )
                    else:
                        try:
                            stoi_before.append(
                                calculate_stoi(noisy_np, clean_np, config.sample_rate)
                            )
                            stoi_after.append(
                                calculate_stoi(enhanced_np, clean_np, config.sample_rate)
                            )
                        except RuntimeError:
                            stoi_ok = False

                # --- PESQ ---
                if pesq_ok:
                    pesq_n_total += 1
                    pesq_reason = _check_perceptual_metrics_validity(
                        clean_np, noisy_np, config.sample_rate
                    )
                    enh_pesq_reason = _check_perceptual_metrics_validity(
                        clean_np, enhanced_np, config.sample_rate
                    )
                    combined_pesq_reason = pesq_reason or enh_pesq_reason
                    if combined_pesq_reason:
                        pesq_skip_counts[combined_pesq_reason] = (
                            pesq_skip_counts.get(combined_pesq_reason, 0) + 1
                        )
                    else:
                        try:
                            pesq_before.append(
                                calculate_pesq(noisy_np, clean_np, config.sample_rate)
                            )
                            pesq_after.append(
                                calculate_pesq(enhanced_np, clean_np, config.sample_rate)
                            )
                        except RuntimeError:
                            pesq_ok = False

    def m(values: List[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    # Report validity summary to console for monitoring during training.
    stoi_n_valid = len(stoi_before)
    stoi_n_skipped = sum(stoi_skip_counts.values())
    if stoi_n_total > 0:
        skip_detail = ", ".join(
            f"{r}={c}" for r, c in sorted(stoi_skip_counts.items())
        )
        skip_str = f" [{skip_detail}]" if skip_detail else ""
        print(
            f"    STOI valid: {stoi_n_valid}/{stoi_n_total}"
            f"  skipped: {stoi_n_skipped}/{stoi_n_total}{skip_str}"
        )
    pesq_n_valid = len(pesq_before)
    pesq_n_skipped = sum(pesq_skip_counts.values())
    if pesq_n_total > 0:
        pesq_skip_detail = ", ".join(
            f"{r}={c}" for r, c in sorted(pesq_skip_counts.items())
        )
        pesq_skip_str = f" [{pesq_skip_detail}]" if pesq_skip_detail else ""
        print(
            f"    PESQ valid: {pesq_n_valid}/{pesq_n_total}"
            f"  skipped: {pesq_n_skipped}/{pesq_n_total}{pesq_skip_str}"
        )

    return {
        "val_si_snr_before_db_full": m(si_before),
        "val_si_snr_after_db_full": m(si_after),
        "val_si_snr_improvement_db_full": m(si_after) - m(si_before)
        if si_before and si_after
        else float("nan"),
        "val_reconstruction_snr_db_full": m(recon_snr),
        "val_stoi_before": m(stoi_before),
        "val_stoi_after": m(stoi_after),
        "val_stoi_improvement": m(stoi_after) - m(stoi_before) if stoi_ok and stoi_before else float("nan"),
        "val_pesq_before": m(pesq_before),
        "val_pesq_after": m(pesq_after),
        "val_pesq_improvement": m(pesq_after) - m(pesq_before) if pesq_ok and pesq_before else float("nan"),
        # Validity counts for downstream logging.
        "val_stoi_n_valid": float(stoi_n_valid),
        "val_stoi_n_skipped": float(stoi_n_skipped),
        "val_pesq_n_valid": float(pesq_n_valid),
        "val_pesq_n_skipped": float(pesq_n_skipped),
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Perceptual-metric (STOI / PESQ) validity constants
# ---------------------------------------------------------------------------

# Safety guard: absolute minimum array length.
# pystoi's analysis window is ~256 ms; 4000 samples (0.25 s) is the hard
# floor.  In practice every evaluation segment is padded to 32 000 samples
# (2 s), so this check almost never fires — it exists to catch edge cases.
_MIN_SAMPLES_FOR_STOI: int = 4000

# Global-RMS guards.  If the whole-segment RMS is below these thresholds the
# signal is silence or a collapsed model output and STOI/PESQ are undefined.
_MIN_CLEAN_RMS: float = 1e-4
_MIN_ESTIMATE_RMS: float = 1e-6

# Active-frame-ratio guard (the primary check for zero-padded 2-second
# segments).  pystoi works by splitting the signal into short-time frames,
# removing frames it classifies as silent, then computing intelligibility
# from the remaining frames.  When a 2-second segment contains only, say,
# 0.3 s of real speech followed by 1.7 s of zero-padding, most frames are
# removed and pystoi emits "Not enough STFT frames … Returning 1e-5".
#
# This check replicates that frame-activity logic BEFORE calling pystoi so
# that such segments are skipped rather than scored with a spurious 1e-5.
#
# Frame length: 256 samples = 16 ms at 16 kHz (standard short-time energy
# analysis window; shorter than pystoi's own 256 ms window so the check
# is conservative — we only skip when the vast majority of frames are silent).
_ACTIVITY_FRAME_LEN: int = 256          # 16 ms at 16 kHz
_ACTIVITY_RMS_THRESHOLD: float = 1e-4   # same scale as _MIN_CLEAN_RMS
_MIN_ACTIVE_FRAME_RATIO: float = 0.10   # at least 10 % of frames must be active


def _check_perceptual_metrics_validity(
    clean: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int,
) -> Optional[str]:
    """Return None when (clean, estimate) are suitable for STOI/PESQ.

    Return a short reason string when the pair should be SKIPPED:
      - 'too_short'             : fewer than _MIN_SAMPLES_FOR_STOI samples
      - 'length_mismatch'       : clean and estimate have different lengths
      - 'silent_clean'          : whole-segment RMS < _MIN_CLEAN_RMS
      - 'silent_estimate'       : whole-segment estimate RMS < _MIN_ESTIMATE_RMS
      - 'insufficient_activity' : < _MIN_ACTIVE_FRAME_RATIO of 16 ms frames
                                  in the clean reference exceed the activity
                                  threshold — the segment is predominantly
                                  zero-padded and pystoi would return 1e-5

    The 'insufficient_activity' check is the primary defence against the
    pystoi warning "Not enough STFT frames to compute intermediate
    intelligibility measure after removing silent frames. Returning 1e-5."
    A 2-second segment is 32,000 samples and always passes 'too_short', but
    can still be overwhelmingly zero-padded if the source utterance was short.
    """
    # 1. Absolute length guard (safety net for unexpected short arrays).
    if len(clean) < _MIN_SAMPLES_FOR_STOI or len(estimate) < _MIN_SAMPLES_FOR_STOI:
        return "too_short"

    # 2. Arrays must be the same length for aligned comparison.
    if len(clean) != len(estimate):
        return "length_mismatch"

    clean_f = clean.astype(np.float64)

    # 3. Global clean RMS — catches completely silent references.
    clean_rms = float(np.sqrt(np.mean(clean_f ** 2)))
    if clean_rms < _MIN_CLEAN_RMS:
        return "silent_clean"

    # 4. Global estimate RMS — catches fully collapsed model outputs.
    est_rms = float(np.sqrt(np.mean(estimate.astype(np.float64) ** 2)))
    if est_rms < _MIN_ESTIMATE_RMS:
        return "silent_estimate"

    # 5. Frame-level activity check (primary guard for zero-padded segments).
    #    Split the clean reference into non-overlapping 16 ms frames and count
    #    how many are energetic.  pystoi does the same internally and fails
    #    when too few active frames remain; catching it here avoids 1e-5 scores.
    n_frames = len(clean_f) // _ACTIVITY_FRAME_LEN
    if n_frames == 0:
        return "too_short"
    frames = clean_f[: n_frames * _ACTIVITY_FRAME_LEN].reshape(n_frames, _ACTIVITY_FRAME_LEN)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    active_ratio = float(np.sum(frame_rms > _ACTIVITY_RMS_THRESHOLD) / n_frames)
    if active_ratio < _MIN_ACTIVE_FRAME_RATIO:
        return "insufficient_activity"

    return None  # valid


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

def calculate_waveform_snr(clean: np.ndarray, estimate: np.ndarray) -> float:
    """Waveform-based SNR: 10*log10(clean_power / error_power).

    Both input and output SNR use this SAME formula so the improvement
    (output_snr - input_snr) is mathematically meaningful.

      INPUT_SNR  = calculate_waveform_snr(clean, noisy)
      OUTPUT_SNR = calculate_waveform_snr(clean, enhanced)
      IMPROVEMENT = OUTPUT_SNR - INPUT_SNR

    The dataset metadata field 'measured_snr_db' (target_snr_db) is reported
    separately as 'target_snr_db' and 'metadata_measured_snr_db' for reference
    only; it is NOT used in the improvement calculation.
    """
    clean_f = clean.astype(np.float64)
    estimate_f = estimate.astype(np.float64)
    error = estimate_f - clean_f
    return float(
        10.0 * np.log10(
            (np.mean(clean_f ** 2) + 1e-12) / (np.mean(error ** 2) + 1e-12)
        )
    )


def evaluate_test_set(
    model: nn.Module,
    dataset: SpeechDataset,
    config: Config,
    device: torch.device,
) -> List[Dict[str, object]]:
    """Evaluate the held-out test split once without gradients or model selection.

    BOTH input and output SNR are computed from waveforms using the same
    formula: 10*log10(clean_power / error_power).  The metadata
    'measured_snr_db' is reported as 'metadata_measured_snr_db' for reference
    but is NOT mixed with the waveform-based SNR improvement.
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

        # EXPERIMENT 3: use the mask-returning variant for statistics.
        enhanced_tensor, mask_mag = enhance_waveform_with_mask(
            model, noisy.unsqueeze(0), config, device
        )
        enhanced_tensor = enhanced_tensor.squeeze(0)
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

        # Waveform SNR — BOTH input and output use the SAME formula:
        #   SNR = 10*log10(clean_power / error_power)
        # This makes the improvement (output - input) mathematically valid.
        waveform_input_snr = calculate_waveform_snr(clean_np, noisy_np)
        waveform_output_snr = calculate_waveform_snr(clean_np, enhanced_np)
        waveform_snr_improvement = _safe_diff(waveform_output_snr, waveform_input_snr)

        # Dataset metadata SNR: reported for reference only.
        # NOT mixed into the waveform_snr_improvement calculation.
        try:
            metadata_measured_snr = float(metadata["measured_snr_db"])
        except (KeyError, ValueError):
            metadata_measured_snr = float("nan")

        try:
            target_snr = float(metadata["target_snr_db"])
        except (KeyError, ValueError):
            target_snr = float("nan")

        input_stoi = output_stoi = float("nan")
        stoi_skip_reason: Optional[str] = None
        if stoi_available:
            # Check both (clean, noisy) and (clean, enhanced) before calling
            # pystoi.  If either pair fails the validity check the sample is
            # skipped for STOI; its reason is stored and counted at the end.
            # This prevents pystoi's 1e-5 fallback value from entering the
            # aggregate mean when audio is too short or nearly silent.
            stoi_reason = _check_perceptual_metrics_validity(
                clean_np, noisy_np, config.sample_rate
            )
            enh_stoi_reason = _check_perceptual_metrics_validity(
                clean_np, enhanced_np, config.sample_rate
            )
            stoi_skip_reason = stoi_reason or enh_stoi_reason
            if stoi_skip_reason is None:
                try:
                    input_stoi = calculate_stoi(noisy_np, clean_np, config.sample_rate)
                    output_stoi = calculate_stoi(enhanced_np, clean_np, config.sample_rate)
                except RuntimeError as error:
                    stoi_available = False
                    stoi_error_msg = str(error)
                    print(f"  STOI unavailable: {error}")

        input_pesq = output_pesq = float("nan")
        pesq_skip_reason: Optional[str] = None
        if pesq_available:
            pesq_reason = _check_perceptual_metrics_validity(
                clean_np, noisy_np, config.sample_rate
            )
            enh_pesq_reason = _check_perceptual_metrics_validity(
                clean_np, enhanced_np, config.sample_rate
            )
            pesq_skip_reason = pesq_reason or enh_pesq_reason
            if pesq_skip_reason is None:
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
                # target_snr_db: nominal SNR level from dataset generation.
                "target_snr_db": target_snr,
                # metadata_measured_snr_db: mixture SNR recorded at generation time.
                # Reported for reference only; NOT used in snr_improvement calculation.
                "metadata_measured_snr_db": metadata_measured_snr,
                # waveform_input_snr_db: 10*log10(clean_power / (noisy-clean)^2 power).
                "waveform_input_snr_db": waveform_input_snr,
                # waveform_output_snr_db: 10*log10(clean_power / (enhanced-clean)^2 power).
                "waveform_output_snr_db": waveform_output_snr,
                # waveform_snr_improvement_db: waveform_output_snr - waveform_input_snr.
                # Both use the SAME formula — the improvement is meaningful.
                "waveform_snr_improvement_db": waveform_snr_improvement,
                # SI-SNR (scale-invariant, dB, higher=better)
                "input_si_snr_db": input_si,
                "output_si_snr_db": output_si,
                "si_snr_improvement_db": _safe_diff(output_si, input_si),
                "input_stoi": input_stoi,
                "output_stoi": output_stoi,
                "stoi_improvement": _safe_diff(output_stoi, input_stoi),
                # stoi_skip_reason: None = evaluated; string = why skipped.
                "stoi_skip_reason": stoi_skip_reason or "",
                "input_pesq": input_pesq,
                "output_pesq": output_pesq,
                "pesq_improvement": _safe_diff(output_pesq, input_pesq),
                # pesq_skip_reason: None = evaluated; string = why skipped.
                "pesq_skip_reason": pesq_skip_reason or "",
                "noise_category": metadata.get("noise_category", ""),
                "max_amplitude_enhanced": max_amp,
                "rms_enhanced": rms,
                # rms_ratio_enhanced_over_clean: target ~1.0.
                "rms_ratio_enhanced_over_clean": rms / (float(np.sqrt(np.mean(clean_np ** 2))) + 1e-12),
                "clipped_for_wav": clipped,
                # Internal waveform arrays (excluded from CSV by prefix '_').
                "_noisy": noisy_np,
                "_clean": clean_np,
                "_enhanced": enhanced_np,
                # EXPERIMENT 3: per-sample mask statistics.
                # mask_mag is [F, T] float32 array of |M| values.
                "mask_mean_magnitude":   float(np.mean(mask_mag)),
                "mask_median_magnitude": float(np.median(mask_mag)),
                "mask_p95_magnitude":    float(np.percentile(mask_mag, 95)),
                "mask_max_magnitude":    float(np.max(mask_mag)),
                # Store array for group-level analysis (excluded from CSV).
                "_mask_mag": mask_mag,
            }
        )

    # Aggregate STOI/PESQ validity summary over the full test set.
    # The mean STOI/PESQ printed later uses only samples where skip_reason == "".
    # This block reports the counts explicitly so the user can see exactly
    # how many samples were scored vs. skipped and why.
    n_total = len(results)
    stoi_skip_counts: Dict[str, int] = {}
    pesq_skip_counts: Dict[str, int] = {}
    for r in results:
        sr = str(r.get("stoi_skip_reason", ""))
        if sr:
            stoi_skip_counts[sr] = stoi_skip_counts.get(sr, 0) + 1
        pr = str(r.get("pesq_skip_reason", ""))
        if pr:
            pesq_skip_counts[pr] = pesq_skip_counts.get(pr, 0) + 1

    stoi_n_skipped = sum(stoi_skip_counts.values())
    stoi_n_valid = n_total - stoi_n_skipped
    pesq_n_skipped = sum(pesq_skip_counts.values())
    pesq_n_valid = n_total - pesq_n_skipped

    def _pct(n: int, total: int) -> str:
        return f"{100.0 * n / total:.2f}%" if total > 0 else "N/A"

    print(f"\n  STOI evaluation coverage:")
    print(f"    Valid  : {stoi_n_valid}/{n_total} ({_pct(stoi_n_valid, n_total)})")
    print(f"    Skipped: {stoi_n_skipped}/{n_total} ({_pct(stoi_n_skipped, n_total)})", end="")
    if stoi_skip_counts:
        detail = ", ".join(f"{r}={c}" for r, c in sorted(stoi_skip_counts.items()))
        print(f"  [{detail}]", end="")
    print()

    print(f"  PESQ evaluation coverage:")
    print(f"    Valid  : {pesq_n_valid}/{n_total} ({_pct(pesq_n_valid, n_total)})")
    print(f"    Skipped: {pesq_n_skipped}/{n_total} ({_pct(pesq_n_skipped, n_total)})", end="")
    if pesq_skip_counts:
        detail = ", ".join(f"{r}={c}" for r, c in sorted(pesq_skip_counts.items()))
        print(f"  [{detail}]", end="")
    print()

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
        "metadata_measured_snr_db",
        "waveform_input_snr_db", "waveform_output_snr_db", "waveform_snr_improvement_db",
        "input_si_snr_db", "output_si_snr_db", "si_snr_improvement_db",
        "input_stoi", "output_stoi", "stoi_improvement",
        "input_pesq", "output_pesq", "pesq_improvement",
        "rms_ratio_enhanced_over_clean",
        # EXPERIMENT 3: mask magnitude metrics.
        "mask_mean_magnitude", "mask_median_magnitude",
        "mask_p95_magnitude", "mask_max_magnitude",
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
# EXPERIMENT 3: Mask statistics analysis
# ---------------------------------------------------------------------------

def evaluate_mask_statistics(
    results: List[Dict[str, object]],
    output_path: Path,
) -> List[Dict[str, object]]:
    """Aggregate |M| statistics by target SNR group and write a CSV.

    Expected qualitative trend (not a hard requirement):
        -5 dB  → larger |M|   (strong denoising needed)
        +20 dB → smaller |M|  (near-identity expected)

    If this trend is absent the model has not learned input-dependent
    enhancement strength; the mask statistics reveal this before bothering
    with STOI/PESQ analysis.
    """
    groups: Dict[float, List[Dict[str, object]]] = {}
    for row in results:
        try:
            k = float(row["target_snr_db"])  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError):
            k = float("nan")
        groups.setdefault(k, []).append(row)

    summary: List[Dict[str, object]] = []
    for snr, items in sorted(groups.items()):
        all_mask_mags: List[float] = []
        for item in items:
            mm = item.get("_mask_mag")
            if mm is not None and isinstance(mm, np.ndarray):
                all_mask_mags.extend(mm.ravel().tolist())

        if all_mask_mags:
            arr = np.array(all_mask_mags, dtype=np.float64)
            row_out: Dict[str, object] = {
                "target_snr_db": snr,
                "sample_count": len(items),
                "mean_mask_magnitude": float(np.mean(arr)),
                "median_mask_magnitude": float(np.median(arr)),
                "p95_mask_magnitude": float(np.percentile(arr, 95)),
                "max_mask_magnitude": float(np.max(arr)),
                # Per-sample means (for interpretability)
                "mean_of_sample_means": _mean([float(r.get("mask_mean_magnitude", float("nan"))) for r in items]),  # type: ignore[arg-type]
                "mean_of_sample_p95s": _mean([float(r.get("mask_p95_magnitude", float("nan"))) for r in items]),   # type: ignore[arg-type]
            }
        else:
            row_out = {
                "target_snr_db": snr,
                "sample_count": len(items),
                "mean_mask_magnitude": float("nan"),
                "median_mask_magnitude": float("nan"),
                "p95_mask_magnitude": float("nan"),
                "max_mask_magnitude": float("nan"),
                "mean_of_sample_means": float("nan"),
                "mean_of_sample_p95s": float("nan"),
            }
        summary.append(row_out)

    if summary:
        _assert_exp3_path(output_path)
        _write_csv(output_path, summary, list(summary[0].keys()))
        print(f"\nMask statistics by SNR group saved to {output_path}")
        print("  (Expected trend: mean |M| decreases as target SNR increases)")
        for row_out in summary:
            print(
                f"  {row_out['target_snr_db']:>5} dB"
                f" | n={row_out['sample_count']}"
                f" | mean|M|={row_out['mean_mask_magnitude']:.4f}"
                f" | p95|M|={row_out['p95_mask_magnitude']:.4f}"
                f" | max|M|={row_out['max_mask_magnitude']:.4f}"
            )
    return summary


def run_clean_input_test(
    model: nn.Module,
    config: Config,
    device: torch.device,
    output_path: Path,
) -> Dict[str, object]:
    """Pass clean speech (no added noise) through the trained model.

    Expected behaviour for an identity-preserving model:
        waveform correlation ≈ 1.0
        RMS ratio           ≈ 1.0
        mean |M|            ≈ small (model barely modifies clean speech)

    Samples are drawn from the test split clean reference files.
    Only clean audio is used — no noisy mixture is involved here.
    """
    print("\n" + "=" * 56)
    print("IDENTITY / CLEAN-INPUT TEST (Experiment 3)")
    print("=" * 56)

    split_path = config.root / "splits" / "test.csv"
    if not split_path.exists():
        print("  Clean-input test skipped: test split CSV not found.")
        return {}

    import csv as _csv
    with split_path.open(newline="", encoding="utf-8") as fh:
        test_rows = list(_csv.DictReader(fh))

    # Pick up to 10 deterministic samples (sorted by sample_id).
    test_rows.sort(key=lambda r: r.get("sample_id", ""))
    sample_rows = test_rows[:10]

    results_clean: List[Dict[str, object]] = []
    for row in sample_rows:
        generated = config.root / "clean_reference" / f"{row['sample_id']}_clean_ref.wav"
        source = config.root / "clean_reference" / row["clean_filename"]
        cpath = generated if generated.exists() else (source if source.exists() else None)
        if cpath is None:
            continue
        import soundfile as _sf
        clean_wav, sr = _sf.read(str(cpath), dtype="float32", always_2d=True)
        clean_wav = clean_wav.mean(axis=1)
        L = config.segment_samples
        if len(clean_wav) >= L:
            clean_wav = clean_wav[:L]
        else:
            clean_wav = np.pad(clean_wav, (0, L - len(clean_wav)))
        clean_tensor = torch.from_numpy(clean_wav.copy()).unsqueeze(0)

        enhanced_tensor, mask_mag = enhance_waveform_with_mask(
            model, clean_tensor, config, device
        )
        enhanced_np = enhanced_tensor.squeeze(0).numpy()
        clean_np = clean_wav

        corr = float(np.corrcoef(clean_np, enhanced_np)[0, 1]) if clean_np.std() > 1e-8 else float("nan")
        clean_rms = float(np.sqrt(np.mean(clean_np.astype(np.float64) ** 2)))
        enh_rms = float(np.sqrt(np.mean(enhanced_np.astype(np.float64) ** 2)))
        rms_ratio = enh_rms / (clean_rms + 1e-12)
        wf_snr = calculate_waveform_snr(clean_np, enhanced_np)
        clip_pct = float((np.abs(enhanced_np) > 1.0).mean() * 100.0)

        results_clean.append({
            "sample_id": row["sample_id"],
            "waveform_snr_db": wf_snr,
            "rms_ratio": rms_ratio,
            "waveform_correlation": corr,
            "mean_mask_magnitude": float(np.mean(mask_mag)),
            "p95_mask_magnitude": float(np.percentile(mask_mag, 95)),
            "max_mask_magnitude": float(np.max(mask_mag)),
            "clipping_pct": clip_pct,
        })

    if not results_clean:
        print("  No samples could be processed for the clean-input test.")
        return {}

    def _m(field: str) -> float:
        return _mean([float(r[field]) for r in results_clean])

    summary_clean = {
        "n_samples": len(results_clean),
        "mean_waveform_snr_db": _m("waveform_snr_db"),
        "mean_rms_ratio": _m("rms_ratio"),
        "mean_waveform_correlation": _m("waveform_correlation"),
        "mean_mask_magnitude": _m("mean_mask_magnitude"),
        "mean_p95_mask_magnitude": _m("p95_mask_magnitude"),
        "mean_max_mask_magnitude": _m("max_mask_magnitude"),
        "mean_clipping_pct": _m("clipping_pct"),
        "per_sample": results_clean,
    }

    print(f"  Samples tested         : {len(results_clean)}")
    print(f"  Mean waveform SNR (dB) : {summary_clean['mean_waveform_snr_db']:+.2f}  (higher = better preserved)")
    print(f"  Mean RMS ratio         : {summary_clean['mean_rms_ratio']:.4f}  (target ~1.0)")
    print(f"  Mean correlation       : {summary_clean['mean_waveform_correlation']:.4f}  (target ~1.0)")
    print(f"  Mean |M| (mask magn.)  : {summary_clean['mean_mask_magnitude']:.4f}  (lower = more identity)")
    print(f"  Mean p95 |M|           : {summary_clean['mean_p95_mask_magnitude']:.4f}")
    print(f"  Clipping               : {summary_clean['mean_clipping_pct']:.3f} %")

    _assert_exp3_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(summary_clean, fh, indent=2)
    print(f"  Clean-input test results saved to {output_path}")
    return summary_clean


def run_unseen_snr_evaluation(
    model: nn.Module,
    config: Config,
    device: torch.device,
    output_path: Path,
) -> List[Dict[str, object]]:
    """Evaluate at interpolated SNR levels not seen during training.

    Creates temporary noisy mixtures at -2.5, +2.5, +7.5, +12.5, +17.5, +25 dB
    using clean and noise files from the test split.
    These samples are NOT used for training or model selection.

    Purpose: verify that enhancement strength changes smoothly rather than
    behaving like 6 discrete SNR-specific modes.
    """
    print("\n" + "=" * 56)
    print("UNSEEN SNR EVALUATION (Experiment 3)")
    print("=" * 56)

    unseen_snrs = [-2.5, 2.5, 7.5, 12.5, 17.5, 25.0]

    split_path = config.root / "splits" / "test.csv"
    if not split_path.exists():
        print("  Skipped: test split CSV not found.")
        return []

    import csv as _csv, soundfile as _sf
    with split_path.open(newline="", encoding="utf-8") as fh:
        test_rows = list(_csv.DictReader(fh))
    test_rows.sort(key=lambda r: r.get("sample_id", ""))

    results_unseen: List[Dict[str, object]] = []

    for row in test_rows[:30]:  # Use up to 30 test samples as source material
        # Load clean reference
        gen = config.root / "clean_reference" / f"{row['sample_id']}_clean_ref.wav"
        src = config.root / "clean_reference" / row["clean_filename"]
        cpath = gen if gen.exists() else (src if src.exists() else None)
        if cpath is None:
            continue

        clean_wav, _ = _sf.read(str(cpath), dtype="float32", always_2d=True)
        clean_wav = clean_wav.mean(axis=1)

        # Find the paired noisy file to extract the noise component.
        target_snr = int(float(row["target_snr_db"]))
        npath = config.root / "noisy_speech" / f"snr_{target_snr}dB" / row["noisy_filename"]
        if not npath.exists():
            continue
        noisy_wav, _ = _sf.read(str(npath), dtype="float32", always_2d=True)
        noisy_wav = noisy_wav.mean(axis=1)

        L = min(len(clean_wav), len(noisy_wav))
        clean_wav, noisy_wav = clean_wav[:L], noisy_wav[:L]

        # Recover noise component from the known mix: noise = noisy - clean
        # (valid because the dataset uses additive noise mixing).
        noise_wav = noisy_wav - clean_wav[:L]

        # Trim/pad to 2 s segment.
        SEG = config.segment_samples
        if L >= SEG:
            clean_seg = clean_wav[:SEG]
            noise_seg = noise_wav[:SEG]
        else:
            clean_seg = np.pad(clean_wav, (0, SEG - L))
            noise_seg = np.pad(noise_wav, (0, SEG - L))

        clean_power = np.mean(clean_seg.astype(np.float64) ** 2)
        noise_power = np.mean(noise_seg.astype(np.float64) ** 2)
        if clean_power < 1e-12 or noise_power < 1e-12:
            continue  # silent segment — skip

        for unseen_snr in unseen_snrs:
            # Scale noise to achieve the desired SNR:
            # SNR = 10*log10(clean_power / (alpha*noise_power))
            # alpha = sqrt(clean_power / (noise_power * 10^(SNR/10)))
            alpha = float(np.sqrt(clean_power / (noise_power * 10.0 ** (unseen_snr / 10.0))))
            noisy_seg = clean_seg + alpha * noise_seg

            noisy_tensor = torch.from_numpy(noisy_seg.astype(np.float32)).unsqueeze(0)
            enhanced_tensor, mask_mag = enhance_waveform_with_mask(
                model, noisy_tensor, config, device
            )
            enhanced_np = enhanced_tensor.squeeze(0).numpy()

            wf_in_snr = calculate_waveform_snr(clean_seg, noisy_seg)
            wf_out_snr = calculate_waveform_snr(clean_seg, enhanced_np)

            results_unseen.append({
                "sample_id": row["sample_id"],
                "unseen_snr_db": unseen_snr,
                "actual_input_snr_db": wf_in_snr,
                "waveform_output_snr_db": wf_out_snr,
                "waveform_snr_improvement_db": wf_out_snr - wf_in_snr,
                "mean_mask_magnitude": float(np.mean(mask_mag)),
                "p95_mask_magnitude": float(np.percentile(mask_mag, 95)),
            })

    if not results_unseen:
        print("  No unseen-SNR samples could be generated.")
        return []

    # Print summary by unseen SNR level.
    unseen_groups: Dict[float, List[Dict[str, object]]] = {}
    for r in results_unseen:
        unseen_groups.setdefault(float(r["unseen_snr_db"]), []).append(r)

    print(f"  {'Unseen SNR':>10} | {'N':>4} | {'In SNR':>8} | {'Out SNR':>8} | {'SNR Imp':>8} | {'Mean|M|':>8}")
    print("  " + "-" * 70)
    for snr in sorted(unseen_groups):
        grp = unseen_groups[snr]
        print(
            f"  {snr:>10.1f} dB"
            f" | {len(grp):>4}"
            f" | {_mean([float(r['actual_input_snr_db']) for r in grp]):>+8.2f} dB"
            f" | {_mean([float(r['waveform_output_snr_db']) for r in grp]):>+8.2f} dB"
            f" | {_mean([float(r['waveform_snr_improvement_db']) for r in grp]):>+8.2f} dB"
            f" | {_mean([float(r['mean_mask_magnitude']) for r in grp]):>8.4f}"
        )

    _assert_exp3_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_path, results_unseen,
        ["sample_id", "unseen_snr_db", "actual_input_snr_db",
         "waveform_output_snr_db", "waveform_snr_improvement_db",
         "mean_mask_magnitude", "p95_mask_magnitude"],
    )
    print(f"  Unseen-SNR results saved to {output_path}")
    return results_unseen


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
                    "metadata_measured_snr_db": row["metadata_measured_snr_db"],
                    "waveform_input_snr_db": row["waveform_input_snr_db"],
                    "waveform_output_snr_db": row["waveform_output_snr_db"],
                    "waveform_snr_improvement_db": row["waveform_snr_improvement_db"],
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

        # 8. Waveform SNR improvement by target SNR
        # Both input and output SNR use 10*log10(clean_power/error_power).
        snr_imp = [
            float(r.get("mean_waveform_snr_improvement_db", float("nan")))
            for r in snr_summary
        ]
        in_snr = [
            float(r.get("mean_waveform_input_snr_db", float("nan")))
            for r in snr_summary
        ]
        out_snr = [
            float(r.get("mean_waveform_output_snr_db", float("nan")))
            for r in snr_summary
        ]
        fig, ax = _fig(
            "Waveform SNR Improvement by Target SNR\n"
            "(output_snr \u2212 input_snr, both waveform-based)",
            "Mean Waveform SNR Improvement (dB)",
        )
        colors = ["#2176AE" if v >= 0 else "#E63946" for v in snr_imp]
        ax.bar(snr_labels, snr_imp, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Target SNR (dB)", fontsize=11)
        fig.tight_layout()
        fig.savefig(plot_dir / "waveform_snr_improvement_by_target_snr.png", dpi=150)
        plt.close(fig)

        # 8b. Input vs Output waveform SNR by target SNR (line plot)
        x_snr = list(range(len(snr_labels)))
        fig, ax = _fig(
            "Waveform SNR: Input vs Output by Target SNR",
            "Mean Waveform SNR (dB)",
        )
        ax.plot(snr_labels, in_snr, "o-", color="#2176AE", lw=1.8, label="Input SNR")
        ax.plot(snr_labels, out_snr, "s-", color="#F7882F", lw=1.8, label="Output SNR")
        ax.set_xlabel("Target SNR (dB)", fontsize=11)
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(plot_dir / "waveform_snr_input_vs_output.png", dpi=150)
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

    # EXPERIMENT 2 ADDITION: RMS ratio and amplitude-loss curves. This is the
    # single most important new plot in this file - it is what would have
    # caught the Experiment-1 amplitude bug within the first few epochs
    # instead of after a full, ~3-hour, 100-epoch run.
    epochs = [int(h["epoch"]) for h in history]
    if all("val_rms_ratio" in h for h in history):
        fig, ax = _fig("RMS Ratio (Enhanced / Clean) Over Training", "RMS Ratio")
        ax.plot(epochs, [h["train_rms_ratio"] for h in history], color="#2176AE", lw=1.8, label="Train")
        ax.plot(epochs, [h["val_rms_ratio"] for h in history], color="#F7882F", lw=1.8, label="Validation")
        ax.axhline(1.0, color="#26A96C", ls="--", lw=1.5, label="Target (1.0)")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(plot_dir / "rms_ratio_over_training.png", dpi=150)
        plt.close(fig)

    if all("val_amplitude_loss_db" in h for h in history):
        fig, ax = _fig("Amplitude Consistency Loss Over Training", "Amplitude Loss (dB)")
        ax.plot(epochs, [h["train_amplitude_loss_db"] for h in history], color="#7B2D8B", lw=1.8, label="Train")
        ax.plot(epochs, [h["val_amplitude_loss_db"] for h in history], color="#C77DFF", lw=1.8, label="Validation")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(plot_dir / "amplitude_loss_over_training.png", dpi=150)
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

    best_path = config.output_dir / "checkpoints" / "Experiment_03_best.pth"
    ckpt_mb = (
        round(best_path.stat().st_size / (1024 ** 2), 4)
        if best_path.exists()
        else None
    )

    report = {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_3_change": "Bounded CRM: S_enh = (1+M)*S_noisy, M=tanh(decoder/bound)*bound, zero-init",
        "model_description": "Tiny GRU-based DCCRN-inspired speech enhancement model",
        "mask_bound": config.mask_bound,
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
        "amplitude_loss_weight": config.amplitude_loss_weight,
        "waveform_loss_weight": config.waveform_loss_weight,
        "snr_sampling_weights": config.snr_sampling_weights,
        "use_snr_weighted_sampling": config.use_snr_weighted_sampling,
        "random_seed": config.random_seed,
        "device": str(device),
        "number_of_epochs_completed": len(history),
        "best_epoch": int(best["epoch"]),  # type: ignore[arg-type]
        "best_validation_loss": float(best["val_total_loss"]),  # type: ignore[arg-type]
        "dataset_counts": dataset_counts,
        "test_set_metrics": {
            "mean_metadata_measured_snr_db": _agg("metadata_measured_snr_db"),
            "mean_waveform_input_snr_db": _agg("waveform_input_snr_db"),
            "mean_waveform_output_snr_db": _agg("waveform_output_snr_db"),
            "mean_waveform_snr_improvement_db": _agg("waveform_snr_improvement_db"),
            "mean_input_si_snr_db": _agg("input_si_snr_db"),
            "mean_output_si_snr_db": _agg("output_si_snr_db"),
            "mean_si_snr_improvement_db": _agg("si_snr_improvement_db"),
            "mean_input_stoi": _agg("input_stoi"),
            "mean_output_stoi": _agg("output_stoi"),
            "mean_stoi_improvement": _agg("stoi_improvement"),
            "mean_input_pesq": _agg("input_pesq"),
            "mean_output_pesq": _agg("output_pesq"),
            "mean_pesq_improvement": _agg("pesq_improvement"),
            "mean_rms_ratio_enhanced_over_clean": _agg("rms_ratio_enhanced_over_clean"),
            # EXPERIMENT 3: mask magnitude metrics.
            "mean_mask_mean_magnitude": _agg("mask_mean_magnitude"),
            "mean_mask_p95_magnitude": _agg("mask_p95_magnitude"),
        },
        "waveform_snr_formula_note": (
            "waveform_snr = 10*log10(mean(clean^2) / mean((estimate-clean)^2)). "
            "BOTH input SNR (noisy as estimate) and output SNR (enhanced as estimate) "
            "use this SAME formula so waveform_snr_improvement = output_snr - input_snr "
            "is mathematically valid. metadata_measured_snr_db is reported separately "
            "and is NOT mixed into the improvement calculation."
        ),
    }
    report_path = config.output_dir / "results" / "experiment_03_summary.json"
    _assert_exp3_path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Model report saved to {report_path}")


def save_run_config(config: Config, device: torch.device) -> None:
    """Persist all configuration and environment details for reproducibility."""
    gpu_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    )
    run_cfg = {
        "experiment_name": EXPERIMENT_NAME,
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
    # Write to config/ subdirectory (req 22) AND results/ for finalize_evaluation.
    for path in [
        config.output_dir / "config" / "config.json",
        config.output_dir / "config" / "run_config.json",
    ]:
        _assert_exp3_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(run_cfg, handle, indent=2)
    print(f"Run configuration saved to {config.output_dir / 'config' / 'config.json'}")


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
    print("  WAVEFORM SNR  (both input and output: 10*log10(clean_power/error_power))")
    print(f"    Metadata Measured SNR (reference) : {_fmt_stat('metadata_measured_snr_db', ' dB', True)}")
    print(f"    Waveform Input SNR                : {_fmt_stat('waveform_input_snr_db', ' dB', True)}")
    print(f"    Waveform Output SNR               : {_fmt_stat('waveform_output_snr_db', ' dB', True)}")
    print(f"    Waveform SNR Improvement          : {_fmt_stat('waveform_snr_improvement_db', ' dB', True)}")
    print()
    print("  SI-SNR  (dB, scale-invariant, higher is better)")
    print(f"    Input SI-SNR                      : {_fmt_stat('input_si_snr_db', ' dB', True)}")
    print(f"    Output SI-SNR                     : {_fmt_stat('output_si_snr_db', ' dB', True)}")
    print(f"    SI-SNR Improvement                : {_fmt_stat('si_snr_improvement_db', ' dB', True)}")
    print()
    print("  STOI  (0 to 1, higher is better)")
    print(f"    Input STOI                        : {_fmt_stat('input_stoi')}")
    print(f"    Output STOI                       : {_fmt_stat('output_stoi')}")
    print(f"    STOI Improvement                  : {_fmt_stat('stoi_improvement', signed=True)}")
    print()
    print("  PESQ  (wideband, higher is better)")
    print(f"    Input PESQ                        : {_fmt_stat('input_pesq')}")
    print(f"    Output PESQ                       : {_fmt_stat('output_pesq')}")
    print(f"    PESQ Improvement                  : {_fmt_stat('pesq_improvement', signed=True)}")
    print()
    print("  AMPLITUDE / SCALE  (target ratio ~= 1.0; this is what Experiment 1 got wrong)")
    print(f"    RMS ratio (enhanced / clean)      : {_fmt_stat('rms_ratio_enhanced_over_clean')}")
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
    All output paths are asserted to be inside Experiment_03_Identity_Residual/.
    """
    results_dir = config.output_dir / "results"
    _assert_exp3_path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results = evaluate_test_set(model, test_dataset, config, device)

    # Per-sample CSV — Exp3-named, skip internal arrays.
    csv_columns = [k for k in results[0] if not k.startswith("_")]
    test_csv = results_dir / "experiment_03_test_results.csv"
    _assert_exp3_path(test_csv)
    _write_csv(test_csv, results, csv_columns)

    snr_csv = results_dir / "experiment_03_snr_group_results.csv"
    _assert_exp3_path(snr_csv)
    snr_summary = evaluate_by_group(results, "target_snr_db", snr_csv)

    cat_csv = results_dir / "noise_category_summary.csv"
    _assert_exp3_path(cat_csv)
    category_summary = evaluate_by_group(results, "noise_category", cat_csv)

    # EXPERIMENT 3: mask statistics by SNR group.
    mask_csv = results_dir / "experiment_03_mask_statistics.csv"
    evaluate_mask_statistics(results, mask_csv)

    # EXPERIMENT 3: clean-input identity test.
    clean_test_path = results_dir / "experiment_03_clean_input_test.json"
    run_clean_input_test(model, config, device, clean_test_path)

    # EXPERIMENT 3: unseen-SNR evaluation.
    unseen_csv = results_dir / "experiment_03_unseen_snr_results.csv"
    run_unseen_snr_evaluation(model, config, device, unseen_csv)

    save_audio_examples(results, config)
    save_plots(history, results, snr_summary, config)

    dataset_counts = _dataset_counts(config)
    save_model_report(model, config, device, history, results, dataset_counts)
    save_run_config(config, device)

    best_record = min(history, key=lambda h: float(h["val_total_loss"]))  # type: ignore[arg-type]
    best_epoch = int(best_record["epoch"])  # type: ignore[arg-type]
    best_val_loss = float(best_record["val_total_loss"])  # type: ignore[arg-type]

    detect_overfitting(history)

    print("\nSNR GROUP SUMMARY (with mask magnitude):")
    print("  waveform_snr_improvement = waveform_output_snr - waveform_input_snr  (same formula)")
    for row in snr_summary:
        print(
            f"  Target {row.get('target_snr_db', '?'):>4} dB"
            f" | n={row['sample_count']:>3}"
            f" | In SNR={row.get('mean_waveform_input_snr_db', float('nan')):+.1f} dB"
            f" | Out SNR={row.get('mean_waveform_output_snr_db', float('nan')):+.1f} dB"
            f" | SNR Imp={row.get('mean_waveform_snr_improvement_db', float('nan')):+.2f} dB"
            f" | SI-SNR Imp={row.get('mean_si_snr_improvement_db', float('nan')):+.2f} dB"
            f" | RMS={row.get('mean_rms_ratio_enhanced_over_clean', float('nan')):.2f}"
            f" | |M|={row.get('mean_mask_mean_magnitude', float('nan')):.4f}"
        )

    print("\nNOISE CATEGORY SUMMARY:")
    for row in category_summary:
        print(
            f"  {str(row.get('noise_category', '?')):<32}"
            f" | n={row['sample_count']:>3}"
            f" | In SNR={row.get('mean_waveform_input_snr_db', float('nan')):+.1f} dB"
            f" | Out SNR={row.get('mean_waveform_output_snr_db', float('nan')):+.1f} dB"
            f" | SNR Imp={row.get('mean_waveform_snr_improvement_db', float('nan')):+.2f} dB"
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

    # Isolation banner — printed before any file I/O so the user sees the
    # path contract before any potential error messages.
    print("\n" + "=" * 64)
    print(f"EXPERIMENT 3: {EXPERIMENT_NAME}")
    print(f"Strategy : Bounded CRM  S_enh = (1 + M) * S_noisy")
    print(f"           mask_bound = {Config().mask_bound}  (tanh saturation value)")
    print(f"           Final decoder layer: ZERO-INITIALISED")
    print("=" * 64)
    print(f"Output root     : {_EXP3_ROOT}")
    print(f"Checkpoints     : {_EXP3_ROOT / 'checkpoints'}")
    print(f"Results         : {_EXP3_ROOT / 'results'}")
    print(f"Logs            : {_EXP3_ROOT / 'logs'}")
    print(f"Config          : {_EXP3_ROOT / 'config'}")
    print("ISOLATION       : Exp1 (dccrn_outputs/) and Exp2 (models/Experiment_02_SNR_Aware/)")
    print("                  are PROTECTED by _assert_exp3_path() guards.")
    print("=" * 64)

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
        marker_path = config.output_dir / "short_training_passed.json"
        _assert_exp3_path(marker_path)
        with marker_path.open("w", encoding="utf-8") as handle:
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
        # History is now in logs/ subdirectory.
        history_path = config.output_dir / "logs" / "training_history.json"
        if history_path.exists():
            with history_path.open(encoding="utf-8") as handle:
                prior_history = json.load(handle)
            if len(prior_history) >= config.short_epochs and all(
                math.isfinite(item["train_total_loss"])
                and math.isfinite(item["val_total_loss"])
                for item in prior_history[: config.short_epochs]
            ):
                _assert_exp3_path(marker)
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
    best_path = config.output_dir / "checkpoints" / "Experiment_03_best.pth"
    checkpoint = torch.load(best_path, map_location=device)
    full_model.load_state_dict(checkpoint["model_state_dict"])
    best_ckpt_epoch = checkpoint["epoch"]
    best_ckpt_val_loss = checkpoint["best_validation_loss"]
    print(f"\nLoaded best checkpoint from epoch : {best_ckpt_epoch}")
    print(f"Best validation loss              : {best_ckpt_val_loss:.6f}")
    print(f"Checkpoint path                   : {best_path}")

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