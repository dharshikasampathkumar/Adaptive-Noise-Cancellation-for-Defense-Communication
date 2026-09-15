"""dccrn_interface.py — Experiment 3 DCCRN deployment inference wrapper.

Deployment pipeline
-------------------
    Primary audio / waveform (float32, mono, 16 kHz, range [-1.0, 1.0])
    ↓  STFT  (n_fft=512, hop=128, win=512, Hann, center=True)
    ↓  TinyDCCRN forward pass
    ↓  apply_crm: S_enh = (1 + M) * S_noisy
    ↓  iSTFT
    ↓  enhanced waveform (float32, mono, 16 kHz)

Input Requirements
------------------
    - Sample rate: 16,000 Hz mono float32.
    - Resampling to 16,000 Hz must be performed prior to invoking this interface.

Reference microphone note
-------------------------
    The DCCRN accepts ONLY the primary (speech + noise) microphone signal.
    A separate reference microphone feeds the NLMS stage AFTER DCCRN.
    Do NOT pass stereo audio containing primary + reference channels into this interface.
    If stereo audio is passed to enhance_file(), it is assumed to be a multi-channel
    recording of the primary microphone array and averaged to mono.

Streaming note
--------------
    Audio transport chunk = 512 samples = 32 ms at 16 kHz.
    DCCRNStreamer maintains a rolling 2.0-second (32,000-sample) context window
    matching the training segment size, with Hanning boundary cross-fading across
    consecutive 512-sample chunks.
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from torch import Tensor

from dccrn_model import (
    TinyDCCRN,
    apply_crm,
    build_model,
    load_checkpoint,
    DEFAULT_MASK_BOUND,
    DEFAULT_N_FFT,
    DEFAULT_HOP_LENGTH,
    DEFAULT_WIN_LENGTH,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SEGMENT_SAMPLES,
)

# ---------------------------------------------------------------------------
# Audio constants  (match train_dccrn_3.py Config exactly)
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = DEFAULT_SAMPLE_RATE          # 16 000 Hz
N_FFT: int = DEFAULT_N_FFT                      # 512
HOP_LENGTH: int = DEFAULT_HOP_LENGTH            # 128
WIN_LENGTH: int = DEFAULT_WIN_LENGTH            # 512
MASK_BOUND: float = DEFAULT_MASK_BOUND          # 2.0

# Streaming transport chunk (audio hardware buffer size)
STREAM_CHUNK: int = 512                         # samples = 32 ms


# ---------------------------------------------------------------------------
# Low-level STFT / iSTFT helpers (identical to train_dccrn_3.py)
# ---------------------------------------------------------------------------

def _stft(waveform: Tensor, device: torch.device) -> Tensor:
    """Run STFT on a [B, T] waveform tensor. Returns [B, F, Frames] complex."""
    window = torch.hann_window(WIN_LENGTH, device=device)
    return torch.stft(
        waveform, N_FFT, HOP_LENGTH, WIN_LENGTH, window,
        return_complex=True, center=True,
    )


def _istft(spectrogram: Tensor, length: int, device: torch.device) -> Tensor:
    """Run iSTFT on a [B, F, Frames] complex tensor. Returns [B, T]."""
    window = torch.hann_window(WIN_LENGTH, device=device)
    return torch.istft(
        spectrogram, N_FFT, HOP_LENGTH, WIN_LENGTH, window,
        length=length, center=True,
    )


# ---------------------------------------------------------------------------
# Whole-file / segment inference
# ---------------------------------------------------------------------------

class DCCRNInference:
    """Single-shot (offline) inference for a loaded Experiment 3 model.

    Usage::

        inf = DCCRNInference(checkpoint_path="dccrn_quantized.pth")
        enhanced_wav, sr = inf.enhance_file("noisy_primary.wav")
        sf.write("enhanced.wav", enhanced_wav, sr, subtype="PCM_16")

    Or from a numpy array::

        enhanced_np = inf.enhance_array(noisy_np, sample_rate=16000)
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model: Optional[TinyDCCRN] = None,
        device: Optional[torch.device] = None,
        mask_bound: float = MASK_BOUND,
    ):
        """Provide either checkpoint_path (to load from disk) or a pre-loaded model."""
        self.device = device or torch.device("cpu")
        self.mask_bound = mask_bound

        if model is not None:
            self.model = model.to(self.device)
        elif checkpoint_path is not None:
            self.model, self._ckpt_meta = load_checkpoint(checkpoint_path, self.device)
        else:
            raise ValueError("Provide either checkpoint_path or model.")

        self.model.eval()

    # ------------------------------------------------------------------
    # Core inference on a [1, T] float32 tensor
    # ------------------------------------------------------------------

    def enhance_tensor(self, waveform: Tensor) -> Tensor:
        """Enhance a [1, T] float32 waveform tensor. Returns [1, T]."""
        waveform = waveform.to(self.device)
        length = waveform.shape[-1]
        with torch.inference_mode():
            noisy_spec = _stft(waveform, self.device)            # [1, F, Frames]
            n_frames = noisy_spec.shape[-1]
            pad_frames = (4 - (n_frames % 4)) % 4
            if pad_frames > 0:
                noisy_spec_padded = torch.nn.functional.pad(noisy_spec, (0, pad_frames))
            else:
                noisy_spec_padded = noisy_spec

            model_input = torch.stack(
                (noisy_spec_padded.real, noisy_spec_padded.imag), dim=1       # [1, 2, F, Frames_padded]
            )
            raw_output = self.model(model_input)                 # [1, 2, F, Frames_padded]

            if pad_frames > 0:
                raw_output = raw_output[..., :-pad_frames]

            enhanced_spec, _ = apply_crm(raw_output, noisy_spec, self.mask_bound)
            enhanced = _istft(enhanced_spec, length, self.device)
        return enhanced.cpu()

    # ------------------------------------------------------------------
    # Convenience: numpy array
    # ------------------------------------------------------------------

    def enhance_array(self, waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """Enhance a primary-microphone float32 numpy array [T,]. Returns float32 [T,].

        Args:
            waveform    : float32 1-D numpy array, 16 kHz mono primary mic audio.
            sample_rate : must equal 16000 Hz.

        Returns:
            enhanced float32 1-D numpy array of the same length.
        """
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"DCCRN requires {SAMPLE_RATE} Hz; got {sample_rate}. "
                "Resample audio to 16,000 Hz before passing to enhance_array()."
            )
        waveform = waveform.astype(np.float32)
        t = torch.from_numpy(waveform).unsqueeze(0)  # [1, T]
        enhanced = self.enhance_tensor(t)
        return enhanced.squeeze(0).numpy()

    # ------------------------------------------------------------------
    # Convenience: WAV file
    # ------------------------------------------------------------------

    def enhance_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """Load a primary-mic WAV, enhance it, optionally save, and return (array, sr).

        The file MUST be 16 kHz mono. Multi-channel primary-mic WAVs are averaged to mono.
        Do NOT pass stereo files containing primary + reference channels.
        """
        audio, sr = sf.read(input_path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)   # downmix multi-channel primary mic array to mono
        if sr != SAMPLE_RATE:
            raise ValueError(
                f"Expected {SAMPLE_RATE} Hz WAV, got {sr} Hz: {input_path}. "
                "Resample first with sox or scipy before processing."
            )
        enhanced = self.enhance_array(audio, sr)
        if output_path is not None:
            sf.write(output_path, enhanced, SAMPLE_RATE, subtype="PCM_16")
        return enhanced, SAMPLE_RATE


# ---------------------------------------------------------------------------
# Real-time streaming interface
# ---------------------------------------------------------------------------

class DCCRNStreamer:
    """Real-time streaming wrapper for Experiment 3 DCCRN.

    Audio transport chunk: 512 samples = 32 ms at 16 kHz.

    Maintains a rolling 2.0-second (32,000-sample) context window matching the
    training segment size (251 STFT frames) to give the model full temporal context,
    with Hanning boundary cross-fading across consecutive 512-sample chunks.

    Guarantees:
        - 1-to-1 matching: 512 samples in -> 512 enhanced samples out.
        - Zero dropped or duplicated samples.
        - Windowed overlap-add streaming context across consecutive 512-sample chunks.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model: Optional[TinyDCCRN] = None,
        device: Optional[torch.device] = None,
        mask_bound: float = MASK_BOUND,
        context_samples: int = DEFAULT_SEGMENT_SAMPLES, # 32,000 samples = 2.0 s
        fade_samples: int = 128,
    ):
        self.device = device or torch.device("cpu")
        self.mask_bound = mask_bound
        self.context_samples = context_samples
        self.fade_samples = fade_samples

        if model is not None:
            self.model = model.to(self.device)
        elif checkpoint_path is not None:
            self.model, _ = load_checkpoint(checkpoint_path, self.device)
        else:
            raise ValueError("Provide either checkpoint_path or model.")

        self.model.eval()
        self.window = torch.hann_window(WIN_LENGTH, device=self.device)

        if self.fade_samples > 0:
            fade = 0.5 * (1.0 - np.cos(np.pi * np.arange(self.fade_samples) / self.fade_samples))
            self.fade_in = fade.astype(np.float32)
            self.fade_out = (1.0 - fade).astype(np.float32)
        else:
            self.fade_in = None

        self.reset()

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all internal rolling buffers and state between sessions."""
        self._input_fifo = np.zeros(self.context_samples, dtype=np.float32)
        self._prev_tail = np.zeros(self.fade_samples, dtype=np.float32)

    # ------------------------------------------------------------------

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Process one 512-sample chunk (32 ms). Returns a 512-sample enhanced chunk.

        Args:
            chunk : float32 numpy array of shape (512,) — primary mic only.

        Returns:
            enhanced_chunk : float32 numpy array of shape (512,).
        """
        if len(chunk) != STREAM_CHUNK:
            c_arr = np.zeros(STREAM_CHUNK, dtype=np.float32)
            c_arr[:min(len(chunk), STREAM_CHUNK)] = chunk[:STREAM_CHUNK]
            chunk = c_arr
        else:
            chunk = chunk.astype(np.float32)

        # Shift input FIFO left by 512 and append new 512-sample chunk
        self._input_fifo[:-STREAM_CHUNK] = self._input_fifo[STREAM_CHUNK:]
        self._input_fifo[-STREAM_CHUNK:] = chunk

        # Compute STFT on rolling 2-second context window
        win_tensor = torch.from_numpy(self._input_fifo).unsqueeze(0).to(self.device)
        stft_spec = torch.stft(
            win_tensor, N_FFT, HOP_LENGTH, WIN_LENGTH, self.window,
            return_complex=True, center=True
        )

        model_in = torch.stack((stft_spec.real, stft_spec.imag), dim=1)

        with torch.inference_mode():
            raw_out = self.model(model_in)
            enh_spec, _ = apply_crm(raw_out, stft_spec, self.mask_bound)
            synth_audio = torch.istft(
                enh_spec, N_FFT, HOP_LENGTH, WIN_LENGTH, self.window,
                length=self.context_samples, center=True
            ).squeeze(0).cpu().numpy()

        # Extract output corresponding to the latest 512-sample chunk
        raw_chunk = synth_audio[-STREAM_CHUNK:].copy()

        # Smooth boundary cross-fade with previous chunk's tail
        if self.fade_in is not None:
            raw_chunk[:self.fade_samples] = (
                raw_chunk[:self.fade_samples] * self.fade_in +
                self._prev_tail * self.fade_out
            )
            self._prev_tail = raw_chunk[-self.fade_samples:].copy()

        return raw_chunk

    # ------------------------------------------------------------------

    @property
    def latency_samples(self) -> int:
        """Algorithmic transport chunk latency in samples (512)."""
        return STREAM_CHUNK

    @property
    def latency_ms(self) -> float:
        """Algorithmic transport chunk latency in milliseconds (32.0 ms)."""
        return (STREAM_CHUNK / SAMPLE_RATE) * 1000.0
