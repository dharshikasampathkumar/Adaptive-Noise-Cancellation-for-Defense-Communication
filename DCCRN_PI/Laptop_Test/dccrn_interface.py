"""dccrn_interface.py — Experiment 3 DCCRN deployment inference wrapper.

Deployment pipeline
-------------------
    Primary WAV / waveform (float32, mono, 16 kHz)
    ↓  preprocess (resample, normalise)
    ↓  STFT  (n_fft=512, hop=128, win=512, Hann, center=True)
    ↓  TinyDCCRN forward pass
    ↓  apply_crm: S_enh = (1 + M) * S_noisy
    ↓  iSTFT
    ↓  enhanced waveform (float32, mono, 16 kHz)

Reference microphone note
-------------------------
    The DCCRN accepts ONLY the primary (speech + noise) microphone.
    A separate reference microphone feeds the NLMS stage AFTER DCCRN.
    Do NOT pass reference audio into this interface.

Streaming note
--------------
    Audio transport chunk = 512 samples = 32 ms at 16 kHz.
    512 samples is NOT a complete STFT input; DCCRNStreamer maintains an
    internal overlap-add buffer so the GRU sees full temporal context.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
import soundfile as sf
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

        inf = DCCRNInference(checkpoint_path="experiment3_best.pth")
        enhanced_wav, sr = inf.enhance_file("noisy.wav")
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
        """
        Provide either checkpoint_path (to load from disk) or a pre-loaded model.
        """
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
                (noisy_spec_padded.real, noisy_spec_padded.imag), dim=1       # [1, 2, F, Frames]
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
        """Enhance a mono float32 numpy array [T,]. Returns float32 [T,].

        Args:
            waveform    : float32 1-D numpy array, 16 kHz mono
            sample_rate : must equal 16000; checked but not used for resampling

        Returns:
            enhanced float32 1-D numpy array of the same length
        """
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"DCCRN requires {SAMPLE_RATE} Hz; got {sample_rate}. "
                "Resample before passing to enhance_array()."
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
        """Load a WAV, enhance it, optionally save, and return (array, sr).

        The file is expected to be 16 kHz mono.  Stereo files are downmixed.
        """
        audio, sr = sf.read(input_path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)   # downmix to mono
        if sr != SAMPLE_RATE:
            raise ValueError(
                f"Expected {SAMPLE_RATE} Hz WAV, got {sr} Hz: {input_path}. "
                "Convert first with: sox input.wav -r 16000 output.wav"
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

    The DCCRN was trained on 2-second (32 000-sample) segments.  To handle
    streaming chunks properly the streamer maintains an overlap-add buffer:

        - Incoming chunks accumulate in an input ring-buffer.
        - When at least one full processing block is available the model
          runs inference on that block.
        - The iSTFT output is overlap-added into an output buffer using
          the hop_length stride.
        - Enhanced samples are dequeued in exact 512-sample chunks.

    The model is loaded ONCE and stays resident between chunks.
    Do NOT recreate this object per chunk — that destroys GRU hidden state.

    Streaming design
    ----------------
    - Block size = 32 × HOP_LENGTH = 32 × 128 = 4096 samples.
      (A full 4096-sample block gives 32 STFT frames, which is a reasonable
      context size; the model was trained on 251 frames but can generalise
      to shorter blocks at the cost of some context.  A larger block reduces
      latency variation.)
    - Stride = HOP_LENGTH = 128 (one new STFT frame per hop).
    - The actual algorithmic latency introduced by DCCRN is the block size.

    Usage::

        streamer = DCCRNStreamer(checkpoint_path="experiment3_best.pth")
        for chunk in audio_hardware_chunks:
            enhanced_chunk = streamer.process_chunk(chunk)
            audio_output.write(enhanced_chunk)
        streamer.reset()   # between utterances / calls
    """

    BLOCK_SIZE: int = 32 * HOP_LENGTH   # 4096 samples = 256 ms context

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model: Optional[TinyDCCRN] = None,
        device: Optional[torch.device] = None,
        mask_bound: float = MASK_BOUND,
        block_size: Optional[int] = None,
    ):
        self.device = device or torch.device("cpu")
        self.mask_bound = mask_bound
        self.block_size = block_size or self.BLOCK_SIZE

        if model is not None:
            self.model = model.to(self.device)
        elif checkpoint_path is not None:
            self.model, _ = load_checkpoint(checkpoint_path, self.device)
        else:
            raise ValueError("Provide either checkpoint_path or model.")

        self.model.eval()
        self.reset()

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all internal buffers and hidden state. Call between sessions."""
        self._input_buffer: list[float] = []
        self._output_buffer: list[float] = []
        # Overlap-add accumulator: stores partially assembled output frames.
        self._ola_buffer = np.zeros(self.block_size + N_FFT, dtype=np.float32)
        # Track how many samples of the ola buffer have been committed.
        self._ola_write_pos: int = 0

    # ------------------------------------------------------------------

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Process one 512-sample chunk. Returns a 512-sample enhanced chunk.

        Args:
            chunk : float32 numpy array of shape (512,) — primary mic only.

        Returns:
            enhanced_chunk : float32 numpy array of shape (512,).
            If the internal buffer has not yet accumulated enough context,
            the chunk is returned with a zero-latency pass-through until
            the first full block is ready.
        """
        if chunk.shape != (STREAM_CHUNK,):
            # Accept any length; pad/trim for robustness.
            if len(chunk) < STREAM_CHUNK:
                chunk = np.pad(chunk, (0, STREAM_CHUNK - len(chunk)))
            else:
                chunk = chunk[:STREAM_CHUNK]

        self._input_buffer.extend(chunk.tolist())
        output = np.zeros(STREAM_CHUNK, dtype=np.float32)

        while len(self._input_buffer) >= self.block_size:
            block = np.array(self._input_buffer[:self.block_size], dtype=np.float32)
            # Keep overlap: slide by HOP_LENGTH to preserve context.
            self._input_buffer = self._input_buffer[HOP_LENGTH:]

            enhanced_block = self._process_block(block)
            self._output_buffer.extend(enhanced_block.tolist())

        # Drain output buffer into the 512-sample return chunk.
        if len(self._output_buffer) >= STREAM_CHUNK:
            output = np.array(self._output_buffer[:STREAM_CHUNK], dtype=np.float32)
            self._output_buffer = self._output_buffer[STREAM_CHUNK:]
        else:
            # Not enough output yet: pass-through until buffer fills.
            available = len(self._output_buffer)
            output[:available] = np.array(self._output_buffer, dtype=np.float32)
            self._output_buffer = []

        return output

    # ------------------------------------------------------------------

    def _process_block(self, block: np.ndarray) -> np.ndarray:
        """Run DCCRN on a single block, return enhanced block (same length)."""
        waveform = torch.from_numpy(block).unsqueeze(0).to(self.device)
        length = waveform.shape[-1]
        with torch.inference_mode():
            noisy_spec = _stft(waveform, self.device)
            model_input = torch.stack((noisy_spec.real, noisy_spec.imag), dim=1)
            raw_output = self.model(model_input)
            enhanced_spec, _ = apply_crm(raw_output, noisy_spec, self.mask_bound)
            enhanced = _istft(enhanced_spec, length, self.device)
        return enhanced.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------

    @property
    def latency_samples(self) -> int:
        """Algorithmic latency introduced by the streaming buffer (samples)."""
        return self.block_size

    @property
    def latency_ms(self) -> float:
        """Algorithmic latency in milliseconds."""
        return 1000.0 * self.block_size / SAMPLE_RATE
