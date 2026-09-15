"""test_2s_context_streamer.py — Test 2-second context rolling window streamer."""
import sys
import time
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

sys.path.insert(0, 'C:/SIH/DCCRN_PI')
from dccrn_model import load_checkpoint, TinyDCCRN, apply_crm
from dccrn_interface import SAMPLE_RATE, N_FFT, HOP_LENGTH, WIN_LENGTH, MASK_BOUND, STREAM_CHUNK, DEFAULT_SEGMENT_SAMPLES


class ContextDCCRNStreamer:
    """Stateful streaming wrapper maintaining a 2.0-second (32,000-sample) context window.

    Matches the exact 2-second STFT frame shape (F=257, T=251) expected by TinyDCCRN.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device("cpu"),
        mask_bound: float = MASK_BOUND,
        context_samples: int = DEFAULT_SEGMENT_SAMPLES, # 32,000 samples = 2.0s
        fade_samples: int = 128,
    ):
        self.model = model.to(device)
        self.device = device
        self.mask_bound = mask_bound
        self.context_samples = context_samples
        self.fade_samples = fade_samples
        self.model.eval()

        self.window = torch.hann_window(WIN_LENGTH, device=self.device)
        # Hanning cross-fade window for smooth overlap at chunk boundaries
        if self.fade_samples > 0:
            fade = 0.5 * (1.0 - np.cos(np.pi * np.arange(self.fade_samples) / self.fade_samples))
            self.fade_in = fade.astype(np.float32)
            self.fade_out = (1.0 - fade).astype(np.float32)
        else:
            self.fade_in = None

        self.reset()

    def reset(self) -> None:
        """Reset internal rolling buffer and overlap state."""
        self._input_fifo = np.zeros(self.context_samples, dtype=np.float32)
        self._prev_tail = np.zeros(self.fade_samples, dtype=np.float32)

    def process_chunk(self, chunk_512: np.ndarray) -> np.ndarray:
        """Process one 512-sample chunk (32 ms). Returns 512-sample enhanced output chunk."""
        if len(chunk_512) != STREAM_CHUNK:
            c_arr = np.zeros(STREAM_CHUNK, dtype=np.float32)
            c_arr[:min(len(chunk_512), STREAM_CHUNK)] = chunk_512[:STREAM_CHUNK]
            chunk = c_arr
        else:
            chunk = chunk_512.astype(np.float32)

        # Shift input FIFO left by 512 and append new chunk
        self._input_fifo[:-STREAM_CHUNK] = self._input_fifo[STREAM_CHUNK:]
        self._input_fifo[-STREAM_CHUNK:] = chunk

        # Compute STFT on 2-second (32,000-sample) context -> shape [1, 257, 251]
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

        # Extract output corresponding to the latest 512 samples
        raw_chunk = synth_audio[-STREAM_CHUNK:].copy()

        # Smooth boundary cross-fade with previous chunk's tail
        if self.fade_in is not None:
            raw_chunk[:self.fade_samples] = (
                raw_chunk[:self.fade_samples] * self.fade_in +
                self._prev_tail * self.fade_out
            )
            self._prev_tail = raw_chunk[-self.fade_samples:].copy()

        return raw_chunk


def main():
    model, _ = load_checkpoint('C:/SIH/Experiment_03_Identity_Residual/checkpoints/Experiment_03_best.pth')
    streamer = ContextDCCRNStreamer(model)

    sr = 16000
    n_chunks = 64  # 64 * 512 = 32,768 samples (2.048 seconds)
    total_samples = n_chunks * 512
    t = np.linspace(0, total_samples / sr, total_samples, endpoint=False, dtype=np.float32)
    sig = 0.5 * np.sin(2 * np.pi * 440 * t)

    chunks = [sig[i*512:(i+1)*512] for i in range(n_chunks)]
    out_chunks = []

    t0 = time.perf_counter()
    for c in chunks:
        out_chunks.append(streamer.process_chunk(c))
    proc_time = time.perf_counter() - t0

    res = np.concatenate(out_chunks)
    print(f"Total chunks processed : {n_chunks}")
    print(f"Input total samples    : {len(sig)}")
    print(f"Output total samples   : {len(res)}")
    print(f"Exact sample match     : {len(sig) == len(res)}")
    print(f"Proc time              : {proc_time*1000:.1f} ms for {total_samples/sr:.3f} s audio (RTF: {proc_time/(total_samples/sr):.4f})")
    print(f"Per-chunk compute time : {proc_time*1000/n_chunks:.2f} ms per 512-sample chunk (Budget: 32.0 ms)")
    print(f"NaN count              : {np.isnan(res).sum()}")
    print(f"Inf count              : {np.isinf(res).sum()}")


if __name__ == "__main__":
    main()
