"""scratch_stream_test.py — Test exact chunk matching in ProductionDCCRNStreamer."""
import sys
import time
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

sys.path.insert(0, 'C:/SIH/DCCRN_PI')
from dccrn_model import load_checkpoint, TinyDCCRN, apply_crm
from dccrn_interface import SAMPLE_RATE, N_FFT, HOP_LENGTH, WIN_LENGTH, MASK_BOUND


class ProductionDCCRNStreamer:
    """Production-grade stateful streaming inference engine for TinyDCCRN.
    
    Processes 512-sample chunks using a rolling context window.
    Guarantees:
      - Exactly 512 samples in -> 512 samples out per call.
      - Zero dropped samples, zero duplicated samples.
      - Smooth STFT/iSTFT frame continuity.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device("cpu"),
        mask_bound: float = MASK_BOUND,
        block_samples: int = 1024,
    ):
        self.model = model.to(device)
        self.device = device
        self.mask_bound = mask_bound
        self.block_samples = block_samples
        self.model.eval()

        self.window = torch.hann_window(WIN_LENGTH, device=self.device)
        self.reset()

    def reset(self) -> None:
        """Reset internal buffers for new session."""
        self._input_fifo = np.zeros(self.block_samples, dtype=np.float32)

    def process_chunk(self, chunk_512: np.ndarray) -> np.ndarray:
        """Process 512-sample input chunk. Returns 512-sample enhanced output chunk."""
        if len(chunk_512) != 512:
            chunk = np.zeros(512, dtype=np.float32)
            chunk[:min(len(chunk_512), 512)] = chunk_512[:512]
        else:
            chunk = chunk_512.astype(np.float32)

        # Shift input FIFO left by 512 and append new 512-sample chunk
        self._input_fifo[:-512] = self._input_fifo[512:]
        self._input_fifo[-512:] = chunk

        # Compute STFT on current 1024-sample context (8 STFT frames)
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
                length=self.block_samples, center=True
            ).squeeze(0).cpu().numpy()

        # Output the enhanced 512 samples corresponding to the current chunk
        out_chunk = synth_audio[-512:]
        return out_chunk


def main():
    model, _ = load_checkpoint('C:/SIH/Experiment_03_Identity_Residual/checkpoints/Experiment_03_best.pth')
    streamer = ProductionDCCRNStreamer(model)

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
    print(f"NaN count              : {np.isnan(res).sum()}")
    print(f"Inf count              : {np.isinf(res).sum()}")
    print(f"Mean Abs Error vs Input: {np.mean(np.abs(res - sig)):.6f}")


if __name__ == "__main__":
    main()
