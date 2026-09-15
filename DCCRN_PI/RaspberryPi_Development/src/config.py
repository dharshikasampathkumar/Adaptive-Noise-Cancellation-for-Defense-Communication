"""config.py — Experiment 3 DCCRN audio and model configuration constants."""

SAMPLE_RATE: int = 16000          # Hz
N_FFT: int = 512                  # FFT points
HOP_LENGTH: int = 128             # Hop size (75% overlap)
WIN_LENGTH: int = 512             # Hann window length
MASK_BOUND: float = 2.0           # Complex Ratio Mask bound
STREAM_CHUNK: int = 512           # Hardware audio transport chunk size (32 ms)
