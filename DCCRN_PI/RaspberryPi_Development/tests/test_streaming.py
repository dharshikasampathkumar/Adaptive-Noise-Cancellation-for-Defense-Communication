"""test_streaming.py — 512-sample streaming chunk test for Raspberry Pi 3B+."""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import soundfile as sf
import torch

_HERE = Path(__file__).parent.resolve()
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dccrn_model import load_quantized_model, load_checkpoint
from dccrn_interface import DCCRNStreamer, SAMPLE_RATE, STREAM_CHUNK
from test_pi_inference import get_process_memory_mb


def run_streaming_test(model_path: Path, wav_path: Path) -> Dict[str, Any]:
    """Run real-time 512-sample streaming test and measure per-chunk latency and RTF."""
    print("============================================================")
    print("  RASPBERRY PI 3B+ -- REAL-TIME 512-SAMPLE STREAMING TEST")
    print("============================================================")
    print(f"  Model checkpoint : {model_path}")
    print(f"  Input WAV        : {wav_path}")

    device = torch.device("cpu")
    if "quantized" in model_path.name.lower():
        base_ckpt = model_path.parent / "dccrn_fp32_best.pth"
        if not base_ckpt.exists():
            base_ckpt = Path("C:/SIH/Experiment_03_Identity_Residual/checkpoints/Experiment_03_best.pth")
        model, _ = load_quantized_model(str(model_path), base_checkpoint_path=str(base_ckpt), device=device)
    else:
        model, _ = load_checkpoint(str(model_path), device=device)

    streamer = DCCRNStreamer(model=model, device=device)

    # Load audio
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)

    n_chunks = len(audio) // STREAM_CHUNK
    truncated_audio = audio[:n_chunks * STREAM_CHUNK]

    chunk_times_ms = []
    output_chunks = []

    print(f"  Streaming Chunks : {n_chunks} ({n_chunks * STREAM_CHUNK / SAMPLE_RATE:.2f} s audio)")
    print(f"  Chunk Size       : {STREAM_CHUNK} samples (32.0 ms)")

    t_start = time.perf_counter()
    for i in range(n_chunks):
        chunk_in = truncated_audio[i * STREAM_CHUNK : (i + 1) * STREAM_CHUNK]
        t0 = time.perf_counter()
        chunk_out = streamer.process_chunk(chunk_in)
        t_chunk = (time.perf_counter() - t0) * 1000.0
        chunk_times_ms.append(t_chunk)
        output_chunks.append(chunk_out)

    t_total = time.perf_counter() - t_start

    full_output = np.concatenate(output_chunks)
    audio_dur = len(truncated_audio) / SAMPLE_RATE
    stream_rtf = t_total / max(audio_dur, 1e-6)

    # Verification checks
    passed = True
    len_match = len(full_output) == len(truncated_audio)
    passed &= len_match

    no_nan_inf = np.isfinite(full_output).all()
    passed &= no_nan_inf

    avg_chunk_ms = float(np.mean(chunk_times_ms))
    max_chunk_ms = float(np.max(chunk_times_ms))

    # Real-time requirement: 512 samples at 16 kHz = 32.0 ms budget per chunk
    rt_capable = stream_rtf < 1.0 and max_chunk_ms < 32.0

    print("\n  --- Diagnostics ---")
    print(f"  Exact Length Match : {'PASS' if len_match else 'FAIL'} ({len(full_output):,} vs {len(truncated_audio):,})")
    print(f"  NaN / Inf Check     : {'PASS' if no_nan_inf else 'FAIL'}")
    print(f"  Algorithmic Latency : {streamer.latency_ms:.1f} ms")
    print(f"  Avg Chunk Compute   : {avg_chunk_ms:.2f} ms / 32.0 ms budget")
    print(f"  Max Chunk Compute   : {max_chunk_ms:.2f} ms")
    print(f"  Stream RTF          : {stream_rtf:.4f}")
    print(f"  Real-Time Status    : {'REAL-TIME CAPABLE' if rt_capable else ('NEAR REAL-TIME' if stream_rtf < 1.0 else 'NOT REAL-TIME')}")

    return {
        "passed": passed,
        "n_chunks": n_chunks,
        "avg_chunk_ms": avg_chunk_ms,
        "max_chunk_ms": max_chunk_ms,
        "stream_rtf": stream_rtf,
        "rt_capable": rt_capable,
        "latency_ms": streamer.latency_ms,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Raspberry Pi Streaming Test")
    parser.add_argument("--model", type=str, default="../model/dccrn_quantized.pth")
    parser.add_argument("--wav", type=str, required=True)
    args = parser.parse_args()

    res = run_streaming_test(Path(args.model), Path(args.wav))
    print(f"\nFinal Result: {'PASS' if res['passed'] else 'FAIL'}")
