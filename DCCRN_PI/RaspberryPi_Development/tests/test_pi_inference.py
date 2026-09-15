"""test_pi_inference.py — Offline WAV inference test for Raspberry Pi 3B+."""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import soundfile as sf
import torch

# Add src to path
_HERE = Path(__file__).parent.resolve()
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dccrn_model import load_quantized_model, load_checkpoint
from dccrn_interface import DCCRNInference, SAMPLE_RATE


def get_process_memory_mb() -> float:
    """Get current process memory usage (RSS) in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        # Fallback for Linux /proc/self/status
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024.0
        except Exception:
            return 0.0
    return 0.0


def run_offline_test(model_path: Path, wav_path: Path) -> Dict[str, Any]:
    """Run offline WAV inference and record latency, RTF, RAM usage."""
    print("============================================================")
    print("  RASPBERRY PI 3B+ -- OFFLINE WAV INFERENCE TEST")
    print("============================================================")
    print(f"  Model checkpoint : {model_path}")
    print(f"  Input WAV        : {wav_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    mem_before = get_process_memory_mb()

    # Load model
    t0 = time.perf_counter()
    device = torch.device("cpu")
    if "quantized" in model_path.name.lower():
        # Find base checkpoint if available, or load directly
        base_ckpt = model_path.parent / "dccrn_fp32_best.pth"
        if not base_ckpt.exists():
            base_ckpt = Path("C:/SIH/Experiment_03_Identity_Residual/checkpoints/Experiment_03_best.pth")
        model, _ = load_quantized_model(str(model_path), base_checkpoint_path=str(base_ckpt), device=device)
    else:
        model, _ = load_checkpoint(str(model_path), device=device)

    load_time_sec = time.perf_counter() - t0
    mem_after = get_process_memory_mb()
    model_ram_mb = max(0.0, mem_after - mem_before)

    print(f"  Model Load Time  : {load_time_sec*1000:.1f} ms")
    print(f"  RAM Usage (RSS)  : {mem_after:.1f} MB (Delta: +{model_ram_mb:.1f} MB)")

    # Load WAV
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1) # Mono
    if sr != SAMPLE_RATE:
        raise ValueError(f"WAV must be {SAMPLE_RATE} Hz, got {sr} Hz")

    duration_sec = len(audio) / SAMPLE_RATE
    print(f"  Audio Duration   : {duration_sec:.2f} s ({len(audio):,} samples)")

    # Run inference
    engine = DCCRNInference(model=model, device=device)
    t0 = time.perf_counter()
    enhanced = engine.enhance_array(audio, sample_rate=SAMPLE_RATE)
    infer_time_sec = time.perf_counter() - t0

    rtf = infer_time_sec / max(duration_sec, 1e-6)

    # Verification checks
    passed = True
    len_match = len(enhanced) == len(audio)
    passed &= len_match

    no_nan_inf = np.isfinite(enhanced).all()
    passed &= no_nan_inf

    clip_pct = float(100.0 * np.mean(np.abs(enhanced) > 1.0))
    enh_rms = float(np.sqrt(np.mean(enhanced.astype(np.float64)**2)))

    print("\n  --- Diagnostics ---")
    print(f"  Length Match     : {'PASS' if len_match else 'FAIL'} ({len(enhanced)} samples)")
    print(f"  NaN / Inf Check  : {'PASS' if no_nan_inf else 'FAIL'}")
    print(f"  Output RMS       : {enh_rms:.6f}")
    print(f"  Clipping Pct     : {clip_pct:.3f}%")
    print(f"  Inference Time   : {infer_time_sec*1000:.1f} ms")
    print(f"  Real-Time Factor : {rtf:.4f}")

    return {
        "passed": passed,
        "load_time_ms": load_time_sec * 1000.0,
        "infer_time_ms": infer_time_sec * 1000.0,
        "duration_sec": duration_sec,
        "rtf": rtf,
        "ram_mb": mem_after,
        "output_rms": enh_rms,
        "clipping_pct": clip_pct,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Raspberry Pi Offline Inference Test")
    parser.add_argument("--model", type=str, default="../model/dccrn_quantized.pth")
    parser.add_argument("--wav", type=str, required=True)
    args = parser.parse_args()

    res = run_offline_test(Path(args.model), Path(args.wav))
    print(f"\nFinal Result: {'PASS' if res['passed'] else 'FAIL'}")
