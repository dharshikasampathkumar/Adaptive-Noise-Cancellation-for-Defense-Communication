"""run_pi_test.py — Single-command Raspberry Pi 3B+ validation launcher.

Runs both offline WAV enhancement and 512-sample real-time streaming tests,
measures CPU / RAM usage on Pi, and prints the explicit final real-time verdict.

Usage:
    python3 run_pi_test.py --wav sample.wav
"""

import argparse
import os
import platform
import sys
import time
from pathlib import Path

# Add src and tests to path
_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE / "src"))
sys.path.insert(0, str(_HERE / "tests"))

import torch
from test_pi_inference import run_offline_test, get_process_memory_mb
from test_streaming import run_streaming_test


def print_system_header():
    print("=" * 60)
    print("  RASPBERRY PI 3B+ VALIDATION LAUNCHER")
    print("  Experiment 3 -- Dynamic INT8 Quantized DCCRN")
    print("=" * 60)
    print(f"  OS / Kernel      : {platform.system()} {platform.release()}")
    print(f"  Architecture     : {platform.machine()}")
    print(f"  Python Version   : {platform.python_version()}")
    print(f"  PyTorch Version  : {torch.__version__}")
    print(f"  CPU Core Count   : {os.cpu_count()}")
    
    # Check 64-bit ARM OS requirement
    arch = platform.machine().lower()
    if "aarch64" in arch or "arm64" in arch or "x86_64" in arch or "amd64" in arch:
        print(f"  64-Bit OS Check  : PASS ({arch})")
    else:
        print(f"  [WARNING] Architecture is '{arch}'. Official PyTorch Pi packages require 64-bit aarch64 OS.")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run complete Raspberry Pi validation suite")
    parser.add_argument("--model", type=str, default=str(_HERE / "model" / "dccrn_quantized.pth"),
                        help="Path to dccrn_quantized.pth checkpoint")
    parser.add_argument("--wav", type=str, required=True,
                        help="Path to 16 kHz mono test WAV file")
    args = parser.parse_args()

    model_p = Path(args.model)
    wav_p = Path(args.wav)

    print_system_header()

    # 1. Offline Test
    off_res = run_offline_test(model_p, wav_p)
    print()

    # 2. Real-Time Streaming Test
    st_res = run_streaming_test(model_p, wav_p)
    print()

    # Final Verdict Synthesis
    print("=" * 60)
    print("  FINAL RASPBERRY PI 3B+ DEPLOYMENT VERDICT")
    print("=" * 60)
    
    rtf_off = off_res["rtf"]
    rtf_st = st_res["stream_rtf"]
    max_chunk = st_res["max_chunk_ms"]
    passed = off_res["passed"] and st_res["passed"]

    print(f"  Offline Inference RTF : {rtf_off:.4f}")
    print(f"  Streaming RTF         : {rtf_st:.4f}")
    print(f"  Max Chunk Compute     : {max_chunk:.2f} ms (Budget: 32.0 ms)")
    print(f"  Memory Footprint      : {off_res['ram_mb']:.1f} MB RSS")

    if passed and rtf_st < 1.0 and max_chunk < 32.0:
        verdict = "REAL-TIME CAPABLE ON RASPBERRY PI 3B+"
    elif passed and rtf_st < 1.0:
        verdict = "NEAR-REAL-TIME CAPABLE (RTF < 1.0, occasional chunk jitter)"
    else:
        verdict = "NOT REAL-TIME CAPABLE ON THIS SYSTEM"

    print("\n  ==========================================================")
    print(f"  FINAL CLASSIFICATION: {verdict}")
    print("  ==========================================================\n")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
