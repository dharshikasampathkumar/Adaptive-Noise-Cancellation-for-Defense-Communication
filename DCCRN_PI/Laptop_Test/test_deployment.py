"""test_deployment.py — Verify Experiment 3 DCCRN deployment on laptop.

Usage
-----
    # Minimal (no clean reference needed for deployment):
    python test_deployment.py --checkpoint path/to/Experiment_03_best.pth \\
                              --input path/to/noisy_test.wav

    # With optional clean reference for quality metrics:
    python test_deployment.py --checkpoint path/to/Experiment_03_best.pth \\
                              --input path/to/noisy_test.wav \\
                              --clean  path/to/clean_ref.wav \\
                              --output enhanced_output.wav

Tests performed
---------------
    1.  Load architecture and checkpoint.
    2.  Strict state_dict compatibility check (missing / unexpected keys).
    3.  Parameter count.
    4.  Load + format-check input WAV.
    5.  Run whole-file inference.
    6.  Save enhanced WAV.
    7.  Output length verification.
    8.  NaN / Inf check.
    9.  RMS calculation.
    10. Clipping percentage.
    11. Inference time and real-time factor.
    12. (Optional) waveform SNR, SI-SNR, STOI, PESQ vs clean reference.
    13. Streaming mode: 512-sample chunk test.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Import from deployment directory (ensure this script runs from DCCRN_PI/ or
# the DCCRN_PI path is on sys.path).
_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dccrn_model import load_checkpoint, DEFAULT_SAMPLE_RATE
from dccrn_interface import DCCRNInference, DCCRNStreamer, STREAM_CHUNK, SAMPLE_RATE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def _clipping_pct(x: np.ndarray) -> float:
    return float(100.0 * np.mean(np.abs(x) > 1.0))


def _waveform_snr(clean: np.ndarray, estimate: np.ndarray) -> float:
    c = clean.astype(np.float64)
    e = estimate.astype(np.float64)
    err = e - c
    return float(10.0 * np.log10((np.mean(c**2) + 1e-12) / (np.mean(err**2) + 1e-12)))


def _si_snr(estimate: np.ndarray, target: np.ndarray) -> float:
    e = estimate.astype(np.float64) - estimate.mean()
    t = target.astype(np.float64) - target.mean()
    dot = np.dot(e, t)
    t_energy = np.dot(t, t) + 1e-8
    proj = max(dot / t_energy, 0.0) * t
    noise = e - proj
    return float(10.0 * np.log10((np.dot(proj, proj) + 1e-8) / (np.dot(noise, noise) + 1e-8)))


def _try_stoi(estimate: np.ndarray, clean: np.ndarray, sr: int) -> Optional[float]:
    try:
        from pystoi import stoi  # type: ignore
        return float(stoi(clean, estimate, sr, extended=False))
    except Exception as exc:
        print(f"  [STOI unavailable: {exc}]")
        return None


def _try_pesq(estimate: np.ndarray, clean: np.ndarray, sr: int) -> Optional[float]:
    try:
        from pesq import pesq  # type: ignore
        return float(pesq(sr, clean, estimate, "wb"))
    except Exception as exc:
        print(f"  [PESQ unavailable: {exc}]")
        return None


def _print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _pass_fail(condition: bool, label: str) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


# Optional type used in helpers
from typing import Optional


# ---------------------------------------------------------------------------
# Main test routine
# ---------------------------------------------------------------------------

def run_tests(args: argparse.Namespace) -> int:
    """Run all tests. Returns 0 on success, 1 on any failure."""
    all_passed = True
    device = torch.device("cpu")

    # ------------------------------------------------------------------
    # 1. Load checkpoint
    # ------------------------------------------------------------------
    _print_section("1. CHECKPOINT LOADING & ARCHITECTURE VERIFICATION")
    print(f"  Checkpoint : {args.checkpoint}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"  [FAIL] Checkpoint not found: {ckpt_path}")
        return 1

    ckpt_size_mb = ckpt_path.stat().st_size / (1024 ** 2)
    print(f"  File size  : {ckpt_size_mb:.2f} MB")

    t0 = time.perf_counter()
    if getattr(args, "quantized", False):
        from dccrn_model import load_quantized_model
        print("  Loading mode : Dynamic INT8 Quantized")
        model, ckpt_meta = load_quantized_model(str(ckpt_path), device=device)
    else:
        print("  Loading mode : Standard FP32 / Pruned")
        model, ckpt_meta = load_checkpoint(str(ckpt_path), device=device, strict=True)
    load_time = time.perf_counter() - t0
    print(f"  Load time  : {load_time*1000:.1f} ms")

    mc = ckpt_meta.get("model_config", {})
    sc = ckpt_meta.get("stft_config", {})
    print(f"  Epoch      : {ckpt_meta.get('epoch', '?')}")
    print(f"  Val loss   : {ckpt_meta.get('best_validation_loss', '?')}")
    print(f"  Channels   : {mc.get('channels')}")
    print(f"  GRU hidden : {mc.get('gru_hidden_size')}")
    print(f"  GRU layers : {mc.get('gru_layers')}")
    print(f"  n_fft      : {sc.get('n_fft')}   hop: {sc.get('hop_length')}   win: {sc.get('win_length')}")

    # ------------------------------------------------------------------
    # 2. State dict compatibility
    # ------------------------------------------------------------------
    _print_section("2. STATE_DICT COMPATIBILITY")
    raw_ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    current_keys = set(model.state_dict().keys())
    saved_dict   = raw_ckpt.get("model_state_dict", raw_ckpt) if isinstance(raw_ckpt, dict) else raw_ckpt
    saved_keys   = set(saved_dict.keys())
    missing   = current_keys - saved_keys
    unexpected = saved_keys - current_keys
    all_passed &= _pass_fail(len(missing)    == 0, f"No missing keys      (found {len(missing)})")
    all_passed &= _pass_fail(len(unexpected) == 0, f"No unexpected keys   (found {len(unexpected)})")
    if missing:
        for k in sorted(missing):
            print(f"    MISSING  : {k}")
    if unexpected:
        for k in sorted(unexpected):
            print(f"    UNEXPECTED: {k}")

    # ------------------------------------------------------------------
    # 3. Parameter count
    # ------------------------------------------------------------------
    _print_section("3. PARAMETER COUNT")
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters     : {total:,}")
    print(f"  Trainable parameters : {trainable:,}")
    print(f"  FP32 weight size     : {total * 4 / (1024**2):.2f} MB")
    all_passed &= _pass_fail(total > 0, "Parameter count > 0")

    # ------------------------------------------------------------------
    # 4. Load and verify input WAV
    # ------------------------------------------------------------------
    _print_section("4. INPUT WAV")
    print(f"  File : {args.input}")
    audio, sr = sf.read(args.input, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    all_passed &= _pass_fail(sr == SAMPLE_RATE, f"Sample rate == {SAMPLE_RATE} Hz (got {sr})")
    all_passed &= _pass_fail(audio.ndim == 1, "Audio is 1-D (mono)")
    print(f"  Samples    : {len(audio):,}  ({len(audio)/sr:.3f} s)")
    print(f"  Input RMS  : {_rms(audio):.6f}")
    print(f"  Input clip : {_clipping_pct(audio):.3f} %")

    # ------------------------------------------------------------------
    # 5–11. Inference + diagnostics
    # ------------------------------------------------------------------
    _print_section("5–11. INFERENCE & OUTPUT DIAGNOSTICS")
    inf = DCCRNInference(model=model, device=device)
    t0 = time.perf_counter()
    enhanced = inf.enhance_array(audio, sr)
    inference_time = time.perf_counter() - t0
    audio_duration = len(audio) / SAMPLE_RATE
    rtf = inference_time / audio_duration

    all_passed &= _pass_fail(len(enhanced) == len(audio),
                              f"Output length == input length ({len(audio)} samples)")
    all_passed &= _pass_fail(np.isfinite(enhanced).all(), "No NaN or Inf in output")
    all_passed &= _pass_fail(not np.isnan(enhanced).any(), "No NaN")
    all_passed &= _pass_fail(not np.isinf(enhanced).any(), "No Inf")

    enh_rms  = _rms(enhanced)
    clip_pct = _clipping_pct(enhanced)
    print(f"  Output RMS       : {enh_rms:.6f}")
    print(f"  Clipping         : {clip_pct:.3f} %")
    print(f"  Inference time   : {inference_time*1000:.1f} ms")
    print(f"  Audio duration   : {audio_duration*1000:.1f} ms")
    print(f"  Real-time factor : {rtf:.4f}  ({'real-time capable' if rtf < 1.0 else 'SLOWER THAN REAL-TIME on this device'})")
    print(f"  NOTE: RTF on laptop is a development benchmark only.")
    print(f"        Final real-time validation MUST occur on Raspberry Pi 3B+.")

    # Save enhanced WAV
    if args.output:
        sf.write(args.output, enhanced, SAMPLE_RATE, subtype="PCM_16")
        print(f"\n  Enhanced WAV saved : {args.output}")

    # ------------------------------------------------------------------
    # 12. Optional quality metrics vs clean reference
    # ------------------------------------------------------------------
    if args.clean:
        _print_section("12. QUALITY METRICS VS CLEAN REFERENCE (Optional)")
        clean, clean_sr = sf.read(args.clean, dtype="float32", always_2d=True)
        clean = clean.mean(axis=1)
        if clean_sr != SAMPLE_RATE:
            print(f"  [SKIP] Clean reference is {clean_sr} Hz (need {SAMPLE_RATE} Hz)")
        else:
            L = min(len(clean), len(audio), len(enhanced))
            clean_c, noisy_c, enh_c = clean[:L], audio[:L], enhanced[:L]

            in_snr  = _waveform_snr(clean_c, noisy_c)
            out_snr = _waveform_snr(clean_c, enh_c)
            in_si   = _si_snr(noisy_c, clean_c)
            out_si  = _si_snr(enh_c,   clean_c)
            stoi_in  = _try_stoi(noisy_c, clean_c, SAMPLE_RATE)
            stoi_out = _try_stoi(enh_c,   clean_c, SAMPLE_RATE)
            pesq_in  = _try_pesq(noisy_c, clean_c, SAMPLE_RATE)
            pesq_out = _try_pesq(enh_c,   clean_c, SAMPLE_RATE)

            print(f"  Waveform SNR  : {in_snr:+.2f} -> {out_snr:+.2f} dB  (d {out_snr-in_snr:+.2f} dB)")
            print(f"  SI-SNR        : {in_si:+.2f}  -> {out_si:+.2f}  dB  (d {out_si-in_si:+.2f} dB)")
            if stoi_in is not None and stoi_out is not None:
                print(f"  STOI          : {stoi_in:.4f} -> {stoi_out:.4f}  (d {stoi_out-stoi_in:+.4f})")
            if pesq_in is not None and pesq_out is not None:
                print(f"  PESQ (wb)     : {pesq_in:.4f} -> {pesq_out:.4f}  (d {pesq_out-pesq_in:+.4f})")

    # ------------------------------------------------------------------
    # 13. Streaming mode test (512-sample chunks)
    # ------------------------------------------------------------------
    _print_section("13. STREAMING MODE (512-sample chunks)")
    streamer = DCCRNStreamer(model=model, device=device)
    n_chunks = max(1, len(audio) // STREAM_CHUNK)
    stream_outputs: list[np.ndarray] = []
    t0 = time.perf_counter()
    for i in range(n_chunks):
        chunk = audio[i * STREAM_CHUNK : (i + 1) * STREAM_CHUNK].copy()
        if len(chunk) < STREAM_CHUNK:
            chunk = np.pad(chunk, (0, STREAM_CHUNK - len(chunk)))
        out_chunk = streamer.process_chunk(chunk)
        stream_outputs.append(out_chunk)
    stream_time = time.perf_counter() - t0
    stream_audio_dur = n_chunks * STREAM_CHUNK / SAMPLE_RATE
    stream_rtf = stream_time / stream_audio_dur

    stream_full = np.concatenate(stream_outputs)
    all_passed &= _pass_fail(np.isfinite(stream_full).all(), "Streaming output has no NaN/Inf")
    print(f"  Chunks processed : {n_chunks}")
    print(f"  Latency (buffer) : {streamer.latency_ms:.1f} ms")
    print(f"  Stream RTF       : {stream_rtf:.4f}")
    print(f"  NOTE: RTF < 1.0 means real-time capable on this device.")
    print(f"        Must be confirmed independently on Raspberry Pi 3B+.")
    streamer.reset()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _print_section("SUMMARY")
    print(f"  {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print(f"  Total parameters   : {total:,}")
    print(f"  Checkpoint size    : {ckpt_size_mb:.2f} MB")
    print(f"  Inference RTF      : {rtf:.4f}")
    return 0 if all_passed else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Experiment 3 DCCRN deployment on laptop before Pi deployment."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to Experiment_03_best.pth")
    parser.add_argument("--input",      required=True,
                        help="Path to noisy test WAV (16 kHz mono PCM_16)")
    parser.add_argument("--clean",      default=None,
                        help="(Optional) Path to clean reference WAV for quality metrics")
    parser.add_argument("--output",     default="enhanced_output.wav",
                        help="Path to save enhanced WAV (default: enhanced_output.wav)")
    parser.add_argument("--quantized",  action="store_true",
                        help="Load model using dynamic INT8 quantization structure")
    args = parser.parse_args()
    sys.exit(run_tests(args))


if __name__ == "__main__":
    main()
