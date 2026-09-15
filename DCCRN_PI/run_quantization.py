"""run_quantization.py — Dynamic INT8 Quantization for Experiment 3 DCCRN.

Applies PyTorch dynamic INT8 quantization (torch.ao.quantization.quantize_dynamic) to
nn.GRU and nn.Linear layers. Conv2d and ConvTranspose2d remain FP32 for numerical stability.

Evaluates the quantized model on the held-out 100-sample test set.

Outputs:
    C:/SIH/Experiment_03_Identity_Residual/results/quantization/quantized_model.pth
    C:/SIH/Experiment_03_Identity_Residual/results/quantization/quantization_comparison.csv
    C:/SIH/Experiment_03_Identity_Residual/results/quantization/quantization_report.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

try:
    import torch.ao.quantization as quantization
except ImportError:
    import torch.quantization as quantization

_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dccrn_model import TinyDCCRN, load_checkpoint
from dccrn_interface import DCCRNInference, SAMPLE_RATE
from test_dataset_evaluation import _waveform_snr, _si_snr, _check_perceptual_validity, _safe_stoi, _safe_pesq


def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """Apply dynamic INT8 quantization to GRU and Linear layers only."""
    model.eval()
    quantized_model = quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.GRU, nn.Linear},
        dtype=torch.qint8
    )
    return quantized_model


def main():
    parser = argparse.ArgumentParser(description="Dynamic INT8 Quantization for DCCRN")
    parser.add_argument("--checkpoint", type=str, default="C:/SIH/Experiment_03_Identity_Residual/checkpoints/Experiment_03_best.pth",
                        help="Path to FP32 or pruned checkpoint to quantize.")
    parser.add_argument("--test-dataset", type=str, default="C:/SIH/synthetic_dataset")
    parser.add_argument("--output-dir", type=str, default="C:/SIH/Experiment_03_Identity_Residual/results/quantization")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    dataset_dir = Path(args.test_dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_csv = dataset_dir / "splits" / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found at {test_csv}")

    print("=== Dynamic INT8 Quantization ===")
    print(f"Loading model from: {checkpoint_path}")

    model, ckpt = load_checkpoint(checkpoint_path, device="cpu")
    stft_config = ckpt.get("stft_config", {})

    print("Quantizing GRU & Linear layers to INT8...")
    q_model = apply_dynamic_quantization(model)

    q_ckpt_path = output_dir / "quantized_model.pth"
    torch.save(q_model.state_dict(), q_ckpt_path)
    file_size_mb = os.path.getsize(q_ckpt_path) / (1024 * 1024)
    print(f"Saved quantized state_dict to {q_ckpt_path} ({file_size_mb:.2f} MB)")

    print("\nEvaluating quantized model on 100 test samples...")
    q_model.eval()
    engine = DCCRNInference(model=q_model, device=torch.device("cpu"))

    with open(test_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        samples = list(reader)

    input_snrs, output_snrs, snr_imps = [], [], []
    input_sisnrs, output_sisnrs, sisnr_imps = [], [], []
    stois, stoi_imps = [], []
    pesqs, pesq_imps = [], []
    rms_ratios = []
    rtfs, latencies_ms = [], []

    # Warmup
    dummy = np.zeros(32000, dtype=np.float32)
    for _ in range(5):
        engine.enhance_array(dummy)

    for idx, row in enumerate(samples, 1):
        sample_id = row.get("sample_id", str(idx))
        target_snr = float(row.get("target_snr_db", float("nan")))

        snr_int = int(target_snr) if math.isfinite(target_snr) else 0
        noisy_path = dataset_dir / "noisy_speech" / f"snr_{snr_int}dB" / row["noisy_filename"]
        gen_clean  = dataset_dir / "clean_reference" / f"{sample_id}_clean_ref.wav"
        src_clean  = dataset_dir / "clean_reference" / row["clean_filename"]
        clean_path = gen_clean if gen_clean.exists() else (src_clean if src_clean.exists() else None)

        if not noisy_path.exists() or clean_path is None:
            continue

        noisy, sr = sf.read(str(noisy_path), dtype="float32", always_2d=True)
        clean, _  = sf.read(str(clean_path), dtype="float32", always_2d=True)
        noisy = noisy.mean(axis=1)
        clean = clean.mean(axis=1)

        L = min(len(noisy), len(clean))
        noisy, clean = noisy[:L], clean[:L]

        seg = int(SAMPLE_RATE * 2.0)
        if L >= seg:
            noisy, clean = noisy[:seg], clean[:seg]
        else:
            noisy = np.pad(noisy, (0, seg - L))
            clean = np.pad(clean, (0, seg - L))

        t0 = time.perf_counter()
        enhanced = engine.enhance_array(noisy, SAMPLE_RATE)
        t_proc = time.perf_counter() - t0

        dur = len(noisy) / SAMPLE_RATE
        rtf = t_proc / max(dur, 1e-6)
        lat_ms = (t_proc / max(len(noisy), 1)) * 1000.0

        rtfs.append(rtf)
        latencies_ms.append(lat_ms)

        in_snr = _waveform_snr(clean, noisy)
        out_snr = _waveform_snr(clean, enhanced)
        in_sisnr = _si_snr(noisy, clean)
        out_sisnr = _si_snr(enhanced, clean)

        input_snrs.append(in_snr)
        output_snrs.append(out_snr)
        snr_imps.append(out_snr - in_snr)
        input_sisnrs.append(in_sisnr)
        output_sisnrs.append(out_sisnr)
        sisnr_imps.append(out_sisnr - in_sisnr)

        c_rms = np.sqrt(np.mean(clean.astype(np.float64)**2))
        e_rms = np.sqrt(np.mean(enhanced.astype(np.float64)**2))
        rms_ratios.append(e_rms / (c_rms + 1e-12))

        if _check_perceptual_validity(clean):
            in_s = _safe_stoi(noisy, clean, SAMPLE_RATE)
            out_s = _safe_stoi(enhanced, clean, SAMPLE_RATE)
            if in_s is not None and out_s is not None:
                stois.append(out_s)
                stoi_imps.append(out_s - in_s)

            in_p = _safe_pesq(noisy, clean, SAMPLE_RATE)
            out_p = _safe_pesq(enhanced, clean, SAMPLE_RATE)
            if in_p is not None and out_p is not None:
                pesqs.append(out_p)
                pesq_imps.append(out_p - in_p)

    results = {
        "model_type": "Quantized INT8 (GRU+Linear)",
        "source_checkpoint": str(checkpoint_path),
        "checkpoint_mb": round(file_size_mb, 2),
        "mean_input_snr_db": round(float(np.mean(input_snrs)), 4),
        "mean_output_snr_db": round(float(np.mean(output_snrs)), 4),
        "mean_snr_improvement_db": round(float(np.mean(snr_imps)), 4),
        "mean_input_sisnr_db": round(float(np.mean(input_sisnrs)), 4),
        "mean_output_sisnr_db": round(float(np.mean(output_sisnrs)), 4),
        "mean_sisnr_improvement_db": round(float(np.mean(sisnr_imps)), 4),
        "mean_stoi": round(float(np.mean(stois)), 4) if stois else 0.0,
        "mean_stoi_improvement": round(float(np.mean(stoi_imps)), 4) if stoi_imps else 0.0,
        "mean_pesq": round(float(np.mean(pesqs)), 4) if pesqs else 0.0,
        "mean_pesq_improvement": round(float(np.mean(pesq_imps)), 4) if pesq_imps else 0.0,
        "mean_rms_ratio": round(float(np.mean(rms_ratios)), 4),
        "mean_rtf": round(float(np.mean(rtfs)), 4),
        "mean_latency_ms": round(float(np.mean(latencies_ms)), 4),
    }

    print("\n=== Quantized Model Metrics ===")
    print(f"  Checkpoint Size: {file_size_mb:.2f} MB")
    print(f"  SNR Improvement: {results['mean_snr_improvement_db']:+.2f} dB")
    print(f"  SI-SNR Improvement: {results['mean_sisnr_improvement_db']:+.2f} dB")
    print(f"  STOI: {results['mean_stoi']:.4f} ({results['mean_stoi_improvement']:+.4f})")
    print(f"  PESQ: {results['mean_pesq']:.4f} ({results['mean_pesq_improvement']:+.4f})")
    print(f"  RTF: {results['mean_rtf']:.4f}")

    csv_path = output_dir / "quantization_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results.keys()))
        writer.writeheader()
        writer.writerow(results)

    json_path = output_dir / "quantization_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Summary CSV:  {csv_path}")
    print(f"Summary JSON: {json_path}")


if __name__ == "__main__":
    main()
