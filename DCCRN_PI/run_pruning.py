"""run_pruning.py — Progressive Unstructured Weight Pruning for Experiment 3 DCCRN.

Evaluates 10%, 20%, 30%, 40% L1-unstructured weight pruning on the held-out 100-sample test set.
Performs global magnitude pruning across target weight tensors (GRU, Linear, Conv2d, ConvTranspose2d).
Saves pruned checkpoints with pruning masks baked in (using prune.remove) so state_dict remains standard.

Outputs:
    C:/SIH/Experiment_03_Identity_Residual/results/pruning/pruned_10pct.pth
    C:/SIH/Experiment_03_Identity_Residual/results/pruning/pruned_20pct.pth
    C:/SIH/Experiment_03_Identity_Residual/results/pruning/pruned_30pct.pth
    C:/SIH/Experiment_03_Identity_Residual/results/pruning/pruned_40pct.pth
    C:/SIH/Experiment_03_Identity_Residual/results/pruning/pruning_comparison.csv
    C:/SIH/Experiment_03_Identity_Residual/results/pruning/pruning_report.json
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
from typing import Dict, List, Tuple, Any

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dccrn_model import TinyDCCRN, load_checkpoint
from dccrn_interface import DCCRNInference, SAMPLE_RATE
from test_dataset_evaluation import _waveform_snr, _si_snr, _check_perceptual_validity, _safe_stoi, _safe_pesq


def get_prunable_parameters(model: nn.Module) -> List[Tuple[nn.Module, str]]:
    """Collect all target weight parameters in GRU, Linear, Conv2d, ConvTranspose2d modules."""
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            if hasattr(module, 'weight') and module.weight is not None:
                parameters_to_prune.append((module, 'weight'))
        elif isinstance(module, (nn.GRU, nn.LSTM)):
            for param_name, _ in module.named_parameters():
                if 'weight' in param_name:
                    parameters_to_prune.append((module, param_name))
    return parameters_to_prune


def count_sparsity(model: nn.Module) -> Tuple[int, int, float]:
    """Count total weights, non-zero weights, and overall sparsity percentage."""
    total_params = 0
    zero_params = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            zero_params += int((param == 0).sum().item())
    nonzero_params = total_params - zero_params
    sparsity_pct = 100.0 * zero_params / max(total_params, 1)
    return total_params, nonzero_params, sparsity_pct


def apply_global_pruning(model: nn.Module, amount: float) -> nn.Module:
    """Apply global L1 unstructured pruning to the specified sparsity amount (0.0 to 1.0)."""
    params_to_prune = get_prunable_parameters(model)
    if not params_to_prune:
        raise ValueError("No prunable parameters found in model!")

    prune.global_unstructured(
        params_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    for module, param_name in params_to_prune:
        prune.remove(module, param_name)

    return model


def evaluate_model_on_test_set(
    model: nn.Module,
    test_csv_path: Path,
    dataset_root: Path,
    num_warmup: int = 5,
) -> Dict[str, Any]:
    """Evaluate a DCCRN model on the test dataset using exact test_dataset_evaluation logic."""
    model.eval()
    engine = DCCRNInference(model=model, device=torch.device("cpu"))

    with open(test_csv_path, "r", encoding="utf-8") as f:
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
    for _ in range(num_warmup):
        engine.enhance_array(dummy)

    for idx, row in enumerate(samples, 1):
        sample_id = row.get("sample_id", str(idx))
        target_snr = float(row.get("target_snr_db", float("nan")))

        snr_int = int(target_snr) if math.isfinite(target_snr) else 0
        noisy_path = dataset_root / "noisy_speech" / f"snr_{snr_int}dB" / row["noisy_filename"]
        gen_clean  = dataset_root / "clean_reference" / f"{sample_id}_clean_ref.wav"
        src_clean  = dataset_root / "clean_reference" / row["clean_filename"]
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

        # Enhance
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

    return {
        "mean_input_snr_db": float(np.mean(input_snrs)),
        "mean_output_snr_db": float(np.mean(output_snrs)),
        "mean_snr_improvement_db": float(np.mean(snr_imps)),
        "mean_input_sisnr_db": float(np.mean(input_sisnrs)),
        "mean_output_sisnr_db": float(np.mean(output_sisnrs)),
        "mean_sisnr_improvement_db": float(np.mean(sisnr_imps)),
        "mean_stoi": float(np.mean(stois)) if stois else 0.0,
        "mean_stoi_improvement": float(np.mean(stoi_imps)) if stoi_imps else 0.0,
        "mean_pesq": float(np.mean(pesqs)) if pesqs else 0.0,
        "mean_pesq_improvement": float(np.mean(pesq_imps)) if pesq_imps else 0.0,
        "mean_rms_ratio": float(np.mean(rms_ratios)),
        "mean_rtf": float(np.mean(rtfs)),
        "mean_latency_ms": float(np.mean(latencies_ms)),
    }


def main():
    parser = argparse.ArgumentParser(description="Pruning evaluation for Exp3 DCCRN")
    parser.add_argument("--checkpoint", type=str, default="C:/SIH/Experiment_03_Identity_Residual/checkpoints/Experiment_03_best.pth")
    parser.add_argument("--test-dataset", type=str, default="C:/SIH/synthetic_dataset")
    parser.add_argument("--output-dir", type=str, default="C:/SIH/Experiment_03_Identity_Residual/results/pruning")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    dataset_dir = Path(args.test_dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_csv = dataset_dir / "splits" / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found at {test_csv}")

    sparsity_levels = [0.10, 0.20, 0.30, 0.40]
    results = []

    print(f"=== Starting Pruning Evaluation ===")
    print(f"Source checkpoint: {checkpoint_path}")
    print(f"Target ratios: {sparsity_levels}\n")

    for level in sparsity_levels:
        pct = int(level * 100)
        print(f"--- Processing {pct}% Pruning ---")
        
        model, ckpt = load_checkpoint(checkpoint_path, device="cpu")
        stft_config = ckpt.get("stft_config", {})

        model = apply_global_pruning(model, amount=level)
        total_p, nonzero_p, actual_sparsity = count_sparsity(model)
        
        pruned_ckpt_path = output_dir / f"pruned_{pct}pct.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "stft_config": stft_config,
            "sparsity_target": level,
            "actual_sparsity_pct": actual_sparsity,
            "total_params": total_p,
            "nonzero_params": nonzero_p,
        }, pruned_ckpt_path)
        
        file_size_mb = os.path.getsize(pruned_ckpt_path) / (1024 * 1024)

        eval_metrics = evaluate_model_on_test_set(
            model=model,
            test_csv_path=test_csv,
            dataset_root=dataset_dir
        )

        res_row = {
            "pruning_level_pct": pct,
            "target_sparsity": level,
            "actual_sparsity_pct": round(actual_sparsity, 2),
            "total_params": total_p,
            "nonzero_params": nonzero_p,
            "checkpoint_mb": round(file_size_mb, 2),
            **eval_metrics
        }
        results.append(res_row)

        print(f"  Actual Sparsity: {actual_sparsity:.2f}% | Non-zero params: {nonzero_p:,}")
        print(f"  SNR Imp: {eval_metrics['mean_snr_improvement_db']:+.2f} dB | SI-SNR Imp: {eval_metrics['mean_sisnr_improvement_db']:+.2f} dB")
        print(f"  STOI: {eval_metrics['mean_stoi']:.3f} ({eval_metrics['mean_stoi_improvement']:+.3f}) | PESQ: {eval_metrics['mean_pesq']:.2f} ({eval_metrics['mean_pesq_improvement']:+.3f})")
        print(f"  RTF: {eval_metrics['mean_rtf']:.4f} | Saved: {pruned_ckpt_path}\n")

    csv_path = output_dir / "pruning_comparison.csv"
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    json_path = output_dir / "pruning_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"=== Pruning Evaluation Complete ===")
    print(f"Saved summary CSV:  {csv_path}")
    print(f"Saved summary JSON: {json_path}")


if __name__ == "__main__":
    main()
