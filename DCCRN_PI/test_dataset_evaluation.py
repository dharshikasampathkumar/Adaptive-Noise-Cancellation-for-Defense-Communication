"""test_dataset_evaluation.py — FP32 Experiment 3 baseline evaluation.

IMPORTANT — DATA SPLIT RULE
----------------------------
    TRAIN      : fine-tuning only
    VALIDATION : model selection (checkpoint is already chosen before this script runs)
    TEST        : THIS SCRIPT ONLY — never used for selection

This script establishes the FP32 baseline against which pruning and
quantization are compared.  It must be run ONCE on the best validation
checkpoint.  Never run it repeatedly to search for a better epoch.

Usage
-----
    python test_dataset_evaluation.py \\
        --checkpoint C:/SIH/Experiment_03_Identity_Residual/checkpoints/Experiment_03_best.pth \\
        --test-dataset C:/SIH/synthetic_dataset \\
        --output-dir  C:/SIH/Experiment_03_Identity_Residual/results/fp32_baseline

The script:
    1.  Loads the Experiment 3 architecture from the checkpoint.
    2.  Reads test.csv from <dataset>/splits/test.csv.
    3.  Evaluates every sample once (no gradients, model never updated).
    4.  Saves:
            per_sample_results.csv
            snr_group_summary.csv
            noise_category_summary.csv     (if metadata present)
            overall_summary.json
    5.  Prints an overall summary table.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch

_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dccrn_model import load_checkpoint, DEFAULT_SAMPLE_RATE
from dccrn_interface import DCCRNInference, SAMPLE_RATE


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _waveform_snr(clean: np.ndarray, estimate: np.ndarray) -> float:
    c = clean.astype(np.float64)
    e = estimate.astype(np.float64)
    return float(10.0 * np.log10(
        (np.mean(c**2) + 1e-12) / (np.mean((e - c)**2) + 1e-12)
    ))


def _si_snr(estimate: np.ndarray, target: np.ndarray) -> float:
    e = estimate.astype(np.float64) - estimate.mean()
    t = target.astype(np.float64)  - target.mean()
    dot = np.dot(e, t)
    t_e = np.dot(t, t) + 1e-8
    proj = max(dot / t_e, 0.0) * t
    noise = e - proj
    return float(10.0 * np.log10(
        (np.dot(proj, proj) + 1e-8) / (np.dot(noise, noise) + 1e-8)
    ))


def _check_perceptual_validity(clean: np.ndarray, min_rms: float = 1e-4,
                                activity_threshold: float = 0.01,
                                min_activity_fraction: float = 0.05) -> bool:
    """Return True if the segment has enough active speech for STOI/PESQ."""
    rms = float(np.sqrt(np.mean(clean.astype(np.float64)**2)))
    if rms < min_rms:
        return False
    peak = float(np.max(np.abs(clean)))
    if peak < 1e-6:
        return False
    active = float(np.mean(np.abs(clean) > activity_threshold * peak))
    return active >= min_activity_fraction


def _safe_stoi(estimate: np.ndarray, clean: np.ndarray, sr: int) -> Optional[float]:
    try:
        import warnings
        from pystoi import stoi  # type: ignore
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = float(stoi(clean, estimate, sr, extended=False))
        return None if v < 1e-4 else v
    except Exception:
        return None


def _safe_pesq(estimate: np.ndarray, clean: np.ndarray, sr: int) -> Optional[float]:
    try:
        from pesq import pesq  # type: ignore
        return float(pesq(sr, clean, estimate, "wb"))
    except Exception:
        return None


def _mean(values: List[float]) -> Optional[float]:
    fin = [v for v in values if math.isfinite(v)]
    return float(sum(fin) / len(fin)) if fin else None


def _write_csv(path: Path, rows: List[Dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model -------------------------------------------------------
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, ckpt_meta = load_checkpoint(args.checkpoint, device=device, strict=True)
    print(f"  Epoch : {ckpt_meta.get('epoch', '?')}")
    print(f"  Val loss : {ckpt_meta.get('best_validation_loss', '?'):.6f}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {total_params:,}")

    inf = DCCRNInference(model=model, device=device)

    # --- Load test split ---------------------------------------------------
    dataset_root = Path(args.test_dataset)
    test_csv = dataset_root / "splits" / "test.csv"
    if not test_csv.exists():
        print(f"ERROR: test split not found: {test_csv}")
        sys.exit(1)

    with test_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"\nTest samples : {len(rows)}")
    print("IMPORTANT: test set used for final evaluation ONLY, not model selection.")

    # --- Evaluate ----------------------------------------------------------
    results: List[Dict] = []
    stoi_valid = stoi_invalid = 0
    pesq_valid = pesq_invalid = 0

    for idx, row in enumerate(rows, 1):
        sample_id = row.get("sample_id", str(idx))
        target_snr = float(row.get("target_snr_db", float("nan")))
        noise_cat  = row.get("noise_category", row.get("noise_type", "unknown"))

        # Resolve paths
        snr_int = int(target_snr) if math.isfinite(target_snr) else 0
        noisy_path = dataset_root / "noisy_speech" / f"snr_{snr_int}dB" / row["noisy_filename"]
        gen_clean  = dataset_root / "clean_reference" / f"{sample_id}_clean_ref.wav"
        src_clean  = dataset_root / "clean_reference" / row["clean_filename"]
        clean_path = gen_clean if gen_clean.exists() else (src_clean if src_clean.exists() else None)

        if not noisy_path.exists() or clean_path is None:
            print(f"  [SKIP {idx:4d}] Missing files for {sample_id}")
            continue

        # Load
        noisy, sr = sf.read(str(noisy_path), dtype="float32", always_2d=True)
        clean, _  = sf.read(str(clean_path), dtype="float32", always_2d=True)
        noisy = noisy.mean(axis=1);  clean = clean.mean(axis=1)
        L = min(len(noisy), len(clean))
        noisy, clean = noisy[:L], clean[:L]

        # Segment to 2 s (pad if shorter)
        seg = int(SAMPLE_RATE * 2.0)
        if L >= seg:
            noisy, clean = noisy[:seg], clean[:seg]
        else:
            noisy = np.pad(noisy, (0, seg - L))
            clean = np.pad(clean, (0, seg - L))

        # Enhance
        t0 = time.perf_counter()
        enhanced = inf.enhance_array(noisy, SAMPLE_RATE)
        infer_sec = time.perf_counter() - t0

        # Waveform SNR
        in_wsnr  = _waveform_snr(clean, noisy)
        out_wsnr = _waveform_snr(clean, enhanced)

        # SI-SNR
        in_sisnr  = _si_snr(noisy,    clean)
        out_sisnr = _si_snr(enhanced, clean)

        # RMS ratio
        clean_rms = float(np.sqrt(np.mean(clean.astype(np.float64)**2)))
        enh_rms   = float(np.sqrt(np.mean(enhanced.astype(np.float64)**2)))
        rms_ratio = enh_rms / (clean_rms + 1e-12)

        # Clipping
        clip_pct = float(100.0 * np.mean(np.abs(enhanced) > 1.0))

        # Perceptual metrics (skip if silent)
        valid_percep = _check_perceptual_validity(clean)
        in_stoi = out_stoi = in_pesq = out_pesq = float("nan")
        if valid_percep:
            v_in_stoi = _safe_stoi(noisy, clean, SAMPLE_RATE)
            v_out_stoi = _safe_stoi(enhanced, clean, SAMPLE_RATE)
            if v_in_stoi is not None and v_out_stoi is not None:
                in_stoi, out_stoi = v_in_stoi, v_out_stoi
                stoi_valid += 1
            else:
                stoi_invalid += 1
            v_in_pesq = _safe_pesq(noisy, clean, SAMPLE_RATE)
            v_out_pesq = _safe_pesq(enhanced, clean, SAMPLE_RATE)
            if v_in_pesq is not None and v_out_pesq is not None:
                in_pesq, out_pesq = v_in_pesq, v_out_pesq
                pesq_valid += 1
            else:
                pesq_invalid += 1
        else:
            stoi_invalid += 1
            pesq_invalid += 1

        # metadata measured snr
        meta_snr = float(row.get("measured_snr_db", row.get("metadata_measured_snr_db", float("nan"))))

        results.append({
            "sample_id":                  sample_id,
            "target_snr_db":              target_snr,
            "metadata_measured_snr_db":   meta_snr,
            "noise_category":             noise_cat,
            "waveform_input_snr_db":      round(in_wsnr,  4),
            "waveform_output_snr_db":     round(out_wsnr, 4),
            "waveform_snr_improvement_db":round(out_wsnr - in_wsnr, 4),
            "input_si_snr_db":            round(in_sisnr,  4),
            "output_si_snr_db":           round(out_sisnr, 4),
            "si_snr_improvement_db":      round(out_sisnr - in_sisnr, 4),
            "input_stoi":                 round(in_stoi,  6) if math.isfinite(in_stoi)  else "",
            "output_stoi":                round(out_stoi, 6) if math.isfinite(out_stoi) else "",
            "stoi_improvement":           round(out_stoi - in_stoi, 6) if math.isfinite(in_stoi) and math.isfinite(out_stoi) else "",
            "input_pesq":                 round(in_pesq,  6) if math.isfinite(in_pesq)  else "",
            "output_pesq":                round(out_pesq, 6) if math.isfinite(out_pesq) else "",
            "pesq_improvement":           round(out_pesq - in_pesq, 6) if math.isfinite(in_pesq) and math.isfinite(out_pesq) else "",
            "rms_ratio_enhanced_over_clean": round(rms_ratio, 6),
            "clipping_pct":               round(clip_pct, 4),
            "valid_percep":               int(valid_percep),
            "inference_sec":              round(infer_sec, 4),
        })

        if idx % 10 == 0 or idx == len(rows):
            print(f"  [{idx:4d}/{len(rows)}]  SNR imp={out_wsnr-in_wsnr:+.2f} dB  "
                  f"STOI {in_stoi:.4f}->{out_stoi:.4f}" if math.isfinite(in_stoi) else
                  f"  [{idx:4d}/{len(rows)}]  SNR imp={out_wsnr-in_wsnr:+.2f} dB")

    # --- Save per-sample CSV ----------------------------------------------
    fields = list(results[0].keys()) if results else []
    _write_csv(output_dir / "per_sample_results.csv", results, fields)

    # --- Group summaries --------------------------------------------------
    def _group_summary(results: List[Dict], key: str, out_path: Path) -> List[Dict]:
        groups: Dict[str, List[Dict]] = defaultdict(list)
        for r in results:
            groups[str(r.get(key, "?"))].append(r)
        summary = []
        metric_fields = [
            "waveform_input_snr_db", "waveform_output_snr_db", "waveform_snr_improvement_db",
            "input_si_snr_db", "output_si_snr_db", "si_snr_improvement_db",
            "input_stoi", "output_stoi", "stoi_improvement",
            "input_pesq", "output_pesq", "pesq_improvement",
            "rms_ratio_enhanced_over_clean", "clipping_pct",
        ]
        for grp_key in sorted(groups.keys(), key=lambda s: (float(s) if s.replace('.','',1).replace('-','',1).lstrip().isdigit() else 0, s)):
            items = groups[grp_key]
            row: Dict = {key: grp_key, "sample_count": len(items)}
            for mf in metric_fields:
                vals = [float(r[mf]) for r in items if r.get(mf) not in ("", None) and str(r.get(mf, "")).strip() != ""]
                row[f"mean_{mf}"] = round(_mean(vals) or float("nan"), 4)
            summary.append(row)
        _write_csv(out_path, summary, list(summary[0].keys()) if summary else [])
        return summary

    snr_summary = _group_summary(results, "target_snr_db",
                                 output_dir / "snr_group_summary.csv")
    cat_summary = _group_summary(results, "noise_category",
                                 output_dir / "noise_category_summary.csv")

    # --- Overall summary --------------------------------------------------
    def _agg(field: str) -> Optional[float]:
        vals = [float(r[field]) for r in results
                if r.get(field) not in ("", None) and str(r.get(field, "")).strip() != ""]
        return round(_mean(vals), 6) if vals else None

    overall = {
        "checkpoint":             args.checkpoint,
        "model_type":             "Experiment_3_TinyDCCRN_CRM",
        "total_parameters":       total_params,
        "fp32_weight_size_mb":    round(total_params * 4 / (1024**2), 4),
        "n_samples_evaluated":    len(results),
        "stoi_valid":             stoi_valid,
        "stoi_invalid":           stoi_invalid,
        "pesq_valid":             pesq_valid,
        "pesq_invalid":           pesq_invalid,
        "mean_waveform_input_snr_db":  _agg("waveform_input_snr_db"),
        "mean_waveform_output_snr_db": _agg("waveform_output_snr_db"),
        "mean_waveform_snr_improvement_db": _agg("waveform_snr_improvement_db"),
        "mean_input_si_snr_db":   _agg("input_si_snr_db"),
        "mean_output_si_snr_db":  _agg("output_si_snr_db"),
        "mean_si_snr_improvement_db": _agg("si_snr_improvement_db"),
        "mean_input_stoi":        _agg("input_stoi"),
        "mean_output_stoi":       _agg("output_stoi"),
        "mean_stoi_improvement":  _agg("stoi_improvement"),
        "mean_input_pesq":        _agg("input_pesq"),
        "mean_output_pesq":       _agg("output_pesq"),
        "mean_pesq_improvement":  _agg("pesq_improvement"),
        "mean_rms_ratio":         _agg("rms_ratio_enhanced_over_clean"),
        "mean_clipping_pct":      _agg("clipping_pct"),
        "fp32_baseline_note": (
            "This is the FP32 baseline. Compare pruned/quantized models against these numbers. "
            "Test set was NOT used for model selection."
        ),
    }

    with (output_dir / "overall_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(overall, fh, indent=2)

    # --- Print summary ----------------------------------------------------
    print("\n" + "="*60)
    print("  FP32 EXPERIMENT 3 BASELINE -- OVERALL")
    print("="*60)
    for k, v in overall.items():
        if k not in ("checkpoint", "fp32_baseline_note", "model_type"):
            print(f"  {k:<42}: {v}")

    print("\nSNR GROUP BREAKDOWN:")
    for r in snr_summary:
        print(
            f"  {r['target_snr_db']:>5} dB | n={r['sample_count']:>3}"
            f" | SNR imp={r.get('mean_waveform_snr_improvement_db', float('nan')):+.2f} dB"
            f" | STOI {r.get('mean_input_stoi', float('nan')):.4f}->{r.get('mean_output_stoi', float('nan')):.4f}"
            f" | PESQ {r.get('mean_input_pesq', float('nan')):.3f}->{r.get('mean_output_pesq', float('nan')):.3f}"
        )

    print(f"\nResults saved to: {output_dir}")
    print(f"  per_sample_results.csv")
    print(f"  snr_group_summary.csv")
    print(f"  noise_category_summary.csv")
    print(f"  overall_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Experiment 3 FP32 checkpoint on the held-out test set."
    )
    parser.add_argument("--checkpoint",   required=True,
                        help="Path to Experiment_03_best.pth")
    parser.add_argument("--test-dataset", required=True, dest="test_dataset",
                        help="Root of the synthetic dataset (contains splits/test.csv)")
    parser.add_argument("--output-dir",   required=True, dest="output_dir",
                        help="Directory to write evaluation outputs")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
