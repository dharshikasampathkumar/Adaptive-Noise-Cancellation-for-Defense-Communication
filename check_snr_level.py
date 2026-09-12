import pandas as pd
import numpy as np

# ============================================================
# PATH TO YOUR METADATA CSV
# ============================================================

CSV_PATH = r"C:\SIH\synthetic_dataset\metadata\dataset_metadata.csv"

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

print("Total samples:", len(df))
print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# SNR LEVELS TO CHECK
# ============================================================

SNR_LEVELS = [-5, 0, 5, 10, 15, 20]

# Acceptable error range
TOLERANCE = 1.0


# ============================================================
# CHECK EACH SNR LEVEL
# ============================================================

results = []

for target in SNR_LEVELS:

    subset = df[df["target_snr_db"] == target]

    measured = subset["measured_snr_db"].dropna()

    if len(measured) == 0:
        print(f"\nNo data found for {target} dB")
        continue

    average_snr = measured.mean()
    minimum_snr = measured.min()
    maximum_snr = measured.max()

    average_error = subset["snr_error_db"].abs().mean()
    maximum_error = subset["snr_error_db"].abs().max()

    # PASS if actual SNR is within ±1 dB of target
    passed = (
        (subset["measured_snr_db"] >= target - TOLERANCE) &
        (subset["measured_snr_db"] <= target + TOLERANCE)
    )

    pass_count = passed.sum()
    fail_count = len(subset) - pass_count

    results.append({
        "Target SNR (dB)": target,
        "Number of Files": len(subset),
        "Average Actual SNR (dB)": average_snr,
        "Minimum Actual SNR (dB)": minimum_snr,
        "Maximum Actual SNR (dB)": maximum_snr,
        "Mean Absolute Error (dB)": average_error,
        "Maximum Error (dB)": maximum_error,
        "PASS": pass_count,
        "FAIL": fail_count
    })


# ============================================================
# DISPLAY RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 110)
print("SNR VERIFICATION RESULTS")
print("=" * 110)

print(results_df.to_string(index=False))

print("\n")
print("=" * 110)
print("OVERALL RESULTS")
print("=" * 110)

overall_average = df["measured_snr_db"].mean()
overall_error = df["snr_error_db"].abs().mean()

print(f"Total samples              : {len(df)}")
print(f"Overall average actual SNR : {overall_average:.4f} dB")
print(f"Overall mean absolute error: {overall_error:.4f} dB")

# Overall PASS / FAIL
overall_pass = (
    abs(df["measured_snr_db"] - df["target_snr_db"]) <= TOLERANCE
)

print(f"Samples within ±{TOLERANCE} dB : {overall_pass.sum()}")
print(f"Samples outside ±{TOLERANCE} dB: {(~overall_pass).sum()}")
print(f"Overall pass percentage    : {overall_pass.mean()*100:.2f}%")