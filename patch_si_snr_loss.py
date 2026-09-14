"""
PROPOSED MINIMUM FIX — round-3 audit, section 6/9.

Drop-in replacement for si_snr_loss() in dccrn_train.py. This is the ONLY
function this patch touches. Nothing else in the file changes.

WHAT CHANGES: the SI-SNR projection scalar alpha = <estimate,target>/<target,
target> is clamped to be non-negative before it is used to build the
"target component" of the SI-SNR decomposition.

WHY: verified empirically (see AUDIT_REPORT_ROUND3.md, sections 1-3) that
the Experiment-2/3 checkpoint's enhanced output is strongly ANTI-correlated
with clean speech (Pearson r = -0.78 on the fixed diagnostic sample,
confirmed structurally across all 100 test-set samples via the negative
correlation between rms_ratio and waveform_output_snr_db, r = -0.84). The
original si_snr_loss formula scores a sign-flipped reconstruction exactly
as well as a correctly-signed one of the same magnitude (verified
numerically below) because the (alpha^2) term in the SI-SNR ratio does not
depend on the sign of alpha. Clamping alpha >= 0 leaves every legitimately
positive-correlation case completely unchanged (see verification below)
and makes negative-correlation solutions score catastrophically, which is
the necessary condition for gradient descent to have a real incentive to
avoid that region instead of getting trapped in it early in training (see
report section 4 - the trap forms by epoch 2-3 in the current run).

VERIFIED EFFECT (synthetic test, see report appendix):
  correctly-signed estimate : original loss -15.621  ->  fixed loss -15.621  (UNCHANGED)
  sign-flipped estimate     : original loss -15.611  ->  fixed loss +121.286 (correctly penalised)

IMPORTANT — READ BEFORE RE-RUNNING:
  - This changes the numeric meaning of si_snr_loss / compute_si_snr_metric
    for any future checkpoint (it now enforces the sign convention). Do NOT
    directly compare a future SI-SNR number to Experiment 2/3's reported
    SI-SNR without noting this - they are not computed with the same
    formula. For a fair comparison, recompute Experiment 2/3's checkpoint
    under this same fixed formula first.
  - No other file, weight, architecture, or hyperparameter is touched by
    this patch, per the brief's explicit "do not change anything else yet"
    instruction. Run the smoke test / short training test after applying
    before committing to a full run.
"""

import torch
from torch import Tensor


def si_snr_loss(estimate: Tensor, target: Tensor) -> Tensor:
    """SI-SNR loss (lower is better; used for gradient optimisation).

    ROOT-CAUSE FIX (round-3 audit): the projection scalar
        alpha = <estimate, target> / <target, target>
    is now clamped to be non-negative. A negative alpha means the model's
    output is correlated with -clean rather than clean - a polarity
    inversion - which the original (unclamped) formula scored identically
    to a correctly-signed reconstruction of the same magnitude, because the
    ratio depends on alpha^2, not alpha. Clamping removes that loophole:
    a negatively-correlated estimate now scores as though it contains
    almost no usable signal (projection -> 0), instead of scoring as well
    as a correct reconstruction.
    """
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    dot = (estimate * target).sum(-1, keepdim=True)
    target_energy = target.pow(2).sum(-1, keepdim=True) + 1e-8
    alpha = torch.clamp(dot / target_energy, min=0.0)  # <-- the fix
    projection = alpha * target
    noise = estimate - projection
    score = 10 * torch.log10(
        (projection.pow(2).sum(-1) + 1e-8) / (noise.pow(2).sum(-1) + 1e-8)
    )
    return -score.mean()


if __name__ == "__main__":
    # Self-contained verification, matches the numbers quoted above.
    torch.manual_seed(0)
    target = torch.randn(1, 16000)
    est_good = 0.9 * target + 0.15 * torch.randn(1, 16000)
    est_flipped = -0.9 * target + 0.15 * torch.randn(1, 16000)

    def original(estimate, target):
        estimate = estimate - estimate.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)
        projection = (estimate * target).sum(-1, keepdim=True) * target
        projection = projection / (target.pow(2).sum(-1, keepdim=True) + 1e-8)
        noise = estimate - projection
        score = 10 * torch.log10((projection.pow(2).sum(-1) + 1e-8) / (noise.pow(2).sum(-1) + 1e-8))
        return -score.mean()

    for name, est in [("correctly-signed", est_good), ("sign-flipped", est_flipped)]:
        print(f"{name}: original={original(est, target).item():.3f}  fixed={si_snr_loss(est, target).item():.3f}")