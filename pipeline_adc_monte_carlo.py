# -*- coding: utf-8 -*-
"""
4-Stage Pipeline ADC -- Monte Carlo (100 runs)
- 2 bits per stage, no redundancy, 8-bit total
- Comparator offsets:  Gaussian sigma = 30 mV
- Residue amp gain mismatch: Gaussian sigma = 30 mV/V
- VECTORIZED for performance
- Averages SNDR over 100 independent random draws
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# == ADC constants ===========================================================
VREF        = 1.0
N_STAGES    = 4
BITS_STAGE  = 2
GAIN_IDEAL  = 2 ** BITS_STAGE       # 4 V/V
TOTAL_BITS  = N_STAGES * BITS_STAGE  # 8
LEVELS      = 2 ** TOTAL_BITS        # 256
LSB         = 2 * VREF / LEVELS      # 7.8125 mV

OFFSET_SIGMA    = 0.030             # 30 mV comparator offset std-dev
GAIN_MIS_SIGMA  = 0.030             # 30 mV/V residue amp gain-error std-dev
N_COMP          = 3                 # comparators per stage (2-bit flash)

NOMINAL_THR = np.array([-VREF / 2, 0.0, VREF / 2])
DAC_LEVELS  = np.array([-3 * VREF / 4, -VREF / 4, VREF / 4, 3 * VREF / 4])

# == Simulation parameters ===================================================
FS    = 1000000
N_FFT = 4096
M     = 97
fin   = M * FS / N_FFT
t     = np.arange(N_FFT, dtype=np.float64) / FS

N_RUNS    = 100                          # Monte Carlo iterations
AMP_PTS   = 40                           # amplitude sweep points (reduced for speed)
amp_dB    = np.linspace(-60, -0.05, AMP_PTS)
IDEAL_PEAK = 6.02 * TOTAL_BITS + 1.76


# == VECTORIZED stage model ==================================================
def _stage_vec(vin_arr, thr, gain):
    """Process entire array through one pipeline stage (vectorized)."""
    codes = np.zeros(len(vin_arr), dtype=np.int32)
    codes += (vin_arr >= thr[0]).astype(np.int32)
    codes += (vin_arr >= thr[1]).astype(np.int32)
    codes += (vin_arr >= thr[2]).astype(np.int32)
    codes = np.clip(codes, 0, 3)
    res = (vin_arr - DAC_LEVELS[codes]) * gain
    return codes, res


def pipeline_adc_vec(x, thresholds, gains):
    """
    Vectorized pipeline ADC -- processes all samples simultaneously.
    thresholds: (N_STAGES, 3) array of comparator thresholds
    gains:      (N_STAGES,) array of residue amplifier gains
    """
    x = np.asarray(x, dtype=np.float64)
    digital = np.zeros(len(x), dtype=np.int32)
    residue = x.copy()

    for s in range(N_STAGES):
        residue = np.clip(residue, -VREF, VREF)
        sc, residue = _stage_vec(residue, thresholds[s], gains[s])
        digital = (digital << BITS_STAGE) | sc

    return digital


def code2volt(c):
    """Convert integer codes to voltage."""
    return (2 * c - (LEVELS - 1)) / float(LEVELS) * VREF


# == SNDR computation ========================================================
def compute_sndr(sig, fs):
    """Compute SNDR using a Hann-windowed FFT with 3-bin signal lobe."""
    N = len(sig)
    win = np.hanning(N)
    W = np.sum(win ** 2)
    spec = np.fft.rfft(sig * win)
    pwr = np.abs(spec) ** 2 / W
    sb = int(np.argmax(pwr[1:])) + 1
    lobe = [b for b in (sb - 1, sb, sb + 1) if 0 < b < len(pwr)]
    Ps = np.sum(pwr[lobe])
    mask = np.ones(len(pwr), dtype=bool)
    mask[0] = False
    for b in lobe:
        mask[b] = False
    Pn = np.sum(pwr[mask])
    sndr = 10.0 * np.log10(Ps / max(Pn, 1e-30))
    return sndr


# == Monte Carlo loop ========================================================
# Storage for results
sndr_mc_ideal    = np.zeros((N_RUNS, AMP_PTS))
sndr_mc_offsets  = np.zeros((N_RUNS, AMP_PTS))
sndr_mc_off_gain = np.zeros((N_RUNS, AMP_PTS))

# Ideal thresholds and gains (constant across runs)
ideal_thr  = np.tile(NOMINAL_THR, (N_STAGES, 1))   # (4, 3)
ideal_gain = np.full(N_STAGES, GAIN_IDEAL)          # (4,)

print("=" * 60)
print("  MONTE CARLO: {} runs x {} amplitude points".format(N_RUNS, AMP_PTS))
print("  sigma_offset = {:.0f} mV    sigma_gain = {:.0f} mV/V".format(
    OFFSET_SIGMA * 1e3, GAIN_MIS_SIGMA * 1e3))
print("=" * 60)

for run in range(N_RUNS):
    rng = np.random.default_rng(run + 1000)

    # Draw new random offsets and gain errors for this run
    mc_comp_offsets = rng.normal(0.0, OFFSET_SIGMA, size=(N_STAGES, N_COMP))
    mc_gain_errors  = rng.normal(0.0, GAIN_MIS_SIGMA, size=N_STAGES)
    mc_eff_thr      = NOMINAL_THR[np.newaxis, :] + mc_comp_offsets  # (4, 3)
    mc_eff_gain     = GAIN_IDEAL + mc_gain_errors                    # (4,)

    # Thresholds for offsets-only mode (ideal gains)
    for j, a in enumerate(10.0 ** (amp_dB / 20.0)):
        si = a * np.sin(2.0 * np.pi * fin * t)

        # Ideal (no errors)
        c0 = pipeline_adc_vec(si, ideal_thr, ideal_gain)
        sndr_mc_ideal[run, j] = compute_sndr(code2volt(c0), FS)

        # Offsets only (ideal gain)
        c1 = pipeline_adc_vec(si, mc_eff_thr, ideal_gain)
        sndr_mc_offsets[run, j] = compute_sndr(code2volt(c1), FS)

        # Offsets + Gain mismatch
        c2 = pipeline_adc_vec(si, mc_eff_thr, mc_eff_gain)
        sndr_mc_off_gain[run, j] = compute_sndr(code2volt(c2), FS)

    if (run + 1) % 10 == 0:
        print("  Completed {}/{} runs".format(run + 1, N_RUNS))

# == Statistics ==============================================================
avg_sndr_ideal    = np.mean(sndr_mc_ideal, axis=0)
avg_sndr_offsets  = np.mean(sndr_mc_offsets, axis=0)
std_sndr_offsets  = np.std(sndr_mc_offsets, axis=0)
avg_sndr_off_gain = np.mean(sndr_mc_off_gain, axis=0)
std_sndr_off_gain = np.std(sndr_mc_off_gain, axis=0)

# Full-scale stats (last amplitude point)
fs_sndr_off  = sndr_mc_offsets[:, -1]
fs_sndr_all  = sndr_mc_off_gain[:, -1]

print("\n" + "-" * 60)
print("  Monte Carlo Full-Scale SNDR Statistics ({} runs)".format(N_RUNS))
print("-" * 60)
print("  +Offsets only:    Mean = {:.2f} dB,  Std = {:.2f} dB,  ENOB = {:.2f} bits".format(
    np.mean(fs_sndr_off), np.std(fs_sndr_off), (np.mean(fs_sndr_off) - 1.76) / 6.02))
print("  +Offsets+Gain:    Mean = {:.2f} dB,  Std = {:.2f} dB,  ENOB = {:.2f} bits".format(
    np.mean(fs_sndr_all), np.std(fs_sndr_all), (np.mean(fs_sndr_all) - 1.76) / 6.02))
print("  Ideal:            {:.2f} dB,  ENOB = {} bits".format(IDEAL_PEAK, TOTAL_BITS))
print("-" * 60)

# == Plot ====================================================================
DARK   = "#0A0E14"
GRID   = "#1C2330"
CYAN   = "#00E5FF"
AMBER  = "#FFB300"
RED    = "#FF4560"
GREEN  = "#00E396"
GHOST  = "#2A3545"
TEXT   = "#CDD6E0"
MUTED  = "#6E7D8C"
ORANGE = "#FF6D00"
PURPLE = "#CE93D8"

plt.rcParams.update({
    "figure.facecolor": DARK,
    "axes.facecolor": DARK,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "text.color": TEXT,
    "font.family": "monospace",
    "font.size": 10,
})

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

fig.suptitle(
    "4-STAGE PIPELINE ADC  |  Monte Carlo ({} runs)  |  "
    "sigma_offset={}mV  |  sigma_gain={}mV/V".format(
        N_RUNS, int(OFFSET_SIGMA * 1e3), int(GAIN_MIS_SIGMA * 1e3)),
    fontsize=12, fontweight="bold", color=CYAN, fontfamily="monospace", y=0.96)

# -- [0,0] Average SNDR vs Amplitude ----------------------------------------
ax = axes[0, 0]
ax.plot(amp_dB, avg_sndr_ideal, color=GHOST, lw=1.5, ls="--",
        label="Ideal {}-bit".format(TOTAL_BITS))
ax.plot(amp_dB, avg_sndr_offsets, color=ORANGE, lw=1.8,
        label="+Offsets (mean)")
ax.fill_between(amp_dB,
                avg_sndr_offsets - std_sndr_offsets,
                avg_sndr_offsets + std_sndr_offsets,
                color=ORANGE, alpha=0.2, label="+/-1 sigma")
ax.plot(amp_dB, avg_sndr_off_gain, color=PURPLE, lw=2.0,
        label="+Offsets+Gain (mean)")
ax.fill_between(amp_dB,
                avg_sndr_off_gain - std_sndr_off_gain,
                avg_sndr_off_gain + std_sndr_off_gain,
                color=PURPLE, alpha=0.2, label="+/-1 sigma")
ax.set_xlabel("Input Amplitude (dBFS)")
ax.set_ylabel("SNDR (dB)")
ax.set_title("Average SNDR vs Input Amplitude ({} runs)".format(N_RUNS), fontsize=10)
ax.legend(fontsize=7.5, framealpha=0.3, loc="upper left")
ax.set_xlim(amp_dB[0], 0)
ax.set_ylim(0, IDEAL_PEAK + 6)
ax.grid(True, alpha=0.3)

# -- [0,1] Histogram of full-scale SNDR -------------------------------------
ax = axes[0, 1]
ax.hist(fs_sndr_off, bins=15, color=ORANGE, alpha=0.6,
        edgecolor=ORANGE, label="+Offsets only")
ax.hist(fs_sndr_all, bins=15, color=PURPLE, alpha=0.6,
        edgecolor=PURPLE, label="+Offsets+Gain")
ax.axvline(np.mean(fs_sndr_off), color=ORANGE, lw=2, ls="--",
           label="Mean={:.1f} dB".format(np.mean(fs_sndr_off)))
ax.axvline(np.mean(fs_sndr_all), color=PURPLE, lw=2, ls="--",
           label="Mean={:.1f} dB".format(np.mean(fs_sndr_all)))
ax.axvline(IDEAL_PEAK, color=CYAN, lw=1.5, ls=":",
           label="Ideal={:.1f} dB".format(IDEAL_PEAK))
ax.set_xlabel("SNDR (dB)")
ax.set_ylabel("Count")
ax.set_title("Full-Scale SNDR Distribution ({} runs)".format(N_RUNS), fontsize=10)
ax.legend(fontsize=7, framealpha=0.3)
ax.grid(True, alpha=0.3)

# -- [1,0] Average ENOB vs Amplitude ----------------------------------------
ax = axes[1, 0]
enob_ideal    = (avg_sndr_ideal - 1.76) / 6.02
enob_avg_off  = (avg_sndr_offsets - 1.76) / 6.02
enob_avg_all  = (avg_sndr_off_gain - 1.76) / 6.02
ax.plot(amp_dB, enob_ideal, color=GHOST, lw=1.5, ls="--",
        label="Ideal {}-bit".format(TOTAL_BITS))
ax.plot(amp_dB, enob_avg_off, color=ORANGE, lw=1.8,
        label="+Offsets (mean)")
ax.plot(amp_dB, enob_avg_all, color=PURPLE, lw=2.0,
        label="+Offsets+Gain (mean)")
ax.axhline(TOTAL_BITS, color=CYAN, lw=0.8, ls=":", alpha=0.5)
ax.set_xlabel("Input Amplitude (dBFS)")
ax.set_ylabel("ENOB (bits)")
ax.set_title("Average ENOB vs Input Amplitude ({} runs)".format(N_RUNS), fontsize=10)
ax.legend(fontsize=7.5, framealpha=0.3, loc="upper left")
ax.set_xlim(amp_dB[0], 0)
ax.set_ylim(0, TOTAL_BITS + 1)
ax.grid(True, alpha=0.3)

# -- [1,1] All individual runs (waterfall) + mean ---------------------------
ax = axes[1, 1]
for run in range(N_RUNS):
    ax.plot(amp_dB, sndr_mc_off_gain[run], color=PURPLE, lw=0.3, alpha=0.15)
ax.plot(amp_dB, avg_sndr_off_gain, color=AMBER, lw=2.5,
        label="Mean SNDR (+Off+Gain)")
ax.plot(amp_dB, avg_sndr_ideal, color=CYAN, lw=1.2, ls="--",
        label="Ideal {}-bit".format(TOTAL_BITS))
ax.set_xlabel("Input Amplitude (dBFS)")
ax.set_ylabel("SNDR (dB)")
ax.set_title("All {} Runs (individual + mean)".format(N_RUNS), fontsize=10)
ax.legend(fontsize=7.5, framealpha=0.3, loc="upper left")
ax.set_xlim(amp_dB[0], 0)
ax.set_ylim(0, IDEAL_PEAK + 6)
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("pipeline_adc_monte_carlo_100runs.png",
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.show()
print("\nSaved -> pipeline_adc_monte_carlo_100runs.png")