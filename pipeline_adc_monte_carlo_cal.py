# -*- coding: utf-8 -*-
"""
4-Stage Pipeline ADC -- Monte Carlo (100 runs) WITH FOREGROUND CALIBRATION
────────────────────────────────────────────────────────────────────────────
Architecture
  • 2 bits per stage, no redundancy, 8-bit total
  • 4 pipeline stages → GAIN_IDEAL = 4 V/V per stage

Non-idealities modelled
  • Comparator offsets  : Gaussian σ = 30 mV
  • Residue-amp gain mismatch : Gaussian σ = 30 mV/V

Calibration hardware (per stage)
  ┌──────────────────────────────────────────────────────────┐
  │ Comparator trim knob (cal_thr[s,k])                      │
  │   A small trim-DAC injects a correction current into     │
  │   each comparator's differential input pair, effectively │
  │   shifting its trip voltage by cal_thr[s,k].            │
  │                                                          │
  │ Residue-amp gain trim knob (cal_gain[s])                 │
  │   A programmable switched-capacitor ratio adjusts the    │
  │   closed-loop gain of the residue amplifier.  The knob   │
  │   multiplies the effective gain by cal_gain[s].         │
  └──────────────────────────────────────────────────────────┘

Calibration algorithm (foreground, runs at power-up / periodically)
  Phase A – Comparator offset calibration (binary search)
    For each comparator k in stage s:
      1. Drive a test voltage to the stage input and sweep the
         trim-DAC code using bisection (n_bits=12 steps → ≈1.5 µV
         resolution over a ±6σ search window).
      2. Find the voltage at which the comparator output transitions.
      3. Set cal_thr[s,k] = nominal_thr[k] − measured_trip_point.
         After calibration, effective threshold ≈ nominal_thr[k].

  Phase B – Gain calibration (two-point injection)
    After offset calibration for stage s:
      1. Inject two known test voltages v_a, v_b into the stage.
      2. Read the residue at the stage output via a test mux.
      3. Estimate actual gain = (res_b − res_a) / (v_b − v_a).
      4. Set cal_gain[s] = GAIN_IDEAL / estimated_gain.
         After calibration, effective gain ≈ GAIN_IDEAL.

Three scenarios compared in Monte Carlo:
  (A) Ideal     – perfect thresholds and gains
  (B) Uncal     – comparator offsets + gain mismatch, no calibration
  (C) Calibrated – same errors but with trim knobs applied
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── ADC constants ────────────────────────────────────────────────────────────
VREF        = 1.0
N_STAGES    = 4
BITS_STAGE  = 2
GAIN_IDEAL  = 2 ** BITS_STAGE        # 4 V/V
TOTAL_BITS  = N_STAGES * BITS_STAGE  # 8
LEVELS      = 2 ** TOTAL_BITS        # 256
LSB         = 2 * VREF / LEVELS      # 7.8125 mV

OFFSET_SIGMA   = 0.030   # 30 mV comparator offset std-dev
GAIN_MIS_SIGMA = 0.030   # 30 mV/V residue-amp gain-error std-dev
N_COMP         = 3       # comparators per stage (2-bit flash)

NOMINAL_THR = np.array([-VREF / 2, 0.0, VREF / 2])
DAC_LEVELS  = np.array([-3 * VREF / 4, -VREF / 4, VREF / 4, 3 * VREF / 4])

# ── Simulation parameters ────────────────────────────────────────────────────
FS    = 1_000_000
N_FFT = 4096
M     = 97
fin   = M * FS / N_FFT
t     = np.arange(N_FFT, dtype=np.float64) / FS

N_RUNS   = 100
AMP_PTS  = 40
amp_dB   = np.linspace(-60, -0.05, AMP_PTS)
IDEAL_PEAK = 6.02 * TOTAL_BITS + 1.76

# ── Calibration parameters ───────────────────────────────────────────────────
CAL_SEARCH_BITS = 12          # bisection steps for threshold search
CAL_SEARCH_NSIG = 6           # ±N sigma search window for binary search
CAL_V_A = -0.10               # two-point gain cal: test voltage A
CAL_V_B = +0.10               # two-point gain cal: test voltage B


# ── Vectorised stage model ───────────────────────────────────────────────────
def _stage_vec(vin_arr, thr, gain):
    """Process entire sample array through one pipeline stage."""
    codes  = (vin_arr >= thr[0]).astype(np.int32)
    codes += (vin_arr >= thr[1]).astype(np.int32)
    codes += (vin_arr >= thr[2]).astype(np.int32)
    codes  = np.clip(codes, 0, 3)
    res    = (vin_arr - DAC_LEVELS[codes]) * gain
    return codes, res


def pipeline_adc_vec(x, thresholds, gains):
    """
    Vectorised pipeline ADC – processes all samples simultaneously.
    thresholds : (N_STAGES, 3) effective comparator thresholds (V)
    gains      : (N_STAGES,)  effective residue-amp gains (V/V)
    """
    x       = np.asarray(x, dtype=np.float64)
    digital = np.zeros(len(x), dtype=np.int32)
    residue = x.copy()
    for s in range(N_STAGES):
        residue = np.clip(residue, -VREF, VREF)
        sc, residue = _stage_vec(residue, thresholds[s], gains[s])
        digital = (digital << BITS_STAGE) | sc
    return digital


def code2volt(c):
    return (2 * c - (LEVELS - 1)) / float(LEVELS) * VREF


# ── SNDR computation ─────────────────────────────────────────────────────────
def compute_sndr(sig, fs):
    N   = len(sig)
    win = np.hanning(N)
    W   = np.sum(win ** 2)
    spec = np.fft.rfft(sig * win)
    pwr  = np.abs(spec) ** 2 / W
    sb   = int(np.argmax(pwr[1:])) + 1
    lobe = [b for b in (sb - 1, sb, sb + 1) if 0 < b < len(pwr)]
    Ps   = np.sum(pwr[lobe])
    mask = np.ones(len(pwr), dtype=bool)
    mask[0] = False
    for b in lobe:
        mask[b] = False
    Pn   = np.sum(pwr[mask])
    return 10.0 * np.log10(Ps / max(Pn, 1e-30))


# ════════════════════════════════════════════════════════════════════════════
# CALIBRATION ENGINE
# ════════════════════════════════════════════════════════════════════════════

def calibrate_adc(mc_eff_thr, mc_eff_gain):
    """
    Foreground calibration – returns trim knob arrays.

    Parameters
    ----------
    mc_eff_thr  : (N_STAGES, 3)  actual comparator thresholds (hardware)
    mc_eff_gain : (N_STAGES,)    actual residue-amp gains (hardware)

    Returns
    -------
    cal_thr  : (N_STAGES, 3)  threshold trim knobs  [V]
               Applied as:  eff_thr  = mc_eff_thr  + cal_thr
    cal_gain : (N_STAGES,)   gain scale trim knobs  [dimensionless]
               Applied as:  eff_gain = mc_eff_gain * cal_gain
    """
    cal_thr  = np.zeros((N_STAGES, N_COMP))
    cal_gain = np.ones(N_STAGES)

    for s in range(N_STAGES):

        # ── Phase A: Comparator offset trim (binary search) ──────────────────
        #
        # Sweep the correction DAC from lo→hi using bisection.
        # The trip point of comparator k is where:
        #   v_inject >= mc_eff_thr[s, k]  transitions from False to True.
        # The correction knob shifts the input by -offset so the comparator
        # trips at exactly NOMINAL_THR[k].
        #
        for k in range(N_COMP):
            search_half = CAL_SEARCH_NSIG * OFFSET_SIGMA
            lo = NOMINAL_THR[k] - search_half
            hi = NOMINAL_THR[k] + search_half

            for _ in range(CAL_SEARCH_BITS):       # 12 steps → ~0.1% residual
                mid   = (lo + hi) * 0.5
                fires = mid >= mc_eff_thr[s, k]    # comparator output
                if fires:
                    hi = mid                        # trip is at or below mid
                else:
                    lo = mid                        # trip is above mid

            measured_trip   = (lo + hi) * 0.5
            # Trim knob cancels measured offset:  nominal - actual_trip ≈ -offset
            cal_thr[s, k]   = NOMINAL_THR[k] - measured_trip

        # ── Phase B: Gain trim (two-point injection) ──────────────────────────
        #
        # Inject v_a and v_b into this stage (via a foreground test mux).
        # Use calibrated thresholds for code detection.
        # Read residue at stage output (via test-mux / output register).
        # Estimated gain = ΔV_residue / ΔV_in.
        # Correction knob = GAIN_IDEAL / estimated_gain.
        #
        eff_thr_cal = mc_eff_thr[s] + cal_thr[s]   # thresholds after offset trim

        code_a = int(np.clip(np.sum(CAL_V_A >= eff_thr_cal), 0, 3))
        code_b = int(np.clip(np.sum(CAL_V_B >= eff_thr_cal), 0, 3))

        # Actual (measured) residue from hardware
        res_a = (CAL_V_A - DAC_LEVELS[code_a]) * mc_eff_gain[s]
        res_b = (CAL_V_B - DAC_LEVELS[code_b]) * mc_eff_gain[s]

        # Ideal residue (what we expect with GAIN_IDEAL)
        res_a_ideal = (CAL_V_A - DAC_LEVELS[code_a]) * GAIN_IDEAL
        res_b_ideal = (CAL_V_B - DAC_LEVELS[code_b]) * GAIN_IDEAL

        delta_meas  = res_b - res_a
        delta_ideal = res_b_ideal - res_a_ideal

        if abs(delta_meas) > 1e-9:
            # Scale trim knob drives actual gain toward GAIN_IDEAL
            cal_gain[s] = delta_ideal / delta_meas

    return cal_thr, cal_gain


# ── Monte Carlo storage ──────────────────────────────────────────────────────
sndr_mc_ideal      = np.zeros((N_RUNS, AMP_PTS))
sndr_mc_offsets    = np.zeros((N_RUNS, AMP_PTS))  # offsets only (no gain mis)
sndr_mc_uncal      = np.zeros((N_RUNS, AMP_PTS))  # offsets + gain, no cal
sndr_mc_calibrated = np.zeros((N_RUNS, AMP_PTS))  # offsets + gain, WITH cal

# Store calibration knob values for post-run analysis
cal_thr_log  = np.zeros((N_RUNS, N_STAGES, N_COMP))
cal_gain_log = np.zeros((N_RUNS, N_STAGES))

# Ideal reference (constant across runs)
ideal_thr  = np.tile(NOMINAL_THR, (N_STAGES, 1))
ideal_gain = np.full(N_STAGES, GAIN_IDEAL)

print("=" * 66)
print("  MONTE CARLO: {} runs × {} amplitude points".format(N_RUNS, AMP_PTS))
print("  σ_offset = {:.0f} mV    σ_gain = {:.0f} mV/V".format(
    OFFSET_SIGMA * 1e3, GAIN_MIS_SIGMA * 1e3))
print("  Cal knob resolution: {}-bit binary search ({:.2f} µV LSB)".format(
    CAL_SEARCH_BITS,
    2 * CAL_SEARCH_NSIG * OFFSET_SIGMA / 2**CAL_SEARCH_BITS * 1e6))
print("=" * 66)

# ── Monte Carlo loop ─────────────────────────────────────────────────────────
for run in range(N_RUNS):
    rng = np.random.default_rng(run + 1000)

    # Draw hardware errors for this run
    mc_comp_offsets = rng.normal(0.0, OFFSET_SIGMA,   size=(N_STAGES, N_COMP))
    mc_gain_errors  = rng.normal(0.0, GAIN_MIS_SIGMA, size=N_STAGES)
    mc_eff_thr      = NOMINAL_THR[np.newaxis, :] + mc_comp_offsets   # (4,3)
    mc_eff_gain     = GAIN_IDEAL + mc_gain_errors                     # (4,)

    # Offsets-only thresholds (for comparison scenario B)
    mc_eff_thr_offonly = mc_eff_thr.copy()

    # ── Run foreground calibration ──────────────────────────────────────────
    cal_thr, cal_gain = calibrate_adc(mc_eff_thr, mc_eff_gain)
    cal_thr_log[run]  = cal_thr
    cal_gain_log[run] = cal_gain

    # Effective parameters after applying trim knobs
    cal_eff_thr  = mc_eff_thr + cal_thr          # hardware + threshold trim
    cal_eff_gain = mc_eff_gain * cal_gain         # hardware × gain trim

    # ── Amplitude sweep ─────────────────────────────────────────────────────
    for j, a in enumerate(10.0 ** (amp_dB / 20.0)):
        si = a * np.sin(2.0 * np.pi * fin * t)

        # A – Ideal
        c_ideal = pipeline_adc_vec(si, ideal_thr, ideal_gain)
        sndr_mc_ideal[run, j] = compute_sndr(code2volt(c_ideal), FS)

        # B – Offsets only (no gain mismatch, no calibration)
        c_off = pipeline_adc_vec(si, mc_eff_thr_offonly, ideal_gain)
        sndr_mc_offsets[run, j] = compute_sndr(code2volt(c_off), FS)

        # C – Offsets + Gain mismatch, NO calibration
        c_uncal = pipeline_adc_vec(si, mc_eff_thr, mc_eff_gain)
        sndr_mc_uncal[run, j] = compute_sndr(code2volt(c_uncal), FS)

        # D – Offsets + Gain mismatch, WITH calibration knobs applied
        c_cal = pipeline_adc_vec(si, cal_eff_thr, cal_eff_gain)
        sndr_mc_calibrated[run, j] = compute_sndr(code2volt(c_cal), FS)

    if (run + 1) % 10 == 0:
        print("  Completed {}/{} runs".format(run + 1, N_RUNS))

# ── Statistics ───────────────────────────────────────────────────────────────
avg_sndr_ideal    = np.mean(sndr_mc_ideal,      axis=0)
avg_sndr_uncal    = np.mean(sndr_mc_uncal,       axis=0)
std_sndr_uncal    = np.std(sndr_mc_uncal,        axis=0)
avg_sndr_cal      = np.mean(sndr_mc_calibrated,  axis=0)
std_sndr_cal      = np.std(sndr_mc_calibrated,   axis=0)

fs_sndr_uncal = sndr_mc_uncal[:, -1]
fs_sndr_cal   = sndr_mc_calibrated[:, -1]

print("\n" + "─" * 66)
print("  Monte Carlo Full-Scale SNDR Statistics ({} runs)".format(N_RUNS))
print("─" * 66)
print("  Ideal:             {:.2f} dB   ENOB = {} bits".format(
    IDEAL_PEAK, TOTAL_BITS))
print("  Uncalibrated:      Mean = {:6.2f} dB  Std = {:.2f} dB  ENOB = {:.2f} bits".format(
    np.mean(fs_sndr_uncal), np.std(fs_sndr_uncal),
    (np.mean(fs_sndr_uncal) - 1.76) / 6.02))
print("  Calibrated:        Mean = {:6.2f} dB  Std = {:.2f} dB  ENOB = {:.2f} bits".format(
    np.mean(fs_sndr_cal), np.std(fs_sndr_cal),
    (np.mean(fs_sndr_cal) - 1.76) / 6.02))
print("  SNDR recovery:     {:.2f} dB  ({:.2f} bits ENOB recovered)".format(
    np.mean(fs_sndr_cal) - np.mean(fs_sndr_uncal),
    (np.mean(fs_sndr_cal) - np.mean(fs_sndr_uncal)) / 6.02))

# Calibration knob analysis
print("\n  Calibration Knob Statistics ({} stages × {} runs)".format(N_STAGES, N_RUNS))
print("─" * 66)
for s in range(N_STAGES):
    thr_rms  = np.sqrt(np.mean(cal_thr_log[:, s, :] ** 2))
    gain_mean = np.mean(cal_gain_log[:, s])
    gain_std  = np.std(cal_gain_log[:, s])
    print("  Stage {}: thr trim RMS = {:+.1f} mV   "
          "gain knob = {:.4f} ± {:.4f}".format(
        s + 1, thr_rms * 1e3, gain_mean, gain_std))
print("─" * 66)


# ── Plot ─────────────────────────────────────────────────────────────────────
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
LIME   = "#B9F542"

plt.rcParams.update({
    "figure.facecolor":  DARK,
    "axes.facecolor":    DARK,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   TEXT,
    "axes.titlecolor":   TEXT,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "grid.color":        GRID,
    "grid.linewidth":    0.5,
    "text.color":        TEXT,
    "font.family":       "monospace",
    "font.size":         10,
})

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor(DARK)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.32)

fig.suptitle(
    "4-STAGE PIPELINE ADC  |  Monte Carlo ({} runs)  |  "
    "σ_offset={}mV  |  σ_gain={}mV/V  |  FOREGROUND CALIBRATION".format(
        N_RUNS, int(OFFSET_SIGMA * 1e3), int(GAIN_MIS_SIGMA * 1e3)),
    fontsize=11, fontweight="bold", color=CYAN, fontfamily="monospace", y=0.97)

# ─── [0,0]  Average SNDR vs Amplitude ───────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.plot(amp_dB, avg_sndr_ideal, color=GHOST, lw=1.5, ls="--",
        label="Ideal {}-bit".format(TOTAL_BITS))
ax.plot(amp_dB, avg_sndr_uncal, color=ORANGE, lw=1.8,
        label="Uncal (mean)")
ax.fill_between(amp_dB,
                avg_sndr_uncal - std_sndr_uncal,
                avg_sndr_uncal + std_sndr_uncal,
                color=ORANGE, alpha=0.18)
ax.plot(amp_dB, avg_sndr_cal, color=LIME, lw=2.2,
        label="Calibrated (mean)")
ax.fill_between(amp_dB,
                avg_sndr_cal - std_sndr_cal,
                avg_sndr_cal + std_sndr_cal,
                color=LIME, alpha=0.18, label="±1σ bands")
ax.set_xlabel("Input Amplitude (dBFS)")
ax.set_ylabel("SNDR (dB)")
ax.set_title("Avg SNDR vs Amplitude", fontsize=10)
ax.legend(fontsize=7.5, framealpha=0.3, loc="upper left")
ax.set_xlim(amp_dB[0], 0)
ax.set_ylim(0, IDEAL_PEAK + 6)
ax.grid(True, alpha=0.3)

# ─── [0,1]  Full-Scale SNDR Histogram ───────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
ax.hist(fs_sndr_uncal, bins=18, color=ORANGE, alpha=0.55,
        edgecolor=ORANGE, label="Uncal")
ax.hist(fs_sndr_cal,   bins=18, color=LIME,   alpha=0.55,
        edgecolor=LIME,   label="Calibrated")
ax.axvline(np.mean(fs_sndr_uncal), color=ORANGE, lw=2.0, ls="--",
           label="Mean={:.1f} dB".format(np.mean(fs_sndr_uncal)))
ax.axvline(np.mean(fs_sndr_cal),   color=LIME,   lw=2.0, ls="--",
           label="Mean={:.1f} dB".format(np.mean(fs_sndr_cal)))
ax.axvline(IDEAL_PEAK, color=CYAN, lw=1.2, ls=":",
           label="Ideal={:.1f} dB".format(IDEAL_PEAK))
ax.set_xlabel("SNDR (dB)")
ax.set_ylabel("Count")
ax.set_title("Full-Scale SNDR Distribution", fontsize=10)
ax.legend(fontsize=7, framealpha=0.3)
ax.grid(True, alpha=0.3)

# ─── [0,2]  Calibration Knob Distributions ──────────────────────────────────
#
# Left sub-panel:  threshold trim knobs  cal_thr[run, stage, comp]
# Right sub-panel: gain trim knobs       cal_gain[run, stage]
#
ax = fig.add_subplot(gs[0, 2])

stage_labels = ["S{}".format(s + 1) for s in range(N_STAGES)]
x_thr  = np.arange(N_STAGES)          # x positions for threshold
x_gain = x_thr + N_STAGES + 1         # x positions for gain (offset right)

# Box plots – threshold trim (average across 3 comparators per stage)
thr_trim_mean = cal_thr_log.mean(axis=2)   # (N_RUNS, N_STAGES)
bp1 = ax.boxplot(
    [thr_trim_mean[:, s] * 1e3 for s in range(N_STAGES)],
    positions=x_thr, widths=0.5,
    patch_artist=True,
    medianprops=dict(color=DARK, lw=2),
    whiskerprops=dict(color=ORANGE),
    capprops=dict(color=ORANGE),
    flierprops=dict(marker=".", color=ORANGE, alpha=0.4, ms=4),
    boxprops=dict(facecolor=ORANGE, alpha=0.6))

# Box plots – gain trim knobs (per stage)
bp2 = ax.boxplot(
    [(cal_gain_log[:, s] - 1.0) * 1e3 for s in range(N_STAGES)],
    positions=x_gain, widths=0.5,
    patch_artist=True,
    medianprops=dict(color=DARK, lw=2),
    whiskerprops=dict(color=LIME),
    capprops=dict(color=LIME),
    flierprops=dict(marker=".", color=LIME, alpha=0.4, ms=4),
    boxprops=dict(facecolor=LIME, alpha=0.6))

ax.axhline(0, color=MUTED, lw=0.8, ls="--", alpha=0.5)
ax.set_xticks(list(x_thr) + list(x_gain))
ax.set_xticklabels(
    ["T-{}".format(s + 1) for s in range(N_STAGES)] +
    ["G-{}".format(s + 1) for s in range(N_STAGES)],
    fontsize=8)
ax.set_ylabel("Trim value (mV  or  mV/V ×10⁻³)")
ax.set_title("Calibration Knob Distributions\n"
             "T-n = threshold trim [mV]    G-n = (gain knob−1)×10³", fontsize=9)
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor=ORANGE, alpha=0.7, label="Thr trim knob [mV]"),
    Patch(facecolor=LIME,   alpha=0.7, label="Gain trim knob (×10³)")],
    fontsize=7.5, framealpha=0.3)
ax.grid(True, alpha=0.3, axis="y")

# ─── [1,0]  Average ENOB vs Amplitude ───────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
enob_ideal = (avg_sndr_ideal - 1.76) / 6.02
enob_uncal = (avg_sndr_uncal - 1.76) / 6.02
enob_cal   = (avg_sndr_cal   - 1.76) / 6.02
ax.plot(amp_dB, enob_ideal, color=GHOST,  lw=1.5, ls="--",
        label="Ideal {}-bit".format(TOTAL_BITS))
ax.plot(amp_dB, enob_uncal, color=ORANGE, lw=1.8,
        label="Uncal (mean)")
ax.plot(amp_dB, enob_cal,   color=LIME,   lw=2.2,
        label="Calibrated (mean)")
ax.fill_between(amp_dB,
                (avg_sndr_cal - std_sndr_cal - 1.76) / 6.02,
                (avg_sndr_cal + std_sndr_cal - 1.76) / 6.02,
                color=LIME, alpha=0.15)
ax.axhline(TOTAL_BITS, color=CYAN, lw=0.8, ls=":", alpha=0.5)
ax.set_xlabel("Input Amplitude (dBFS)")
ax.set_ylabel("ENOB (bits)")
ax.set_title("Average ENOB vs Amplitude", fontsize=10)
ax.legend(fontsize=7.5, framealpha=0.3, loc="upper left")
ax.set_xlim(amp_dB[0], 0)
ax.set_ylim(0, TOTAL_BITS + 1)
ax.grid(True, alpha=0.3)

# ─── [1,1]  Waterfall – Uncalibrated ────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
for run in range(N_RUNS):
    ax.plot(amp_dB, sndr_mc_uncal[run], color=ORANGE, lw=0.3, alpha=0.15)
ax.plot(amp_dB, avg_sndr_uncal, color=AMBER, lw=2.5,
        label="Mean (Uncal)")
ax.plot(amp_dB, avg_sndr_ideal, color=CYAN,  lw=1.2, ls="--",
        label="Ideal {}-bit".format(TOTAL_BITS))
ax.set_xlabel("Input Amplitude (dBFS)")
ax.set_ylabel("SNDR (dB)")
ax.set_title("All {} Runs – Uncalibrated".format(N_RUNS), fontsize=10)
ax.legend(fontsize=7.5, framealpha=0.3, loc="upper left")
ax.set_xlim(amp_dB[0], 0)
ax.set_ylim(0, IDEAL_PEAK + 6)
ax.grid(True, alpha=0.3)

# ─── [1,2]  Waterfall – Calibrated ──────────────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
for run in range(N_RUNS):
    ax.plot(amp_dB, sndr_mc_calibrated[run], color=LIME, lw=0.3, alpha=0.15)
ax.plot(amp_dB, avg_sndr_cal, color=LIME, lw=2.5,
        label="Mean (Calibrated)")
ax.plot(amp_dB, avg_sndr_ideal, color=CYAN, lw=1.2, ls="--",
        label="Ideal {}-bit".format(TOTAL_BITS))
ax.set_xlabel("Input Amplitude (dBFS)")
ax.set_ylabel("SNDR (dB)")
ax.set_title("All {} Runs – Calibrated".format(N_RUNS), fontsize=10)
ax.legend(fontsize=7.5, framealpha=0.3, loc="upper left")
ax.set_xlim(amp_dB[0], 0)
ax.set_ylim(0, IDEAL_PEAK + 6)
ax.grid(True, alpha=0.3)

out_png = "pipeline_adc_monte_carlo_calibrated.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.show()
print("\nSaved → {}".format(out_png))
