"""
4-Stage Pipeline ADC  —  2.5 bit/stage  —  Foreground Calibration
══════════════════════════════════════════════════════════════════
Impairments
  • Comparator offsets   σ = 30 mV   (7 comp × 4 stages = 28 total)
  • Gain mismatch        σ = 30 mV/V (4 stages)

Calibration knobs (added directly to hardware)
  ┌─ Comparator trim DAC ──────────────────────────────────────────┐
  │  knob_thr[s,k]  — voltage added to comparator k reference     │
  │  6-bit trim over ±5σ  →  step ≈ 4.7 mV                       │
  │  Protocol: binary search per comparator per stage              │
  │    inject V_test at stage s input; observe sub-ADC code;       │
  │    bisect until code boundary found; set knob = Vnominal − V*  │
  └────────────────────────────────────────────────────────────────┘
  ┌─ Gain stage trim multiplier ────────────────────────────────────┐
  │  knob_gain[s]   — scalar multiplying residue amp output        │
  │  6-bit trim over ±2.5%  →  step ≈ 0.08%                       │
  │  Protocol: two-point injection in center sub-range             │
  │    residue slope = (r2−r1)/(v2−v1)  →  knob = Gideal / slope  │
  └────────────────────────────────────────────────────────────────┘

Monte Carlo: 100 independent draws of (offsets, gain errors)
  For each draw: calibrate → measure knobs → run calibrated ADC
  Plot: mean ± 1σ for (uncal, calibrated) + knob distributions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

# ── ADC constants ──────────────────────────────────────────────────────────
VREF          = 1.0
N_STAGES      = 4
BITS_STAGE    = 2
BITS_RAW      = BITS_STAGE + 1
GAIN_IDEAL    = float(2 ** BITS_STAGE)     # 4 V/V
N_COMP        = 2 ** BITS_RAW - 1          # 7 comparators/stage
N_DAC         = 2 ** BITS_RAW              # 8 DAC levels
TOTAL_BITS    = N_STAGES * BITS_STAGE      # 8 effective bits
LEVELS        = 2 ** TOTAL_BITS            # 256
LSB           = 2 * VREF / LEVELS          # 7.8125 mV
REDUNDANCY_V  = VREF / N_DAC              # ±125 mV DEC budget

OFFSET_SIGMA   = 0.030    # 30 mV  comparator offset σ
GAIN_MIS_SIGMA = 0.030    # 30 mV/V gain-error σ
N_MC           = 100

NOMINAL_THR = np.array([-3,-2,-1,0,1,2,3], dtype=float) * VREF / 4
DAC_LEVELS  = np.array([-7,-5,-3,-1,1,3,5,7], dtype=float) * VREF / 8

# ── Trim knob resolution (quantisation) ───────────────────────────────────
THR_TRIM_BITS  = 6
THR_TRIM_RANGE = 5 * OFFSET_SIGMA               # ±150 mV
THR_TRIM_STEP  = 2*THR_TRIM_RANGE / (2**THR_TRIM_BITS - 1)   # ≈ 4.76 mV

GAIN_TRIM_BITS  = 8                                   # 8-bit trim for fine resolution
GAIN_TRIM_RANGE = 5.0 * GAIN_MIS_SIGMA / GAIN_IDEAL  # ±3.75% relative (covers ±5σ)
GAIN_TRIM_STEP  = 2*GAIN_TRIM_RANGE / (2**GAIN_TRIM_BITS - 1)  # ≈ 0.029%

# ── Signal ─────────────────────────────────────────────────────────────────
FS = 1_000_000; N_FFT = 4096; M = 97
fin = M * FS / N_FFT
t   = np.arange(N_FFT) / FS
A   = 0.999 * VREF

IDEAL_PEAK = 6.02 * TOTAL_BITS + 1.76
amp_dB     = np.linspace(-60, -0.05, 120)
amp_linear = 10 ** (amp_dB / 20)
ideal_sndr = np.minimum(amp_dB + IDEAL_PEAK, IDEAL_PEAK)

thr_ideal  = np.tile(NOMINAL_THR, (N_STAGES, 1))
gain_ideal = np.full(N_STAGES, GAIN_IDEAL)

# ─────────────────────────────────────────────────────────────────────────
# ADC kernel — vectorised over samples
# ─────────────────────────────────────────────────────────────────────────
def pipeline_adc_vec(x, thr, gain):
    """DEC reconstruction.  thr:(N_S,N_C)  gain:(N_S,) → voltages (N,)"""
    r   = np.clip(x, -VREF, VREF).copy()
    rec = np.zeros(len(x))
    for s in range(N_STAGES):
        codes = np.clip(np.sum(r[:,None] > thr[s][None,:], axis=1), 0, N_DAC-1)
        dac   = DAC_LEVELS[codes]
        rec  += dac / (GAIN_IDEAL ** s)   # DEC weighted sum (uses GAIN_IDEAL always)
        r     = np.clip((r - dac) * gain[s], -VREF, VREF)
    return rec

# ─────────────────────────────────────────────────────────────────────────
# SNDR — 3-bin Hann main-lobe
# ─────────────────────────────────────────────────────────────────────────
def compute_sndr(sig):
    win = np.hanning(len(sig)); W = np.sum(win**2)
    pwr = np.abs(np.fft.rfft(sig * win))**2 / W
    sb  = int(np.argmax(pwr[1:])) + 1
    lobe = [b for b in (sb-1, sb, sb+1) if 0 < b < len(pwr)]
    Ps   = np.sum(pwr[lobe])
    mask = np.ones(len(pwr), bool); mask[0] = False
    for b in lobe: mask[b] = False
    return 10 * np.log10(Ps / np.sum(pwr[mask]))

# ─────────────────────────────────────────────────────────────────────────
# CALIBRATION KNOB MEASUREMENT
# ─────────────────────────────────────────────────────────────────────────
def calibrate_knobs(thr_r, gain_r, n_bisect=40):
    """
    Foreground calibration — measures and returns trim knobs.

    Threshold knobs  (knob_thr):
      Stage-by-stage binary search: inject V_test at stage s input;
      observe sub-ADC code; bisect to find exact threshold crossing V*.
      knob_thr[s,k] = NOMINAL_THR[k] − V*   → trims threshold back to nominal.
      Quantised to THR_TRIM_STEP (6-bit trim DAC).

    Gain knobs  (knob_gain):
      After threshold cal, inject v1=−Vref/12 and v2=+Vref/12 (both in
      center sub-range). Measure residue r1,r2; slope = (r2−r1)/(v2−v1).
      knob_gain[s] = GAIN_IDEAL / slope   → rescales residue amp to ideal.
      Quantised to GAIN_TRIM_STEP (6-bit trim multiplier).

    Returns
      knob_thr  : (N_STAGES, N_COMP)  voltages to add to each comparator ref
      knob_gain : (N_STAGES,)          multipliers on each residue amp output
      thr_eff   : calibrated effective thresholds (for diagnostics)
      gain_eff  : calibrated effective gains       (for diagnostics)
    """
    knob_thr  = np.zeros((N_STAGES, N_COMP))
    knob_gain = np.ones(N_STAGES)

    for s in range(N_STAGES):
        # ── Step 1: threshold trim — binary search per comparator ──────────
        for k in range(N_COMP):
            lo = NOMINAL_THR[k] - THR_TRIM_RANGE
            hi = NOMINAL_THR[k] + THR_TRIM_RANGE
            for _ in range(n_bisect):
                mid  = (lo + hi) * 0.5
                code = int(np.clip(np.sum(mid > thr_r[s]), 0, N_DAC-1))
                # code > k  ⟹  mid has crossed threshold k → tighten from above
                if code > k:
                    hi = mid
                else:
                    lo = mid
            measured_thr = (lo + hi) * 0.5
            # Quantise to trim DAC resolution
            raw_knob     = NOMINAL_THR[k] - measured_thr
            knob_thr[s,k] = np.round(raw_knob / THR_TRIM_STEP) * THR_TRIM_STEP

        # ── Step 2: gain trim — two-point injection ────────────────────────
        # CRITICAL: both points must land in the SAME sub-range (same code).
        # Use v1=-0.20V, v2=-0.06V: both fall in [-VREF/4, 0) → code 3.
        # Threshold[2]=−0.25V and Threshold[3]=0V bracket this region.
        thr_s_cal = thr_r[s] + knob_thr[s]   # effective threshold after trim
        v1, v2 = NOMINAL_THR[2] + 0.05*VREF, NOMINAL_THR[2] + 0.19*VREF
        # = −0.20V and −0.06V — both safely inside code-3 sub-range
        c1 = int(np.clip(np.sum(v1 > thr_s_cal), 0, N_DAC-1))
        c2 = int(np.clip(np.sum(v2 > thr_s_cal), 0, N_DAC-1))
        if c1 == c2:
            # Residues: r = (v − DAC[c]) × gain_r  →  slope = gain_r
            r1 = (v1 - DAC_LEVELS[c1]) * gain_r[s]
            r2 = (v2 - DAC_LEVELS[c2]) * gain_r[s]
            slope    = (r2 - r1) / (v2 - v1)        # measured ≈ gain_r[s]
            raw_knob = GAIN_IDEAL / slope            # correction factor
            rel_dev  = raw_knob - 1.0
            rel_dev_q = np.clip(
                np.round(rel_dev / GAIN_TRIM_STEP) * GAIN_TRIM_STEP,
                -GAIN_TRIM_RANGE, +GAIN_TRIM_RANGE
            )
            knob_gain[s] = 1.0 + rel_dev_q
        else:
            # Fallback: single-point measurement (v1 only)
            dv = v1 - DAC_LEVELS[c1]
            if abs(dv) > 0.01:
                slope    = (v1 - DAC_LEVELS[c1]) * gain_r[s] / dv
                raw_knob = GAIN_IDEAL / slope
                rel_dev_q = np.clip(
                    np.round((raw_knob-1.0) / GAIN_TRIM_STEP) * GAIN_TRIM_STEP,
                    -GAIN_TRIM_RANGE, +GAIN_TRIM_RANGE)
                knob_gain[s] = 1.0 + rel_dev_q

    thr_eff  = thr_r  + knob_thr            # should be ≈ NOMINAL_THR
    gain_eff = gain_r * knob_gain           # should be ≈ GAIN_IDEAL
    return knob_thr, knob_gain, thr_eff, gain_eff

# ─────────────────────────────────────────────────────────────────────────
# Monte Carlo
# ─────────────────────────────────────────────────────────────────────────
print(f"Running {N_MC}-run Monte Carlo …")
rng = np.random.default_rng(7)

mc_uncal = np.zeros((N_MC, len(amp_dB)))   # uncalibrated (offsets + gain)
mc_cal   = np.zeros((N_MC, len(amp_dB)))   # after calibration knobs applied

# Store knobs from every run for the diagnostic plots
all_knob_thr  = np.zeros((N_MC, N_STAGES, N_COMP))  # (100, 4, 7) mV
all_knob_gain = np.zeros((N_MC, N_STAGES))           # (100, 4)

# Also track residual errors (how well did cal correct?)
all_thr_residual  = np.zeros((N_MC, N_STAGES, N_COMP))  # thr_eff − nominal
all_gain_residual = np.zeros((N_MC, N_STAGES))           # gain_eff − GAIN_IDEAL

for run in range(N_MC):
    # Draw fresh impairments
    co     = rng.normal(0, OFFSET_SIGMA,   (N_STAGES, N_COMP))
    ge     = rng.normal(0, GAIN_MIS_SIGMA,  N_STAGES)
    thr_r  = NOMINAL_THR[None,:] + co
    gain_r = GAIN_IDEAL + ge

    # Measure calibration knobs via the protocol
    knob_thr, knob_gain, thr_eff, gain_eff = calibrate_knobs(thr_r, gain_r)

    all_knob_thr[run]  = knob_thr
    all_knob_gain[run] = knob_gain
    all_thr_residual[run]  = thr_eff  - NOMINAL_THR[None,:]
    all_gain_residual[run] = gain_eff - GAIN_IDEAL

    # Sweep amplitudes
    for j, a in enumerate(amp_linear):
        si = a * np.sin(2*np.pi*fin*t)
        mc_uncal[run, j] = compute_sndr(pipeline_adc_vec(si, thr_r,   gain_r   ))
        mc_cal  [run, j] = compute_sndr(pipeline_adc_vec(si, thr_eff, gain_eff ))

    if (run+1) % 25 == 0:
        print(f"  {run+1}/{N_MC}")

# Statistics
mean_uncal = mc_uncal.mean(0);  std_uncal = mc_uncal.std(0)
mean_cal   = mc_cal.mean(0);    std_cal   = mc_cal.std(0)

pk_uncal   = mean_uncal.max(); pk_uncal_s = std_uncal[mean_uncal.argmax()]
pk_cal     = mean_cal.max();   pk_cal_s   = std_cal[mean_cal.argmax()]

# Ideal reference
sv_ideal = np.array([
    compute_sndr(pipeline_adc_vec(a * np.sin(2*np.pi*fin*t), thr_ideal, gain_ideal))
    for a in amp_linear
])

print(f"\n{'─'*60}")
print(f"  {'':30s}  Peak SNDR        ENOB")
print(f"  {'Ideal (no errors)':30s}  {sv_ideal.max():.2f} dB       {(sv_ideal.max()-1.76)/6.02:.2f}")
print(f"  {'Uncalibrated (+off+gain)':30s}  {pk_uncal:.2f} ± {pk_uncal_s:.2f} dB  {(pk_uncal-1.76)/6.02:.2f}")
print(f"  {'Calibrated':30s}  {pk_cal:.2f} ± {pk_cal_s:.2f} dB  {(pk_cal-1.76)/6.02:.2f}")
print(f"  SNDR recovery: {pk_cal - pk_uncal:.2f} dB")
print(f"\n  Trim knob residuals (across all {N_MC} runs × all stages):")
print(f"    Threshold : σ_residual = {all_thr_residual.std()*1e3:.3f} mV  "
      f"(DEC window = ±{REDUNDANCY_V*1e3:.0f} mV)")
print(f"    Gain      : σ_residual = {all_gain_residual.std()*1e3:.3f} mV/V  "
      f"(ideal = {GAIN_IDEAL:.1f})")
print(f"{'─'*60}")

# ─────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────
DARK   = "#0A0E14"; GRID  = "#1C2330"; GHOST = "#2A3545"
TEXT   = "#CDD6E0"; MUTED = "#6E7D8C"
CYAN   = "#00E5FF"; AMBER = "#FFB300"; RED   = "#FF4560"
GREEN  = "#00E396"; ORANGE= "#FF6D00"; PURPLE= "#CE93D8"

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": DARK,
    "axes.edgecolor":   GRID, "axes.labelcolor": TEXT,
    "axes.titlecolor":  TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize":  8,    "ytick.labelsize": 8,
    "grid.color":       GRID, "grid.linewidth":  0.5,
    "text.color":       TEXT, "font.family":     "monospace", "font.size": 9,
})

fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(
    2, 3, figure=fig,
    height_ratios=[1.8, 1],
    hspace=0.52, wspace=0.35,
    left=0.06, right=0.97, top=0.90, bottom=0.06
)

# ── Header ────────────────────────────────────────────────────────────────
fig.text(0.5, 0.963,
    "4-STAGE PIPELINE ADC  ·  2.5 BIT/STAGE  ·  1-BIT REDUNDANCY + DEC  ·  "
    "FOREGROUND CALIBRATION (TRIM KNOBS ON COMPARATORS + GAIN STAGES)",
    ha="center", fontsize=11.5, fontweight="bold", color=CYAN, fontfamily="monospace")
fig.text(0.5, 0.938,
    f"σ_offset = {OFFSET_SIGMA*1e3:.0f} mV / comparator  │  "
    f"σ_gain = {GAIN_MIS_SIGMA*1e3:.0f} mV/V / stage  │  "
    f"N_MC = {N_MC}  │  "
    f"Thr trim: 6-bit ±{THR_TRIM_RANGE*1e3:.0f}mV (step={THR_TRIM_STEP*1e3:.1f}mV)  │  "
    f"Gain trim: 6-bit ±{GAIN_TRIM_RANGE*100:.2f}% (step={GAIN_TRIM_STEP*100:.3f}%)",
    ha="center", fontsize=8.5, color=AMBER, fontfamily="monospace")

def sax(ax, title, xlabel, ylabel, minorx=None, minory=None):
    ax.set_title(title, fontsize=9, pad=5)
    ax.set_xlabel(xlabel, fontsize=8); ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, which="major", lw=0.5)
    ax.grid(True, which="minor", lw=0.22, alpha=0.35)
    if minory: ax.yaxis.set_minor_locator(MultipleLocator(minory))
    if minorx: ax.xaxis.set_minor_locator(MultipleLocator(minorx))

# ═══════════════════════════════════════════════════════════════════════════
# [0, 0:3]  MAIN: SNDR vs amplitude — MC mean ± 1σ
# ═══════════════════════════════════════════════════════════════════════════
ax_sndr = fig.add_subplot(gs[0, :])

# Faint individual traces
stride = max(1, N_MC // 30)
for run in range(0, N_MC, stride):
    ax_sndr.plot(amp_dB, mc_uncal[run], color=PURPLE, lw=0.35, alpha=0.09)
    ax_sndr.plot(amp_dB, mc_cal[run],   color=GREEN,  lw=0.35, alpha=0.09)

# ±1σ shaded bands
ax_sndr.fill_between(amp_dB, mean_uncal-std_uncal, mean_uncal+std_uncal,
                     color=PURPLE, alpha=0.18)
ax_sndr.fill_between(amp_dB, mean_cal-std_cal,     mean_cal+std_cal,
                     color=GREEN,  alpha=0.22)

# Reference lines
ax_sndr.plot(amp_dB, ideal_sndr, color=GHOST,  lw=1.3, ls="--",
             label="Ideal 8-bit (theoretical)")
ax_sndr.plot(amp_dB, sv_ideal,   color=CYAN,   lw=1.8, alpha=0.75,
             label=f"ADC ideal — no errors  ({sv_ideal.max():.1f} dB pk)")

# Mean curves
ax_sndr.plot(amp_dB, mean_uncal, color=PURPLE, lw=2.2,
             label=f"Uncalibrated +Offsets+Gain  "
                   f"({pk_uncal:.1f} ± {pk_uncal_s:.1f} dB)")
ax_sndr.plot(amp_dB, mean_cal,   color=GREEN,  lw=2.4,
             label=f"Calibrated (trim knobs applied)  "
                   f"({pk_cal:.1f} ± {pk_cal_s:.1f} dB)")

# Peak annotations with error bars
for mvec, svec, col, dy in [
        (mean_uncal, std_uncal, PURPLE,  -9),
        (mean_cal,   std_cal,   GREEN,   +4)]:
    idx = mvec.argmax(); xp = amp_dB[idx]; yp = mvec[idx]; sp = svec[idx]
    ax_sndr.errorbar(xp, yp, yerr=sp, fmt='o', color=col,
                     capsize=7, capthick=1.5, ms=7, zorder=8)
    enob = (yp - 1.76) / 6.02
    ax_sndr.annotate(
        f"  {yp:.2f} ± {sp:.2f} dB\n  ENOB = {enob:.2f} bits",
        xy=(xp, yp), xytext=(xp-18, yp+dy),
        fontsize=9, color=col,
        arrowprops=dict(arrowstyle="->", color=col, lw=1.1),
        bbox=dict(boxstyle="round,pad=0.4", fc=DARK, ec=col, lw=1.0, alpha=0.93))

# Recovery arrow
x_arr  = -5.0
y_unc  = np.interp(x_arr, amp_dB, mean_uncal)
y_cal  = np.interp(x_arr, amp_dB, mean_cal)
ax_sndr.annotate("", xy=(x_arr, y_cal), xytext=(x_arr, y_unc),
                 arrowprops=dict(arrowstyle="<->", color=AMBER, lw=1.8))
ax_sndr.text(x_arr + 1.0, (y_unc+y_cal)/2,
             f"  +{pk_cal-pk_uncal:.1f} dB\n  recovered",
             fontsize=8.5, color=AMBER, va="center",
             bbox=dict(boxstyle="round,pad=0.3", fc=DARK, ec=AMBER, lw=0.8, alpha=0.9))

ax_sndr.axvline(20*np.log10(A), color=AMBER, lw=0.9, ls=":", alpha=0.6)

# Legend with ±1σ patches
handles, labels = ax_sndr.get_legend_handles_labels()
handles += [Patch(fc=PURPLE, alpha=0.35, label="±1σ  Uncalibrated"),
            Patch(fc=GREEN,  alpha=0.40, label="±1σ  Calibrated")]
labels  += ["±1σ  Uncalibrated", "±1σ  Calibrated"]
ax_sndr.legend(handles, labels, fontsize=8.5, framealpha=0.25,
               loc="upper left", ncol=2)

sax(ax_sndr,
    f"Average SNDR vs Input Amplitude  —  {N_MC}-Run Monte Carlo  "
    f"[σ_offset={OFFSET_SIGMA*1e3:.0f}mV, σ_gain={GAIN_MIS_SIGMA*1e3:.0f}mV/V]  "
    f"│  faint traces = individual runs  │  band = ±1σ",
    "Amplitude (dBFS)", "SNDR (dB)", minorx=5, minory=5)
ax_sndr.set_xlim(amp_dB[0], 0)
ax_sndr.set_ylim(0, IDEAL_PEAK + 8)

# ═══════════════════════════════════════════════════════════════════════════
# [1, 0]  Comparator threshold trim knob distribution
# ═══════════════════════════════════════════════════════════════════════════
ax_thr = fig.add_subplot(gs[1, 0])

# Show knob values vs comparator index, coloured by stage
stage_colors = [CYAN, ORANGE, PURPLE, GREEN]
comp_labels  = [f"C{k+1}" for k in range(N_COMP)]
x_base       = np.arange(N_COMP)

for s in range(N_STAGES):
    vals_mV = all_knob_thr[:, s, :] * 1e3   # (100, 7) in mV
    # Box plot per comparator
    bp = ax_thr.boxplot(
        [vals_mV[:, k] for k in range(N_COMP)],
        positions=x_base + (s - 1.5)*0.18,
        widths=0.14, patch_artist=True, manage_ticks=False,
        medianprops=dict(color=DARK, lw=1.5),
        whiskerprops=dict(color=stage_colors[s], lw=0.8),
        capprops=dict(color=stage_colors[s], lw=1.0),
        flierprops=dict(marker='.', ms=2, color=stage_colors[s], alpha=0.4),
        boxprops=dict(facecolor=stage_colors[s], alpha=0.55, linewidth=0.6,
                      edgecolor=stage_colors[s])
    )

# Overlay: true offsets from first MC run (to show what we're correcting)
for s in range(N_STAGES):
    ax_thr.plot(x_base, -all_knob_thr[0, s, :]*1e3,
                'x', ms=5, color=RED, alpha=0.4, zorder=5)

ax_thr.axhline(0, color=MUTED, lw=0.6)
ax_thr.axhline( THR_TRIM_STEP*1e3/2, color=MUTED, lw=0.5, ls=":", alpha=0.5,
                label=f"±½ step ({THR_TRIM_STEP*1e3/2:.1f}mV)")
ax_thr.axhline(-THR_TRIM_STEP*1e3/2, color=MUTED, lw=0.5, ls=":", alpha=0.5)
ax_thr.axhspan(-REDUNDANCY_V*1e3, REDUNDANCY_V*1e3, color=CYAN, alpha=0.04,
               label=f"DEC window ±{REDUNDANCY_V*1e3:.0f}mV")

for s in range(N_STAGES):
    ax_thr.plot([], [], color=stage_colors[s], lw=5,
                alpha=0.55, label=f"Stage {s+1}")
ax_thr.plot([], [], 'x', color=RED, ms=5, alpha=0.6, label="True offset (run 0)")

ax_thr.set_xticks(x_base); ax_thr.set_xticklabels(comp_labels, fontsize=7)
sax(ax_thr, "Threshold Trim Knobs  (knob_thr, mV)",
    "Comparator", "Knob Value (mV)", minory=10)
ax_thr.legend(fontsize=6.5, framealpha=0.2, ncol=2, loc="upper right")
ax_thr.set_xlim(-0.5, N_COMP-0.5)

# ═══════════════════════════════════════════════════════════════════════════
# [1, 1]  Gain trim knob distribution
# ═══════════════════════════════════════════════════════════════════════════
ax_gain = fig.add_subplot(gs[1, 1])

x_stages = np.arange(N_STAGES)
knob_gain_pct = (all_knob_gain - 1.0) * 100   # relative % deviation from 1.0

# Violin-style scatter + box per stage
for s in range(N_STAGES):
    vals = knob_gain_pct[:, s]
    jitter = np.random.default_rng(s).uniform(-0.12, 0.12, N_MC)
    ax_gain.scatter(np.full(N_MC, s) + jitter, vals,
                    color=stage_colors[s], s=10, alpha=0.45, zorder=4)
    ax_gain.boxplot(vals, positions=[s], widths=0.28, patch_artist=True,
                    medianprops=dict(color=DARK, lw=2),
                    whiskerprops=dict(color=stage_colors[s], lw=1.0),
                    capprops=dict(color=stage_colors[s], lw=1.2),
                    flierprops=dict(marker='', ms=0),
                    boxprops=dict(facecolor=stage_colors[s], alpha=0.35,
                                  linewidth=0.8, edgecolor=stage_colors[s]),
                    manage_ticks=False)

ax_gain.axhline(0, color=MUTED, lw=0.7, label="Target (0% deviation)")
ax_gain.axhspan(-GAIN_TRIM_STEP*100/2, GAIN_TRIM_STEP*100/2,
                color=GREEN, alpha=0.12, label=f"±½ step ({GAIN_TRIM_STEP*100/2:.3f}%)")
ax_gain.axhspan(-GAIN_MIS_SIGMA/GAIN_IDEAL*100,
                 GAIN_MIS_SIGMA/GAIN_IDEAL*100,
                color=ORANGE, alpha=0.08, label=f"±σ gain error ({GAIN_MIS_SIGMA/GAIN_IDEAL*100:.2f}%)")

for s in range(N_STAGES):
    ax_gain.plot([], [], color=stage_colors[s], lw=5, alpha=0.45, label=f"Stage {s+1}")

sax(ax_gain, "Gain Trim Knobs  (knob_gain − 1, %)",
    "Stage", "Knob Value − 1 (%)", minory=0.25)
ax_gain.set_xticks(x_stages)
ax_gain.set_xticklabels([f"S{s+1}" for s in range(N_STAGES)])
ax_gain.legend(fontsize=6.5, framealpha=0.2, ncol=2, loc="upper right")

# ═══════════════════════════════════════════════════════════════════════════
# [1, 2]  Calibration residual error + SNDR improvement bar
# ═══════════════════════════════════════════════════════════════════════════
ax_res = fig.add_subplot(gs[1, 2])

# Residual errors after calibration (histogram)
thr_res_flat  = all_thr_residual.flatten()  * 1e3    # mV, all stages/comps/runs
gain_res_flat = all_gain_residual.flatten() * 1e3    # mV/V, all stages/runs

bins_t = np.linspace(thr_res_flat.min()*1.05,  thr_res_flat.max()*1.05,  40)
bins_g = np.linspace(gain_res_flat.min()*1.05, gain_res_flat.max()*1.05, 40)

ax_res.hist(thr_res_flat,  bins=bins_t, color=ORANGE, alpha=0.65,
            density=True, label=f"Thr residual  σ={thr_res_flat.std():.2f} mV")
ax_res.hist(gain_res_flat, bins=bins_g, color=PURPLE, alpha=0.55,
            density=True, label=f"Gain residual σ={gain_res_flat.std():.3f} mV/V")
ax_res.axvline(0, color=MUTED, lw=0.8)
ax_res.axvline( THR_TRIM_STEP*1e3/2, color=ORANGE, lw=0.7, ls="--", alpha=0.7,
                label=f"±½ thr step ({THR_TRIM_STEP*1e3/2:.1f}mV)")
ax_res.axvline(-THR_TRIM_STEP*1e3/2, color=ORANGE, lw=0.7, ls="--", alpha=0.7)

# Inset: SNDR improvement bar
ax_ins = ax_res.inset_axes([0.56, 0.40, 0.40, 0.55])
ax_ins.patch.set_facecolor(DARK)
xs    = [0, 1]
vals  = [pk_uncal, pk_cal]
colrs = [PURPLE, GREEN]
bars  = ax_ins.bar(xs, vals, color=colrs, alpha=0.75, width=0.5, zorder=3)
ax_ins.set_ylim(0, IDEAL_PEAK + 6)
ax_ins.set_xticks([0, 1]); ax_ins.set_xticklabels(["Uncal", "Cal"], fontsize=6.5)
ax_ins.axhline(sv_ideal.max(), color=CYAN, lw=1.0, ls="--", alpha=0.7)
ax_ins.text(0.5, sv_ideal.max()+0.8, "Ideal", ha="center",
            fontsize=6.5, color=CYAN, transform=ax_ins.get_xaxis_transform())
for b, v in zip(bars, vals):
    ax_ins.text(b.get_x()+b.get_width()/2, v+0.5, f"{v:.1f}",
                ha="center", fontsize=7, color=TEXT)
ax_ins.set_ylabel("SNDR (dB)", fontsize=6); ax_ins.tick_params(labelsize=6)
ax_ins.yaxis.set_minor_locator(MultipleLocator(5))
ax_ins.grid(True, lw=0.4, alpha=0.5); ax_ins.grid(True, which="minor", lw=0.2, alpha=0.25)
ax_ins.set_facecolor(DARK)
for sp in ax_ins.spines.values(): sp.set_edgecolor(GRID)
ax_ins.tick_params(colors=MUTED)
ax_ins.yaxis.label.set_color(MUTED)

sax(ax_res, "Calibration Residual Errors  (post-trim)",
    "Residual Error (mV  or  mV/V)", "Density", minory=None)
ax_res.legend(fontsize=7.0, framealpha=0.25, loc="upper left")

plt.savefig("/mnt/user-data/outputs/pipeline_adc_calibrated.png",
            dpi=150, bbox_inches="tight", facecolor=DARK)
print("\nSaved → /mnt/user-data/outputs/pipeline_adc_calibrated.png")
