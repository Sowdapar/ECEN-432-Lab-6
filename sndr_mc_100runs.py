"""
100-run Monte Carlo  —  Average SNDR vs Input Amplitude
4-Stage Pipeline ADC, 2.5 bit/stage, 1-bit redundancy + DEC
  σ_offset = 30 mV  |  σ_gain = 30 mV/V
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

# ── ADC constants ──────────────────────────────────────────────────────────
VREF          = 1.0
N_STAGES      = 4
BITS_STAGE    = 2
BITS_RAW      = BITS_STAGE + 1
GAIN_IDEAL    = float(2 ** BITS_STAGE)       # 4 V/V
N_COMP        = 2 ** BITS_RAW - 1            # 7
N_DAC         = 2 ** BITS_RAW                # 8
TOTAL_BITS    = N_STAGES * BITS_STAGE        # 8
LEVELS        = 2 ** TOTAL_BITS              # 256
IDEAL_PEAK    = 6.02 * TOTAL_BITS + 1.76    # ~50 dB theoretical max

OFFSET_SIGMA   = 0.030   # 30 mV
GAIN_MIS_SIGMA = 0.030   # 30 mV/V
N_MC           = 100

NOMINAL_THR = np.array([-3,-2,-1,0,1,2,3], dtype=float) * VREF / 4
DAC_LEVELS  = np.array([-7,-5,-3,-1,1,3,5,7], dtype=float) * VREF / 8

# ── Signal ─────────────────────────────────────────────────────────────────
FS = 1_000_000; N_FFT = 4096; M = 97
fin = M * FS / N_FFT
t   = np.arange(N_FFT) / FS

amp_dB     = np.linspace(-60, -0.05, 120)
amp_linear = 10 ** (amp_dB / 20)
ideal_sndr = np.minimum(amp_dB + IDEAL_PEAK, IDEAL_PEAK)

# ── Vectorised ADC kernel ──────────────────────────────────────────────────
def pipeline_adc_vec(x, thr, gain):
    r   = np.clip(x, -VREF, VREF).copy()
    rec = np.zeros(len(x))
    for s in range(N_STAGES):
        codes = np.clip(np.sum(r[:,None] > thr[s][None,:], axis=1), 0, N_DAC-1)
        dac   = DAC_LEVELS[codes]
        rec  += dac / (GAIN_IDEAL ** s)
        r     = np.clip((r - dac) * gain[s], -VREF, VREF)
    return rec

# ── SNDR (3-bin Hann exclusion) ────────────────────────────────────────────
def compute_sndr(sig):
    win = np.hanning(len(sig)); W = np.sum(win**2)
    pwr = np.abs(np.fft.rfft(sig * win))**2 / W
    sb  = int(np.argmax(pwr[1:])) + 1
    lobe = [b for b in (sb-1, sb, sb+1) if 0 < b < len(pwr)]
    Ps   = np.sum(pwr[lobe])
    mask = np.ones(len(pwr), bool); mask[0] = False
    for b in lobe: mask[b] = False
    return 10 * np.log10(Ps / np.sum(pwr[mask]))

# ── Ideal reference (no errors) ───────────────────────────────────────────
thr_ideal  = np.tile(NOMINAL_THR, (N_STAGES, 1))
gain_ideal = np.full(N_STAGES, GAIN_IDEAL)

sv_ideal = np.array([
    compute_sndr(pipeline_adc_vec(a * np.sin(2*np.pi*fin*t), thr_ideal, gain_ideal))
    for a in amp_linear
])

# ── 100-run Monte Carlo ────────────────────────────────────────────────────
rng = np.random.default_rng(7)

mc_off = np.zeros((N_MC, len(amp_dB)))   # offsets only  (DEC active)
mc_all = np.zeros((N_MC, len(amp_dB)))   # offsets + gain mismatch

print(f"Running {N_MC}-run Monte Carlo …")
for run in range(N_MC):
    co      = rng.normal(0, OFFSET_SIGMA,   (N_STAGES, N_COMP))
    ge      = rng.normal(0, GAIN_MIS_SIGMA,  N_STAGES)
    thr_r   = NOMINAL_THR[None,:] + co
    gain_r  = GAIN_IDEAL + ge

    for j, a in enumerate(amp_linear):
        si = a * np.sin(2*np.pi*fin*t)
        mc_off[run, j] = compute_sndr(pipeline_adc_vec(si, thr_r, gain_ideal))
        mc_all[run, j] = compute_sndr(pipeline_adc_vec(si, thr_r, gain_r))

    if (run+1) % 25 == 0:
        print(f"  {run+1}/{N_MC}")

# Statistics
mean_off = mc_off.mean(0);  std_off = mc_off.std(0)
mean_all = mc_all.mean(0);  std_all = mc_all.std(0)

pk_off = mean_off.max();  pk_off_s = std_off[mean_off.argmax()]
pk_all = mean_all.max();  pk_all_s = std_all[mean_all.argmax()]

print(f"\nPeak mean SNDR — +Offsets (DEC):    {pk_off:.2f} ± {pk_off_s:.2f} dB")
print(f"Peak mean SNDR — +Offsets+Gain:     {pk_all:.2f} ± {pk_all_s:.2f} dB  "
      f"(ENOB = {(pk_all-1.76)/6.02:.2f})")

# ── Plot ───────────────────────────────────────────────────────────────────
DARK  = "#0A0E14"; GRID  = "#1C2330"; GHOST = "#2A3545"
TEXT  = "#CDD6E0"; MUTED = "#6E7D8C"
CYAN  = "#00E5FF"; AMBER = "#FFB300"; RED   = "#FF4560"
ORANGE= "#FF6D00"; PURPLE= "#CE93D8"

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": DARK,
    "axes.edgecolor":   GRID, "axes.labelcolor": TEXT,
    "axes.titlecolor":  TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color":       GRID, "grid.linewidth": 0.5,
    "text.color":       TEXT, "font.family": "monospace", "font.size": 10,
})

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor(DARK)

# ── faint individual runs ──────────────────────────────────────────────────
for run in range(N_MC):
    ax.plot(amp_dB, mc_off[run], color=ORANGE, lw=0.35, alpha=0.08)
    ax.plot(amp_dB, mc_all[run], color=PURPLE, lw=0.35, alpha=0.08)

# ── ±1σ bands ──────────────────────────────────────────────────────────────
ax.fill_between(amp_dB, mean_off-std_off, mean_off+std_off, color=ORANGE, alpha=0.20)
ax.fill_between(amp_dB, mean_all-std_all, mean_all+std_all, color=PURPLE, alpha=0.20)

# ── mean lines ─────────────────────────────────────────────────────────────
ax.plot(amp_dB, ideal_sndr, color=GHOST,  lw=1.4, ls="--",
        label="Ideal 8-bit  (theoretical)")
ax.plot(amp_dB, sv_ideal,   color=CYAN,   lw=1.8, alpha=0.75,
        label=f"ADC ideal — no errors  ({sv_ideal.max():.1f} dB pk)")
ax.plot(amp_dB, mean_off,   color=ORANGE, lw=2.4,
        label=f"Mean  +Offsets → DEC  ({pk_off:.1f} ± {pk_off_s:.1f} dB pk)")
ax.plot(amp_dB, mean_all,   color=PURPLE, lw=2.4,
        label=f"Mean  +Offsets + Gain  ({pk_all:.1f} ± {pk_all_s:.1f} dB pk)")

# ── peak error-bar annotations ─────────────────────────────────────────────
for mvec, svec, col, dy in [(mean_off, std_off, ORANGE, +4),
                             (mean_all, std_all, PURPLE, -7)]:
    idx = mvec.argmax(); xp = amp_dB[idx]; yp = mvec[idx]; sp = svec[idx]
    ax.errorbar(xp, yp, yerr=sp, fmt='o', color=col,
                capsize=6, capthick=1.5, ms=7, zorder=8)
    ax.annotate(
        f"  {yp:.2f} ± {sp:.2f} dB\n  ENOB = {(yp-1.76)/6.02:.2f} bits",
        xy=(xp, yp), xytext=(xp - 16, yp + dy),
        fontsize=9.5, color=col,
        arrowprops=dict(arrowstyle="->", color=col, lw=1.0),
        bbox=dict(boxstyle="round,pad=0.4", fc=DARK, ec=col, lw=1.0, alpha=0.93))

# ── full-scale marker ──────────────────────────────────────────────────────
ax.axvline(-0.009, color=AMBER, lw=0.9, ls=":", alpha=0.7, label="Full-scale (−0 dBFS)")

# ── legend (add ±1σ band patches) ─────────────────────────────────────────
handles, labels = ax.get_legend_handles_labels()
handles += [Patch(fc=ORANGE, alpha=0.35, label="±1σ  +Offsets (DEC)"),
            Patch(fc=PURPLE, alpha=0.35, label="±1σ  +Offsets+Gain")]
labels  += ["±1σ  +Offsets (DEC)", "±1σ  +Offsets+Gain"]
ax.legend(handles, labels, fontsize=9, framealpha=0.25,
          loc="upper left", ncol=2)

# ── axes dressing ──────────────────────────────────────────────────────────
ax.set_xlim(amp_dB[0], 0)
ax.set_ylim(0, IDEAL_PEAK + 8)
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(5))
ax.grid(True, which="major", lw=0.55)
ax.grid(True, which="minor", lw=0.22, alpha=0.35)
ax.set_xlabel("Input Amplitude (dBFS)", fontsize=11)
ax.set_ylabel("SNDR (dB)", fontsize=11)
ax.set_title(
    f"Average SNDR vs Input Amplitude  —  {N_MC}-Run Monte Carlo\n"
    f"4-Stage Pipeline ADC · 2.5 bit/stage · 7 comparators/stage · Gain = 4 V/V · DEC\n"
    f"σ_offset = {OFFSET_SIGMA*1e3:.0f} mV per comparator  │  "
    f"σ_gain = {GAIN_MIS_SIGMA*1e3:.0f} mV/V per stage  │  "
    f"faint traces = individual runs  │  band = ±1σ",
    fontsize=10.5, color=CYAN, pad=10)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/sndr_mc_100runs.png",
            dpi=150, bbox_inches="tight", facecolor=DARK)
print("\nSaved → /mnt/user-data/outputs/sndr_mc_100runs.png")
