"""
4-Stage Pipeline ADC Model
- 2 bits per stage, no redundancy, 8-bit total
- Comparator offsets:  Gaussian σ = 30 mV  (3 comp × 4 stages = 12 total)
- Residue amp gain mismatch: Gaussian σ = 30 mV/V around ideal gain = 4 V/V
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator

# ── ADC constants ──────────────────────────────────────────────────────────
VREF        = 1.0
N_STAGES    = 4
BITS_STAGE  = 2
GAIN_IDEAL  = 2 ** BITS_STAGE       # 4 V/V
TOTAL_BITS  = N_STAGES * BITS_STAGE # 8
LEVELS      = 2 ** TOTAL_BITS       # 256
LSB         = 2 * VREF / LEVELS     # 7.8125 mV

OFFSET_SIGMA    = 0.030             # 30 mV  comparator offset std-dev
GAIN_MIS_SIGMA  = 0.030             # 30 mV/V residue amp gain-error std-dev
N_COMP          = 3                 # comparators per stage (2-bit flash)

NOMINAL_THR = np.array([-VREF/2, 0.0, VREF/2])
DAC_LEVELS  = np.array([-3*VREF/4, -VREF/4, VREF/4, 3*VREF/4])

RNG = np.random.default_rng(42)

# ── Draw errors once ───────────────────────────────────────────────────────
comp_offsets  = RNG.normal(0.0, OFFSET_SIGMA,   size=(N_STAGES, N_COMP))
gain_errors   = RNG.normal(0.0, GAIN_MIS_SIGMA, size=N_STAGES)   # delta-G per stage
eff_thr       = NOMINAL_THR[np.newaxis, :] + comp_offsets         # (4,3)
eff_gain      = GAIN_IDEAL + gain_errors                           # (4,)

print(f"\nComparator offsets (mV):")
for s in range(N_STAGES):
    v = comp_offsets[s]*1e3
    print(f"  Stage {s+1}: [{v[0]:+.1f}, {v[1]:+.1f}, {v[2]:+.1f}] mV")

print(f"\nResidue amp gains (V/V)  [ideal = {GAIN_IDEAL:.3f}]:")
for s in range(N_STAGES):
    print(f"  Stage {s+1}: {eff_gain[s]:.4f}  (Δ = {gain_errors[s]*1e3:+.1f} mV/V)")

# ── Stage models ───────────────────────────────────────────────────────────
def _stage(vin, stage_idx, use_offset, use_gain):
    thr  = eff_thr[stage_idx]  if use_offset else NOMINAL_THR
    gain = eff_gain[stage_idx] if use_gain   else GAIN_IDEAL
    code = int(np.clip(np.searchsorted(thr, vin, side='right'), 0, 3))
    res  = (vin - DAC_LEVELS[code]) * gain
    return code, res

def pipeline_adc(x, use_offset=False, use_gain=False):
    out = np.empty(len(x), dtype=np.int32)
    for i, v in enumerate(x):
        d = 0; r = float(v)
        for s in range(N_STAGES):
            r = np.clip(r, -VREF, VREF)
            sc, r = _stage(r, s, use_offset, use_gain)
            d = (d << BITS_STAGE) | sc
        out[i] = d
    return out

def code2volt(c):
    return (2*c - (LEVELS-1)) / LEVELS * VREF

# ── SNDR (3-bin Hann main-lobe) ────────────────────────────────────────────
def compute_sndr(sig, fs):
    N   = len(sig); win = np.hanning(N); W = np.sum(win**2)
    spec = np.fft.rfft(sig * win)
    pwr  = np.abs(spec)**2 / W
    sb   = int(np.argmax(pwr[1:])) + 1
    lobe = [b for b in (sb-1, sb, sb+1) if 0 < b < len(pwr)]
    Ps   = np.sum(pwr[lobe])
    mask = np.ones(len(pwr), bool); mask[0] = False
    for b in lobe: mask[b] = False
    Pn  = np.sum(pwr[mask])
    sndr = 10*np.log10(Ps/Pn); enob = (sndr-1.76)/6.02
    freqs  = np.fft.rfftfreq(N, 1/fs)
    mdb    = 10*np.log10(np.maximum(pwr, 1e-30)); mdb -= mdb[sb]
    return dict(sndr=sndr, enob=enob, freqs=freqs, mag_db=mdb, sig_bin=sb)

# ── Simulation ─────────────────────────────────────────────────────────────
FS = 1_000_000; N_FFT = 4096; M = 97
fin = M * FS / N_FFT
t   = np.arange(N_FFT) / FS
A   = 0.999 * VREF
vin = A * np.sin(2*np.pi*fin*t)

c_id   = pipeline_adc(vin, False, False); v_id   = code2volt(c_id)
c_off  = pipeline_adc(vin, True,  False); v_off  = code2volt(c_off)
c_all  = pipeline_adc(vin, True,  True ); v_all  = code2volt(c_all)

r_id  = compute_sndr(v_id,  FS)
r_off = compute_sndr(v_off, FS)
r_all = compute_sndr(v_all, FS)

# SNDR vs amplitude (3 variants)
amp_dB  = np.linspace(-60, -0.05, 120)
sv_id, sv_off, sv_all = [], [], []
for a in 10**(amp_dB/20):
    si = a * np.sin(2*np.pi*fin*t)
    sv_id.append( compute_sndr(code2volt(pipeline_adc(si, False, False)), FS)['sndr'])
    sv_off.append(compute_sndr(code2volt(pipeline_adc(si, True,  False)), FS)['sndr'])
    sv_all.append(compute_sndr(code2volt(pipeline_adc(si, True,  True )), FS)['sndr'])
sv_id  = np.array(sv_id)
sv_off = np.array(sv_off)
sv_all = np.array(sv_all)

IDEAL_PEAK = 6.02*TOTAL_BITS + 1.76
ideal_sndr = np.minimum(amp_dB + IDEAL_PEAK, IDEAL_PEAK)

# DC residue traces (stage 1, all three modes)
vin_dc = np.linspace(-VREF, VREF, 8000)

def residue_trace(vin_arr, use_offset, use_gain, stage=0):
    out = []
    for v in vin_arr:
        r = float(v)
        for s in range(stage+1):
            r = np.clip(r, -VREF, VREF)
            _, r = _stage(r, s, use_offset, use_gain)
        out.append(r)
    return np.array(out)

rt_id  = residue_trace(vin_dc, False, False)
rt_off = residue_trace(vin_dc, True,  False)
rt_all = residue_trace(vin_dc, True,  True )

# DNL / INL
codes_dc_id  = pipeline_adc(vin_dc, False, False)
codes_dc_off = pipeline_adc(vin_dc, True,  False)
codes_dc_all = pipeline_adc(vin_dc, True,  True )

def inl_dnl(codes_dc, vin_dc):
    trans = {}
    for i,c in enumerate(codes_dc):
        if c not in trans: trans[c] = vin_dc[i]
    sc = sorted(trans)
    dnl, inl, acc, axis = [], [], 0.0, []
    for i in range(1, len(sc)-1):
        ck, ck1 = sc[i], sc[i+1] if i+1<len(sc) else None
        if ck1 is None: break
        d = (trans[ck1]-trans[ck]-LSB)/LSB
        acc += d; dnl.append(d); inl.append(acc); axis.append(ck)
    return np.array(dnl), np.array(inl), np.array(axis)

dnl_id,  inl_id,  ax_id  = inl_dnl(codes_dc_id,  vin_dc)
dnl_off, inl_off, ax_off = inl_dnl(codes_dc_off, vin_dc)
dnl_all, inl_all, ax_all = inl_dnl(codes_dc_all, vin_dc)

sndr_id  = r_id['sndr'];  enob_id  = r_id['enob']
sndr_off = r_off['sndr']; enob_off = r_off['enob']
sndr_all = r_all['sndr']; enob_all = r_all['enob']

print(f"\n{'─'*56}")
print(f"  σ_offset  = {OFFSET_SIGMA*1e3:.0f} mV      σ/LSB = {OFFSET_SIGMA/LSB:.2f}")
print(f"  σ_gain    = {GAIN_MIS_SIGMA*1e3:.0f} mV/V   σ/G_ideal = {GAIN_MIS_SIGMA/GAIN_IDEAL*100:.2f}%")
print(f"{'─'*56}")
print(f"  {'':28s} Ideal    +Offsets  +Gain")
print(f"  {'SNDR (dB)':28s} {sndr_id:.2f}    {sndr_off:.2f}     {sndr_all:.2f}")
print(f"  {'ENOB (bits)':28s} {enob_id:.2f}    {enob_off:.2f}      {enob_all:.2f}")
print(f"  {'DNL peak (LSB)':28s} {max(abs(dnl_id)):.3f}   {max(abs(dnl_off)):.3f}    {max(abs(dnl_all)):.3f}")
print(f"  {'INL peak (LSB)':28s} {max(abs(inl_id)):.3f}   {max(abs(inl_off)):.3f}   {max(abs(inl_all)):.3f}")
print(f"{'─'*56}")

# ── Plot ───────────────────────────────────────────────────────────────────
DARK  = "#0A0E14"; GRID = "#1C2330"
CYAN  = "#00E5FF"; AMBER = "#FFB300"; RED = "#FF4560"
GREEN = "#00E396"; GHOST = "#2A3545"; TEXT = "#CDD6E0"
MUTED = "#6E7D8C"; ORANGE = "#FF6D00"; PURPLE = "#CE93D8"

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": DARK,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "grid.color": GRID, "grid.linewidth": 0.5,
    "text.color": TEXT, "font.family": "monospace", "font.size": 9,
})

fig = plt.figure(figsize=(18, 13))
gs  = gridspec.GridSpec(3, 3, figure=fig,
                        hspace=0.62, wspace=0.38,
                        left=0.06, right=0.97, top=0.91, bottom=0.06)

fig.text(0.5, 0.963,
    "4-STAGE PIPELINE ADC   ·   2 BIT/STAGE   ·   NO REDUNDANCY   ·   "
    "COMPARATOR OFFSETS σ=30mV   ·   GAIN MISMATCH σ=30mV/V",
    ha="center", fontsize=12, fontweight="bold", color=CYAN, fontfamily="monospace")

info = (f"fin={fin/1e3:.2f} kHz   fs={FS/1e6:.0f} MS/s   N={N_FFT}   "
        f"σ_off/LSB={OFFSET_SIGMA/LSB:.2f}   σ_G/G={GAIN_MIS_SIGMA/GAIN_IDEAL*100:.2f}%   │   "
        f"SNDR: ideal={sndr_id:.1f} → +offsets={sndr_off:.1f} → +gain={sndr_all:.1f} dB   "
        f"ENOB={enob_all:.2f} bits")
fig.text(0.5, 0.933, info, ha="center", fontsize=8.8, color=AMBER, fontfamily="monospace")

def sax(ax, title, xlabel, ylabel, miny=None):
    ax.set_title(title, fontsize=9, pad=5)
    ax.set_xlabel(xlabel, fontsize=8); ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, which="major", lw=0.5)
    ax.grid(True, which="minor", lw=0.22, alpha=0.35)
    if miny: ax.yaxis.set_minor_locator(MultipleLocator(miny))

LBL_ID  = f"Ideal  ({sndr_id:.1f} dB)"
LBL_OFF = f"+Offsets ({sndr_off:.1f} dB)"
LBL_ALL = f"+Offsets+Gain ({sndr_all:.1f} dB)"

ns = 256

# [0,0] Time-domain
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(t[:ns]*1e6, vin[:ns],    color=MUTED,  lw=1.1, label="Input (analog)", zorder=1)
ax0.step(t[:ns]*1e6, v_id[:ns],   color=CYAN,   lw=0.85, where="post", alpha=0.5, label=LBL_ID,  zorder=2)
ax0.step(t[:ns]*1e6, v_off[:ns],  color=ORANGE, lw=0.85, where="post", alpha=0.7, label=LBL_OFF, zorder=3)
ax0.step(t[:ns]*1e6, v_all[:ns],  color=PURPLE, lw=0.9,  where="post", alpha=0.9, label=LBL_ALL, zorder=4)
sax(ax0, "Time-Domain  (first 256 samples)", "Time (µs)", "Voltage (V)", miny=0.125)
ax0.legend(fontsize=6.5, framealpha=0.2, loc="upper right")
ax0.set_ylim(-1.15, 1.15)

# [0,1] Quantisation error
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(t[:ns]*1e6, (v_id[:ns] -vin[:ns])/LSB, color=CYAN,   lw=0.8, alpha=0.6, label=LBL_ID)
ax1.plot(t[:ns]*1e6, (v_off[:ns]-vin[:ns])/LSB, color=ORANGE, lw=0.8, alpha=0.8, label=LBL_OFF)
ax1.plot(t[:ns]*1e6, (v_all[:ns]-vin[:ns])/LSB, color=PURPLE, lw=0.8, alpha=0.9, label=LBL_ALL)
ax1.axhline( 0.5, color=MUTED, lw=0.6, ls=":", alpha=0.6)
ax1.axhline(-0.5, color=MUTED, lw=0.6, ls=":", alpha=0.6)
sax(ax1, "Quantisation Error  (LSB units)", "Time (µs)", "Error (LSB)", miny=0.5)
ax1.legend(fontsize=6.5, framealpha=0.2)

# [0,2] Gain mismatch bar chart
ax2 = fig.add_subplot(gs[0, 2])
x = np.arange(N_STAGES)
bars = ax2.bar(x, eff_gain, color=PURPLE, alpha=0.8, width=0.5, label="Actual gain", zorder=3)
ax2.axhline(GAIN_IDEAL, color=CYAN, lw=1.2, ls="--", label=f"Ideal gain = {GAIN_IDEAL}", zorder=2)
ax2.axhspan(GAIN_IDEAL - GAIN_MIS_SIGMA, GAIN_IDEAL + GAIN_MIS_SIGMA,
            color=CYAN, alpha=0.08, label=f"±σ = ±{GAIN_MIS_SIGMA*1e3:.0f} mV/V")
for b, g in zip(bars, eff_gain):
    dg = g - GAIN_IDEAL
    ax2.text(b.get_x()+b.get_width()/2, g + 0.002,
             f"{dg*1e3:+.1f}\nmV/V", ha="center", va="bottom", fontsize=7, color=PURPLE)

# Annotate stage weight: error in stage s is amplified by GAIN^(N-1-s) at output
ax2_r = ax2.twinx()
weights = [GAIN_IDEAL**(N_STAGES-1-s) for s in range(N_STAGES)]
ax2_r.plot(x, weights, color=RED, marker='o', ms=5, lw=1.0, ls=':', label="Output weight ×")
ax2_r.set_ylabel("Stage error weight at output", fontsize=7, color=RED)
ax2_r.tick_params(axis='y', colors=RED, labelsize=7)
ax2_r.spines['right'].set_color(RED)

# Merge legends
lines1, labs1 = ax2.get_legend_handles_labels()
lines2, labs2 = ax2_r.get_legend_handles_labels()
ax2.legend(lines1+lines2, labs1+labs2, fontsize=6.5, framealpha=0.2, loc="lower right")

sax(ax2, "Residue Amp Gain Mismatch per Stage", "Stage", "Actual Gain (V/V)")
ax2.set_xticks(x); ax2.set_xticklabels([f"S{s+1}" for s in range(N_STAGES)])
ax2.set_ylim(GAIN_IDEAL - 6*GAIN_MIS_SIGMA, GAIN_IDEAL + 6*GAIN_MIS_SIGMA)

# [1, 0:2] Output spectra — all three
ax3 = fig.add_subplot(gs[1, :2])
f_kHz = r_id['freqs']/1e3
ax3.plot(f_kHz, r_id['mag_db'],  color=CYAN,   lw=0.6, alpha=0.6,  label=LBL_ID)
ax3.plot(f_kHz, r_off['mag_db'], color=ORANGE, lw=0.7, alpha=0.75, label=LBL_OFF)
ax3.plot(f_kHz, r_all['mag_db'], color=PURPLE, lw=0.8, alpha=0.95, label=LBL_ALL)
ax3.fill_between(f_kHz, r_all['mag_db'], -135, color=PURPLE, alpha=0.05)

sb = r_all['sig_bin']
ax3.axvline(r_all['freqs'][sb]/1e3, color=AMBER, lw=0.7, ls="--", alpha=0.4)
ax3.annotate(
    f" Ideal:       {sndr_id:.2f} dB\n"
    f" +Offsets:   {sndr_off:.2f} dB\n"
    f" +Gain:      {sndr_all:.2f} dB\n"
    f" ENOB:       {enob_all:.2f} bits",
    xy=(r_all['freqs'][sb]/1e3, 0), xytext=(r_all['freqs'][sb]/1e3+65, -14),
    fontsize=8, color=AMBER,
    arrowprops=dict(arrowstyle="->", color=AMBER, lw=0.9),
    bbox=dict(boxstyle="round,pad=0.4", fc=DARK, ec=AMBER, lw=0.9, alpha=0.93)
)
sax(ax3, "Output Spectrum  (Hann-windowed FFT  ·  progressive error budget)",
    "Frequency (kHz)", "Magnitude (dBFS)", miny=10)
ax3.legend(fontsize=7.5, framealpha=0.2)
ax3.set_xlim(0, FS/2e3); ax3.set_ylim(-130, 10)

# [1,2] Stage 1 residue transfer — ideal vs full errors
ax4 = fig.add_subplot(gs[1, 2])
ax4.plot(vin_dc, rt_id,  color=CYAN,   lw=1.1, alpha=0.6, label="Ideal")
ax4.plot(vin_dc, rt_off, color=ORANGE, lw=1.0, alpha=0.8, label="+Offsets")
ax4.plot(vin_dc, rt_all, color=PURPLE, lw=1.0, alpha=0.9, label="+Offsets+Gain")
for th in eff_thr[0]:
    ax4.axvline(th, color=RED, lw=0.5, ls=":", alpha=0.5)
ax4.axhline( VREF, color=GHOST, lw=0.8, ls="--"); ax4.axhline(-VREF, color=GHOST, lw=0.8, ls="--")

# Annotate actual gain vs ideal
mid = vin_dc[len(vin_dc)//4]
slope_all  = eff_gain[0]
ax4.annotate(f"G₁={slope_all:.4f}\n(ΔG={gain_errors[0]*1e3:+.1f}mV/V)",
             xy=(0.3, rt_all[int(0.35*len(vin_dc))]),
             xytext=(0.55, 0.55), fontsize=7, color=PURPLE,
             arrowprops=dict(arrowstyle='->', color=PURPLE, lw=0.8))
sax(ax4, "Stage 1 Residue Transfer  (gain slope changes)",
    "Input (V)", "Residue (V)", miny=0.25)
ax4.legend(fontsize=7, framealpha=0.2)
ax4.set_ylim(-1.40, 1.40)

# [2,0] DNL
ax5 = fig.add_subplot(gs[2, 0])
ax5.bar(ax_id,  dnl_id,  color=CYAN,   alpha=0.45, width=0.7, label="Ideal")
ax5.bar(ax_off, dnl_off, color=ORANGE, alpha=0.6,  width=0.6, label="+Offsets")
ax5.bar(ax_all, dnl_all, color=PURPLE, alpha=0.75, width=0.5, label="+Offsets+Gain")
ax5.axhline( 1.0, color=RED,  lw=0.7, ls=":", alpha=0.7, label="+1 LSB (miss)")
ax5.axhline(-1.0, color=RED,  lw=0.7, ls=":", alpha=0.7)
ax5.axhline(0, color=MUTED, lw=0.5)
sax(ax5, "DNL  (Differential Non-Linearity)", "Code", "DNL (LSB)", miny=1.0)
ax5.legend(fontsize=6.5, framealpha=0.2, ncol=2)
ax5.set_xlim(0, LEVELS)

# [2,1] INL
ax6 = fig.add_subplot(gs[2, 1])
ax6.plot(ax_id,  inl_id,  color=CYAN,   lw=1.0, alpha=0.6, label="Ideal")
ax6.plot(ax_off, inl_off, color=ORANGE, lw=1.1, alpha=0.8, label="+Offsets")
ax6.plot(ax_all, inl_all, color=PURPLE, lw=1.2, alpha=0.9, label="+Offsets+Gain")
ax6.axhline(0, color=MUTED, lw=0.5)
sax(ax6, "INL  (Integral Non-Linearity)", "Code", "INL (LSB)", miny=2.0)
ax6.legend(fontsize=7, framealpha=0.2)
ax6.set_xlim(0, LEVELS)

# [2,2] SNDR vs amplitude
ax7 = fig.add_subplot(gs[2, 2])
ax7.plot(amp_dB, ideal_sndr, color=GHOST,  lw=1.1, ls="--", label=f"Ideal 8-bit")
ax7.plot(amp_dB, sv_id,      color=CYAN,   lw=1.3, alpha=0.65, label=LBL_ID)
ax7.plot(amp_dB, sv_off,     color=ORANGE, lw=1.4, alpha=0.80, label=LBL_OFF)
ax7.plot(amp_dB, sv_all,     color=PURPLE, lw=1.8, label=LBL_ALL)
ax7.axvline(20*np.log10(A), color=AMBER, lw=0.7, ls=":", alpha=0.8)

# Annotate SNDR floor set by gain mismatch at high amplitude
gain_noise_floor = sv_all[-5]   # near full scale
ax7.axhline(gain_noise_floor, color=PURPLE, lw=0.5, ls=":", alpha=0.6)
ax7.text(amp_dB[0]+0.5, gain_noise_floor+1.2, f"{gain_noise_floor:.1f} dB", fontsize=7, color=PURPLE)

sax(ax7, "SNDR vs Input Amplitude", "Amplitude (dBFS)", "SNDR (dB)", miny=5)
ax7.legend(fontsize=6.5, framealpha=0.2, loc="upper left")
ax7.set_xlim(amp_dB[0], 0); ax7.set_ylim(0, IDEAL_PEAK+6)

plt.savefig("/mnt/user-data/outputs/pipeline_adc_gain_mismatch.png",
            dpi=150, bbox_inches="tight", facecolor=DARK)
print("\nSaved -> /mnt/user-data/outputs/pipeline_adc_gain_mismatch.png")
