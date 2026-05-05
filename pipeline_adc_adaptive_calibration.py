# -*- coding: utf-8 -*-
"""
4-Stage Pipeline ADC  ──  Adaptive Calibration Methods Comparison
═══════════════════════════════════════════════════════════════════
Iteratively adjusts cal_thr[s,k] and cal_gain[s] knobs until
comparator offsets and residue-amp gain mismatches are fully
compensated.  Four methods compared against a foreground reference:

  Method 1 │ LMS Gradient Descent
           │ Widrow-Hoff sign-LMS on binary comparator decisions +
           │ normalised LMS (NLMS) on residue-amp output voltages.
           │ Each sample propagates through all stages; both knob
           │ types updated every step via instantaneous gradients.
  ─────────┼─────────────────────────────────────────────────────
  Method 2 │ Multi-Centroid AI-Loops + Parallel Thompson Sampling
           │ K=8 input-voltage centroids partition [-VREF, VREF].
           │ P=3 parallel Bayesian bandit agents each maintain a
           │ Beta(α,β) posterior per (stage, centroid) reflecting
           │ calibration-improvement probability.  Thompson Sampling
           │ selects the most promising centroid to probe; posteriors
           │ updated on success/failure; agents merged by confidence.
  ─────────┼─────────────────────────────────────────────────────
  Method 3 │ Reinforcement Learning  (per-knob Q-Learning, ε-greedy)
           │ Independent tabular Q-agent per (stage, knob) observes
           │ hardware-derived binary states, selects discrete delta
           │ actions, and updates Q-values via Bellman TD(0).
           │ ε decays exponentially from 0.95 → 0.05 over episodes.
  ─────────┼─────────────────────────────────────────────────────
  Method 4 │ Agentic AI  (4 specialised agents + shared message bus)
           │   OracleAgent     – injects test voltages, collects
           │                     comparator and residue measurements
           │   ThresholdAgent  – decodes comparator messages,
           │                     applies sign-corrected LMS updates
           │   GainAgent       – two-point gain estimation + soft
           │                     closed-loop update toward GAIN_IDEAL
           │   SupervisorAgent – watches convergence, anneals μ
           │                     when improvement stalls
  ─────────┼─────────────────────────────────────────────────────
  Reference│ Foreground binary-search (12-bit bisection) +
           │ two-point gain injection (baseline script algorithm)

All adaptive methods use ONLY hardware-observable quantities:
  • comparator binary output  hw_cmp(x, s, k, cal_thr)
  • residue-amp output        hw_res(x, s, cal_thr, cal_gain)
The hidden mc_eff_thr / mc_eff_gain parameters are NEVER accessed
directly inside the calibration functions (only via the hw_ oracles).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════════
#  ADC Constants & Simulation Parameters
# ═══════════════════════════════════════════════════════════════════════════════
VREF           = 1.0
N_STAGES       = 4
BITS_STAGE     = 2
GAIN_IDEAL     = 2 ** BITS_STAGE          # 4 V/V
TOTAL_BITS     = N_STAGES * BITS_STAGE    # 8
LEVELS         = 2 ** TOTAL_BITS          # 256
LSB            = 2 * VREF / LEVELS        # ~7.8 mV
OFFSET_SIGMA   = 0.030                    # 30 mV comparator offset σ
GAIN_MIS_SIGMA = 0.030                    # 30 mV/V residue-amp gain-error σ
N_COMP         = 3                        # comparators per stage
NOMINAL_THR    = np.array([-VREF / 2, 0.0, VREF / 2])
DAC_LEVELS     = np.array([-3 * VREF / 4, -VREF / 4, VREF / 4, 3 * VREF / 4])

FS         = 1_000_000
N_FFT      = 4096
M          = 97
fin        = M * FS / N_FFT
t          = np.arange(N_FFT, dtype=np.float64) / FS
N_RUNS     = 30          # MC runs (increase for tighter statistics)
AMP_PTS    = 20
amp_dB     = np.linspace(-60, -0.05, AMP_PTS)
IDEAL_PEAK = 6.02 * TOTAL_BITS + 1.76    # 49.92 dB ideal SNDR


# ═══════════════════════════════════════════════════════════════════════════════
#  ADC Model  (vectorised, identical to baseline)
# ═══════════════════════════════════════════════════════════════════════════════
def _stage_vec(vin_arr, thr, gain):
    codes  = (vin_arr >= thr[0]).astype(np.int32)
    codes += (vin_arr >= thr[1]).astype(np.int32)
    codes += (vin_arr >= thr[2]).astype(np.int32)
    codes  = np.clip(codes, 0, 3)
    return codes, (vin_arr - DAC_LEVELS[codes]) * gain


def pipeline_adc_vec(x, thresholds, gains):
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


def compute_sndr(sig):
    N    = len(sig)
    win  = np.hanning(N)
    W    = np.sum(win ** 2)
    pwr  = np.abs(np.fft.rfft(sig * win)) ** 2 / W
    sb   = int(np.argmax(pwr[1:])) + 1
    lobe = [b for b in (sb - 1, sb, sb + 1) if 0 < b < len(pwr)]
    Ps   = np.sum(pwr[lobe])
    mask = np.ones(len(pwr), bool)
    mask[0] = False
    for b in lobe:
        mask[b] = False
    return 10.0 * np.log10(Ps / max(np.sum(pwr[mask]), 1e-30))


# ═══════════════════════════════════════════════════════════════════════════════
#  Hardware Oracle  ── the ONLY interface to hardware used by all cal methods
# ═══════════════════════════════════════════════════════════════════════════════
def hw_cmp(x, s, k, cal_thr, mc_eff_thr):
    """Comparator binary output: 1 if fires, 0 otherwise."""
    return 1 if x >= mc_eff_thr[s, k] + cal_thr[s, k] else 0


def hw_res(x, s, cal_thr, cal_gain, mc_eff_thr, mc_eff_gain):
    """Residue-amp output, v_in, and ADC code for stage s at input x."""
    eff_thr = mc_eff_thr[s] + cal_thr[s]
    code    = int(np.clip(np.sum(x >= eff_thr), 0, 3))
    v_in    = x - DAC_LEVELS[code]
    return v_in * mc_eff_gain[s] * cal_gain[s], v_in, code


def _cal_rms(cal_thr, cal_gain, mc_eff_thr, mc_eff_gain):
    """Calibration error metrics for convergence tracking."""
    thr_rms  = float(np.sqrt(np.mean((mc_eff_thr + cal_thr - NOMINAL_THR) ** 2)))
    gain_rms = float(np.sqrt(np.mean((mc_eff_gain * cal_gain - GAIN_IDEAL) ** 2)))
    return thr_rms, gain_rms


# ═══════════════════════════════════════════════════════════════════════════════
#  METHOD 1 – LMS Gradient Descent  (Widrow-Hoff)
# ═══════════════════════════════════════════════════════════════════════════════
def calibrate_lms(mc_eff_thr, mc_eff_gain,
                  n_iter=1400, mu_thr=0.9e-3, mu_gain=0.10, seed=0):
    """
    Threshold  — sign-LMS:
        For each stage s and comparator k, a random test voltage v is
        injected.  The desired decision is d = (v >= NOMINAL_THR[k]).
        The actual decision is y = (v >= eff_thr[s,k]).
        sign-LMS update:
            cal_thr[s,k] += mu_thr * (d – y)          (e ∈ {-2,0,+2})

    Gain  — NLMS on residue error:
        After the threshold update the stage residue is observed:
            res_obs  = v_in * mc_eff_gain[s] * cal_gain[s]
            res_des  = v_in * GAIN_IDEAL
            e_g      = res_des – res_obs
        Gradient w.r.t. cal_gain[s]:  d(res)/d(cal_gain) ≈ v_in·(eff_g/cal_g)
        Normalised update:
            cal_gain[s] += mu_gain * e_g * v_in * mc_proxy / (v_in² + ε)
        where mc_proxy ≈ mc_eff_gain[s] estimated as eff_gain/cal_gain.

    Both knobs are updated on every sample as the signal cascades through
    all pipeline stages.
    """
    rng      = np.random.default_rng(seed)
    cal_thr  = np.zeros((N_STAGES, N_COMP))
    cal_gain = np.ones(N_STAGES)
    hist     = np.zeros((n_iter, 2))

    x_seq = rng.uniform(-VREF * 0.95, VREF * 0.95, n_iter)

    for i, x in enumerate(x_seq):
        v = x
        for s in range(N_STAGES):
            v = np.clip(v, -VREF, VREF)
            eff_thr   = mc_eff_thr[s] + cal_thr[s]
            eff_gain_s = mc_eff_gain[s] * cal_gain[s]

            # ── sign-LMS: threshold update ─────────────────────────────────
            for k in range(N_COMP):
                d = 1.0 if v >= NOMINAL_THR[k] else 0.0
                y = 1.0 if v >= eff_thr[k]     else 0.0
                cal_thr[s, k] += mu_thr * (d - y)   # e ∈ {-1, 0, +1}

            # ── NLMS: gain update  ─────────────────────────────────────────
            eff_thr_now = mc_eff_thr[s] + cal_thr[s]
            code  = int(np.clip(np.sum(v >= eff_thr_now), 0, 3))
            v_in  = v - DAC_LEVELS[code]
            res_obs = v_in * mc_eff_gain[s] * cal_gain[s]
            e_g   = v_in * GAIN_IDEAL - res_obs
            if abs(v_in) > 1e-6:
                mc_proxy = eff_gain_s / max(abs(cal_gain[s]), 1e-4) * np.sign(cal_gain[s])
                cal_gain[s] += mu_gain * e_g * v_in * mc_proxy / (v_in ** 2 + 1e-9)
                cal_gain[s]  = np.clip(cal_gain[s], 0.4, 2.5)

            # Pass calibrated residue to next stage
            v = np.clip(res_obs, -VREF, VREF)

        hist[i] = _cal_rms(cal_thr, cal_gain, mc_eff_thr, mc_eff_gain)

    return cal_thr, cal_gain, hist


# ═══════════════════════════════════════════════════════════════════════════════
#  METHOD 2 – Multi-Centroid AI-Loops + Parallel Thompson Sampling
# ═══════════════════════════════════════════════════════════════════════════════
def calibrate_thompson(mc_eff_thr, mc_eff_gain,
                       n_iter=900, K=8, P=3,
                       mu_thr=1.1e-3, mu_gain=0.12, seed=0):
    """
    Architecture
    ────────────
    K=8 centroid voltages partition the input range.  Each centroid c
    defines a probe point x_c at which the hardware oracle is queried.

    P=3 parallel AI loops run simultaneously.  Each loop p maintains
    an independent Beta(α_{p,s,c}, β_{p,s,c}) posterior per
    (agent p, stage s, centroid c).  The posterior encodes the agent's
    belief about how useful probing centroid c is for stage s.

    Per iteration (one pass for all P agents):
      1. Thompson sample: θ_{p,s,c} ~ Beta(α_{p,s,c}, β_{p,s,c})
      2. Select centroid c* = argmax_c θ_{p,s,c}  per stage
      3. Inject x_{c*} into stage s, compute gradient, update knobs
      4. Evaluate improvement; update Beta posteriors accordingly:
           improved → α_{p,s,c*} += 1   (success)
           stalled  → β_{p,s,c*} += 1   (failure)

    Merge: final knob values = confidence-weighted average of P agents
           weight_p ∝ mean_{s,c}[α_{p,s,c} / (α_{p,s,c} + β_{p,s,c})]
    Soft re-sync every 100 steps prevents agent divergence.
    """
    rng       = np.random.default_rng(seed)
    centroids = np.linspace(-VREF * 0.88, VREF * 0.88, K)

    # Per-agent knob estimates
    thr_p  = [np.zeros((N_STAGES, N_COMP)) for _ in range(P)]
    gain_p = [np.ones(N_STAGES)             for _ in range(P)]

    # Beta posteriors: shape (P, N_STAGES, K)
    alpha  = np.ones((P, N_STAGES, K), dtype=float)
    beta_  = np.ones((P, N_STAGES, K), dtype=float)

    hist   = np.zeros((n_iter, 2))

    for i in range(n_iter):

        for p in range(P):
            # ── Thompson Sampling: pick best centroid per stage ────────────
            theta  = rng.beta(alpha[p], beta_[p])        # (N_STAGES, K)
            sel_c  = np.argmax(theta, axis=1)             # (N_STAGES,)

            for s in range(N_STAGES):
                c  = int(sel_c[s])
                xp = centroids[c]

                eff_thr    = mc_eff_thr[s] + thr_p[p][s]
                eff_gain_s = mc_eff_gain[s] * gain_p[p][s]
                err_thr_old  = np.sum(np.abs(eff_thr - NOMINAL_THR))
                err_gain_old = abs(eff_gain_s - GAIN_IDEAL)

                # ── sign-LMS threshold gradient at centroid xp ────────────
                for k in range(N_COMP):
                    d = 1.0 if xp >= NOMINAL_THR[k] else 0.0
                    y = 1.0 if xp >= eff_thr[k]     else 0.0
                    thr_p[p][s, k] += mu_thr * (d - y)

                # ── NLMS gain gradient at centroid xp ─────────────────────
                eff_thr_now = mc_eff_thr[s] + thr_p[p][s]
                code  = int(np.clip(np.sum(xp >= eff_thr_now), 0, 3))
                v_in  = xp - DAC_LEVELS[code]
                if abs(v_in) > 1e-6:
                    res_obs = v_in * mc_eff_gain[s] * gain_p[p][s]
                    e_g     = v_in * GAIN_IDEAL - res_obs
                    mc_proxy = eff_gain_s / max(abs(gain_p[p][s]), 1e-4)
                    gain_p[p][s] += mu_gain * e_g * v_in * mc_proxy / (v_in ** 2 + 1e-9)
                    gain_p[p][s]  = np.clip(gain_p[p][s], 0.4, 2.5)

                # ── Update Beta posteriors ─────────────────────────────────
                new_eff_thr  = mc_eff_thr[s] + thr_p[p][s]
                new_eff_gain = mc_eff_gain[s] * gain_p[p][s]
                improved = (np.sum(np.abs(new_eff_thr - NOMINAL_THR)) < err_thr_old - 1e-9 or
                            abs(new_eff_gain - GAIN_IDEAL) < err_gain_old - 1e-9)
                if improved:
                    alpha[p, s, c] += 1.0
                else:
                    beta_[p, s, c] += 1.0

        # ── Confidence-weighted merge of P agents ──────────────────────────
        conf    = (alpha / (alpha + beta_)).mean(axis=(1, 2))    # (P,)
        w       = conf / (conf.sum() + 1e-12)
        cal_thr  = sum(w[p] * thr_p[p]  for p in range(P))
        cal_gain = sum(w[p] * gain_p[p] for p in range(P))

        # Soft re-sync agents every 100 steps (prevents drift)
        if i > 0 and i % 100 == 0:
            for p in range(P):
                thr_p[p]  = 0.75 * thr_p[p]  + 0.25 * cal_thr
                gain_p[p] = 0.75 * gain_p[p] + 0.25 * cal_gain

        hist[i] = _cal_rms(cal_thr, cal_gain, mc_eff_thr, mc_eff_gain)

    return cal_thr, cal_gain, hist


# ═══════════════════════════════════════════════════════════════════════════════
#  METHOD 3 – Reinforcement Learning  (per-knob Q-Learning, ε-greedy)
# ═══════════════════════════════════════════════════════════════════════════════
_RL_DITHER = 1.0e-3    # 1 mV dither for hardware-observable state probe

def _obs_thr_state(s, k, cal_thr, mc_eff_thr):
    """
    Hardware-observable threshold state via two dithered probe voltages:
      Probe at NOMINAL_THR[k]          → should fire  if calibrated
      Probe at NOMINAL_THR[k] – dither → should NOT fire if calibrated

    Returns:
      0 – eff_thr too HIGH (probe at nominal does not fire)
      1 – calibrated       (probe at nominal fires, probe below doesn't)
      2 – eff_thr too LOW  (probe below nominal still fires)
    """
    x_nom = NOMINAL_THR[k]
    x_lo  = NOMINAL_THR[k] - _RL_DITHER
    f_nom = hw_cmp(x_nom, s, k, cal_thr, mc_eff_thr)
    f_lo  = hw_cmp(x_lo,  s, k, cal_thr, mc_eff_thr)
    if   f_nom == 0: return 0   # eff_thr > nominal → too high
    elif f_lo  == 1: return 2   # eff_thr < nominal - dither → too low
    else:            return 1   # calibrated


def _obs_gain_state(s, cal_thr, cal_gain, mc_eff_thr, mc_eff_gain):
    """
    Hardware-observable gain state via two-point residue measurement.
    Probes at ±0.12 V; ratio gives obs_gain = mc_eff_gain * cal_gain.
    Returns:
      0 – obs_gain < GAIN_IDEAL (too low)
      1 – calibrated
      2 – obs_gain > GAIN_IDEAL (too high)
    """
    res_a, v_in_a, _ = hw_res(-0.12, s, cal_thr, cal_gain, mc_eff_thr, mc_eff_gain)
    res_b, v_in_b, _ = hw_res(+0.12, s, cal_thr, cal_gain, mc_eff_thr, mc_eff_gain)
    dv = v_in_b - v_in_a
    if abs(dv) < 1e-9:
        return 1
    obs_gain = (res_b - res_a) / dv    # = mc_eff_gain[s] * cal_gain[s]
    err = obs_gain - GAIN_IDEAL
    if   err < -1e-3: return 0
    elif err >  1e-3: return 2
    else:             return 1


def calibrate_rl(mc_eff_thr, mc_eff_gain,
                 n_episodes=90, n_steps=30,
                 lr_q=0.20, gamma=0.88,
                 eps_start=0.95, eps_end=0.05, seed=0):
    """
    One independent Q-learning agent per (stage, knob):
      N_STAGES * (N_COMP + 1) = 16 agents total.

    State space  (3 discrete states per agent):
      Threshold agent (s,k):  0=too-high | 1=ok | 2=too-low
                               (observed via hw_cmp dithered probes)
      Gain agent (s):          0=too-low  | 1=ok | 2=too-high
                               (observed via two-point residue ratio)

    Action space  (5 discrete step sizes):
      Threshold: Δ ∈ {−5, −1, 0, +1, +5} mV
      Gain:      Δ ∈ {−0.020, −0.005, 0, +0.005, +0.020}

    Reward:  r = prev_abs_error – new_abs_error  (+ for improvement)

    Update:  Q[s,a] += α · (r + γ · max_a' Q[s',a'] – Q[s,a])

    Policy:  ε-greedy with exponential decay over episodes.
    """
    rng   = np.random.default_rng(seed)
    N_S   = 3     # number of observable states
    N_A   = 5     # number of discrete actions

    # Action tables
    thr_delta  = np.array([-5e-3, -1e-3, 0.0, +1e-3, +5e-3])
    gain_delta = np.array([-0.020, -0.005, 0.0, +0.005, +0.020])

    # Q-tables  (stage, comp/gain, state, action)
    Q_thr  = np.zeros((N_STAGES, N_COMP, N_S, N_A))
    Q_gain = np.zeros((N_STAGES, N_S, N_A))

    cal_thr  = np.zeros((N_STAGES, N_COMP))
    cal_gain = np.ones(N_STAGES)

    eps_decay = (eps_end / eps_start) ** (1.0 / max(n_episodes - 1, 1))
    eps       = eps_start

    hist = np.zeros((n_episodes * n_steps, 2))

    for ep in range(n_episodes):
        for step in range(n_steps):
            for s in range(N_STAGES):

                # ── Threshold knobs ────────────────────────────────────────
                for k in range(N_COMP):
                    st = _obs_thr_state(s, k, cal_thr, mc_eff_thr)

                    # ε-greedy action selection
                    if rng.random() < eps:
                        a = int(rng.integers(N_A))
                    else:
                        a = int(np.argmax(Q_thr[s, k, st]))

                    # Apply action, measure reward
                    err_before = abs(mc_eff_thr[s,k] + cal_thr[s,k] - NOMINAL_THR[k])
                    cal_thr[s, k] += thr_delta[a]
                    err_after  = abs(mc_eff_thr[s,k] + cal_thr[s,k] - NOMINAL_THR[k])
                    reward = err_before - err_after      # + if improved

                    # Bellman TD(0) update
                    nxt = _obs_thr_state(s, k, cal_thr, mc_eff_thr)
                    td  = reward + gamma * np.max(Q_thr[s,k,nxt]) - Q_thr[s,k,st,a]
                    Q_thr[s, k, st, a] += lr_q * td

                # ── Gain knob ──────────────────────────────────────────────
                st = _obs_gain_state(s, cal_thr, cal_gain, mc_eff_thr, mc_eff_gain)

                if rng.random() < eps:
                    a = int(rng.integers(N_A))
                else:
                    a = int(np.argmax(Q_gain[s, st]))

                err_before = abs(mc_eff_gain[s] * cal_gain[s] - GAIN_IDEAL)
                cal_gain[s] += gain_delta[a]
                cal_gain[s]  = np.clip(cal_gain[s], 0.4, 2.5)
                err_after  = abs(mc_eff_gain[s] * cal_gain[s] - GAIN_IDEAL)
                reward = err_before - err_after

                nxt = _obs_gain_state(s, cal_thr, cal_gain, mc_eff_thr, mc_eff_gain)
                td  = reward + gamma * np.max(Q_gain[s, nxt]) - Q_gain[s, st, a]
                Q_gain[s, st, a] += lr_q * td

            # Track convergence at each inner step
            idx = ep * n_steps + step
            hist[idx] = _cal_rms(cal_thr, cal_gain, mc_eff_thr, mc_eff_gain)

        eps *= eps_decay   # decay exploration rate each episode

    return cal_thr, cal_gain, hist


# ═══════════════════════════════════════════════════════════════════════════════
#  METHOD 4 – Agentic AI  (4 specialised agents + message bus)
# ═══════════════════════════════════════════════════════════════════════════════
class _MessageBus:
    """Synchronous single-tick message bus for inter-agent communication."""
    def __init__(self):
        self._q = defaultdict(list)

    def send(self, to, msg_type, payload):
        self._q[to].append({'type': msg_type, 'data': payload})

    def recv(self, agent):
        msgs = self._q[agent]
        self._q[agent] = []
        return msgs


class _OracleAgent:
    """
    Hardware interface: injects calibration test voltages and reports
    comparator decisions + two-point residue measurements to the bus.

    Threshold probe strategy:
      Two probes per comparator k, stage s:
        x_hi = NOMINAL_THR[k] + ε  (should fire  → desired = 1)
        x_lo = NOMINAL_THR[k] - ε  (should NOT fire → desired = 0)
      Together these give an unambiguous gradient signal in both
      directions even when the offset is large.

    Gain probe:
      Two injection voltages (−0.15 V and +0.15 V) → two-point estimate
      of the effective stage gain = mc_eff_gain × cal_gain.
    """
    PROBE_DITHER = 0.8e-3    # 0.8 mV dither around nominal threshold

    def __init__(self, bus, mc_eff_thr, mc_eff_gain):
        self.bus      = bus
        self.mc_thr   = mc_eff_thr
        self.mc_gain  = mc_eff_gain

    def step(self, cal_thr, cal_gain):
        eps = self.PROBE_DITHER
        for s in range(N_STAGES):
            for k in range(N_COMP):
                x_hi = NOMINAL_THR[k] + eps
                x_lo = NOMINAL_THR[k] - eps
                f_hi = hw_cmp(x_hi, s, k, cal_thr, self.mc_thr)
                f_lo = hw_cmp(x_lo, s, k, cal_thr, self.mc_thr)
                self.bus.send('thr', 'THR_PROBE',
                              {'s': s, 'k': k,
                               'f_hi': f_hi, 'f_lo': f_lo,
                               'des_hi': 1,  'des_lo': 0})
            # Two-point gain measurement
            res_a, vin_a, _ = hw_res(-0.15, s, cal_thr, cal_gain, self.mc_thr, self.mc_gain)
            res_b, vin_b, _ = hw_res(+0.15, s, cal_thr, cal_gain, self.mc_thr, self.mc_gain)
            self.bus.send('gain', 'GAIN_PROBE',
                          {'s': s,
                           'res_a': res_a, 'vin_a': vin_a,
                           'res_b': res_b, 'vin_b': vin_b})


class _ThresholdAgent:
    """
    Decodes THR_PROBE messages and applies two-probe sign-LMS updates:
      • x_hi fires but shouldn't → eff_thr too low  → cal_thr[s,k] += μ
      • x_lo doesn't fire & x_hi also doesn't → eff_thr too high → cal_thr -= μ
      • x_hi fires, x_lo doesn't → within deadband  → no update

    Reports aggregate residual error to Supervisor.
    """
    def __init__(self, bus, mu=1.3e-3):
        self.bus = bus
        self.mu  = mu

    def step(self, cal_thr):
        for m in self.bus.recv('thr'):
            if m['type'] != 'THR_PROBE':
                continue
            p = m['data']
            s, k = p['s'], p['k']
            f_hi, f_lo = p['f_hi'], p['f_lo']

            # f_hi should be 1, f_lo should be 0 when calibrated
            if f_hi == 0:
                # eff_thr above x_hi → above NOMINAL → decrease cal_thr
                cal_thr[s, k] -= self.mu
            elif f_lo == 1:
                # eff_thr below x_lo → below NOMINAL → increase cal_thr
                cal_thr[s, k] += self.mu
            # else: within deadband ε → converged for this comparator

        # Report residual to supervisor (observable proxy: mean |delta_f|)
        self.bus.send('sup', 'STATUS',
                      {'agent': 'thr', 'mu': self.mu})
        return cal_thr


class _GainAgent:
    """
    Decodes GAIN_PROBE messages, estimates effective stage gain from the
    two-point residue ratio, then applies a soft closed-loop update:

        obs_gain  = (res_b − res_a) / (vin_b − vin_a)
                  = mc_eff_gain[s] · cal_gain[s]
        target    = GAIN_IDEAL / (obs_gain / cal_gain[s])
                  = GAIN_IDEAL / mc_eff_gain[s]     ← exact optimal
        cal_gain[s] += μ · (target − cal_gain[s])

    This is a first-order IIR loop that converges exponentially toward
    the exact correction in ~1/μ steps.
    """
    def __init__(self, bus, mu=0.18):
        self.bus = bus
        self.mu  = mu

    def step(self, cal_gain):
        for m in self.bus.recv('gain'):
            if m['type'] != 'GAIN_PROBE':
                continue
            p = m['data']
            s = p['s']
            dv  = p['vin_b'] - p['vin_a']
            dr  = p['res_b'] - p['res_a']
            if abs(dv) < 1e-9:
                continue
            obs_gain = dr / dv                        # mc_eff_gain[s] · cal_gain[s]
            mc_proxy = obs_gain / max(abs(cal_gain[s]), 1e-4)  # ≈ mc_eff_gain[s]
            target   = GAIN_IDEAL / max(abs(mc_proxy), 1e-4)   # exact optimal
            cal_gain[s] += self.mu * (target - cal_gain[s])
            cal_gain[s]  = np.clip(cal_gain[s], 0.4, 2.5)

        self.bus.send('sup', 'STATUS',
                      {'agent': 'gain', 'mu': self.mu})
        return cal_gain


class _SupervisorAgent:
    """
    Monitors per-step improvement and anneals learning rates when
    calibration stalls (no improvement for `patience` consecutive steps).
    Triggers agents: ThresholdAgent and GainAgent.
    """
    def __init__(self, bus, patience=25, lr_decay=0.92):
        self.bus       = bus
        self.patience  = patience
        self.lr_decay  = lr_decay
        self._best     = np.inf
        self._stalled  = 0

    def step(self, thr_ag, gain_ag, thr_rms, gain_rms):
        # Consume status messages
        self.bus.recv('sup')
        metric = thr_rms + gain_rms
        if metric < self._best - 5e-8:
            self._best   = metric
            self._stalled = 0
        else:
            self._stalled += 1
        # Anneal both agents' step sizes when stalled
        if self._stalled >= self.patience:
            thr_ag.mu  *= self.lr_decay
            gain_ag.mu *= self.lr_decay
            self._stalled = 0
        return thr_ag, gain_ag


def calibrate_agentic(mc_eff_thr, mc_eff_gain,
                      n_steps=450, mu_thr=1.3e-3, mu_gain=0.18, seed=0):
    """
    Four-agent calibration loop:

        [OracleAgent] ──THR_PROBE──► [ThresholdAgent] ──STATUS──► [SupervisorAgent]
             │                                                           │ μ-anneal
             └──GAIN_PROBE──► [GainAgent] ──STATUS──────────────────────┘

    Tick sequence each step:
        1. Oracle injects probes → messages on bus
        2. ThresholdAgent reads bus → updates cal_thr → posts STATUS
        3. GainAgent reads bus      → updates cal_gain → posts STATUS
        4. SupervisorAgent reads STATUS → anneals μ if stalled
    """
    bus     = _MessageBus()
    oracle  = _OracleAgent(bus, mc_eff_thr, mc_eff_gain)
    thr_ag  = _ThresholdAgent(bus, mu=mu_thr)
    gain_ag = _GainAgent(bus, mu=mu_gain)
    sup_ag  = _SupervisorAgent(bus, patience=28, lr_decay=0.92)

    cal_thr  = np.zeros((N_STAGES, N_COMP))
    cal_gain = np.ones(N_STAGES)
    hist     = np.zeros((n_steps, 2))

    for i in range(n_steps):
        oracle.step(cal_thr, cal_gain)           # 1. measure
        cal_thr  = thr_ag.step(cal_thr)          # 2. update thresholds
        cal_gain = gain_ag.step(cal_gain)         # 3. update gains
        thr_rms, gain_rms = _cal_rms(cal_thr, cal_gain, mc_eff_thr, mc_eff_gain)
        thr_ag, gain_ag  = sup_ag.step(thr_ag, gain_ag, thr_rms, gain_rms)  # 4. supervise
        hist[i] = (thr_rms, gain_rms)

    return cal_thr, cal_gain, hist


# ═══════════════════════════════════════════════════════════════════════════════
#  REFERENCE – Foreground Binary-Search Calibration  (baseline algorithm)
# ═══════════════════════════════════════════════════════════════════════════════
_FG_BITS  = 12
_FG_NSIG  = 6
_FG_VA    = -0.10
_FG_VB    = +0.10

def calibrate_foreground(mc_eff_thr, mc_eff_gain):
    """
    Phase A: 12-bit binary search per comparator (offset trim).
    Phase B: Two-point residue injection per stage (gain trim).
    """
    cal_thr  = np.zeros((N_STAGES, N_COMP))
    cal_gain = np.ones(N_STAGES)
    for s in range(N_STAGES):
        for k in range(N_COMP):
            lo = NOMINAL_THR[k] - _FG_NSIG * OFFSET_SIGMA
            hi = NOMINAL_THR[k] + _FG_NSIG * OFFSET_SIGMA
            for _ in range(_FG_BITS):
                mid = (lo + hi) * 0.5
                if mid >= mc_eff_thr[s, k]:
                    hi = mid
                else:
                    lo = mid
            cal_thr[s, k] = NOMINAL_THR[k] - (lo + hi) * 0.5

        eff_thr_cal = mc_eff_thr[s] + cal_thr[s]
        ca  = int(np.clip(np.sum(_FG_VA >= eff_thr_cal), 0, 3))
        cb  = int(np.clip(np.sum(_FG_VB >= eff_thr_cal), 0, 3))
        r_a = (_FG_VA - DAC_LEVELS[ca]) * mc_eff_gain[s]
        r_b = (_FG_VB - DAC_LEVELS[cb]) * mc_eff_gain[s]
        r_ai = (_FG_VA - DAC_LEVELS[ca]) * GAIN_IDEAL
        r_bi = (_FG_VB - DAC_LEVELS[cb]) * GAIN_IDEAL
        dm, di = r_b - r_a, r_bi - r_ai
        if abs(dm) > 1e-9:
            cal_gain[s] = di / dm
    return cal_thr, cal_gain


# ═══════════════════════════════════════════════════════════════════════════════
#  Monte Carlo Comparison
# ═══════════════════════════════════════════════════════════════════════════════
METHOD_NAMES = ['Ideal', 'Uncal', 'Foreground', 'LMS', 'Thompson-TS', 'RL', 'Agentic']
sndr_mc    = {m: np.zeros((N_RUNS, AMP_PTS)) for m in METHOD_NAMES}
conv_runs  = {m: [] for m in ('LMS', 'Thompson-TS', 'RL', 'Agentic')}

ideal_thr  = np.tile(NOMINAL_THR, (N_STAGES, 1))
ideal_gain = np.full(N_STAGES, GAIN_IDEAL)

print("=" * 76)
print("  Pipeline ADC — 4 Adaptive Calibration Methods  ({} MC runs)".format(N_RUNS))
print("  σ_offset={:.0f} mV    σ_gain={:.0f} mV/V    SNDR ideal={:.2f} dB".format(
    OFFSET_SIGMA*1e3, GAIN_MIS_SIGMA*1e3, IDEAL_PEAK))
print("  Methods: LMS | Thompson-TS | RL | Agentic AI | Foreground ref")
print("=" * 76)

for run in range(N_RUNS):
    rng = np.random.default_rng(run + 1000)
    mc_offsets  = rng.normal(0.0, OFFSET_SIGMA,    (N_STAGES, N_COMP))
    mc_gerrs    = rng.normal(0.0, GAIN_MIS_SIGMA,  N_STAGES)
    mc_eff_thr  = NOMINAL_THR[np.newaxis, :] + mc_offsets
    mc_eff_gain = GAIN_IDEAL + mc_gerrs

    # ── Run all five calibration methods ─────────────────────────────────────
    thr_fg,  gain_fg              = calibrate_foreground(mc_eff_thr, mc_eff_gain)
    thr_lms, gain_lms, h_lms     = calibrate_lms(mc_eff_thr, mc_eff_gain, seed=run)
    thr_ts,  gain_ts,  h_ts      = calibrate_thompson(mc_eff_thr, mc_eff_gain, seed=run)
    thr_rl,  gain_rl,  h_rl      = calibrate_rl(mc_eff_thr, mc_eff_gain, seed=run)
    thr_ag,  gain_ag,  h_ag      = calibrate_agentic(mc_eff_thr, mc_eff_gain, seed=run)

    conv_runs['LMS'].append(h_lms)
    conv_runs['Thompson-TS'].append(h_ts)
    conv_runs['RL'].append(h_rl)
    conv_runs['Agentic'].append(h_ag)

    # ── Effective parameters for each scenario ────────────────────────────────
    scenarios = {
        'Ideal':       (ideal_thr,                ideal_gain),
        'Uncal':       (mc_eff_thr,               mc_eff_gain),
        'Foreground':  (mc_eff_thr + thr_fg,      mc_eff_gain * gain_fg),
        'LMS':         (mc_eff_thr + thr_lms,     mc_eff_gain * gain_lms),
        'Thompson-TS': (mc_eff_thr + thr_ts,      mc_eff_gain * gain_ts),
        'RL':          (mc_eff_thr + thr_rl,      mc_eff_gain * gain_rl),
        'Agentic':     (mc_eff_thr + thr_ag,      mc_eff_gain * gain_ag),
    }

    # ── Amplitude sweep ───────────────────────────────────────────────────────
    for j, a in enumerate(10.0 ** (amp_dB / 20.0)):
        si = a * np.sin(2.0 * np.pi * fin * t)
        for name, (thr, gn) in scenarios.items():
            sndr_mc[name][run, j] = compute_sndr(code2volt(pipeline_adc_vec(si, thr, gn)))

    if (run + 1) % 5 == 0:
        print("  Run {}/{} done".format(run + 1, N_RUNS))

# ── Summary statistics ────────────────────────────────────────────────────────
avg_sndr = {m: sndr_mc[m].mean(0) for m in METHOD_NAMES}
std_sndr = {m: sndr_mc[m].std(0)  for m in METHOD_NAMES}
fs_sndr  = {m: sndr_mc[m][:, -1]  for m in METHOD_NAMES}
conv_avg = {m: np.mean(conv_runs[m], axis=0) for m in conv_runs}

print("\n" + "─" * 76)
print("  Full-Scale SNDR Summary  ({} MC runs)".format(N_RUNS))
print("─" * 76)
print("  {:>14s}  {:>10s}  {:>9s}  {:>8s}  {:>9s}".format(
    "Method", "Mean SNDR", "Std SNDR", "ENOB", "vs Ideal"))
print("─" * 76)
ideal_mean = np.mean(fs_sndr['Ideal'])
for m in METHOD_NAMES:
    mu  = np.mean(fs_sndr[m])
    sig = np.std(fs_sndr[m])
    enob = (mu - 1.76) / 6.02
    print("  {:>14s}  {:>8.2f} dB  {:>7.2f} dB  {:>6.2f} b  {:>+.2f} dB".format(
        m, mu, sig, enob, mu - ideal_mean))
print("─" * 76)


# ═══════════════════════════════════════════════════════════════════════════════
#  Plots
# ═══════════════════════════════════════════════════════════════════════════════
DARK   = "#0A0E14";  GRID   = "#1C2330";  CYAN   = "#00E5FF"
AMBER  = "#FFB300";  RED    = "#FF4560";  GREEN  = "#00E396"
GHOST  = "#2A3545";  TEXT   = "#CDD6E0";  MUTED  = "#6E7D8C"
ORANGE = "#FF6D00";  PURPLE = "#CE93D8";  LIME   = "#B9F542"

METHOD_COLOR = {
    'Ideal': GHOST, 'Uncal': RED,
    'Foreground': CYAN, 'LMS': LIME,
    'Thompson-TS': AMBER, 'RL': PURPLE, 'Agentic': ORANGE,
}
METHOD_LW = {m: (1.4 if m in ('Ideal','Uncal') else 2.2) for m in METHOD_NAMES}
METHOD_LS = {'Ideal': '--', 'Uncal': '-', 'Foreground': '-.',
             'LMS': '-', 'Thompson-TS': '-', 'RL': '-', 'Agentic': '-'}
CONV_COL  = {'LMS': LIME, 'Thompson-TS': AMBER, 'RL': PURPLE, 'Agentic': ORANGE}

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": DARK,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT, "xtick.color": MUTED,
    "ytick.color": MUTED, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "grid.color": GRID,
    "grid.linewidth": 0.5, "text.color": TEXT,
    "font.family": "monospace", "font.size": 10,
})

fig = plt.figure(figsize=(21, 13))
fig.patch.set_facecolor(DARK)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.33)
fig.suptitle(
    "4-STAGE PIPELINE ADC  │  Adaptive Calibration Comparison  │  "
    "MC={} runs  │  σ_off={}mV  σ_gain={}mV/V".format(
        N_RUNS, int(OFFSET_SIGMA*1e3), int(GAIN_MIS_SIGMA*1e3)),
    fontsize=11, fontweight='bold', color=CYAN, y=0.975)


# ── [0,0]  Average SNDR vs Amplitude ─────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
for m in METHOD_NAMES:
    ax.plot(amp_dB, avg_sndr[m],
            color=METHOD_COLOR[m], lw=METHOD_LW[m],
            ls=METHOD_LS[m], label=m, alpha=0.92)
    if m in ('LMS', 'Thompson-TS', 'RL', 'Agentic'):
        ax.fill_between(amp_dB,
                        avg_sndr[m] - std_sndr[m],
                        avg_sndr[m] + std_sndr[m],
                        color=METHOD_COLOR[m], alpha=0.08)
ax.set_xlabel("Input Amplitude (dBFS)")
ax.set_ylabel("SNDR (dB)")
ax.set_title("Average SNDR vs Amplitude", fontsize=10)
ax.legend(fontsize=7.2, framealpha=0.3, loc='upper left')
ax.set_xlim(amp_dB[0], 0)
ax.set_ylim(0, IDEAL_PEAK + 6)
ax.grid(True, alpha=0.3)


# ── [0,1]  Full-Scale SNDR Histogram ─────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
show_methods = ['Uncal', 'Foreground', 'LMS', 'Thompson-TS', 'RL', 'Agentic']
for m in show_methods:
    label = "{}: {:.1f} dB".format(m, np.mean(fs_sndr[m]))
    ax.hist(fs_sndr[m], bins=14, color=METHOD_COLOR[m],
            alpha=0.50, edgecolor=METHOD_COLOR[m], label=label)
ax.axvline(IDEAL_PEAK, color=CYAN, lw=1.2, ls=':', label="Ideal {:.1f} dB".format(IDEAL_PEAK))
ax.set_xlabel("SNDR @ Full Scale (dB)")
ax.set_ylabel("Count")
ax.set_title("Full-Scale SNDR Distribution  (Monte Carlo)", fontsize=10)
ax.legend(fontsize=6.5, framealpha=0.3)
ax.grid(True, alpha=0.3)


# ── [0,2]  Convergence Curves ─────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
for m, h in conv_avg.items():
    n = len(h)
    x_pct = np.linspace(0, 100, n)
    thr_mv  = h[:, 0] * 1e3                          # V → mV
    gain_vv = h[:, 1]                                # V/V
    ax.semilogy(x_pct, thr_mv,  color=CONV_COL[m], lw=2.0,
                label='{} | thr (mV)'.format(m))
    ax.semilogy(x_pct, gain_vv, color=CONV_COL[m], lw=1.1,
                ls='--', alpha=0.65,
                label='{} | gain (V/V)'.format(m))

# Mark the ±1 LSB target line for thresholds
ax.axhline(LSB * 1e3, color=GHOST, lw=0.9, ls=':', alpha=0.7,
           label='1 LSB = {:.1f} mV'.format(LSB*1e3))
ax.set_xlabel("Calibration Progress (%)")
ax.set_ylabel("RMS Calibration Error")
ax.set_title("Knob Convergence  (MC-average)\n"
             "Solid = threshold (mV)   Dashed = gain (V/V)", fontsize=9)
ax.legend(fontsize=5.8, framealpha=0.3, ncol=2)
ax.set_xlim(0, 100)
ax.grid(True, alpha=0.3)


# ── [1,0]  Average ENOB vs Amplitude ─────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
for m in METHOD_NAMES:
    enob = (avg_sndr[m] - 1.76) / 6.02
    ax.plot(amp_dB, enob,
            color=METHOD_COLOR[m], lw=METHOD_LW[m],
            ls=METHOD_LS[m], label=m)
ax.axhline(TOTAL_BITS, color=CYAN, lw=0.8, ls=':', alpha=0.5)
ax.set_xlabel("Input Amplitude (dBFS)")
ax.set_ylabel("ENOB (bits)")
ax.set_title("Average ENOB vs Amplitude", fontsize=10)
ax.legend(fontsize=7.2, framealpha=0.3, loc='upper left')
ax.set_xlim(amp_dB[0], 0)
ax.set_ylim(0, TOTAL_BITS + 1)
ax.grid(True, alpha=0.3)


# ── [1,1]  SNDR Loss from Ideal  (box plots) ─────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
plot_m = ['Foreground', 'LMS', 'Thompson-TS', 'RL', 'Agentic']
loss   = [IDEAL_PEAK - fs_sndr[m] for m in plot_m]
bp     = ax.boxplot(
    loss, patch_artist=True,
    medianprops=dict(color=DARK, lw=2.2),
    whiskerprops=dict(color=TEXT, lw=1.2),
    capprops=dict(color=TEXT, lw=1.2),
    flierprops=dict(marker='.', alpha=0.5, ms=4))
for patch, m in zip(bp['boxes'], plot_m):
    patch.set_facecolor(METHOD_COLOR[m])
    patch.set_alpha(0.72)
ax.set_xticks(range(1, len(plot_m) + 1))
ax.set_xticklabels([m.replace('-', '\n') for m in plot_m], fontsize=8)
ax.set_ylabel("SNDR Loss from Ideal (dB)")
ax.set_title("Full-Scale SNDR Loss vs Ideal\n(lower = better)", fontsize=9)
ax.axhline(0, color=CYAN, lw=0.8, ls=':', alpha=0.5)
ax.grid(True, alpha=0.3, axis='y')


# ── [1,2]  Summary Table ──────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
ax.axis('off')

col_labels = ["Method", "Mean SNDR", "Std", "ENOB", "Loss"]
rows = []
for m in METHOD_NAMES:
    mu   = np.mean(fs_sndr[m])
    sig  = np.std(fs_sndr[m])
    enob = (mu - 1.76) / 6.02
    loss = ideal_mean - mu
    rows.append([m,
                 "{:.2f} dB".format(mu),
                 "{:.2f} dB".format(sig),
                 "{:.2f} b".format(enob),
                 "{:+.2f} dB".format(-loss)])

tbl = ax.table(cellText=rows, colLabels=col_labels,
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.2)
tbl.scale(1.0, 1.65)

for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor(GRID)
        cell.set_text_props(color=CYAN, fontweight='bold')
    else:
        cell.set_facecolor(DARK)
        cell.set_text_props(color=TEXT)
        if c == 0:
            m = METHOD_NAMES[r - 1]
            cell.set_text_props(color=METHOD_COLOR[m], fontweight='bold')
    cell.set_edgecolor(GHOST)

ax.set_title("Monte Carlo Summary — Full-Scale SNDR ({} runs)".format(N_RUNS),
             fontsize=9, pad=12)

out_png = "pipeline_adc_adaptive_calibration.png"
plt.savefig(out_png, dpi=150, bbox_inches='tight', facecolor=DARK)
plt.show()
print("\nSaved → {}".format(out_png))