# Codebase Full Analysis — `temp_cic` Branch
**Repository:** `Manojit-Pal/cache_aided_noma`  
**Branch:** `temp_cic`  
**Analyzed by:** AI Agent (Perplexity)  
**Date:** 03 July 2026  
**Purpose:** Full understanding of module flow + Standard CIC vs Hybrid CIC comparison readiness

---

## 1. Repository Top-Level Structure

```
cache_aided_noma/ (temp_cic branch)
│
├── IEEE_Paper.tex              # LaTeX paper (~40KB)
├── README.md                   # Project documentation (~17KB)
├── INSTALL.md                  # Setup guide
├── requirements.txt            # Python dependencies
├── run_comparison.py           # Entry point: runs all policy comparisons
├── train_and_evaluate_dqn.py   # Root-level DQN training script
├── test_dqn_cache.py           # DQN cache unit tests (~23KB)
├── test_noma_integration.py    # NOMA integration tests (~17KB)
├── test_noma_sim.py            # NOMA simulation tests
│
├── src/                        # CORE source module
│   ├── config.py               # All simulation parameters
│   ├── utils.py                # Plotting, metrics, utilities (~23KB)
│   ├── caching/                # Cache policy implementations
│   ├── noma/                   # NOMA physical layer
│   ├── simulation/             # Main simulation engines
│   └── experiments/            # Comparative analysis scripts
│
├── cic_pairing/                # CIC-specific analysis scripts (new on temp_cic)
│   ├── cic_pairing_analysis.py
│   ├── diagnose_and_fix.py
│   └── quick_cic_check.py
│
├── checkpoints/                # DQN model checkpoints
├── models/                     # Saved model weights
├── results/                    # Raw result outputs
├── results_csv/                # CSV result files
├── results_pdf/                # PDF plots
└── results_pic/                # PNG plots
```

---

## 2. Module-by-Module Deep Analysis

### 2.1 `src/config.py` — Global Configuration

The single source of truth for all parameters. Key settings:

| Parameter | Value | Meaning |
|---|---|---|
| `NUM_FILES` | 2000 | Content catalog size |
| `CACHE_SIZE` | 200 | 10% of catalog |
| `NUM_USERS` | 200 | Users in cell |
| `TX_POWER` | 2.0 W | Base station transmit power |
| `NOISE_POWER` | 1e-9 W | Thermal noise floor |
| `POWER_COEFF_WEAK` | 0.8 | Default weak user power share |
| `POWER_COEFF_STRONG` | 0.2 | Default strong user power share |
| `POWER_ALLOC_METHOD` | `"cache_aware"` | **Key: cache-aware power is default** |
| `SIC_IMPERFECTION` | 0.05 | Residual interference factor ζ |
| `TARGET_RATE_BPS` | 0.3 bps/Hz | SINR threshold = 2^0.3 - 1 ≈ 0.231 |
| `ENABLE_CIC` | True | CIC globally enabled |
| `CIC_PERFECT` | True | Assumes perfect cancellation when cached |
| `FADING_TYPE` | `"mixed"` | Rayleigh + Rician mixed fading |
| `PAIRING_METHOD` | `"extreme"` | Weakest paired with strongest |

**Bug fixes already applied in config:** BUG-5, BUG-7, BUG-8, ROOT-1, ROOT-2, ROOT-4, etc.

---

### 2.2 `src/noma/` — Physical Layer (CORE MODULE)

This is the heart of the system. Contains 4 files:

#### 2.2.1 `src/noma/power_allocation.py`

Implements ALL power allocation strategies via a universal dispatcher `allocate_power()`:

```
allocate_power(gain_w, gain_s, cfg, method=..., weak_cached=..., strong_cached=...)
     │
     ├── 'gridsearch'       → allocate_power_gridsearch()   [exhaustive scan]
     ├── 'closedform'       → allocate_power_closedform()   [analytical SINR constraints]
     ├── 'cache_aware'      → allocate_power_cache_aware()  [THE KEY FUNCTION]
     ├── 'sumrate_max'      → allocate_power_sumrate_max()  [scipy optimize]
     └── 'energy_efficient' → allocate_power_energy_efficient() [green optimization]
```

**`allocate_power_cache_aware()` — Most Critical Function:**

Handles 4 cases based on `(weak_cached, strong_cached)` booleans:

| Scenario | p_w Constraint Change | zeta | Description |
|---|---|---|---|
| No cache | Standard constraints | 0.05 | Falls back to `closedform` |
| `weak_cached=True` | Lower p_w bound: `T*N0/(P*g_w)` (no interference term) | 0.05 | Weak user CIC: no interference from strong |
| `strong_cached=True` | Upper p_w bound: `1 - T*N0/(P*g_s)` (zeta=0) | **0.0** | Perfect SIC: residual=0, upper bound widens |
| Both cached | Combined relaxed constraints, midpoint allocation | **0.0** | Maximum flexibility |

**Standard `closedform` constraints (no cache):**
- Lower: `p_w >= (T/(1+T))*(1 + N0/(P*g_w))` (weak user rate)
- Lower: `p_w >= (T/(1+T))*(1 + N0/(P*g_s))` (strong decodes weak)
- Upper: `p_w <= (1 - T*N0/(P*g_s)) / (1 + T*zeta)` (strong own rate)

**Bug fix applied:** BUG-PA-1 — grid search no longer exits early; full scan guarantees truly optimal sum_SINR.

---

#### 2.2.2 `src/noma/sic.py`

Implements Successive Interference Cancellation (SIC) functions:

| Function | Purpose |
|---|---|
| `sinr_weak_user()` | Standard SINR: `(P*p_w*g_w)/(P*p_s*g_w + N0)` |
| `sinr_weak_user_with_cache()` | CIC SINR: `(P*p_w*g_w)/N0` — NO interference |
| `sinr_strong_decode_weak()` | Strong decodes weak before SIC |
| `sinr_strong_after_sic()` | Strong's SINR after SIC with residual |
| `sinr_strong_after_perfect_sic()` | Strong's SINR with residual=0 (cache-aided) |
| `compute_residual_interference()` | Returns 0 (cached), ζ×signal (imperfect), full signal (failed) |
| `simulate_sic_process()` | Complete wrapper: runs all 4 scenarios end-to-end |

**Key CIC SINR improvement formula (in code):**
```
SINR_w_standard = (P*p_w*g_w) / (P*p_s*g_w + N0)
SINR_w_CIC      = (P*p_w*g_w) / N0
Improvement     = 1 + (P*p_s*g_w / N0)  [typically 2x–10x]
```

---

#### 2.2.3 `src/noma/noma_base.py`

Orchestrates the full NOMA pair simulation:

```
simulate_noma_pair(gain_weak, gain_strong, cfg,
                   p_w=None, p_s=None,
                   weak_cached=False, strong_cached=False,
                   optimize_power=True)
```

**Internal Flow:**
1. **Power Allocation:** If `POWER_ALLOC_METHOD == 'cache_aware'`, calls `allocate_power_cache_aware(weak_cached, strong_cached)`. Otherwise uses config defaults.
2. **Weak user SINR:**
   - `weak_cached=True` → `SINR_w = P*p_w*g_w / N0` (perfect CIC)
   - else → standard formula with interference
3. **Strong decodes weak:** Checks if `SINR_s_decode_w >= threshold`
4. **Residual computation:**
   - `strong_cached=True` → `residual = 0.0` (perfect cache-aided SIC)
   - `can_decode_weak=True` → `residual = zeta * P*p_w*g_s`
   - else → `residual = P*p_w*g_s` (full interference)
5. **Strong user SINR** after SIC with residual
6. **Metrics:** sum_rate, fairness (Jain's index), BER (QPSK model), outage

`simulate_noma_system()` loops over all pairs, aggregates metrics including `cic_benefit_rate`, `weak_cic_count`, `strong_cic_count`.

**Bug fixes applied:** FIX #1 (CIC tracking via list), FIX #2 (cache-aware power integration), BUG-NOMA-3 (divide-by-zero guard in Jain's fairness).

---

#### 2.2.4 `src/noma/channel_model.py` (~21KB)

Handles channel generation:
- Rayleigh, Rician, and Mixed fading models
- Path loss with configurable exponent (3.5)
- Doppler / mobility support
- User placement within cell radius

---

### 2.3 `src/caching/` — Cache Policy Layer

| File | Purpose |
|---|---|
| `cache_base.py` | Abstract base class for all cache policies |
| `static_cache.py` | Top-K (popularity-based), LRU, LFU policies |
| `dynamic_cache.py` | Adaptive/online cache updates |
| `dqn_cache_final.py` | **DQN cache agent** (~37KB) — main ML component |
| `test_caching_policies.py` | Unit tests for all policies |
| `verify_init.py` | Import/init verification script |

**`dqn_cache_final.py` highlights:**
- Binary action space (cache file = 1, evict = 0) — v2 redesign
- State: file popularity, channel gains, CIC status flags
- Reward: +2.0 (cache hit), +1.5 (CIC enabled bonus), deprecated NOMA rewards
- Prioritized Experience Replay (PER) — Schaul et al. ICLR 2016
- Double DQN architecture with soft target updates (τ=0.005)
- Network: [64, 32] hidden dims (v2 compact architecture)

---

### 2.4 `src/simulation/` — Simulation Engines

| File | Purpose |
|---|---|
| `noma_caching_sim.py` | Full NOMA+caching Monte Carlo simulation (~31KB) |
| `stable_dqn_sim.py` | DQN-specific stable simulation loop (~37KB) |
| `train_and_evaluate_dqn.py` | DQN training + evaluation runner |

**`noma_caching_sim.py` flow:**
1. Generate users with random positions in cell
2. Compute channel gains (path loss + fading)
3. Apply cache policy → determine `cache_status[user_id]`
4. Check requested file vs cache → set `weak_cached`, `strong_cached`
5. Call `simulate_noma_pair()` per pair
6. Aggregate: hit rate, outage, sum rate, CIC benefit rate
7. Repeat for NUM_RUNS Monte Carlo iterations

---

### 2.5 `src/experiments/comparative_analysis.py` (~38KB)

The main experiment runner comparing policies:
- Runs: Top-K, LRU, LFU, Stable DQN
- Metrics: hit_rate, outage_probability, sum_rate, cic_benefit_rate, BER
- Generates plots and CSV exports
- Sweeps over: SNR values, cache sizes, user counts, Zipf alpha

---

### 2.6 `cic_pairing/` — CIC Analysis Scripts (temp_cic specific)

| File | Purpose |
|---|---|
| `cic_pairing_analysis.py` | Analyzes CIC effects on user pairing |
| `diagnose_and_fix.py` | Diagnostic tool for CIC-related issues |
| `quick_cic_check.py` | Quick sanity check for CIC conditions |

---

### 2.7 `src/utils.py` (~23KB)

Utility functions:
- Plotting: learning curves, SINR distributions, CDF plots
- Metrics aggregation helpers
- CSV export utilities
- Result formatting

---

## 3. Full System Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    SYSTEM ENTRY POINTS                          │
│  run_comparison.py  │  train_and_evaluate_dqn.py               │
└──────────┬──────────┴──────────────┬──────────────────────────-─┘
           │                         │
           ▼                         ▼
┌──────────────────────┐   ┌──────────────────────────────────────┐
│  src/simulation/     │   │  src/simulation/stable_dqn_sim.py   │
│  noma_caching_sim.py │   │  (DQN training loop)                 │
└──────────┬───────────┘   └──────────────┬───────────────────────┘
           │                              │
           ▼                              ▼
┌──────────────────────┐   ┌──────────────────────────────────────┐
│  src/noma/           │   │  src/caching/dqn_cache_final.py     │
│  channel_model.py    │   │  (DQN agent: state→action→reward)   │
│  → generate gains    │   └──────────────┬───────────────────────┘
└──────────┬───────────┘                  │
           │                              │ cache_status dict
           ▼                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                src/noma/noma_base.py                            │
│                simulate_noma_pair()                             │
│                                                                  │
│  1. Power Allocation ──────────────────────────────────────┐    │
│     src/noma/power_allocation.py                           │    │
│     allocate_power_cache_aware(weak_cached, strong_cached)  │    │
│     → adjusts p_w, p_s based on CIC status                │    │
│                                                             │    │
│  2. Weak User SINR ◄───────────────────────────────────────┘    │
│     if weak_cached: SINR = P*p_w*g_w / N0  (CIC: no interference)│
│     else:           SINR = P*p_w*g_w / (P*p_s*g_w + N0)         │
│                                                                  │
│  3. Strong Decodes Weak → SIC feasibility check                 │
│     src/noma/sic.py: sinr_strong_decode_weak()                  │
│                                                                  │
│  4. Residual Interference                                       │
│     strong_cached=True → residual = 0   (perfect)              │
│     SIC success       → residual = ζ × P*p_w*g_s              │
│     SIC failure       → residual = P*p_w*g_s                   │
│                                                                  │
│  5. Strong User SINR after SIC                                  │
│     SINR_s = P*p_s*g_s / (N0 + residual)                      │
│                                                                  │
│  6. Output: sum_rate, outage, BER, CIC flags                   │
└──────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐
│  src/utils.py        │
│  Metrics, CSV, Plots │
└──────────────────────┘
           │
           ▼
    results/, results_csv/, results_pic/, results_pdf/
```

---

## 4. Standard CIC vs Hybrid CIC — Deep Analysis

### 4.1 Definitions

#### Standard CIC ("General CIC Condition" per teacher)
- Near-far pair is formed based purely on channel gains
- Power is allocated **without** knowledge of the cache (uses `closedform` or fixed `POWER_COEFF_WEAK/STRONG`)
- During transmission, IF the requested file is cached, the interference is cancelled as a byproduct
- The power allocation formula does NOT change regardless of cache status
- SINR formulas used:
  - Weak: `(P*p_w*g_w)/(P*p_s*g_w + N0)` → if cached, retroactively no interference
  - Strong: standard SIC with ζ residual

#### Hybrid CIC (Cache-Aware Power Allocation)
- Near-far pair is formed (same as above)
- Power is allocated **with** knowledge of cache/CIC status BEFORE transmission
- `allocate_power_cache_aware(weak_cached, strong_cached)` is called
- The allocation mathematically adjusts p_w and p_s:
  - `weak_cached=True`: lower p_w minimum (no interference → less power needed for weak)
  - `strong_cached=True`: sets zeta=0, which widens upper p_w bound (more power can safely go to weak)
- Result: massive rate boosts because the system preemptively leverages known interference-free conditions

---

### 4.2 Mathematical Difference

#### Standard CIC Power Constraints:
```
Lower: p_w >= (T/(1+T)) * (1 + N0/(P*g_w))     [weak rate]
Lower: p_w >= (T/(1+T)) * (1 + N0/(P*g_s))     [strong decodes weak]
Upper: p_w <= (1 - T*N0/(P*g_s)) / (1 + T*ζ)  [strong rate, ζ=0.05]
```

#### Hybrid CIC — weak_cached=True:
```
Lower: p_w >= T*N0/(P*g_w)                       [RELAXED — no interference term]
Lower: p_w >= (T/(1+T)) * (1 + N0/(P*g_s))     [strong decodes weak, unchanged]
Upper: p_w <= (1 - T*N0/(P*g_s)) / (1 + T*ζ)  [strong rate, ζ=0.05]
```
Effect: lower minimum p_w → can allocate more power to strong user if needed.

#### Hybrid CIC — strong_cached=True:
```
Lower: p_w >= (T/(1+T)) * (1 + N0/(P*g_w))     [weak rate, unchanged]
Lower: p_w >= (T/(1+T)) * (1 + N0/(P*g_s))     [strong decodes weak, unchanged]
Upper: p_w <= 1 - T*N0/(P*g_s)                  [EXPANDED — ζ=0, no residual]
```
Effect: upper bound widens significantly → more power can go to weak user → higher sum rate.

---

### 4.3 What IS Implemented (✅)

| Feature | Implemented? | Where |
|---|---|---|
| Standard SIC SINR computation | ✅ | `sic.py: sinr_weak_user()`, `sinr_strong_after_sic()` |
| CIC SINR (no interference) | ✅ | `sic.py: sinr_weak_user_with_cache()` |
| Cache-unaware power allocation | ✅ | `power_allocation.py: allocate_power_closedform()` |
| Cache-aware power allocation | ✅ | `power_allocation.py: allocate_power_cache_aware()` |
| `weak_cached` flag propagation | ✅ | `noma_base.py: simulate_noma_pair()` |
| `strong_cached` flag propagation | ✅ | `noma_base.py: simulate_noma_pair()` |
| zeta=0 for strong_cached | ✅ | `power_allocation.py: line 'zeta = 0.0 if strong_cached'` |
| Residual=0 for strong_cached | ✅ | `noma_base.py: residual = 0.0` |
| User pairing (extreme) | ✅ | `noma_base.py: pair_users_extreme()` |
| CIC benefit rate tracking | ✅ | `noma_base.py: simulate_noma_system()` |
| CIC-aware reward in DQN | ✅ | `dqn_cache_final.py: +1.5 CIC bonus` |

---

### 4.4 What is MISSING / NOT Implemented for Clean Comparison (⚠️)

| Feature | Status | Impact |
|---|---|---|
| **Explicit "Standard CIC" mode** | ⚠️ MISSING | No dedicated function that uses `closedform` PA + CIC SINR post-hoc |
| **Explicit "Hybrid CIC" mode** | ✅ Implemented as `cache_aware` | But not labeled/separated for clean comparison |
| **Experiment runner for Std vs Hybrid CIC** | ⚠️ MISSING | `comparative_analysis.py` compares cache policies, not CIC modes |
| **Per-pair CIC mode logging** | ⚠️ PARTIAL | CIC flags tracked but not separated by mode |
| **Sweep across SNR for both modes** | ⚠️ MISSING | No dedicated sweep script for Standard vs Hybrid |

---

### 4.5 How the Current Config is Set Up

Currently, `config.py` has:
```python
POWER_ALLOC_METHOD = "cache_aware"
ENABLE_CIC = True
CIC_PERFECT = True
```

This means the system is currently **always running Hybrid CIC** (cache-aware PA + CIC). To run Standard CIC, you would need to change `POWER_ALLOC_METHOD = "closedform"` (or `"gridsearch"`) while keeping `ENABLE_CIC = True`. The CIC SINR correction in `noma_base.py` is triggered by the `weak_cached` flag regardless of power method.

**Important nuance:** In Standard CIC, the `weak_cached` flag still triggers `SINR_w = P*p_w*g_w / N0` inside `noma_base.py`. So the CIC SINR benefit applies in both modes — the ONLY difference is whether the power allocation was computed knowing about CIC or not.

---

## 5. Answer: Can We Compare Standard CIC vs Hybrid CIC?

### ✅ YES — All Core Physics Is Implemented

The mathematical foundations are fully in place:
- Standard CIC SINR formula: ✅
- Hybrid CIC SINR formula: ✅
- Cache-unaware power allocation (`closedform`): ✅
- Cache-aware power allocation (`cache_aware`): ✅
- Both `weak_cached` and `strong_cached` flags: ✅

### ⚠️ BUT — A Comparison Experiment Script Is Missing

To do a clean, publishable comparison, we need a new script that:
1. Runs the SAME user pairs / channel realizations under two modes:
   - **Standard CIC**: `POWER_ALLOC_METHOD = 'closedform'`, `ENABLE_CIC = True`
   - **Hybrid CIC**: `POWER_ALLOC_METHOD = 'cache_aware'`, `ENABLE_CIC = True`
2. Sweeps over SNR (or TX_POWER) and measures: sum rate, outage, SINR gains
3. Also compares DQN cache vs Top-K under each mode (4 combinations)

### 🔧 What Needs to Be Built (Next Stage)

```
[NEXT TASK] Create: src/experiments/standard_vs_hybrid_cic.py

Scenarios to simulate:
  A. Standard CIC + Top-K cache
  B. Standard CIC + DQN cache
  C. Hybrid CIC + Top-K cache
  D. Hybrid CIC + DQN cache

Metrics per scenario:
  - Average sum rate (bps/Hz)
  - Outage probability
  - Weak user SINR distribution
  - Strong user SINR distribution
  - CIC benefit rate (%)
  - Hit rate (%)
```

---

## 6. Key Insights & Observations

1. **Power allocation is the fundamental differentiator.** In Standard CIC, the power algorithm is "cache-blind" — it sets p_w from analytical SINR constraints assuming interference always exists. In Hybrid CIC, it knowingly exploits the zero-residual condition to widen the feasible power range.

2. **The current default config (`POWER_ALLOC_METHOD = 'cache_aware'`) already implements Hybrid CIC.** To run Standard CIC, simply switch the method to `'closedform'`.

3. **The CIC SINR gain is automatic** — `noma_base.py` always applies it when `weak_cached=True`, regardless of power method. This means even Standard CIC gets the SINR benefit at the receiver — the difference is purely in how power was pre-allocated.

4. **DQN reward has a built-in CIC bonus (+1.5).** This means DQN is already incentivized to prefer cache decisions that enable CIC. Under Hybrid CIC with DQN, this incentive is consistent with the power allocation strategy — a theoretically clean combination.

5. **Bug fixes are comprehensive.** BUG-PA-1, BUG-NOMA-3, FIX #1, FIX #2 are all applied. The codebase is stable for comparison.

6. **cic_pairing/ folder** on this branch contains diagnostic scripts specifically for analyzing CIC pairing effects — useful for the next stage.

---

## 7. File-by-File Summary Table

| File | Role | Lines (approx) | Status |
|---|---|---|---|
| `src/config.py` | Global parameters | ~350 | ✅ Complete, validated |
| `src/noma/power_allocation.py` | All PA algorithms | ~350 | ✅ Complete + BUG-PA-1 fixed |
| `src/noma/sic.py` | SIC / CIC SINR formulas | ~400 | ✅ Complete |
| `src/noma/noma_base.py` | NOMA pair/system simulation | ~350 | ✅ Complete + 3 bugs fixed |
| `src/noma/channel_model.py` | Channel generation | ~500 | ✅ Complete |
| `src/caching/cache_base.py` | Cache abstract base | ~450 | ✅ Complete |
| `src/caching/static_cache.py` | Top-K, LRU, LFU | ~350 | ✅ Complete |
| `src/caching/dynamic_cache.py` | Adaptive cache | ~350 | ✅ Complete |
| `src/caching/dqn_cache_final.py` | DQN agent (v2 binary) | ~900 | ✅ Complete |
| `src/simulation/noma_caching_sim.py` | Main Monte Carlo sim | ~780 | ✅ Complete |
| `src/simulation/stable_dqn_sim.py` | DQN simulation loop | ~950 | ✅ Complete |
| `src/experiments/comparative_analysis.py` | Policy comparison | ~950 | ✅ Complete (policies only) |
| `cic_pairing/cic_pairing_analysis.py` | CIC pairing analysis | ~400 | ✅ Available |
| **MISSING** | Standard vs Hybrid CIC runner | — | ⚠️ Needs to be created |

---

*This document was auto-generated by full codebase analysis on 03 July 2026.*  
*Branch: temp_cic | Repo: Manojit-Pal/cache_aided_noma*
