# 🔄 Cache-Aided Interference Cancellation (CIC) Pairing Guide

## Table of Contents
1. [What is CIC Pairing?](#what-is-cic-pairing)
2. [How CIC Works](#how-cic-works)
3. [Checking CIC Pairing](#checking-cic-pairing)
4. [Understanding the Results](#understanding-the-results)
5. [Common Scenarios](#common-scenarios)
6. [Troubleshooting](#troubleshooting)

---

## What is CIC Pairing?

**Cache-Aided Interference Cancellation (CIC)** is a technique where users with cached content can perfectly cancel interference from other users transmitting that same content.

In NOMA (Non-Orthogonal Multiple Access):
- Two users share the same frequency band
- One user's signal acts as interference to the other
- **With CIC**: If a user has the interfering content cached, they can cancel it perfectly!

### CIC Pairing Types

| Type | Symbol | Description | Benefit |
|------|--------|-------------|---------|
| **No CIC** | `  ` | Neither user has cache | Standard NOMA performance |
| **Weak CIC** | `🔄←` | Weak user has strong user's content cached | Weak user cancels interference from strong user |
| **Strong CIC** | `→🔄` | Strong user has weak user's content cached | Perfect SIC (no residual interference) |
| **Both CIC** | `🔄🔄` | Both users have each other's content | Maximum performance gain |

---

## How CIC Works

### Standard NOMA (No Cache)

```
Weak User:  Receives signal = [Weak Signal] + [Strong Signal (interference)] + Noise
            SINR = (P*p_w*g_w) / (P*p_s*g_w + N0)

Strong User: 1. Decodes weak signal (treats own as interference)
             2. Performs SIC (Successive Interference Cancellation)
             3. Residual = ζ * (weak signal power)  [ζ ≈ 0.05 = imperfection]
             SINR = (P*p_s*g_s) / (N0 + residual)
```

### Cache-Aided NOMA (With CIC)

#### Scenario 1: Weak User Has Cache (🔄←)
```
Weak User:  Has strong user's content cached
            Can reconstruct and subtract strong user's signal perfectly!
            SINR = (P*p_w*g_w) / N0  ← NO INTERFERENCE!
            Improvement: Typically 2x - 10x SINR boost
```

#### Scenario 2: Strong User Has Cache (→🔄)
```
Strong User: Has weak user's content cached
             Perfect SIC (residual = 0)
             SINR = (P*p_s*g_s) / N0  ← PERFECT CANCELLATION!
```

#### Scenario 3: Both Users Have Cache (🔄🔄)
```
Both:       Maximum benefit - both users cancel interference
            Highest sum rate possible
            Improvement: Up to 60% sum rate gain
```

---

## Checking CIC Pairing

### Method 1: Quick Visual Check

**Run the quick check script:**
```bash
python examples/quick_cic_check.py
```

**Output:**
```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       QUICK CIC PAIRING CHECK                        │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                               RESULTS                                │
├──────────────────────────────────────────────────────────────────────────────┤
│ Pair 1: ❌ U04    ❌ U09 | No CIC   | Rate: 0.847 ✅              │
│ Pair 2: ✅ U00 🔄← ❌ U08 | Weak CIC | Rate: 1.234 ✅              │
│ Pair 3: ❌ U05 →🔄 ✅ U07 | Strong CIC| Rate: 1.156 ✅              │
│ Pair 4: ✅ U01 🔄🔄 ✅ U06 | Both CIC | Rate: 1.512 ✅              │
│ Pair 5: ❌ U03    ✅ U02 | Strong CIC| Rate: 1.089 ✅              │
├──────────────────────────────────────────────────────────────────────────────┤
│ Total: 5 pairs | Weak CIC: 1 | Strong CIC: 2 | Both: 1                 │
│ Success Rate: 100.0% | Avg Sum Rate: 1.168 bps/Hz                   │
│ Cache-Aware Power: 4/5 pairs                                         │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Method 2: Programmatic Check

```python
from src.noma import simulate_noma_system
import src.config as cfg

# Your simulation
results = simulate_noma_system(gains, cfg, cache_status=cache_status)

# Check each pair
for pr in results['pair_results']:
    weak_id = pr['weak_idx']
    strong_id = pr['strong_idx']
    cic_users = pr['cic_users']  # List: [], ['weak'], ['strong'], or ['weak', 'strong']
    
    if len(cic_users) == 0:
        print(f"Pair ({weak_id}, {strong_id}): No CIC")
    elif len(cic_users) == 2:
        print(f"Pair ({weak_id}, {strong_id}): Both users benefit from CIC 🔄🔄")
    elif 'weak' in cic_users:
        print(f"Pair ({weak_id}, {strong_id}): Weak user has CIC 🔄←")
    elif 'strong' in cic_users:
        print(f"Pair ({weak_id}, {strong_id}): Strong user has CIC →🔄")
```

### Method 3: Full Analysis

```bash
python examples/cic_pairing_analysis.py
```

This provides:
- ✅ Visual pairing diagram
- ✅ Detailed statistics
- ✅ Performance comparison
- ✅ Top performing pairs
- ✅ Export capabilities

---

## Understanding the Results

### Key Fields in Pair Results

```python
pair_result = {
    'weak_idx': 5,              # Weak user ID
    'strong_idx': 12,           # Strong user ID
    'weak_cached': True,        # Weak user has content in cache
    'strong_cached': False,     # Strong user cache status
    'cic_users': ['weak'],      # Who benefits from CIC
    'cic_applied': True,        # Whether CIC was applied
    
    # Power allocation
    'p_w': 0.65,                # Power coefficient for weak user
    'p_s': 0.35,                # Power coefficient for strong user
    'power_allocation': {       # How power was allocated
        'method': 'cache_aware' # 'cache_aware', 'config_default', etc.
    },
    
    # Performance
    'sinr_w': 2.45,             # Weak user SINR (linear)
    'sinr_s_after': 3.12,       # Strong user SINR after SIC
    'sum_rate': 1.234,          # Sum rate (bps/Hz)
    'weak_success': True,       # Weak user met target rate
    'strong_success': True,     # Strong user met target rate
    'pair_success': True,       # Both users successful
}
```

### System Metrics

```python
metrics = results['system_metrics']

print(f"Total pairs: {metrics['num_pairs']}")
print(f"Weak CIC count: {metrics['weak_cic_count']}")
print(f"Strong CIC count: {metrics['strong_cic_count']}")
print(f"Both CIC count: {metrics['both_cic_count']}")
print(f"Cache-aware power used: {metrics['cache_aware_power_count']} pairs")
```

---

## Common Scenarios

### Scenario 1: No CIC Detected (Despite Having Cached Users)

**Problem**: You have cached users but `cic_users` is always empty.

**Causes**:
1. Cache status not properly set
2. Cached users not paired together
3. Pairing strategy issues

**Solution**:
```python
# Check cache status
print(f"Cached users: {[i for i, c in cache_status.items() if c]}")

# Check pairs
for pair in results['pairs']:
    weak_id, strong_id = pair
    print(f"Pair: {weak_id} (cached={cache_status[weak_id]}) <-> "
          f"{strong_id} (cached={cache_status[strong_id]})")
```

### Scenario 2: Cache-Aware Power Not Being Used

**Problem**: `power_allocation['method']` is `'config_default'` instead of `'cache_aware'`.

**Solution**:
```python
# 1. Check config
import src.config as cfg
print(f"Power method: {cfg.POWER_ALLOC_METHOD}")
# Should be 'cache_aware'

# 2. Enable optimization in simulation
results = simulate_noma_system(
    gains, cfg,
    cache_status=cache_status,
    optimize_power=True  # ← MUST BE TRUE
)
```

### Scenario 3: Both Users Cached But Different Content

**Important**: CIC only works if users have **each other's requested content** cached!

```python
# Example: Both users cached, but different files
user_A_requests = file_5
user_B_requests = file_8

# User A cache contains: [file_1, file_2, file_3]
# User B cache contains: [file_6, file_7, file_9]

# Result: NO CIC! ❌
# Because:
# - User A doesn't have file_8 (B's request)
# - User B doesn't have file_5 (A's request)
```

**Correct CIC scenario**:
```python
# User A requests file_5
# User B requests file_8

# User A cache: [file_1, file_8]  ← Has B's file!
# User B cache: [file_5, file_9]  ← Has A's file!

# Result: Both CIC! 🔄🔄
```

---

## Troubleshooting

### Debug Checklist

```python
def debug_cic_pairing(results, cache_status):
    """
    Comprehensive debug check for CIC pairing.
    """
    print("=" * 60)
    print("CIC PAIRING DEBUG")
    print("=" * 60)
    
    # 1. Cache status
    num_cached = sum(cache_status.values())
    print(f"\n1. Cache Status:")
    print(f"   Total users: {len(cache_status)}")
    print(f"   Cached users: {num_cached} ({num_cached/len(cache_status)*100:.1f}%)")
    print(f"   Cached IDs: {[i for i, c in cache_status.items() if c]}")
    
    # 2. Pairing
    print(f"\n2. Pairing:")
    print(f"   Total pairs: {len(results['pairs'])}")
    for i, (w, s) in enumerate(results['pairs']):
        w_cache = "✅" if cache_status.get(w, False) else "❌"
        s_cache = "✅" if cache_status.get(s, False) else "❌"
        print(f"   Pair {i+1}: {w_cache} U{w:02d} <-> {s_cache} U{s:02d}")
    
    # 3. CIC application
    metrics = results['system_metrics']
    print(f"\n3. CIC Application:")
    print(f"   Weak CIC:   {metrics['weak_cic_count']} pairs")
    print(f"   Strong CIC: {metrics['strong_cic_count']} pairs")
    print(f"   Both CIC:   {metrics['both_cic_count']} pairs")
    
    total_cic = metrics['weak_cic_count'] + metrics['strong_cic_count']
    if total_cic == 0 and num_cached > 0:
        print(f"   ⚠ WARNING: No CIC despite {num_cached} cached users!")
    
    # 4. Power allocation
    print(f"\n4. Power Allocation:")
    if 'cache_aware_power_count' in metrics:
        print(f"   Cache-aware used: {metrics['cache_aware_power_count']}/{metrics['num_pairs']} pairs")
    else:
        print(f"   ⚠ Cache-aware power allocation not tracked")
    
    # 5. Performance
    print(f"\n5. Performance:")
    print(f"   Success rate: {metrics['overall_success_rate']*100:.1f}%")
    print(f"   Avg sum rate: {metrics['average_sum_rate']:.3f} bps/Hz")
    
    print("\n" + "=" * 60)

# Usage
debug_cic_pairing(results, cache_status)
```

### Common Issues and Fixes

| Issue | Symptom | Fix |
|-------|---------|-----|
| **No CIC** | `cic_users` always empty | Check `cache_status` dictionary is passed correctly |
| **Wrong power** | Method is `'config_default'` | Set `optimize_power=True` and `cfg.POWER_ALLOC_METHOD='cache_aware'` |
| **Low improvement** | CIC doesn't improve performance | Check channel gains (may be too similar) |
| **Import error** | `allocate_power_cache_aware` not found | Update to latest version of power_allocation.py |

---

## Quick Reference

### How to Check Which Pairs Have CIC

```python
# Simple one-liner for each CIC type
no_cic = [(r['weak_idx'], r['strong_idx']) for r in results['pair_results'] 
          if len(r.get('cic_users', [])) == 0]

weak_cic = [(r['weak_idx'], r['strong_idx']) for r in results['pair_results'] 
            if 'weak' in r.get('cic_users', [])]

strong_cic = [(r['weak_idx'], r['strong_idx']) for r in results['pair_results'] 
              if 'strong' in r.get('cic_users', [])]

both_cic = [(r['weak_idx'], r['strong_idx']) for r in results['pair_results'] 
            if len(r.get('cic_users', [])) == 2]

print(f"Weak CIC pairs: {weak_cic}")
print(f"Strong CIC pairs: {strong_cic}")
print(f"Both CIC pairs: {both_cic}")
```

### Performance Impact

```python
# Calculate CIC benefit
no_cic_rate = np.mean([r['sum_rate'] for r in results['pair_results'] 
                       if len(r.get('cic_users', [])) == 0])

both_cic_rate = np.mean([r['sum_rate'] for r in results['pair_results'] 
                         if len(r.get('cic_users', [])) == 2])

improvement = (both_cic_rate / no_cic_rate - 1) * 100
print(f"Both CIC improvement: {improvement:.1f}%")
```

---

## 📚 Additional Resources

- **Examples**: See `examples/cic_pairing_analysis.py` for full demo
- **Quick Check**: Run `examples/quick_cic_check.py` for fast verification
- **API Docs**: Check `docs/user_guide.md` for detailed API
- **Theory**: See research paper for mathematical background

---

**Last Updated**: December 2025  
**Version**: 1.0.0
