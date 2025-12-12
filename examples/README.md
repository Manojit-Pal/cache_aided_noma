# Cache-Aided NOMA Examples

This directory contains example scripts demonstrating the usage of the Cache-Aided NOMA simulation framework.

## 📁 Available Examples

### 1. `cic_pairing_analysis.py` - **CIC Pairing Analysis** ⭐

A comprehensive demonstration of Cache-Aided Interference Cancellation (CIC) pairing analysis.

**Features:**
- ✅ Visualize which user pairs benefit from CIC
- ✅ Compare standard NOMA vs cache-aided NOMA
- ✅ Analyze performance gains from cache-aware power allocation
- ✅ Export results for further analysis
- ✅ Detailed statistics and top-performing pair analysis

---

## 🚀 Quick Start

### Running the CIC Pairing Analysis

```bash
# From the project root directory
python examples/cic_pairing_analysis.py
```

### Expected Output

The script produces three main sections:

#### **Part 1: Basic CIC Pairing Analysis**

Shows a visual representation of user pairing:

```
================================================================================
CIC PAIRING VISUALIZATION
================================================================================

Legend: ✅ = Cached | ❌ = Not Cached | 🔄 = CIC Applied
--------------------------------------------------------------------------------
Pair | Weak User            | CIC    | Strong User          | Sum Rate
--------------------------------------------------------------------------------
   1 | ❌ User   0          |        | ❌ User  29          |     0.847 ✅
   2 | ✅ User   3          | 🔄←    | ❌ User  28          |     1.234 ✅
   3 | ❌ User   1          |   →🔄  | ✅ User  27          |     1.156 ✅
   4 | ✅ User   6          | 🔄🔄   | ✅ User  24          |     1.512 ✅
...
```

**Legend Explanation:**
- `✅` next to user = User has cached content
- `❌` next to user = User does not have cached content
- `🔄←` = Weak user benefits from CIC (can cancel strong's interference)
- `→🔄` = Strong user benefits from CIC (perfect SIC)
- `🔄🔄` = Both users benefit from CIC
- `✅` at end = Pair successful (both users meet target rate)
- `❌` at end = Pair in outage

#### **Part 2: Detailed Statistics**

```
================================================================================
CIC STATISTICS
================================================================================

📊 Total Pairs: 15

🔄 CIC Breakdown:
  • No CIC:           7 pairs ( 46.7%)
  • Weak User CIC:    3 pairs ( 20.0%)
  • Strong User CIC:  3 pairs ( 20.0%)
  • Both Users CIC:   2 pairs ( 13.3%)

📈 Performance Metrics:
  • Overall Success Rate: 86.7%
  • Average Sum Rate:     1.124 bps/Hz
  • Outage Probability:   13.3%
  • Cache Hit Rate:       30.0%

⚡ Power Allocation:
  • Cache-Aware Power Used: 8/15 pairs
  • Optimization Enabled:   True

📋 Average Sum Rates by CIC Type:
  • No CIC:          0.952 bps/Hz
  • Weak User CIC:   1.187 bps/Hz (+24.7%)
  • Strong User CIC: 1.203 bps/Hz (+26.4%)
  • Both Users CIC:  1.534 bps/Hz (+61.1%)
================================================================================
```

**Key Insights:**
- **Both Users CIC** provides the best performance (61% improvement)
- **Cache-aware power allocation** is used automatically when CIC is detected
- **Success rate** improves significantly with CIC

#### **Part 3: Performance Comparison**

```
================================================================================
COMPARISON: STANDARD NOMA vs CACHE-AIDED NOMA
================================================================================

🔵 Scenario 1: Standard NOMA (No Cache)...
  • Success Rate: 72.0%
  • Avg Sum Rate:  0.943 bps/Hz
  • Outage Prob:   28.0%

🟠 Scenario 2: Cache-Aided NOMA (Standard Power)...
  • Success Rate: 84.0%
  • Avg Sum Rate:  1.087 bps/Hz
  • Outage Prob:   16.0%
  • CIC Applied:   8 weak, 7 strong, 3 both

🟢 Scenario 3: Cache-Aided NOMA (Cache-Aware Power)...
  • Success Rate: 88.0%
  • Avg Sum Rate:  1.165 bps/Hz
  • Outage Prob:   12.0%
  • CIC Applied:   8 weak, 7 strong, 3 both

📈 Performance Gains:
  • Cache-aided NOMA (std power) vs Standard: +15.3% sum rate
  • Cache-aided NOMA (opt power) vs Standard: +23.5% sum rate
  • Cache-aware power allocation benefit:     +7.2% sum rate
  • Outage probability reduction:             -16.0%
================================================================================
```

**Key Findings:**
- Cache-aided NOMA provides **15-23% sum rate improvement**
- Cache-aware power allocation adds **additional 7% gain**
- Outage probability reduced by **16 percentage points**

---

## 🔧 Customization

### Modify Cache Ratio

```python
# In the script, find:
results = compare_with_without_cache(
    num_users=50,
    cache_ratio=0.3,  # Change this (0.0 to 1.0)
    seed=42
)
```

### Change Pairing Strategy

```python
results = simulate_noma_system(
    gains, cfg,
    cache_status=cache_status,
    pairing_method='extreme',  # Options: 'extreme', 'random', 'sequential'
    optimize_power=True
)
```

### Adjust Number of Users

```python
num_users = 30  # Change to any even number
positions = generate_user_positions(num_users, cfg.CELL_RADIUS, seed=42)
```

### Define Custom Cache Placement

```python
# Option 1: Random cache placement
cache_status = {i: np.random.rand() < 0.3 for i in range(num_users)}

# Option 2: Strategic cache placement (near cell edge)
cache_status = {i: (positions[i][0]**2 + positions[i][1]**2) > 250000 
                for i in range(num_users)}

# Option 3: Specific users
cached_users = [0, 5, 10, 15, 20]  # Manually select
cache_status = {i: (i in cached_users) for i in range(num_users)}
```

---

## 💾 Export Results

### Export to CSV

```python
import pandas as pd

# Create DataFrame from pair results
df = pd.DataFrame(results['pair_results'])

# Save to CSV
df.to_csv('cic_pairing_results.csv', index=False)

print("Results exported to cic_pairing_results.csv")
```

### Export to JSON

```python
import json

# Export system metrics
with open('cic_metrics.json', 'w') as f:
    json.dump(results['system_metrics'], f, indent=2)

print("Metrics exported to cic_metrics.json")
```

### Filter and Export CIC Pairs Only

```python
import pandas as pd

# Get only pairs with CIC
cic_pairs = [pr for pr in results['pair_results'] 
             if len(pr.get('cic_users', [])) > 0]

df_cic = pd.DataFrame(cic_pairs)
df_cic.to_csv('cic_pairs_only.csv', index=False)

print(f"Exported {len(cic_pairs)} CIC pairs")
```

---

## 📊 Understanding the Results

### CIC User Types

| CIC Type | Description | Benefit |
|----------|-------------|----------|
| **No CIC** | Neither user has cache | Standard NOMA performance |
| **Weak CIC** (`🔄←`) | Weak user has strong's content cached | Weak user cancels interference from strong |
| **Strong CIC** (`→🔄`) | Strong user has weak's content cached | Perfect SIC (no residual interference) |
| **Both CIC** (`🔄🔄`) | Both users have each other's content | Maximum performance gain |

### Power Allocation Methods

```python
# Check which power allocation was used for each pair
for pr in results['pair_results']:
    print(f"Pair ({pr['weak_idx']}, {pr['strong_idx']}): "
          f"{pr['power_allocation']['method']}")
```

Possible methods:
- `config_default` - Used default power from config
- `cache_aware` - Cache-aware optimization applied ✅
- `cache_aware_failed_fallback` - Optimization failed, used default

---

## 🎯 Use Cases

### 1. **Baseline Comparison**
```python
# Compare your results against standard NOMA
results_standard = simulate_noma_system(gains, cfg, 
                                        cache_status={i: False for i in range(num_users)},
                                        optimize_power=False)

results_cache = simulate_noma_system(gains, cfg,
                                     cache_status=cache_status,
                                     optimize_power=True)

improvement = (results_cache['system_metrics']['average_sum_rate'] / 
               results_standard['system_metrics']['average_sum_rate'] - 1) * 100

print(f"Cache-aided NOMA improvement: {improvement:.1f}%")
```

### 2. **Cache Placement Optimization**
```python
# Test different cache ratios
for cache_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
    num_cached = int(num_users * cache_ratio)
    cache_status = {i: (i < num_cached) for i in range(num_users)}
    
    results = simulate_noma_system(gains, cfg, cache_status=cache_status)
    
    print(f"Cache ratio {cache_ratio*100:.0f}%: "
          f"Sum rate = {results['system_metrics']['average_sum_rate']:.3f} bps/Hz")
```

### 3. **Pairing Strategy Analysis**
```python
# Compare pairing methods
for method in ['extreme', 'random', 'sequential']:
    results = simulate_noma_system(gains, cfg,
                                   cache_status=cache_status,
                                   pairing_method=method)
    
    print(f"{method.capitalize()} pairing: "
          f"Both CIC = {results['system_metrics']['both_cic_count']} pairs")
```

---

## 🐛 Troubleshooting

### Issue: No CIC pairs appearing
**Solution**: Check cache_status dictionary
```python
print(f"Cached users: {[i for i, cached in cache_status.items() if cached]}")
print(f"Total cached: {sum(cache_status.values())}")
```

### Issue: Cache-aware power not being used
**Solution**: Enable cache-aware power allocation in config
```python
# In src/config.py, ensure:
POWER_ALLOC_METHOD = 'cache_aware'

# And in simulation:
results = simulate_noma_system(gains, cfg, optimize_power=True)
```

### Issue: Low sum rates
**Solution**: Check channel conditions and power settings
```python
print(f"Min channel gain: {np.min(gains):.2e}")
print(f"Max channel gain: {np.max(gains):.2e}")
print(f"TX power: {cfg.TX_POWER} W")
print(f"Noise power: {cfg.NOISE_POWER} W")
```

---

## 📚 Related Documentation

- **User Guide**: `../docs/user_guide.md` - Complete API reference
- **Configuration**: `../src/config.py` - All simulation parameters
- **Theory**: `../docs/theory.md` - Mathematical background

---

## 🤝 Contributing

Want to add more examples? Please:
1. Follow the existing code style
2. Add comprehensive comments
3. Include example outputs in this README
4. Test thoroughly before submitting

---

## 📧 Support

For questions or issues:
- Open an issue on GitHub
- Check the documentation
- Review existing examples

---

**Happy Experimenting! 🚀**
