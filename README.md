# Cache-Aided NOMA with Deep Reinforcement Learning

<div align="center">

**Advanced 6G Wireless Communication System with Intelligent Caching**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-success.svg)](https://github.com/Manojit-Pal/cache_aided_noma)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Algorithm Details](#-algorithm-details)
- [Usage](#-usage)
- [Configuration](#%EF%B8%8F-configuration)
- [Results](#-results)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)

---

## 🎯 Overview

This project implements a **Cache-Aided Non-Orthogonal Multiple Access (NOMA)** system enhanced with **Deep Reinforcement Learning (DRL)** for next-generation 6G wireless networks. The system intelligently caches popular content to enable **Cache-aided Interference Cancellation (CIC)**, significantly improving spectral efficiency and user QoS.

### 🌟 What Makes This Unique?

1. **Cache-Aided Interference Cancellation (CIC)**: When a paired NOMA user has the other user's requested file cached, it can perform **perfect interference cancellation** — the weak user eliminates strong-user interference entirely, and the strong user achieves perfect SIC (ζ=0). This dramatically improves SINR by 2–10×.

2. **Deep Q-Network (DQN) Cache**: Uses a Dueling DQN with Prioritized Experience Replay and a binary action space (`skip` / `cache`) to learn optimal caching policies that maximize CIC opportunities while maintaining high cache hit rates.

3. **Cache-Aware Power Allocation**: Dynamically adjusts NOMA power coefficients based on cache status, exploiting CIC to save transmit power while maintaining QoS targets.

4. **Complete NOMA Implementation**: Full support for Successive Interference Cancellation (SIC) with configurable imperfection, multiple power allocation strategies, extreme user pairing, and multi-fading channel models.

---

## ✨ Key Features

### 🔄 NOMA Components

- **Successive Interference Cancellation (SIC)**
  - Imperfect SIC with configurable residual interference factor (ζ=0.05)
  - Cache-aided perfect SIC when partner's file is cached (ζ=0)
  - Strong user decodes weak user's signal first

- **Cache-Aided Interference Cancellation (CIC)**
  - Weak user CIC: partner's (strong) file is cached → interference fully eliminated
  - Strong user CIC: partner's (weak) file is cached → perfect SIC (ζ=0)
  - Both users can independently benefit from CIC
  - CIC reward bonus in DQN learning (+1.5)

- **Power Allocation Strategies**
  - Closed-form analytical allocation
  - Grid search optimization
  - **Cache-aware dynamic allocation** (novel — adjusts p_w/p_s based on CIC status)
  - Sum-rate maximization
  - Energy-efficient allocation

- **User Pairing**
  - Extreme pairing (weakest ↔ strongest channel) — default
  - Random pairing
  - Sequential pairing

- **Channel Modeling**
  - Rayleigh fading (NLoS)
  - Rician fading (LoS with K-factor)
  - Mixed fading (probabilistic LoS/NLoS)
  - Distance-dependent path loss
  - Time-varying channels (Jake's Doppler model)

### 🧠 Deep Reinforcement Learning

- **Dueling DQN Architecture (v2 — Binary Action)**
  - Binary action space: `0=skip`, `1=cache` (v2 redesign for faster convergence)
  - Separates value and advantage streams
  - Compact network: [64, 32] hidden dims

- **Prioritized Experience Replay (PER)**
  - Samples important transitions more frequently
  - Beta annealing (0.4 → 1.0) over training
  - Alpha priority exponent: 0.6

- **NOMA-Aware State Representation**
  - LRU counters (recency)
  - LFU counters (frequency)
  - File popularity (EMA-tracked)
  - Channel quality (mean, std)
  - NOMA metrics (CIC rate, success rate)
  - Cache occupancy

- **Training Configuration (v2)**
  - 500 episodes × 10,000 steps/episode = 5M total steps
  - Epsilon decay: 1.0 → 0.01 over 2M steps (40%)
  - Learning rate: 0.001 (10× faster than v1)
  - Soft target updates (τ=0.005)
  - Gradient clipping (max norm=10.0)
  - Warm-up: 2,000 steps

### 📊 Caching Policies

- **DQN**: Deep Q-Network with CIC-aware rewards (our approach)
- **TopK**: Cache K most popular files (static baseline)
- **LRU**: Least Recently Used (dynamic baseline)
- **LFU**: Least Frequently Used (dynamic baseline)
- **Random**: Random replacement (dynamic baseline)
- **NO-CACHE**: Traditional NOMA without caching (baseline)

---

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/Manojit-Pal/cache_aided_noma.git
cd cache_aided_noma

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run quick test (~10-15 minutes)
python run_comparison.py --quick

# 4. View results
open results/cache_aided_vs_traditional_noma.png
```

---

## 💻 Installation

### Prerequisites

- Python 3.8, 3.9, 3.10, 3.11, or 3.12
- pip (Python package manager)
- (Optional) NVIDIA GPU with CUDA 11.8 or 12.1

### Basic Installation (CPU)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, numpy; print('✅ Ready!')"
```

### GPU Installation (CUDA 11.8)

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Verify GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

**For detailed installation instructions, see [INSTALL.md](INSTALL.md)**

---

## 📁 Project Structure

```
cache_aided_noma/
│
├── 📄 README.md                        # This file
├── 📄 INSTALL.md                       # Detailed installation guide
├── 📄 requirements.txt                 # Python dependencies
├── 📄 IEEE_Paper.tex                   # Research paper (LaTeX)
│
├── 🚀 run_comparison.py                # Main experiment runner (CLI)
├── 🎓 train_and_evaluate_dqn.py        # DQN training entry point (root)
├── 🧪 test_dqn_cache.py               # DQN unit tests
├── 🧪 test_noma_integration.py        # NOMA integration tests
├── 🧪 test_noma_sim.py                # NOMA simulation tests
│
├── 📦 src/                            # Source code
│   ├── 📄 __init__.py
│   ├── ⚙️  config.py                  # All configuration parameters
│   ├── 🛠️  utils.py                   # Zipf sampling, stats, I/O utilities
│   │
│   ├── 💾 caching/                    # Cache implementations
│   │   ├── __init__.py               # Factory: create_cache()
│   │   ├── cache_base.py             # Abstract base class (NOMA-aware)
│   │   ├── dqn_cache_final.py        # DQN cache agent (v2 binary-action)
│   │   ├── static_cache.py           # StaticTopKCache
│   │   ├── dynamic_cache.py          # LRU, LFU, Random policies
│   │   ├── test_caching_policies.py  # Caching policy tests
│   │   └── verify_init.py            # Init verification script
│   │
│   ├── 📡 noma/                       # NOMA physical layer
│   │   ├── __init__.py               # Public API re-exports
│   │   ├── channel_model.py          # Fading, path loss, mobility
│   │   ├── noma_base.py              # User pairing, NOMA pair simulation
│   │   ├── power_allocation.py       # 5 power allocation strategies
│   │   └── sic.py                    # SIC/CIC, SINR formulas
│   │
│   ├── 🔬 simulation/                 # Simulation engines
│   │   ├── __init__.py               # Conditional imports
│   │   ├── noma_caching_sim.py       # General NOMA + cache simulator
│   │   ├── stable_dqn_sim.py         # DQN trainer + policy evaluator
│   │   └── train_and_evaluate_dqn.py # End-to-end pipeline script
│   │
│   └── 📊 experiments/                # Experiment scripts
│       └── comparative_analysis.py   # SNR-sweep Monte Carlo comparison
│
├── 🔧 cic_pairing/                   # CIC pairing analysis utilities
│   ├── cic_pairing_analysis.py       # CIC pairing visualization
│   ├── diagnose_and_fix.py           # Diagnostic utilities
│   └── quick_cic_check.py            # Quick CIC verification
│
├── 📊 results/                        # Experiment results
├── 📊 results_csv/                    # CSV result archives
├── 📊 results_pdf/                    # PDF result archives
├── 📊 results_pic/                    # Image result archives
├── 🧠 models/                         # Trained DQN models
├── 🔧 checkpoints/                    # Training checkpoints
└── 📚 docs/                           # Documentation
```

---

## 🔬 Algorithm Details

### Successive Interference Cancellation (SIC)

**Standard SIC:**
1. Strong user decodes weak user's signal first
2. Subtracts weak signal with imperfection factor ζ
3. Decodes own signal from residual

**SINR (Weak User):**
```
γ_w = (P · p_w · h_w) / (P · p_s · h_w + N₀)
```

**SINR (Strong User, after SIC):**
```
γ_s = (P · p_s · h_s) / (ζ · P · p_w · h_s + N₀)
```
where ζ = SIC imperfection factor (0.05 default)

### Cache-Aided Interference Cancellation (CIC)

**Weak User CIC — when weak user has strong user's file cached:**
```
γ_w^CIC = (P · p_w · h_w) / N₀    (interference fully eliminated)
```
Improvement factor: 1 + P·p_s·h_w/N₀ → typically 2–10× SINR gain.

**Strong User CIC — when strong user has weak user's file cached:**
```
γ_s^CIC = (P · p_s · h_s) / N₀    (ζ = 0, perfect SIC)
```

### Cache-Aware Power Allocation

Dynamic power adjustment based on CIC status:

| Scenario | Effect |
|----------|--------|
| No cache | Standard closed-form allocation |
| Weak user cached | Lower p_w needed (CIC removes interference) |
| Strong user cached | More power for weak user (perfect SIC, ζ=0) |
| Both cached | Maximum flexibility, balanced midpoint |

### DQN Reward Function (v2 — Immediate Rewards)

```
+2.0   Cache hit (no wireless transmission needed)
+1.5   CIC enabled (cached partner's file → perfect cancellation)
 0.0   Standard cache miss (neutral)

Additional modifiers:
  + popularity_weight    Bonus for caching popular files
  + cic_bonus (1.5)      When file enables CIC for paired user
```

---

## 🎮 Usage

### 1. Quick Comparison (Recommended First Run)

```bash
# Quick test: 100 episodes, ~10-15 minutes
python run_comparison.py --quick
```

### 2. Full Experiment (Research Quality)

```bash
# Full comparison: 2000 episodes, ~2-3 hours
python run_comparison.py --full

# Custom configuration
python run_comparison.py --snr-min 0 --snr-max 30 --policies topk lru dqn
```

### 3. DQN-Specific Training

```bash
# Quick smoke test (~2-3 min)
python -m src.simulation.train_and_evaluate_dqn --debug

# Full training + evaluation pipeline
python -m src.simulation.train_and_evaluate_dqn
```

### 4. Testing

```bash
# Test NOMA integration
python test_noma_integration.py

# Test DQN cache
python test_dqn_cache.py

# Test simulation
python test_noma_sim.py

# Test utilities
python -m src.utils
```

### 5. CLI Options — `run_comparison.py`

```bash
Usage: run_comparison.py [OPTIONS]

Options:
  --quick                  Quick test mode (100 episodes, 10 runs)
  --full                   Full experiment (2000 episodes, 100 runs)
  --no-dqn                 Skip DQN training
  --policies POLICIES      Policies to compare (topk, lru, lfu, random, none, dqn)
  --snr-min SNR_MIN        Minimum SNR in dB (default: -10)
  --snr-max SNR_MAX        Maximum SNR in dB (default: 30)
  --snr-step STEP          SNR step size in dB (default: 2)
  --num-realizations N     Monte Carlo realizations per SNR point (default: 1000)
  --output-dir DIR         Results directory (default: results/)
  --config CONFIG          Preset: default, aggressive, conservative
  --seed SEED              Random seed (default: 2025)
  --verbose                Enable verbose output
```

**Examples:**

```bash
# Compare only TopK and DQN
python run_comparison.py --policies topk dqn

# High SNR regime only
python run_comparison.py --snr-min 10 --snr-max 40

# More Monte Carlo runs for smoother curves
python run_comparison.py --num-realizations 2000

# Conservative learning (more stable)
python run_comparison.py --config conservative
```

---

## ⚙️ Configuration

### Configuration File: `src/config.py`

**Key Parameters (v2 — Binary-Action DQN):**

```python
# Content Catalog
NUM_FILES = 2000          # Total files in catalog
CACHE_SIZE = 200          # ~10% cache penetration
ZIPF_ALPHA = 1.0          # Popularity skew

# NOMA System
NUM_USERS = 200           # Users per cell
REQUESTS_PER_USER = 50    # Requests per user per episode
TX_POWER = 2.0            # Watts
CELL_RADIUS = 500.0       # Meters
SIC_IMPERFECTION = 0.05   # ζ = 5%
TARGET_RATE_BPS = 0.3     # bps/Hz
PAIRING_METHOD = "extreme"       # extreme / random / sequential
POWER_ALLOC_METHOD = "cache_aware" # cache_aware / closedform / gridsearch / sumrate_max / energy_efficient

# DQN Training (v2 — Binary-Action)
RL_TRAINING_EPISODES = 500        # v2: was 2000 (converges faster)
RL_STEPS_PER_EPISODE = 10_000     # 200 users × 50 requests
RL_EPSILON_DECAY_STEPS = 2_000_000
RL_GAMMA = 0.99
RL_LEARNING_RATE = 0.001          # v2: 10× faster (simpler problem)
RL_BATCH_SIZE = 64
RL_HIDDEN_DIMS = [64, 32]         # v2: compact network
RL_REPLAY_BUFFER_SIZE = 50_000    # v2: 5× steps/episode

# Prioritized Replay
RL_USE_PRIORITIZED_REPLAY = True
RL_PRIORITY_ALPHA = 0.6
RL_PRIORITY_BETA_START = 0.4
RL_PRIORITY_BETA_END = 1.0
```

### Configuration Presets

```python
from src import config as cfg

cfg.set_debug_config()              # ~1-2 min (tiny scale, 50 episodes)
cfg.set_quick_test_config()         # ~5-10 min (100 episodes)
cfg.set_full_experiment_config()    # ~2-3 hours (2000 episodes)
cfg.set_aggressive_learning_config()   # Faster, less stable
cfg.set_conservative_learning_config() # Slower, more stable
```

---

## 📊 Results

### Output Files

After running `python run_comparison.py`, you'll get:

**1. Performance Comparison Plot**
```
results/cache_aided_vs_traditional_noma.png
```
9-subplot figure showing:
- Outage probability vs SNR
- Sum rate vs SNR
- Cache hit rate vs SNR
- CIC benefit rate vs SNR
- SIC success rate vs SNR
- BER (weak & strong users) vs SNR
- Spectral efficiency vs SNR
- Energy efficiency vs SNR
- Jain's Fairness Index vs SNR

**2. Numerical Results**
```
results/comparative_analysis_results.csv
```

**3. Summary Statistics**
```
results/performance_summary.txt
```

### Expected Performance Gains

**Cache-Aided NOMA vs Traditional NOMA:**
- Outage probability: −40–60% reduction
- Sum rate: +30–50% improvement
- Strong user SINR: +10–20 dB with CIC

---

## 🧪 Testing

### Test Suite Overview

| Test File | Coverage |
|-----------|----------|
| `test_noma_integration.py` | NOMA modules (channel, SIC, power, pairing) |
| `test_dqn_cache.py` | DQN components (state, action, reward, learning) |
| `test_noma_sim.py` | Simulation engine |
| `src/caching/test_caching_policies.py` | All caching policies |

### Run All Tests

```bash
# NOMA integration
python test_noma_integration.py

# DQN cache
python test_dqn_cache.py

# Simulation
python test_noma_sim.py

# Utilities
python -m src.utils
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Run tests (`python test_*.py`)
4. Commit changes (`git commit -m 'Add YourFeature'`)
5. Push to branch (`git push origin feature/YourFeature`)
6. Open a Pull Request

**Code Standards:**
- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@misc{cache_aided_noma_dqn,
  author = {Manojit Pal},
  title = {Cache-Aided NOMA with Deep Reinforcement Learning for 6G Wireless Networks},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Manojit-Pal/cache_aided_noma}
}
```

**Related Papers:**
- Schaul et al., "Prioritized Experience Replay", ICLR 2016
- Wang et al., "Dueling Network Architectures for Deep RL", ICML 2016
- arXiv:1712.09557, "Cache-Aided Non-Orthogonal Multiple Access", 2018
- arXiv:1909.11074, "Power Allocation in Cache-Aided NOMA", 2019

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Research Papers**: Implementations based on state-of-the-art DRL and NOMA research
- **PyTorch Team**: For the excellent deep learning framework

---

## 📞 Contact

**Author**: Manojit Pal

**Repository**: [https://github.com/Manojit-Pal/cache_aided_noma](https://github.com/Manojit-Pal/cache_aided_noma)

**Issues**: [https://github.com/Manojit-Pal/cache_aided_noma/issues](https://github.com/Manojit-Pal/cache_aided_noma/issues)

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star! ⭐**

**Made with ❤️ for 6G wireless research**

</div>
