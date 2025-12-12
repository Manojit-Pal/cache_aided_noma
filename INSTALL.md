# Installation Guide

Comprehensive installation instructions for the Cache-Aided NOMA project.

---

## 📋 **Prerequisites**

### **System Requirements**
- **Python:** 3.8, 3.9, 3.10, or 3.11
- **RAM:** Minimum 8 GB (16 GB recommended for large-scale experiments)
- **Disk Space:** ~2 GB (for dependencies + experiment results)
- **GPU (Optional):** NVIDIA GPU with CUDA 11.8 or 12.1 for faster training

### **Check Python Version**
```bash
python --version
# or
python3 --version
```

Expected output: `Python 3.8.x`, `Python 3.9.x`, `Python 3.10.x`, or `Python 3.11.x`

---

## 🚀 **Quick Installation (CPU Only)**

For most users who don't have an NVIDIA GPU or just want to get started quickly:

```bash
# 1. Clone the repository
git clone https://github.com/Manojit-Pal/cache_aided_noma.git
cd cache_aided_noma

# 2. Create virtual environment (recommended)
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify installation
python -c "import numpy, pandas, scipy, sklearn, matplotlib, torch; print('✅ All dependencies OK')"
```

**Expected output:**
```
✅ All dependencies OK
```

---

## 🎮 **GPU Installation (NVIDIA CUDA)**

For users with NVIDIA GPUs who want faster DQN training:

### **Step 1: Check CUDA Version**

```bash
nvcc --version
# or
nvidia-smi
```

Note your CUDA version (e.g., 11.8, 12.1)

### **Step 2: Install with GPU Support**

#### **For CUDA 11.8:**
```bash
# Install base dependencies
pip install -r requirements.txt

# Reinstall PyTorch with CUDA 11.8 support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### **For CUDA 12.1:**
```bash
# Install base dependencies
pip install -r requirements.txt

# Reinstall PyTorch with CUDA 12.1 support
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### **Step 3: Verify GPU Setup**

```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
```

**Expected output (GPU):**
```
CUDA Available: True
CUDA Version: 11.8
```

**Expected output (CPU fallback):**
```
CUDA Available: False
CUDA Version: N/A
```

---

## 🛠️ **Development Installation**

For contributors or developers who want testing and linting tools:

```bash
# Install all dependencies including dev tools
pip install -r requirements.txt
pip install pytest black flake8 ipython jupyter tensorboard

# Verify dev environment
python -m pytest --version
black --version
flake8 --version
```

---

## 🧪 **Verify Installation**

### **Quick Test**
```bash
# Run utility module test
python -m src.utils
```

**Expected output:**
```
======================================================================
UTILS MODULE - FUNCTIONALITY TEST
======================================================================

1️⃣  Testing Zipf Distribution:
   Generated 1000 requests
   ...

✅ ALL TESTS PASSED
```

### **Comprehensive Tests**
```bash
# Test NOMA integration
python test_noma_integration.py

# Test DQN cache
python test_dqn_cache.py
```

### **Quick Experiment**
```bash
# Run quick comparison (10-15 minutes)
python run_comparison.py --quick
```

---

## 📦 **Package Versions**

### **Tested Configurations**

| Python | NumPy  | PyTorch | CUDA   | Status |
|--------|--------|---------|--------|--------|
| 3.10   | 1.24   | 2.1     | 11.8   | ✅     |
| 3.11   | 1.26   | 2.2     | CPU    | ✅     |
| 3.9    | 1.23   | 2.0     | 12.1   | ✅     |
| 3.8    | 1.23   | 2.0     | CPU    | ✅     |

### **Core Dependencies**

```
numpy >= 1.23.0, < 2.0.0
pandas >= 1.4.0, < 3.0.0
scipy >= 1.8.0, < 2.0.0
torch >= 2.0.0, < 2.5.0
scikit-learn >= 1.0.0, < 2.0.0
matplotlib >= 3.5.0, < 4.0.0
seaborn >= 0.12.0, < 1.0.0
tqdm >= 4.64.0
```

---

## 🐛 **Troubleshooting**

### **Issue 1: ImportError: No module named 'torch'**

**Solution:**
```bash
pip install torch
```

### **Issue 2: CUDA out of memory**

**Solution 1:** Reduce batch size in `src/config.py`:
```python
RL_BATCH_SIZE = 32  # Default is 64
```

**Solution 2:** Use CPU mode:
```python
RL_USE_NEURAL_NETWORK = True  # Keep neural network
# PyTorch will automatically fall back to CPU if CUDA unavailable
```

### **Issue 3: Slow training on CPU**

**Expected:** CPU training is 5-10x slower than GPU. For quick tests:
```bash
python run_comparison.py --quick  # Uses reduced training episodes
```

**Recommendation:** For full experiments, use a GPU or reduce:
```python
RL_TRAINING_EPISODES = 500  # Default is 2000
```

### **Issue 4: NumPy version conflict**

**Solution:**
```bash
pip install "numpy>=1.23.0,<2.0.0" --force-reinstall
```

### **Issue 5: Matplotlib backend error**

**Solution (Linux/macOS):**
```bash
export MPLBACKEND=Agg
python run_comparison.py
```

**Solution (Windows):**
```cmd
set MPLBACKEND=Agg
python run_comparison.py
```

### **Issue 6: Permission denied (Windows)**

**Solution:** Run command prompt as Administrator or use:
```bash
pip install --user -r requirements.txt
```

---

## 🖥️ **Platform-Specific Notes**

### **Windows**

1. **Virtual Environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Long Path Support:** Enable if you encounter path errors:
   - Run `regedit`
   - Navigate to `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
   - Set `LongPathsEnabled` to `1`

3. **CUDA Installation:**
   - Download CUDA Toolkit from [NVIDIA](https://developer.nvidia.com/cuda-downloads)
   - Install Visual Studio Build Tools if needed

### **Linux (Ubuntu/Debian)**

1. **Install Python development headers:**
   ```bash
   sudo apt-get update
   sudo apt-get install python3-dev python3-pip python3-venv
   ```

2. **Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **CUDA Installation:**
   ```bash
   # Check CUDA availability
   nvidia-smi
   
   # Install CUDA toolkit (if not already installed)
   # Follow: https://developer.nvidia.com/cuda-downloads
   ```

### **macOS**

1. **Homebrew (if not installed):**
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python:**
   ```bash
   brew install python@3.10
   ```

3. **Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Note:** macOS doesn't support CUDA (Apple Silicon or Intel). Use CPU mode.

---

## ✅ **Post-Installation Checklist**

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`pip list`)
- [ ] Import test passed (`import torch, numpy, ...`)
- [ ] GPU detected (if applicable): `torch.cuda.is_available()`
- [ ] Utils test passed: `python -m src.utils`
- [ ] NOMA test passed: `python test_noma_integration.py`
- [ ] DQN test passed: `python test_dqn_cache.py`
- [ ] Quick experiment runs: `python run_comparison.py --quick`

---

## 🎯 **Next Steps**

Once installation is complete:

1. **Run Quick Test:**
   ```bash
   python run_comparison.py --quick
   ```

2. **View Results:**
   ```bash
   # Results saved in results/ directory
   ls results/
   ```

3. **Read Documentation:**
   - `README.md` - Project overview
   - `src/config.py` - Configuration options
   - `test_dqn_cache.py` - DQN testing

4. **Run Full Experiment:**
   ```bash
   python run_comparison.py
   ```

---

## 📞 **Support**

If you encounter issues not covered here:

1. Check existing [GitHub Issues](https://github.com/Manojit-Pal/cache_aided_noma/issues)
2. Create a new issue with:
   - Python version (`python --version`)
   - OS and version
   - Error message (full traceback)
   - Installation method used

---

## 📄 **License**

See `LICENSE` file in the repository root.

---

**Happy experimenting! 🚀**
