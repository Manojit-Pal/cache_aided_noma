#!/usr/bin/env python
# generate_latex_tables.py
"""
Generate LaTeX-formatted CSI tables for inclusion in reports/papers.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src import config as cfg

def generate_latex_system_parameters():
    """Generate LaTeX table for system parameters."""
    
    latex = r"""\begin{table}[h]
\centering
\caption{System Parameters and Simulation Configuration}
\label{tab:system_params}
\begin{tabular}{|l|c|l|}
\hline
\textbf{Parameter} & \textbf{Value} & \textbf{Description} \\
\hline
Number of Users ($N_u$) & """ + str(cfg.NUM_USERS) + r""" & Total users in cell \\
Number of Files ($N_f$) & """ + str(cfg.NUM_FILES) + r""" & Content library size \\
Cache Size ($C$) & """ + str(cfg.CACHE_SIZE) + r""" & Cache storage capacity \\
Cache Ratio & """ + f"{cfg.CACHE_SIZE/cfg.NUM_FILES*100:.1f}" + r"""\% & Percentage of catalog cached \\
Zipf Exponent ($\alpha$) & """ + str(cfg.ZIPF_ALPHA) + r""" & Content popularity skew \\
\hline
Transmit Power ($P_t$) & """ + str(cfg.TX_POWER) + r""" W & Base station Tx power \\
Noise Power ($N_0$) & """ + f"{cfg.NOISE_POWER:.2e}" + r""" W & AWGN power \\
Cell Radius ($R$) & """ + str(cfg.CELL_RADIUS) + r""" m & Cell coverage radius \\
Path Loss Exponent ($\gamma$) & """ + str(cfg.PATHLOSS_EXPONENT) + r""" & Urban/suburban model \\
\hline
Target Rate ($R_{target}$) & """ + str(cfg.TARGET_RATE_BPS) + r""" bps/Hz & QoS requirement \\
Power Coeff. Weak ($p_w$) & """ + str(cfg.POWER_COEFF_WEAK) + r""" & Power to weak user \\
Power Coeff. Strong ($p_s$) & """ + str(cfg.POWER_COEFF_STRONG) + r""" & Power to strong user \\
SIC Imperfection ($\zeta$) & """ + str(cfg.SIC_IMPERFECTION) + r""" & Residual interference \\
\hline
SNR Range & -10 to 28 dB & Signal-to-noise sweep \\
Monte Carlo Runs & """ + str(cfg.NUM_RUNS) + r""" & Statistical replications \\
\hline
\end{tabular}
\end{table}
"""
    
    return latex


def generate_latex_channel_model():
    """Generate LaTeX table for channel model."""
    
    latex = r"""\begin{table}[h]
\centering
\caption{Channel Model Specifications}
\label{tab:channel_model}
\begin{tabular}{|l|l|}
\hline
\textbf{Component} & \textbf{Model/Description} \\
\hline
Large-Scale Fading & Path loss: $PL(d) = d^{-\gamma}$, $\gamma = """ + str(cfg.PATHLOSS_EXPONENT) + r"""$ \\
Small-Scale Fading & Rayleigh: $|h|^2 \sim \text{Exp}(1)$ \\
User Distribution & Uniform in circle, radius """ + str(cfg.CELL_RADIUS) + r""" m \\
Channel Gain & $g = PL(d) \times |h|^2$ \\
\hline
Weak User Gain & $g_w = \min\{g_1, g_2\}$ \\
Strong User Gain & $g_s = \max\{g_1, g_2\}$ \\
\hline
SINR (Weak User) & $\gamma_w = \frac{P_t p_w g_w}{P_t p_s g_w + N_0}$ \\
SINR (Strong User) & $\gamma_s = \frac{P_t p_s g_s}{N_0 + \zeta P_t p_w g_s}$ \\
\hline
Achievable Rate & $R = \log_2(1 + \gamma)$ bps/Hz \\
BER (BPSK) & $P_e = \frac{1}{2}\text{erfc}(\sqrt{\gamma})$ \\
\hline
\end{tabular}
\end{table}
"""
    
    return latex


def generate_latex_noma_operations():
    """Generate LaTeX table describing NOMA operations."""
    
    latex = r"""\begin{table}[h]
\centering
\caption{NOMA Transmission Operations}
\label{tab:noma_ops}
\begin{tabular}{|l|p{8cm}|}
\hline
\textbf{Operation} & \textbf{Description} \\
\hline
User Pairing & Pair weak user (low $g_w$) with strong user (high $g_s$) \\
            & Strategy: Extreme pairing (weakest $\leftrightarrow$ strongest) \\
\hline
Power Allocation & Grid search over $p_w \in [0,1]$, $p_s = 1 - p_w$ \\
                & Maximize users satisfying $R \geq R_{target}$ \\
                & Weak user gets higher power: $p_w = """ + str(cfg.POWER_COEFF_WEAK) + r"""$ \\
\hline
Weak User Decoding & Direct decoding of own signal \\
                   & Treats strong user as interference \\
                   & SINR: $\gamma_w = \frac{P_t p_w g_w}{P_t p_s g_w + N_0}$ \\
\hline
Strong User SIC & 1. Decode weak user signal first \\
                & 2. Subtract (imperfect, factor $\zeta$) \\
                & 3. Decode own signal \\
                & SINR: $\gamma_s = \frac{P_t p_s g_s}{N_0 + \zeta P_t p_w g_s}$ \\
\hline
\end{tabular}
\end{table}
"""
    
    return latex


def generate_latex_cache_operations():
    """Generate LaTeX table for cache-aided NOMA."""
    
    latex = r"""\begin{table}[h]
\centering
\caption{Cache-Aided NOMA Operations}
\label{tab:cache_ops}
\begin{tabular}{|l|p{8cm}|}
\hline
\textbf{Phase} & \textbf{Operation} \\
\hline
\textbf{Placement Phase} & \\
Content Popularity & Model as Zipf distribution: $p_i \propto i^{-\alpha}$ \\
Cache Strategy & Store top-$C$ most popular files \\
Cache Size & $C = """ + str(cfg.CACHE_SIZE) + r"""$ files (""" + f"{cfg.CACHE_SIZE/cfg.NUM_FILES*100:.1f}" + r"""\% of catalog) \\
\hline
\textbf{Delivery Phase} & \\
Request Generation & Users request files according to Zipf \\
Cache Hit & File served locally at rate $R_{cache}$ \\
            & No wireless transmission needed \\
            & Outage probability = 0, BER = 0 \\
\hline
Cache Miss & File transmitted via NOMA \\
           & Subject to channel conditions \\
           & Outage probability $> 0$, BER $> 0$ \\
\hline
\textbf{Performance Metrics} & \\
Sum-Rate & $R_{sum} = R_w + R_s$ (per-request basis) \\
Outage Prob. & $P_{out} = P(R < R_{target})$ \\
BER & $P_e = \frac{1}{2}\text{erfc}(\sqrt{\gamma})$ for BPSK \\
\hline
\end{tabular}
\end{table}
"""
    
    return latex


def generate_all_latex_tables():
    """Generate all LaTeX tables and save to file."""
    
    latex_content = r"""\documentclass{article}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{array}

\begin{document}

% ============================================
% TABLE 1: SYSTEM PARAMETERS
% ============================================
""" + generate_latex_system_parameters() + r"""

\newpage

% ============================================
% TABLE 2: CHANNEL MODEL
% ============================================
""" + generate_latex_channel_model() + r"""

\newpage

% ============================================
% TABLE 3: NOMA OPERATIONS
% ============================================
""" + generate_latex_noma_operations() + r"""

\newpage

% ============================================
% TABLE 4: CACHE-AIDED NOMA
% ============================================
""" + generate_latex_cache_operations() + r"""

\end{document}
"""
    
    # Save to file
    with open('csi_tables_latex.tex', 'w') as f:
        f.write(latex_content)
    
    print("="*70)
    print("LaTeX CSI TABLES GENERATED")
    print("="*70)
    print("\n✅ Saved: csi_tables_latex.tex")
    print("\nThis file contains 4 LaTeX tables:")
    print("   1. System Parameters")
    print("   2. Channel Model")
    print("   3. NOMA Operations")
    print("   4. Cache-Aided NOMA Operations")
    print("\nYou can:")
    print("   - Compile with: pdflatex csi_tables_latex.tex")
    print("   - Copy individual tables into your report/paper")
    print("   - Modify as needed for your presentation")
    
    # Also save individual tables
    tables = {
        'csi_table_1_system_params.tex': generate_latex_system_parameters(),
        'csi_table_2_channel_model.tex': generate_latex_channel_model(),
        'csi_table_3_noma_ops.tex': generate_latex_noma_operations(),
        'csi_table_4_cache_ops.tex': generate_latex_cache_operations()
    }
    
    for filename, content in tables.items():
        with open(filename, 'w') as f:
            f.write(content)
        print(f"   ✅ Saved: {filename}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    generate_all_latex_tables()