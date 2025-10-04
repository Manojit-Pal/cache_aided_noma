#!/usr/bin/env python
# generate_csi_table.py
"""
Generate Channel State Information (CSI) Table for presentation/report.
Shows system parameters, channel model, and typical CSI values.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src import config as cfg
from src.noma import channel_model
from src.utils import set_seed

def generate_system_parameters_table():
    """Generate table of simulation/system parameters."""
    
    data = {
        'Parameter': [
            'Number of Users',
            'Number of Files',
            'Cache Size',
            'Cache Capacity Ratio',
            'Zipf Exponent (α)',
            'Requests per User',
            'Total Requests',
            'Transmit Power',
            'Noise Power',
            'Cell Radius',
            'Path Loss Exponent',
            'Target Rate',
            'Power Allocation (Weak)',
            'Power Allocation (Strong)',
            'SIC Imperfection Factor',
            'SNR Range',
            'Monte Carlo Runs'
        ],
        'Value': [
            cfg.NUM_USERS,
            cfg.NUM_FILES,
            cfg.CACHE_SIZE,
            f'{cfg.CACHE_SIZE/cfg.NUM_FILES*100:.1f}%',
            cfg.ZIPF_ALPHA,
            cfg.REQUESTS_PER_USER,
            cfg.NUM_USERS * cfg.REQUESTS_PER_USER,
            f'{cfg.TX_POWER} W',
            f'{cfg.NOISE_POWER:.2e} W',
            f'{cfg.CELL_RADIUS} m',
            cfg.PATHLOSS_EXPONENT,
            f'{cfg.TARGET_RATE_BPS} bits/s/Hz',
            cfg.POWER_COEFF_WEAK,
            cfg.POWER_COEFF_STRONG,
            cfg.SIC_IMPERFECTION,
            '-10 to 28 dB',
            cfg.NUM_RUNS
        ],
        'Description': [
            'Total users in cell',
            'Content library size',
            'Cache storage capacity',
            'Percentage of catalog cached',
            'Content popularity skew',
            'Requests generated per user',
            'Total simulation requests',
            'Base station transmit power',
            'Additive white Gaussian noise',
            'Cell coverage radius',
            'Urban/suburban environment',
            'QoS requirement per user',
            'Power fraction to weak user',
            'Power fraction to strong user',
            'Residual interference factor',
            'Signal-to-noise ratio sweep',
            'Statistical replications'
        ]
    }
    
    df = pd.DataFrame(data)
    return df


def generate_channel_statistics_table(num_samples=1000):
    """Generate typical channel gain statistics."""
    
    set_seed(cfg.RANDOM_SEED)
    
    # Generate multiple user positions
    all_distances = []
    all_pathloss = []
    all_channel_gains = []
    
    for _ in range(num_samples):
        positions = channel_model.generate_user_positions(2, cfg.CELL_RADIUS)
        distances = positions[:, 2]
        
        pl = np.array([
            channel_model.pathloss(d, cfg.PATHLOSS_EXPONENT, cfg.MIN_DISTANCE)
            for d in distances
        ])
        
        small_scale = channel_model.rayleigh_gain(2)
        gains = pl * small_scale
        
        all_distances.extend(distances)
        all_pathloss.extend(pl)
        all_channel_gains.extend(gains)
    
    all_distances = np.array(all_distances)
    all_pathloss = np.array(all_pathloss)
    all_channel_gains = np.array(all_channel_gains)
    
    # Calculate statistics
    data = {
        'Metric': [
            'Distance (m)',
            'Path Loss',
            'Channel Gain (|h|²)',
            'Channel Gain (dB)'
        ],
        'Mean': [
            f'{np.mean(all_distances):.2f}',
            f'{np.mean(all_pathloss):.2e}',
            f'{np.mean(all_channel_gains):.2e}',
            f'{10*np.log10(np.mean(all_channel_gains)):.2f}'
        ],
        'Std Dev': [
            f'{np.std(all_distances):.2f}',
            f'{np.std(all_pathloss):.2e}',
            f'{np.std(all_channel_gains):.2e}',
            f'{10*np.log10(np.std(all_channel_gains)):.2f}'
        ],
        'Min': [
            f'{np.min(all_distances):.2f}',
            f'{np.min(all_pathloss):.2e}',
            f'{np.min(all_channel_gains):.2e}',
            f'{10*np.log10(np.min(all_channel_gains)):.2f}'
        ],
        'Max': [
            f'{np.max(all_distances):.2f}',
            f'{np.max(all_pathloss):.2e}',
            f'{np.max(all_channel_gains):.2e}',
            f'{10*np.log10(np.max(all_channel_gains)):.2f}'
        ],
        'Median': [
            f'{np.median(all_distances):.2f}',
            f'{np.median(all_pathloss):.2e}',
            f'{np.median(all_channel_gains):.2e}',
            f'{10*np.log10(np.median(all_channel_gains)):.2f}'
        ]
    }
    
    df = pd.DataFrame(data)
    return df


def generate_example_csi_scenarios():
    """Generate example CSI scenarios for weak/strong user pairs."""
    
    set_seed(cfg.RANDOM_SEED)
    
    scenarios = []
    
    # Generate 5 representative scenarios
    for i in range(5):
        positions = channel_model.generate_user_positions(2, cfg.CELL_RADIUS)
        distances = positions[:, 2]
        
        pl = np.array([
            channel_model.pathloss(d, cfg.PATHLOSS_EXPONENT, cfg.MIN_DISTANCE)
            for d in distances
        ])
        
        small_scale = channel_model.rayleigh_gain(2)
        gains = pl * small_scale
        
        # Sort to get weak/strong
        idx_weak = np.argmin(gains)
        idx_strong = np.argmax(gains)
        
        scenarios.append({
            'Scenario': i + 1,
            'Weak User Distance (m)': f'{distances[idx_weak]:.1f}',
            'Strong User Distance (m)': f'{distances[idx_strong]:.1f}',
            'Weak User Gain': f'{gains[idx_weak]:.2e}',
            'Strong User Gain': f'{gains[idx_strong]:.2e}',
            'Weak User Gain (dB)': f'{10*np.log10(gains[idx_weak]):.1f}',
            'Strong User Gain (dB)': f'{10*np.log10(gains[idx_strong]):.1f}',
            'Gain Ratio (Strong/Weak)': f'{gains[idx_strong]/gains[idx_weak]:.1f}'
        })
    
    df = pd.DataFrame(scenarios)
    return df


def generate_snr_csi_table(snr_db_list=[0, 10, 20, 28]):
    """Generate CSI table for specific SNR values."""
    
    set_seed(cfg.RANDOM_SEED)
    
    data = []
    
    for snr_db in snr_db_list:
        # Generate typical channel
        positions = channel_model.generate_user_positions(2, cfg.CELL_RADIUS)
        distances = positions[:, 2]
        
        pl = np.array([
            channel_model.pathloss(d, cfg.PATHLOSS_EXPONENT, cfg.MIN_DISTANCE)
            for d in distances
        ])
        
        small_scale = channel_model.rayleigh_gain(2)
        gains = pl * small_scale
        
        gain_weak = np.min(gains)
        gain_strong = np.max(gains)
        gain_avg = np.mean(gains)
        
        # Calculate noise power for this SNR
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = cfg.TX_POWER * gain_avg / snr_linear
        
        # Calculate SINRs with typical power allocation
        p_w = cfg.POWER_COEFF_WEAK
        p_s = cfg.POWER_COEFF_STRONG
        
        # Weak user SINR
        sinr_weak = (cfg.TX_POWER * p_w * gain_weak) / \
                   (cfg.TX_POWER * p_s * gain_weak + noise_power)
        
        # Strong user SINR (after SIC)
        residual = cfg.SIC_IMPERFECTION * (cfg.TX_POWER * p_w * gain_strong)
        sinr_strong = (cfg.TX_POWER * p_s * gain_strong) / (noise_power + residual)
        
        data.append({
            'SNR (dB)': snr_db,
            'Noise Power': f'{noise_power:.2e}',
            'Weak User Gain': f'{gain_weak:.2e}',
            'Strong User Gain': f'{gain_strong:.2e}',
            'SINR Weak (dB)': f'{10*np.log10(sinr_weak):.2f}',
            'SINR Strong (dB)': f'{10*np.log10(sinr_strong):.2f}',
            'Rate Weak (bps/Hz)': f'{np.log2(1 + sinr_weak):.2f}',
            'Rate Strong (bps/Hz)': f'{np.log2(1 + sinr_strong):.2f}'
        })
    
    df = pd.DataFrame(data)
    return df


def save_all_tables():
    """Generate and save all CSI-related tables."""
    
    print("="*70)
    print("GENERATING CSI TABLES")
    print("="*70)
    
    # 1. System Parameters Table
    print("\n1. Generating System Parameters Table...")
    df_params = generate_system_parameters_table()
    df_params.to_csv('csi_table_system_parameters.csv', index=False)
    print("   ✅ Saved: csi_table_system_parameters.csv")
    print("\n" + df_params.to_string(index=False))
    
    # 2. Channel Statistics Table
    print("\n2. Generating Channel Statistics Table...")
    df_stats = generate_channel_statistics_table()
    df_stats.to_csv('csi_table_channel_statistics.csv', index=False)
    print("   ✅ Saved: csi_table_channel_statistics.csv")
    print("\n" + df_stats.to_string(index=False))
    
    # 3. Example Scenarios
    print("\n3. Generating Example CSI Scenarios...")
    df_scenarios = generate_example_csi_scenarios()
    df_scenarios.to_csv('csi_table_example_scenarios.csv', index=False)
    print("   ✅ Saved: csi_table_example_scenarios.csv")
    print("\n" + df_scenarios.to_string(index=False))
    
    # 4. SNR-specific CSI Table
    print("\n4. Generating SNR-Specific CSI Table...")
    df_snr = generate_snr_csi_table()
    df_snr.to_csv('csi_table_snr_specific.csv', index=False)
    print("   ✅ Saved: csi_table_snr_specific.csv")
    print("\n" + df_snr.to_string(index=False))
    
    print("\n" + "="*70)
    print("ALL CSI TABLES GENERATED SUCCESSFULLY!")
    print("="*70)
    
    return df_params, df_stats, df_scenarios, df_snr


def create_csi_visualization():
    """Create visual representation of CSI distribution."""
    
    set_seed(cfg.RANDOM_SEED)
    
    # Generate many samples
    num_samples = 1000
    weak_gains = []
    strong_gains = []
    
    for _ in range(num_samples):
        positions = channel_model.generate_user_positions(2, cfg.CELL_RADIUS)
        distances = positions[:, 2]
        
        pl = np.array([
            channel_model.pathloss(d, cfg.PATHLOSS_EXPONENT, cfg.MIN_DISTANCE)
            for d in distances
        ])
        
        small_scale = channel_model.rayleigh_gain(2)
        gains = pl * small_scale
        
        weak_gains.append(np.min(gains))
        strong_gains.append(np.max(gains))
    
    weak_gains_db = 10 * np.log10(np.array(weak_gains))
    strong_gains_db = 10 * np.log10(np.array(strong_gains))
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Histogram of weak user gains
    axes[0, 0].hist(weak_gains_db, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_xlabel('Channel Gain (dB)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Weak User Channel Gains')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(np.mean(weak_gains_db), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(weak_gains_db):.1f} dB')
    axes[0, 0].legend()
    
    # 2. Histogram of strong user gains
    axes[0, 1].hist(strong_gains_db, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xlabel('Channel Gain (dB)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Strong User Channel Gains')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(np.mean(strong_gains_db), color='red', linestyle='--',
                       label=f'Mean: {np.mean(strong_gains_db):.1f} dB')
    axes[0, 1].legend()
    
    # 3. Scatter plot
    axes[1, 0].scatter(weak_gains_db, strong_gains_db, alpha=0.3, s=10)
    axes[1, 0].set_xlabel('Weak User Gain (dB)')
    axes[1, 0].set_ylabel('Strong User Gain (dB)')
    axes[1, 0].set_title('Weak vs Strong User Channel Gains')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].plot([weak_gains_db.min(), weak_gains_db.max()],
                    [weak_gains_db.min(), weak_gains_db.max()],
                    'r--', label='Equal Gain Line')
    axes[1, 0].legend()
    
    # 4. CDF comparison
    weak_sorted = np.sort(weak_gains_db)
    strong_sorted = np.sort(strong_gains_db)
    cdf = np.arange(1, len(weak_sorted) + 1) / len(weak_sorted)
    
    axes[1, 1].plot(weak_sorted, cdf, label='Weak User', linewidth=2)
    axes[1, 1].plot(strong_sorted, cdf, label='Strong User', linewidth=2)
    axes[1, 1].set_xlabel('Channel Gain (dB)')
    axes[1, 1].set_ylabel('CDF')
    axes[1, 1].set_title('Cumulative Distribution of Channel Gains')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('csi_distribution_visualization.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: csi_distribution_visualization.png")
    plt.close()


def main():
    """Main execution."""
    
    # Generate all tables
    df_params, df_stats, df_scenarios, df_snr = save_all_tables()
    
    # Create visualization
    print("\n5. Creating CSI Distribution Visualization...")
    create_csi_visualization()
    
    print("\n📊 SUMMARY OF GENERATED FILES:")
    print("   1. csi_table_system_parameters.csv - System configuration")
    print("   2. csi_table_channel_statistics.csv - Statistical summary")
    print("   3. csi_table_example_scenarios.csv - Example user pairs")
    print("   4. csi_table_snr_specific.csv - SNR-dependent CSI")
    print("   5. csi_distribution_visualization.png - Visual analysis")
    
    print("\n💡 USE THESE IN YOUR PRESENTATION:")
    print("   - Table 1: System setup slide")
    print("   - Table 2: Channel model description")
    print("   - Table 3: Example scenarios for explanation")
    print("   - Table 4: Results discussion at specific SNR")
    print("   - Figure: Channel characteristics visualization")
    
    print("\n✅ ALL CSI TABLES AND FIGURES READY!")

if __name__ == "__main__":
    main()