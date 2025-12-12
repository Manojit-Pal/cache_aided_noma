#!/usr/bin/env python3
# run_comparison.py
"""
Cache-Aided NOMA Comparison Runner

Convenient command-line interface for running comprehensive comparative analysis
of cache-aided NOMA systems. This script orchestrates the entire experimental
pipeline with flexible configuration options.

Features:
- Automatic DQN training if checkpoint not found
- Quick test mode for debugging (--quick)
- Custom SNR range selection
- Optional DQN exclusion
- Multiple configuration presets
- Color-coded terminal output

Usage:
    # Basic usage (full experiment)
    python run_comparison.py
    
    # Quick test (10-15 min)
    python run_comparison.py --quick
    
    # Custom SNR range
    python run_comparison.py --snr-min 0 --snr-max 30
    
    # Skip DQN training
    python run_comparison.py --no-dqn
    
    # Custom output directory
    python run_comparison.py --output-dir my_results
    
    # Use conservative learning config
    python run_comparison.py --config conservative

Author: Cache-Aided NOMA Team
Date: December 12, 2025
Version: 2.0 (Enhanced CLI)
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add src to path for imports
SRC_DIR = Path(__file__).parent / 'src'
sys.path.insert(0, str(SRC_DIR))

try:
    from src import config as cfg
    from src.experiments.comparative_analysis import main, CacheAidedNOMAAnalysis
except ImportError as e:
    print(f"\n❌ Error: Could not import required modules.")
    print(f"   Details: {e}")
    print(f"\n   Make sure you are running from the project root directory.")
    print(f"   Directory structure should be:")
    print(f"     project_root/")
    print(f"       ├── run_comparison.py  ← Run from here")
    print(f"       ├── src/")
    print(f"       │   ├── config.py")
    print(f"       │   └── experiments/")
    print(f"       │       └── comparative_analysis.py")
    print(f"       └── ...\n")
    sys.exit(1)


# ============================================================================
# COLOR CODES FOR TERMINAL OUTPUT
# ============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text):
    """Print colored header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.ENDC}\n")


def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}✅ {text}{Colors.ENDC}")


def print_warning(text):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.ENDC}")


def print_error(text):
    """Print error message."""
    print(f"{Colors.RED}❌ {text}{Colors.ENDC}")


def print_info(text):
    """Print info message."""
    print(f"{Colors.CYAN}📄 {text}{Colors.ENDC}")


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

def setup_quick_test_config():
    """Configure for quick testing (10-15 minutes)."""
    print_info("Setting up QUICK TEST configuration...")
    cfg.set_quick_test_config()
    print_success("Quick test mode enabled")
    print(f"   Training: {cfg.RL_TRAINING_EPISODES} episodes")
    print(f"   Evaluation: {cfg.NUM_RUNS} runs")
    print(f"   Expected time: ~10-15 minutes\n")


def setup_full_experiment_config():
    """Configure for full experiment (2-3 hours)."""
    print_info("Setting up FULL EXPERIMENT configuration...")
    cfg.set_full_experiment_config()
    print_success("Full experiment mode enabled")
    print(f"   Training: {cfg.RL_TRAINING_EPISODES} episodes")
    print(f"   Evaluation: {cfg.NUM_RUNS} runs")
    print(f"   Expected time: ~2-3 hours\n")


def setup_custom_config(config_preset):
    """Apply custom configuration preset."""
    if config_preset == 'aggressive':
        print_info("Applying AGGRESSIVE learning configuration...")
        cfg.set_aggressive_learning_config()
    elif config_preset == 'conservative':
        print_info("Applying CONSERVATIVE learning configuration...")
        cfg.set_conservative_learning_config()
    else:
        print_warning(f"Unknown config preset: {config_preset}")
        print_info("Using default configuration")


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Cache-Aided NOMA Comparative Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test for debugging
  python run_comparison.py --quick
  
  # Full experiment with custom SNR range
  python run_comparison.py --snr-min 0 --snr-max 30
  
  # Skip DQN to save time
  python run_comparison.py --no-dqn
  
  # Use conservative learning
  python run_comparison.py --config conservative
  
  # Custom output location
  python run_comparison.py --output-dir experiments/run_001
        """
    )
    
    # Experiment mode
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test mode (100 episodes, 10 runs, ~10-15 min)'
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help='Full experiment mode (2000 episodes, 100 runs, ~2-3 hours)'
    )
    
    # SNR configuration
    parser.add_argument(
        '--snr-min',
        type=int,
        default=-10,
        help='Minimum SNR in dB (default: -10)'
    )
    
    parser.add_argument(
        '--snr-max',
        type=int,
        default=30,
        help='Maximum SNR in dB (default: 30)'
    )
    
    parser.add_argument(
        '--snr-step',
        type=int,
        default=2,
        help='SNR step size in dB (default: 2)'
    )
    
    # Policy selection
    parser.add_argument(
        '--no-dqn',
        action='store_true',
        help='Skip DQN policy (useful for quick baseline comparison)'
    )
    
    parser.add_argument(
        '--policies',
        nargs='+',
        choices=['topk', 'lru', 'lfu', 'random', 'none', 'dqn'],
        help='Specify which policies to evaluate (default: all)'
    )
    
    # Monte Carlo configuration
    parser.add_argument(
        '--num-realizations',
        type=int,
        default=1000,
        help='Number of Monte Carlo realizations per SNR point (default: 1000)'
    )
    
    # Output configuration
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    
    # Learning configuration
    parser.add_argument(
        '--config',
        choices=['default', 'aggressive', 'conservative'],
        default='default',
        help='DQN learning configuration preset (default: default)'
    )
    
    # Reproducibility
    parser.add_argument(
        '--seed',
        type=int,
        default=2025,
        help='Random seed for reproducibility (default: 2025)'
    )
    
    # Verbosity
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main_cli():
    """Main CLI entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Print header
    print_header("CACHE-AIDED NOMA COMPARATIVE ANALYSIS")
    
    # ========================================================================
    # CONFIGURATION SETUP
    # ========================================================================
    
    # Set random seed
    cfg.RANDOM_SEED = args.seed
    print_info(f"Random seed: {cfg.RANDOM_SEED}")
    
    # Apply experiment mode
    if args.quick:
        setup_quick_test_config()
    elif args.full:
        setup_full_experiment_config()
    else:
        print_info("Using default configuration")
        print(f"   Training episodes: {cfg.RL_TRAINING_EPISODES}")
        print(f"   Evaluation runs: {cfg.NUM_RUNS}\n")
    
    # Apply custom config preset
    if args.config != 'default':
        setup_custom_config(args.config)
    
    # ========================================================================
    # PARAMETER SUMMARY
    # ========================================================================
    
    print_header("EXPERIMENT PARAMETERS")
    
    print(f"{Colors.BOLD}SNR Configuration:{Colors.ENDC}")
    print(f"   Range: {args.snr_min} to {args.snr_max} dB (step: {args.snr_step} dB)")
    print(f"   Points: {len(range(args.snr_min, args.snr_max + 1, args.snr_step))}")
    
    print(f"\n{Colors.BOLD}Monte Carlo:{Colors.ENDC}")
    print(f"   Realizations per SNR: {args.num_realizations}")
    
    # Determine policies
    if args.policies:
        policies = args.policies
    elif args.no_dqn:
        policies = ['topk', 'lru', 'lfu', 'random', 'none']
    else:
        policies = ['topk', 'lru', 'lfu', 'random', 'none', 'dqn']
    
    print(f"\n{Colors.BOLD}Policies to Evaluate:{Colors.ENDC}")
    for policy in policies:
        if policy == 'none':
            print(f"   • NO-CACHE (baseline)")
        elif policy == 'dqn':
            print(f"   • DQN (deep reinforcement learning)")
        else:
            print(f"   • {policy.upper()}")
    
    print(f"\n{Colors.BOLD}Output:{Colors.ENDC}")
    print(f"   Directory: {args.output_dir}/")
    
    print()
    
    # ========================================================================
    # EXECUTION CONFIRMATION
    # ========================================================================
    
    # Estimate execution time
    num_snr_points = len(range(args.snr_min, args.snr_max + 1, args.snr_step))
    num_policies = len(policies)
    total_sims = num_snr_points * num_policies * args.num_realizations
    
    est_time_sec = total_sims * 0.001  # Rough estimate: 1ms per simulation
    if 'dqn' in policies and not os.path.exists('models/dqn_cache/dqn_cache_final.pth'):
        est_time_sec += cfg.RL_TRAINING_EPISODES * 0.5  # Add DQN training time
    
    est_time_min = est_time_sec / 60
    
    print_warning(f"Estimated execution time: {est_time_min:.1f} minutes")
    
    if not args.quick and est_time_min > 30:
        response = input(f"\n{Colors.YELLOW}Proceed? [Y/n]: {Colors.ENDC}").strip().lower()
        if response and response not in ['y', 'yes']:
            print_warning("Execution cancelled by user.")
            print_info("Tip: Use --quick for a faster test run.")
            sys.exit(0)
    
    # ========================================================================
    # RUN ANALYSIS
    # ========================================================================
    
    print_header("RUNNING ANALYSIS")
    
    start_time = time.time()
    
    try:
        # Import analysis module
        import numpy as np
        
        # Build SNR range
        snr_range = np.arange(args.snr_min, args.snr_max + 1, args.snr_step)
        
        # If using main() from comparative_analysis.py
        if args.policies is None and not args.no_dqn and \
           args.snr_min == -10 and args.snr_max == 30 and args.snr_step == 2 and \
           args.num_realizations == 1000 and args.output_dir == 'results':
            # Use default main() function
            print_info("Running default comparative analysis...\n")
            main()
        else:
            # Custom configuration - run manually
            print_info("Running custom comparative analysis...\n")
            
            # Check for DQN
            trained_dqn_cache = None
            if 'dqn' in policies:
                from src.experiments.comparative_analysis import check_dqn_checkpoint, load_trained_dqn, train_dqn_automatically
                from src.caching import DQNCache
                
                checkpoint_path = check_dqn_checkpoint()
                if checkpoint_path:
                    print_success(f"Found DQN checkpoint: {checkpoint_path}")
                    trained_dqn_cache = load_trained_dqn(checkpoint_path, cfg)
                else:
                    print_warning("No DQN checkpoint found. Training...")
                    checkpoint_path = train_dqn_automatically(cfg, num_episodes=cfg.RL_TRAINING_EPISODES)
                    if checkpoint_path:
                        trained_dqn_cache = load_trained_dqn(checkpoint_path, cfg)
            
            # Create analyzer
            analyzer = CacheAidedNOMAAnalysis(
                cfg,
                snr_range_db=snr_range,
                num_realizations=args.num_realizations,
                trained_dqn_cache=trained_dqn_cache
            )
            
            # Run comparison
            df = analyzer.run_full_comparison(policies=policies)
            
            # Save results
            os.makedirs(args.output_dir, exist_ok=True)
            analyzer.save_results(df, save_dir=args.output_dir)
            
            # Generate plots
            plot_path = os.path.join(args.output_dir, 'cache_aided_vs_traditional_noma.png')
            analyzer.plot_main_comparison(df, save_path=plot_path)
        
        elapsed_time = time.time() - start_time
        
        # ====================================================================
        # SUCCESS SUMMARY
        # ====================================================================
        
        print_header("ANALYSIS COMPLETE")
        
        print_success(f"Execution time: {elapsed_time/60:.1f} minutes")
        
        print(f"\n{Colors.BOLD}Generated Files:{Colors.ENDC}")
        print(f"   📊 {args.output_dir}/cache_aided_vs_traditional_noma.png")
        print(f"   📄 {args.output_dir}/comparative_analysis_results.csv")
        print(f"   📝 {args.output_dir}/performance_summary.txt")
        
        if 'dqn' in policies:
            print(f"   🧠 models/dqn_cache/dqn_cache_final.pth")
        
        print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
        print(f"   1. View plots: open {args.output_dir}/cache_aided_vs_traditional_noma.png")
        print(f"   2. Analyze data: {args.output_dir}/comparative_analysis_results.csv")
        print(f"   3. Read summary: {args.output_dir}/performance_summary.txt")
        
        print(f"\n{Colors.GREEN}{Colors.BOLD}✅ SUCCESS{Colors.ENDC}\n")
        
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}⚠️  Analysis interrupted by user{Colors.ENDC}\n")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n{Colors.RED}❌ ERROR during analysis:{Colors.ENDC}")
        print(f"   {e}\n")
        
        if args.verbose:
            import traceback
            print(f"{Colors.RED}Traceback:{Colors.ENDC}")
            traceback.print_exc()
        else:
            print(f"   {Colors.CYAN}Tip: Run with --verbose for full error details{Colors.ENDC}")
        
        sys.exit(1)


if __name__ == "__main__":
    main_cli()
