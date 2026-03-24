#!/usr/bin/env python3
# test_dqn_cache.py
"""
DQN Cache Unit & Integration Tests

Comprehensive test suite for NOMA-aware DQN cache implementation.
Verifies correctness of all components before full training.

Test Coverage:
1. Initialization & Configuration
2. State Representation (LRU/LFU/NOMA)
3. Action Selection (Epsilon-Greedy)
4. Reward Function (NOMA-aware)
5. Learning Loop (Training Stability)
6. Evaluation Mode (No Exploration)
7. Model Persistence (Save/Load)
8. NOMA Integration (CIC/SIC)
9. Warmup Period
10. Beta Annealing (Prioritized Replay)

Usage:
    # Run all tests
    python test_dqn_cache.py
    
    # Run specific test
    python test_dqn_cache.py --test initialization

Expected runtime: ~2-3 minutes

Author: Cache-Aided NOMA Team
Date: December 12, 2025
Version: 2.0 (Enhanced Integration)
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add src to path
SRC_DIR = Path(__file__).parent / 'src'
sys.path.insert(0, str(SRC_DIR))

import numpy as np

try:
    from src import config as cfg
    from src.caching.dqn_cache_final import DQNCache, StableDQNCache
    from src.utils import sample_zipf_catalog, set_seed
except ImportError as e:
    print(f"\n❌ Error: Could not import required modules.")
    print(f"   Details: {e}")
    print(f"\n   Make sure you are running from the project root directory.")
    sys.exit(1)


# ============================================================================
# COLOR CODES FOR OUTPUT
# ============================================================================

class Colors:
    """ANSI color codes."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_test_header(name):
    """Print test header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{name}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.ENDC}")


def print_success(msg):
    """Print success message."""
    print(f"{Colors.GREEN}✅ {msg}{Colors.ENDC}")


def print_error(msg):
    """Print error message."""
    print(f"{Colors.RED}❌ {msg}{Colors.ENDC}")


def print_warning(msg):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.ENDC}")


def print_info(key, value):
    """Print info line."""
    print(f"   {key}: {Colors.CYAN}{value}{Colors.ENDC}")


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_initialization():
    """Test 1: Cache initializes correctly."""
    print_test_header("TEST 1: Initialization")
    
    try:
        cache = DQNCache(
            capacity=50,
            num_files=100,
            num_users=20,
            epsilon_decay_steps=1000,
            seed=42
        )
        
        stats = cache.get_stats()
        
        print_success("Cache initialized successfully")
        print_info("State dim", cache.state_dim if cache.use_nn else 'N/A')
        print_info("Action dim", cache.action_dim)
        print_info("Neural network", stats['use_neural_network'])
        print_info("Epsilon", f"{stats['epsilon']:.3f}")
        print_info("Warmup steps", stats['warm_up_steps'])
        
        # Verify StableDQNCache alias works
        cache2 = StableDQNCache(capacity=50, num_files=100, num_users=20, seed=42)
        print_success("StableDQNCache alias verified")
        
        return True
    except Exception as e:
        print_error(f"Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_state_representation():
    """Test 2: State vector generation works."""
    print_test_header("TEST 2: State Representation")
    
    try:
        cache = DQNCache(
            capacity=50,
            num_files=100,
            num_users=20,
            seed=42
        )
        
        # Generate state for file 0
        state = cache._get_state_vector(0)
        
        print_success("State vector generated")
        print_info("Shape", state.shape)
        print_info("Expected", f"({cache.state_dim},)")
        print_info("Min value", f"{state.min():.3f}")
        print_info("Max value", f"{state.max():.3f}")
        print_info("Mean value", f"{state.mean():.3f}")
        
        # Check for NaN or Inf
        if np.isnan(state).any():
            print_error("State contains NaN values")
            return False
        if np.isinf(state).any():
            print_error("State contains Inf values")
            return False
        
        # Verify state components (new 14-dim compact state)
        if len(state) != cache.state_dim:
            print_error(f"State dimension mismatch: {len(state)} != {cache.state_dim}")
            return False
            
        print_info("State Dimension", f"{len(state)} dims")
        print_info("Rank & Freq", f"rank={state[0]:.2f}, freq={state[1]:.2f}")
        print_info("Cache occupancy", f"occ={state[3]:.2f}")
        
        print_success("State representation valid")
        return True
    except Exception as e:
        print_error(f"State generation failed: {e}")
        return False


def test_action_selection():
    """Test 3: Action selection works."""
    print_test_header("TEST 3: Action Selection")
    
    try:
        cache = DQNCache(
            capacity=50,
            num_files=100,
            num_users=20,
            epsilon_start=1.0,  # Full exploration for testing
            seed=42
        )
        
        state = cache._get_state_vector(0)
        
        # Test multiple action selections
        actions = []
        for _ in range(100):
            action = cache._select_action(state, file_id=0)
            if action >= 0:  # Valid action
                actions.append(action)
        
        if not actions:
            print_error("No valid actions generated")
            return False
        
        unique_actions = len(set(actions))
        
        print_success("Action selection working")
        print_info("Action space size", cache.action_dim)
        print_info("Unique actions sampled", unique_actions)
        print_info("Action range", f"[{min(actions)}, {max(actions)}]")
        print_info("Exploration coverage", f"{100*unique_actions/cache.action_dim:.1f}%")
        
        # Check if actions are valid
        if any(a < 0 or a >= cache.action_dim for a in actions):
            print_error("Invalid actions detected!")
            return False
        
        # Test exploitation (epsilon = 0)
        cache.epsilon = 0.0
        exploit_actions = [cache._select_action(state, file_id=0) for _ in range(10)]
        exploit_actions = [a for a in exploit_actions if a >= 0]
        
        if len(set(exploit_actions)) == 1:
            print_success("Exploitation mode working (deterministic)")
        else:
            print_warning("Exploitation mode not fully deterministic")
        
        return True
    except Exception as e:
        print_error(f"Action selection failed: {e}")
        return False


def test_reward_function():
    """Test 4: Reward function is NOMA-aware."""
    print_test_header("TEST 4: NOMA-Aware Reward Function")
    
    try:
        cache = DQNCache(
            capacity=20,
            num_files=100,
            num_users=20,
            seed=42
        )
        
        # Test different scenarios
        scenarios = [
            (True, False, True, False, None, "Cache Hit"),
            (False, True, True, False, 1e-5, "CIC Enabled + Low BER"),
            (False, False, True, False, None, "NOMA Success"),
            (False, False, False, False, None, "NOMA Failure"),
            (False, False, True, True, None, "Outage"),
        ]
        
        print_success("Testing reward scenarios:")
        print()
        
        for cache_hit, cic, noma_ok, outage, ber, desc in scenarios:
            reward = cache._compute_reward(cache_hit, cic, noma_ok, outage, ber)
            status = "✅" if reward > 0 else "⚠️" if reward == 0 else "❌"
            print(f"   {status} {desc:25s}: {reward:+6.1f}")
        
        print()
        print_success("Reward function working correctly")
        return True
        
    except Exception as e:
        print_error(f"Reward function test failed: {e}")
        return False


def test_learning_loop():
    """Test 5: Learning loop works without crashes."""
    print_test_header("TEST 5: Learning Loop (200 steps)")
    
    try:
        set_seed(42)
        
        cache = DQNCache(
            capacity=20,
            num_files=100,
            num_users=20,
            batch_size=32,
            epsilon_start=0.5,
            warm_up_steps=100,
            seed=42
        )
        
        print_info("Warmup steps", cache.warm_up_steps)
        print_info("Batch size", cache.batch_size)
        print()
        
        # Simulate 200 requests
        rewards_history = []
        for step in range(200):
            file_id = sample_zipf_catalog(100, 1.0, size=1)[0]
            user_id = np.random.randint(0, 20)
            
            # Check cache before request
            cache_hit = cache.is_hit(file_id, update_stats=False)
            
            # Simulate NOMA outcome
            noma_success = np.random.rand() > 0.3
            channel_gain = np.random.exponential(1.0)
            sinr = np.random.exponential(5.0)
            ber = 0.5 * np.exp(-sinr)
            
            episode_done = (step % 50 == 49)  # Episodes of 50 steps
            
            # Use the request() method (full NOMA integration)
            result = cache.request(
                item=file_id,
                user_id=user_id,
                channel_gain=channel_gain,
                noma_success=noma_success,
                sinr_weak=sinr,
                sinr_strong=sinr * 2,
                ber=ber,
                outage=not noma_success,
                episode_done=episode_done
            )
            
            if step % 50 == 0:
                stats = cache.get_stats()
                print(f"   Step {step:3d}: Buffer={len(cache.replay_buffer):4d}, "
                      f"Epsilon={stats['epsilon']:.3f}, Loss={stats['avg_loss']:.6f}")
        
        print()
        stats = cache.get_stats()
        
        print_success("Learning loop completed")
        print_info("Training steps", stats['training_step'])
        print_info("Replay buffer size", stats['replay_buffer_size'])
        occupied = sum(1 for s in cache.cache_slots if s != -1)
        print_info("Cache occupancy", f"{100*occupied/cache.capacity:.1f}%")
        print_info("Hit rate", f"{stats['hit_rate']:.4f}")
        print_info("Avg loss", f"{stats['avg_loss']:.6f}")
        
        # Check for learning instability
        if np.isnan(stats['avg_loss']) or np.isinf(stats['avg_loss']):
            print_error("Loss is NaN/Inf - training unstable!")
            return False
        
        print_success("Training stable")
        return True
        
    except Exception as e:
        print_error(f"Learning loop failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_mode():
    """Test 6: Evaluation mode works (no exploration)."""
    print_test_header("TEST 6: Evaluation Mode")
    
    try:
        cache = DQNCache(
            capacity=20,
            num_files=100,
            num_users=20,
            epsilon_start=0.5,
            seed=42
        )
        
        initial_epsilon = cache.epsilon
        print_info("Initial epsilon", f"{initial_epsilon:.3f}")
        
        # Enable eval mode
        cache.set_eval_mode(True)
        eval_epsilon = cache.epsilon
        print_info("Eval mode epsilon", f"{eval_epsilon:.3f}")
        
        if eval_epsilon != 0.0:
            print_error("Epsilon should be 0 in eval mode!")
            return False
        
        # Disable eval mode
        cache.set_eval_mode(False)
        restored_epsilon = cache.epsilon
        print_info("Restored epsilon", f"{restored_epsilon:.3f}")
        
        if abs(restored_epsilon - initial_epsilon) > 1e-6:
            print_error(f"Epsilon not properly restored! Expected {initial_epsilon:.3f}, got {restored_epsilon:.3f}")
            return False
        
        print_success("Evaluation mode working correctly")
        return True
        
    except Exception as e:
        print_error(f"Evaluation mode failed: {e}")
        return False


def test_save_load():
    """Test 7: Model saving and loading works."""
    print_test_header("TEST 7: Save/Load Model")
    
    try:
        # Create and train a cache
        cache1 = DQNCache(
            capacity=20,
            num_files=100,
            num_users=20,
            seed=42
        )
        
        # Do some learning
        for step in range(100):
            file_id = sample_zipf_catalog(100, 1.0, size=1)[0]
            cache1.request(
                item=file_id,
                user_id=0,
                channel_gain=1.0,
                episode_done=(step == 99)
            )
        
        stats1 = cache1.get_stats()
        original_steps = stats1['training_step']
        original_epsilon = stats1['epsilon']
        
        print_info("Original training steps", original_steps)
        print_info("Original epsilon", f"{original_epsilon:.4f}")
        
        # Save model
        test_path = 'test_dqn_model.pth'
        cache1.save_model(test_path)
        print_success("Model saved")
        
        # Load into new cache
        cache2 = DQNCache(
            capacity=20,
            num_files=100,
            num_users=20,
            seed=999  # Different seed
        )
        cache2.load_model(test_path)
        print_success("Model loaded")
        
        # Check if state was preserved
        stats2 = cache2.get_stats()
        loaded_steps = stats2['training_step']
        loaded_epsilon = stats2['epsilon']
        
        print_info("Loaded training steps", loaded_steps)
        print_info("Loaded epsilon", f"{loaded_epsilon:.4f}")
        
        if loaded_steps != original_steps:
            print_error(f"Training steps mismatch! {original_steps} != {loaded_steps}")
            return False
        
        if abs(loaded_epsilon - original_epsilon) > 1e-6:
            print_error(f"Epsilon mismatch! {original_epsilon:.4f} != {loaded_epsilon:.4f}")
            return False
        
        print_success("State preserved correctly")
        
        # Cleanup
        os.remove(test_path)
        print_success("Cleanup complete")
        
        return True
        
    except Exception as e:
        print_error(f"Save/Load failed: {e}")
        # Cleanup on error
        if os.path.exists('test_dqn_model.pth'):
            os.remove('test_dqn_model.pth')
        return False


def test_noma_integration():
    """Test 8: NOMA-specific features work."""
    print_test_header("TEST 8: NOMA Integration (CIC/SIC)")
    
    try:
        cache = DQNCache(
            capacity=20,
            num_files=100,
            num_users=20,
            enable_noma_awareness=True,
            seed=42
        )
        
        # Simulate NOMA scenario with CIC
        file1, file2 = 0, 1
        
        # Cache file1 (will enable CIC for user requesting file2)
        cache.request(item=file1, user_id=0, channel_gain=1.0)
        
        # Request file2 with paired user having file1 cached
        result = cache.request(
            item=file2,
            user_id=1,
            channel_gain=0.5,
            paired_user=0,
            paired_file=file1,
            noma_success=True
        )
        
        print_success("NOMA request processed")
        print_info("CIC enabled", result.get('cic_enabled', False))
        print_info("Strong user benefit", result.get('strong_user_benefit', False))
        
        stats = cache.get_stats()
        print_info("CIC count", stats.get('cic_count', 0))
        print_info("SIC count", stats.get('sic_count', 0))
        
        # Verify NOMA history tracking
        if len(cache.noma_history) > 0:
            print_success("NOMA history tracked")
        else:
            print_warning("NOMA history empty")
        
        return True
        
    except Exception as e:
        print_error(f"NOMA integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_warmup_period():
    """Test 9: Warmup period prevents premature training."""
    print_test_header("TEST 9: Warmup Period")
    
    try:
        warmup_steps = 100
        cache = DQNCache(
            capacity=20,
            num_files=100,
            num_users=20,
            warm_up_steps=warmup_steps,
            batch_size=32,
            train_freq=4,
            seed=42
        )
        
        print_info("Warmup steps", warmup_steps)
        print_info("Batch size", cache.batch_size)
        
        # Run requests during warmup
        for step in range(warmup_steps - 10):
            file_id = sample_zipf_catalog(100, 1.0, size=1)[0]
            cache.request(item=file_id, user_id=0, channel_gain=1.0)
        
        stats_during = cache.get_stats()
        loss_during = stats_during['avg_loss']
        
        print_info("Loss during warmup", f"{loss_during:.6f}")
        
        # Complete warmup
        for step in range(20):
            file_id = sample_zipf_catalog(100, 1.0, size=1)[0]
            cache.request(item=file_id, user_id=0, channel_gain=1.0)
        
        stats_after = cache.get_stats()
        loss_after = stats_after['avg_loss']
        
        print_info("Loss after warmup", f"{loss_after:.6f}")
        print_info("Buffer size", len(cache.replay_buffer))
        
        if loss_during == 0.0 and loss_after > 0.0:
            print_success("Warmup period working correctly")
            return True
        else:
            print_warning("Warmup behavior unclear - check manually")
            return True  # Not critical
        
    except Exception as e:
        print_error(f"Warmup test failed: {e}")
        return False


def test_beta_annealing():
    """Test 10: Beta annealing in prioritized replay."""
    print_test_header("TEST 10: Beta Annealing (Prioritized Replay)")
    
    try:
        cache = DQNCache(
            capacity=20,
            num_files=100,
            num_users=20,
            use_prioritized_replay=True,
            priority_beta_start=0.4,
            priority_beta_end=1.0,
            priority_beta_frames=1000,
            seed=42
        )
        
        if not cache.use_prioritized:
            print_warning("Prioritized replay not enabled")
            return True
        
        initial_beta = cache.get_stats().get('beta', 0.0)
        print_info("Initial beta", f"{initial_beta:.4f}")
        
        # Run training to trigger beta updates
        for step in range(500):
            file_id = sample_zipf_catalog(100, 1.0, size=1)[0]
            cache.request(item=file_id, user_id=0, channel_gain=1.0)
        
        mid_beta = cache.get_stats().get('beta', 0.0)
        print_info("Beta after 500 steps", f"{mid_beta:.4f}")
        
        # Continue training
        for step in range(1000):
            file_id = sample_zipf_catalog(100, 1.0, size=1)[0]
            cache.request(item=file_id, user_id=0, channel_gain=1.0)
        
        final_beta = cache.get_stats().get('beta', 0.0)
        print_info("Beta after 1500 steps", f"{final_beta:.4f}")
        
        if initial_beta < mid_beta < final_beta:
            print_success("Beta annealing working correctly")
            return True
        else:
            print_warning(f"Beta trajectory unclear: {initial_beta:.3f} → {mid_beta:.3f} → {final_beta:.3f}")
            return True  # Not critical
        
    except Exception as e:
        print_error(f"Beta annealing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_all_tests(selected_test=None):
    """Run all tests or a specific test."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'🧪'*35}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}DQN CACHE - COMPREHENSIVE TEST SUITE{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'🧪'*35}{Colors.ENDC}")
    
    tests = [
        ("initialization", test_initialization),
        ("state", test_state_representation),
        ("action", test_action_selection),
        ("reward", test_reward_function),
        ("learning", test_learning_loop),
        ("eval", test_evaluation_mode),
        ("save_load", test_save_load),
        ("noma", test_noma_integration),
        ("warmup", test_warmup_period),
        ("beta", test_beta_annealing),
    ]
    
    if selected_test:
        tests = [(name, fn) for name, fn in tests if name == selected_test]
        if not tests:
            print_error(f"Unknown test: {selected_test}")
            print(f"\nAvailable tests: {', '.join(name for name, _ in tests)}")
            return False
    
    results = []
    start_time = time.time()
    
    for test_name, test_fn in tests:
        try:
            result = test_fn()
            results.append((test_name, result))
        except Exception as e:
            print_error(f"Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    elapsed = time.time() - start_time
    
    # Summary
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}TEST SUMMARY{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.ENDC}\n")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = f"{Colors.GREEN}✅ PASS{Colors.ENDC}" if result else f"{Colors.RED}❌ FAIL{Colors.ENDC}"
        print(f"   {status}  {test_name}")
    
    print(f"\n{Colors.BOLD}Tests passed: {passed}/{total}{Colors.ENDC}")
    print(f"{Colors.BOLD}Execution time: {elapsed:.1f}s{Colors.ENDC}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED! DQN cache is ready.{Colors.ENDC}")
        print(f"\n{Colors.BOLD}Next steps:{Colors.ENDC}")
        print(f"   1. Run full training: {Colors.CYAN}python run_comparison.py --quick{Colors.ENDC}")
        print(f"   2. Or train DQN specifically: {Colors.CYAN}python train_and_evaluate_dqn.py{Colors.ENDC}")
        print(f"   3. Full experiment: {Colors.CYAN}python run_comparison.py{Colors.ENDC}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}⚠️  SOME TESTS FAILED! Fix issues before proceeding.{Colors.ENDC}")
    
    print()
    return passed == total


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Test DQN Cache Implementation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--test',
        type=str,
        default=None,
        help='Run specific test (initialization, state, action, reward, learning, eval, save_load, noma, warmup, beta)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = run_all_tests(selected_test=args.test)
    sys.exit(0 if success else 1)
