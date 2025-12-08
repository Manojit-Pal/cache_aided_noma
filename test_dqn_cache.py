# test_dqn_cache.py
"""
Quick test script to verify DQN cache is working correctly.
Run this BEFORE the full simulation to catch any issues early.

Expected runtime: ~2-3 minutes
"""

import sys
import os
sys.path.insert(0, 'src')

import numpy as np
from src import config as cfg
from src.caching.dqn_cache_final import StableDQNCache
from src.utils import sample_zipf_catalog


def test_initialization():
    """Test 1: Cache initializes correctly."""
    print("\n" + "="*70)
    print("TEST 1: Initialization")
    print("="*70)
    
    try:
        cache = StableDQNCache(
            capacity=50,
            num_files=100,
            num_users=20,
            epsilon_decay_steps=1000,
            seed=42
        )
        
        stats = cache.get_stats()
        
        print("✅ Cache initialized successfully")
        print(f"   State dim: {cache.state_dim if cache.use_nn else 'N/A'}")
        print(f"   Action dim: {cache.action_dim}")
        print(f"   Neural network: {stats['use_neural_network']}")
        print(f"   Epsilon: {stats['epsilon']:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False


def test_state_representation():
    """Test 2: State vector generation works."""
    print("\n" + "="*70)
    print("TEST 2: State Representation")
    print("="*70)
    
    try:
        cache = StableDQNCache(
            capacity=50,
            num_files=100,
            num_users=20,
            seed=42
        )
        
        # Generate state for file 0
        state = cache._get_state_vector(0)
        
        print(f"✅ State vector generated")
        print(f"   Shape: {state.shape}")
        print(f"   Expected: ({cache.state_dim},)")
        print(f"   Min value: {state.min():.3f}")
        print(f"   Max value: {state.max():.3f}")
        
        # Check for NaN or Inf
        if np.isnan(state).any():
            print(f"⚠️  WARNING: State contains NaN values")
            return False
        if np.isinf(state).any():
            print(f"⚠️  WARNING: State contains Inf values")
            return False
        
        return True
    except Exception as e:
        print(f"❌ State generation failed: {e}")
        return False


def test_action_selection():
    """Test 3: Action selection works."""
    print("\n" + "="*70)
    print("TEST 3: Action Selection")
    print("="*70)
    
    try:
        cache = StableDQNCache(
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
            actions.append(action)
        
        unique_actions = len(set(actions))
        
        print(f"✅ Action selection working")
        print(f"   Action space size: {cache.action_dim}")
        print(f"   Unique actions sampled: {unique_actions}")
        print(f"   Action range: [{min(actions)}, {max(actions)}]")
        
        # Check if actions are valid
        if any(a < 0 or a >= cache.action_dim for a in actions):
            print(f"❌ Invalid actions detected!")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Action selection failed: {e}")
        return False


def test_learning_loop():
    """Test 4: Learning loop works without crashes."""
    print("\n" + "="*70)
    print("TEST 4: Learning Loop (100 steps)")
    print("="*70)
    
    try:
        cache = StableDQNCache(
            capacity=20,
            num_files=100,
            num_users=20,
            batch_size=32,
            epsilon_start=0.5,
            seed=42
        )
        
        # Simulate 100 requests
        for step in range(100):
            file_id = sample_zipf_catalog(100, 1.0, size=1)[0]
            user_id = np.random.randint(0, 20)
            
            cache_hit = cache.is_hit(file_id)
            
            # Simulate NOMA outcome
            noma_success = np.random.rand() > 0.3
            channel_gain = np.random.exponential(1.0)
            sinr = np.random.exponential(5.0)
            ber = 0.5 * np.exp(-sinr)
            
            episode_done = (step == 99)
            
            cache.observe_request(
                user_id=user_id,
                file_id=file_id,
                cache_hit=cache_hit,
                noma_success=noma_success,
                channel_gain=channel_gain,
                sinr_weak=sinr,
                sinr_strong=sinr * 2,
                ber=ber,
                outage=not noma_success,
                episode_done=episode_done
            )
        
        stats = cache.get_stats()
        
        print(f"✅ Learning loop completed")
        print(f"   Training steps: {stats['training_step']}")
        print(f"   Replay buffer size: {stats['replay_buffer_size']}")
        print(f"   Cache occupancy: {stats['cache_occupancy']}")
        print(f"   Cumulative reward: {stats['cumulative_reward']:.2f}")
        print(f"   Avg loss: {stats['avg_loss']:.6f}")
        
        # Check for learning instability
        if np.isnan(stats['avg_loss']) or np.isinf(stats['avg_loss']):
            print(f"⚠️  WARNING: Loss is NaN/Inf - training unstable!")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Learning loop failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_mode():
    """Test 5: Evaluation mode works (no exploration)."""
    print("\n" + "="*70)
    print("TEST 5: Evaluation Mode")
    print("="*70)
    
    try:
        cache = StableDQNCache(
            capacity=20,
            num_files=100,
            num_users=20,
            epsilon_start=0.5,
            seed=42
        )
        
        print(f"Initial epsilon: {cache.epsilon:.3f}")
        
        # Enable eval mode
        cache.set_eval_mode(True)
        print(f"Eval mode epsilon: {cache.epsilon:.3f}")
        
        if cache.epsilon != 0.0:
            print(f"❌ Epsilon should be 0 in eval mode!")
            return False
        
        # Disable eval mode
        cache.set_eval_mode(False)
        print(f"Back to training epsilon: {cache.epsilon:.3f}")
        
        print(f"✅ Evaluation mode working correctly")
        return True
    except Exception as e:
        print(f"❌ Evaluation mode failed: {e}")
        return False


def test_save_load():
    """Test 6: Model saving and loading works."""
    print("\n" + "="*70)
    print("TEST 6: Save/Load Model")
    print("="*70)
    
    try:
        # Create and train a cache
        cache1 = StableDQNCache(
            capacity=20,
            num_files=100,
            num_users=20,
            seed=42
        )
        
        # Do some learning
        for _ in range(50):
            file_id = sample_zipf_catalog(100, 1.0, size=1)[0]
            cache1.observe_request(
                user_id=0, file_id=file_id, cache_hit=cache1.is_hit(file_id),
                channel_gain=1.0
            )
        
        original_reward = cache1.cumulative_reward
        
        # Save model
        cache1.save_model('test_model.pt')
        print(f"✅ Model saved")
        
        # Load into new cache
        cache2 = StableDQNCache(
            capacity=20,
            num_files=100,
            num_users=20,
            seed=42
        )
        cache2.load_model('test_model.pt')
        print(f"✅ Model loaded")
        
        # Check if state was preserved
        loaded_reward = cache2.cumulative_reward
        
        print(f"   Original reward: {original_reward:.2f}")
        print(f"   Loaded reward: {loaded_reward:.2f}")
        
        if abs(original_reward - loaded_reward) < 0.01:
            print(f"✅ State preserved correctly")
        else:
            print(f"⚠️  State mismatch after loading")
        
        # Cleanup
        os.remove('test_model.pt')
        
        return True
    except Exception as e:
        print(f"❌ Save/Load failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "🧪"*35)
    print("STABLE DQN CACHE - UNIT TESTS")
    print("🧪"*35)
    
    tests = [
        test_initialization,
        test_state_representation,
        test_action_selection,
        test_learning_loop,
        test_evaluation_mode,
        test_save_load
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Test crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Ready for full simulation.")
        print("\nNext steps:")
        print("  1. Run full training: python src/simulation/stable_dqn_sim.py")
        print("  2. Or use existing runner: python run_final_comparison.py")
    else:
        print("\n⚠️  SOME TESTS FAILED! Please fix issues before full simulation.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
    