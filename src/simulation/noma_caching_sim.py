# src/simulation/noma_caching_sim.py
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from collections import Counter

from src import config
from src.utils import set_seed, sample_zipf_catalog
from src.caching.static_cache import StaticTopKCache

# NOMA imports
from src.noma import channel_model
from src.noma.noma_base import simulate_noma_pair
from src.noma.power_allocation import allocate_power_gridsearch, allocate_power_closedform

# Dynamic caches
from src.caching.dynamic_cache import LRUCache, LFUCache, RandomCache
from src.caching.optimized_cache import JointOptimizationCache, ReinforcementLearningCache

from src.caching.noma_aware_cache import NomaAwarePredictiveCache, MultiObjectiveCache


def compute_popularity(requests):
    cnt = Counter(requests)
    sorted_items = [item for item, _ in cnt.most_common()]
    return sorted_items, cnt


def pair_users_by_method(miss_list, channel_gains, method="extreme"):
    """Pair users for NOMA according to pairing strategy."""
    if len(miss_list) == 0:
        return [], None

    indices = np.array(miss_list)

    if method == "random":
        np.random.shuffle(indices)
        pairs = []
        i = 0
        while i + 1 < len(indices):
            u1, u2 = indices[i], indices[i + 1]
            if channel_gains[u1] < channel_gains[u2]:
                pairs.append((u1, u2))
            else:
                pairs.append((u2, u1))
            i += 2
        leftover = indices[i] if i < len(indices) else None
        return pairs, leftover

    # sort by gain
    order = np.argsort(channel_gains[indices])
    sorted_indices = indices[order]

    if method == "extreme":
        pairs = []
        left, right = 0, len(sorted_indices) - 1
        while left < right:
            u_weak = sorted_indices[left]
            u_strong = sorted_indices[right]
            pairs.append((u_weak, u_strong))
            left += 1
            right -= 1
        leftover = sorted_indices[left] if left == right else None
        return pairs, leftover

    if method == "adjacent":
        pairs = []
        i = 0
        while i + 1 < len(sorted_indices):
            u_weak = sorted_indices[i]
            u_strong = sorted_indices[i + 1]
            if channel_gains[u_weak] <= channel_gains[u_strong]:
                pairs.append((u_weak, u_strong))
            else:
                pairs.append((u_strong, u_weak))
            i += 2
        leftover = sorted_indices[i] if i < len(sorted_indices) else None
        return pairs, leftover

    return [], None


def run_single_experiment(seed, cfg):
    set_seed(seed)

    # --- Channel gains ---
    user_pos = channel_model.generate_user_positions(cfg.NUM_USERS, cfg.CELL_RADIUS, seed=seed)
    distances = user_pos[:, 2]
    pl = np.array([channel_model.pathloss(d, cfg.PATHLOSS_EXPONENT, cfg.MIN_DISTANCE) for d in distances])
    small_scale = channel_model.rayleigh_gain(cfg.NUM_USERS)
    channel_gains = pl * small_scale

    # --- Requests ---
    total_requests = cfg.NUM_USERS * cfg.REQUESTS_PER_USER
    requests = sample_zipf_catalog(cfg.NUM_FILES, cfg.ZIPF_ALPHA, size=total_requests)

    ranking, counts = compute_popularity(requests)

    # --- Cache selection ---
    if cfg.CACHE_POLICY == "topk":
        cache = StaticTopKCache(cfg.CACHE_SIZE)
        cache.populate(ranking)   # pre-populated with top-K files
    elif cfg.CACHE_POLICY == "lru":
        cache = LRUCache(cfg.CACHE_SIZE)
    elif cfg.CACHE_POLICY == "lfu":
        cache = LFUCache(cfg.CACHE_SIZE)
    elif cfg.CACHE_POLICY == "random":
        cache = RandomCache(cfg.CACHE_SIZE)
    else:
        raise ValueError(f"Unknown CACHE_POLICY: {cfg.CACHE_POLICY}")

    # --- Delivery simulation ---
    hits = 0
    misses = []
    requester_ids = np.random.choice(cfg.NUM_USERS, size=total_requests, replace=True)

    for req_idx, file_id in enumerate(requests):
        user_idx = requester_ids[req_idx]

        if cfg.CACHE_POLICY == "topk":
            # static cache: just check
            if cache.is_hit(file_id):
                hits += 1
            else:
                misses.append(user_idx)

        else:
            # dynamic caches (LRU, LFU, Random) - handled inside is_hit
            if cache.is_hit(file_id):
                hits += 1
            else:
                misses.append(user_idx)

    hit_rate = hits / total_requests

    # --- Pair users according to method ---
    pairs, leftover = pair_users_by_method(misses, channel_gains, method=cfg.PAIRING_METHOD)

    pair_outcomes = []
    single_outcomes = []
    allocation_stats = {"pairs_tried": 0, "pairs_both_satisfied": 0, "pairs_partial": 0, "pairs_none": 0}

    for (u_weak, u_strong) in pairs:
        allocation_stats["pairs_tried"] += 1
        gain_w = channel_gains[u_weak]
        gain_s = channel_gains[u_strong]

        # Power allocation strategy
        if cfg.USE_CLOSED_FORM_ALLOC:
            p_w, p_s, feasible, _ = allocate_power_closedform(gain_w, gain_s, cfg)
        else:
            p_w, p_s, feasible, _ = allocate_power_gridsearch(gain_w, gain_s, cfg, grid_points=cfg.POWER_ALLOC_GRID)

        wsucc, ssucc, _, _, _ = simulate_noma_pair(gain_w, gain_s, cfg, p_w=p_w, p_s=p_s)
        pair_outcomes.append((wsucc, ssucc))

        if feasible:
            allocation_stats["pairs_both_satisfied"] += 1
        elif wsucc or ssucc:
            allocation_stats["pairs_partial"] += 1
        else:
            allocation_stats["pairs_none"] += 1

    # Single leftover user
    if leftover is not None:
        u = leftover
        gain = channel_gains[u]
        sinr_single = cfg.TX_POWER * 1.0 * gain / cfg.NOISE_POWER
        sinr_th = 2 ** (cfg.TARGET_RATE_BPS) - 1
        single_outcomes.append(sinr_single >= sinr_th)

    # Outage calculation
    pair_failed = sum((not w) + (not s) for w, s in pair_outcomes)
    single_failed = sum(1 for x in single_outcomes if not x)

    total_miss_users = len(misses)
    total_failed = pair_failed + single_failed
    outage_rate = (total_failed / total_miss_users) if total_miss_users > 0 else 0.0

    return {
        "seed": seed,
        "total_requests": total_requests,
        "hits": hits,
        "hit_rate": hit_rate,
        "num_misses": total_miss_users,
        "outage_rate": outage_rate,
        "pair_count": len(pair_outcomes),
        "single_count": len(single_outcomes),
        "pairs_both_satisfied": allocation_stats["pairs_both_satisfied"],
        "pairs_partial": allocation_stats["pairs_partial"],
        "pairs_none": allocation_stats["pairs_none"],
        "top_popular": ranking[:10],
    }


def run_mc_runs(cfg):
    results = []
    for run in range(cfg.NUM_RUNS):
        seed = cfg.RANDOM_SEED + run
        out = run_single_experiment(seed, cfg)
        results.append(out)
        print(
            f"Run {run+1}/{cfg.NUM_RUNS}: "
            f"policy={cfg.CACHE_POLICY}, "
            f"hit_rate={out['hit_rate']:.4f}, "
            f"outage={out['outage_rate']:.4f}, "
            f"misses={out['num_misses']}, "
            f"pairs_both={out['pairs_both_satisfied']}, "
            f"pairs_partial={out['pairs_partial']}, "
            f"pairs_none={out['pairs_none']}"
        )
    df = pd.DataFrame(results)
    return df

def run_single_experiment_with_learning(seed, cfg):
    """New simulation that includes learning from NOMA outcomes."""
    
    set_seed(seed)
    
    # Generate users and channel conditions
    user_pos = channel_model.generate_user_positions(cfg.NUM_USERS, cfg.CELL_RADIUS, seed)
    distances = user_pos[:, 2]
    pl = np.array([channel_model.pathloss(d, cfg.PATHLOSS_EXPONENT) for d in distances])
    
    # Initialize novel cache based on policy
    if cfg.CACHE_POLICY == "noma_aware":
        cache = NomaAwarePredictiveCache(cfg.CACHE_SIZE, cfg.NUM_FILES, cfg.NUM_USERS)
    elif cfg.CACHE_POLICY == "joint_opt":
        small_scale = channel_model.rayleigh_gain(cfg.NUM_USERS)
        channel_gains = pl * small_scale
        cache = JointOptimizationCache(cfg.CACHE_SIZE, cfg.NUM_FILES, channel_gains, cfg)
    elif cfg.CACHE_POLICY == "multi_obj":
        cache = MultiObjectiveCache(cfg.CACHE_SIZE, cfg.NUM_FILES, cfg.MO_OBJECTIVES)
    elif cfg.CACHE_POLICY == "rl":
        cache = ReinforcementLearningCache(cfg.CACHE_SIZE, cfg.NUM_FILES)
    else:
        # Fallback to existing static cache
        requests_for_popularity = sample_zipf_catalog(cfg.NUM_FILES, cfg.ZIPF_ALPHA, 
                                                    size=cfg.NUM_USERS * cfg.REQUESTS_PER_USER)
        ranking, _ = compute_popularity(requests_for_popularity)
        cache = StaticTopKCache(cfg.CACHE_SIZE)
        cache.populate(ranking)
    
    # Learning-based simulation with time slots
    total_hits = 0
    total_requests = 0
    total_outages = 0
    total_transmissions = 0
    
    learning_stats = {
        "cache_updates": 0,
        "noma_successes": 0,
        "noma_attempts": 0,
        "learning_episodes": 0
    }
    
    for time_slot in range(cfg.TIME_SLOTS):
        # Generate requests for this time slot
        slot_requests = cfg.NUM_USERS * cfg.REQUESTS_PER_USER // cfg.TIME_SLOTS
        if slot_requests == 0:
            slot_requests = 1
            
        requests = sample_zipf_catalog(cfg.NUM_FILES, cfg.ZIPF_ALPHA, size=slot_requests)
        requesting_users = np.random.choice(cfg.NUM_USERS, size=slot_requests, replace=True)
        
        # Update channel gains for this time slot (mobility)
        small_scale = channel_model.rayleigh_gain(cfg.NUM_USERS)
        channel_gains = pl * small_scale
        
        # Process requests
        for req_idx, (file_id, user_id) in enumerate(zip(requests, requesting_users)):
            total_requests += 1
            
            # Check cache
            cache_hit = cache.is_hit(file_id)
            
            if cache_hit:
                total_hits += 1
                # For learning caches, observe the hit
                if hasattr(cache, 'observe_request'):
                    cache.observe_request(user_id, file_id, cache_hit=True)
                elif hasattr(cache, 'request_with_learning'):
                    cache.request_with_learning(file_id, hit=True, outage_occurred=False)
            else:
                # Cache miss - need NOMA transmission
                total_transmissions += 1
                learning_stats["noma_attempts"] += 1
                
                # Find users for pairing
                other_miss_users = [requesting_users[i] for i in range(len(requests)) 
                                  if not cache.is_hit(requests[i]) and requesting_users[i] != user_id]
                
                if other_miss_users and cfg.PAIR_USERS:
                    # NOMA pairing
                    partner_user = np.random.choice(other_miss_users)
                    
                    # Ensure weak/strong ordering
                    if channel_gains[user_id] < channel_gains[partner_user]:
                        weak_user, strong_user = user_id, partner_user
                    else:
                        weak_user, strong_user = partner_user, user_id
                    
                    # Power allocation
                    if cfg.USE_CLOSED_FORM_ALLOC:
                        p_w, p_s, feasible, _ = allocate_power_closedform(
                            channel_gains[weak_user], channel_gains[strong_user], cfg)
                    else:
                        p_w, p_s, feasible, _ = allocate_power_gridsearch(
                            channel_gains[weak_user], channel_gains[strong_user], cfg)
                    
                    # Simulate NOMA transmission
                    weak_success, strong_success, _, _, _ = simulate_noma_pair(
                        channel_gains[weak_user], channel_gains[strong_user], cfg, p_w, p_s)
                    
                    # Determine if current user succeeded
                    noma_success = weak_success if user_id == weak_user else strong_success
                    
                    if not noma_success:
                        total_outages += 1
                    else:
                        learning_stats["noma_successes"] += 1
                        
                else:
                    # Single user transmission (no pairing)
                    sinr_single = cfg.TX_POWER * channel_gains[user_id] / cfg.NOISE_POWER
                    sinr_threshold = 2 ** cfg.TARGET_RATE_BPS - 1
                    noma_success = sinr_single >= sinr_threshold
                    
                    if not noma_success:
                        total_outages += 1
                    else:
                        learning_stats["noma_successes"] += 1
                
                # Learning from outcome
                if hasattr(cache, 'observe_request'):
                    cache.observe_request(user_id, file_id, cache_hit=False, 
                                        noma_success=noma_success)
                elif hasattr(cache, 'request_with_learning'):
                    cache.request_with_learning(file_id, hit=False, 
                                              outage_occurred=not noma_success)
        
        # Periodic cache optimization
        if time_slot % cfg.CACHE_UPDATE_INTERVAL == 0 and time_slot > 0:
            if hasattr(cache, 'populate'):
                if hasattr(cache, 'update_popularity'):
                    # For caches that need request history
                    all_requests = sample_zipf_catalog(cfg.NUM_FILES, cfg.ZIPF_ALPHA, 
                                                     size=cfg.CACHE_UPDATE_INTERVAL * slot_requests)
                    cache.update_popularity(all_requests.tolist())
                cache.populate()
                learning_stats["cache_updates"] += 1
        
        learning_stats["learning_episodes"] = time_slot + 1
    
    # Calculate final metrics
    hit_rate = total_hits / total_requests if total_requests > 0 else 0
    outage_rate = total_outages / total_transmissions if total_transmissions > 0 else 0
    noma_success_rate = learning_stats["noma_successes"] / learning_stats["noma_attempts"] \
                       if learning_stats["noma_attempts"] > 0 else 0
    
    result = {
        "seed": seed,
        "policy": cfg.CACHE_POLICY,
        "total_requests": total_requests,
        "hits": total_hits,
        "hit_rate": hit_rate,
        "total_transmissions": total_transmissions,
        "outages": total_outages,
        "outage_rate": outage_rate,
        "noma_success_rate": noma_success_rate,
        "cache_updates": learning_stats["cache_updates"],
        "learning_episodes": learning_stats["learning_episodes"]
    }
    
    # Add cache-specific metrics
    if hasattr(cache, 'get_cache_stats'):
        cache_stats = cache.get_cache_stats()
        result.update({f"cache_{k}": v for k, v in cache_stats.items()})
    
    return result

# 3. Create comparison framework
def compare_novel_algorithms(cfg):
    """Compare novel algorithms against baselines."""
    
    policies_to_test = [
        "topk",           # Baseline
        "lru",            # Baseline  
        "noma_aware",     # Novel 1
        "joint_opt",      # Novel 2
        "multi_obj",      # Novel 3
        "rl"              # Novel 4
    ]
    
    results = {}
    
    for policy in policies_to_test:
        print(f"\n=== Testing {policy} policy ===")
        cfg.CACHE_POLICY = policy
        
        if policy in ["noma_aware", "joint_opt", "multi_obj", "rl"]:
            # Use new learning-based simulation
            runs = []
            for run in range(cfg.NUM_RUNS):
                seed = cfg.RANDOM_SEED + run
                result = run_single_experiment_with_learning(seed, cfg)
                runs.append(result)
                print(f"Run {run+1}: hit_rate={result['hit_rate']:.3f}, "
                      f"outage={result['outage_rate']:.3f}, "
                      f"noma_success={result['noma_success_rate']:.3f}")
            
            results[policy] = pd.DataFrame(runs)
        else:
            # Use original simulation for baselines
            results[policy] = run_mc_runs(cfg)
            results[policy]["noma_success_rate"] = 0.5  # Default value
            results[policy]["cache_updates"] = 0
            results[policy]["learning_episodes"] = 0
    
    return results

# 4. Advanced metrics and analysis
def analyze_novel_contributions(results_dict):
    """Analyze the contributions of novel algorithms."""
    
    metrics = {}
    
    for policy, df in results_dict.items():
        metrics[policy] = {
            "mean_hit_rate": df["hit_rate"].mean(),
            "std_hit_rate": df["hit_rate"].std(),
            "mean_outage": df["outage_rate"].mean(), 
            "std_outage": df["outage_rate"].std(),
            "mean_noma_success": df["noma_success_rate"].mean(),
            "adaptation_capability": df["cache_updates"].mean() if "cache_updates" in df else 0,
        }
    
    # Calculate improvements over baseline
    baseline_hit = metrics["topk"]["mean_hit_rate"]
    baseline_outage = metrics["topk"]["mean_outage"]
    
    improvements = {}
    for policy in metrics:
        if policy != "topk":
            hit_improvement = (metrics[policy]["mean_hit_rate"] - baseline_hit) / baseline_hit * 100
            outage_improvement = (baseline_outage - metrics[policy]["mean_outage"]) / baseline_outage * 100
            
            improvements[policy] = {
                "hit_rate_improvement_pct": hit_improvement,
                "outage_reduction_pct": outage_improvement,
                "joint_metric": hit_improvement + outage_improvement  # Combined benefit
            }
    
    return metrics, improvements

# 5. Visualization for novel contributions
def plot_novel_algorithm_performance(results_dict, improvements):
    """Create visualizations showing novel algorithm benefits."""
    
    # Performance comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    policies = list(results_dict.keys())
    hit_rates = [results_dict[p]["hit_rate"].mean() for p in policies]
    outage_rates = [results_dict[p]["outage_rate"].mean() for p in policies]
    
    # Bar plots
    ax1.bar(policies, hit_rates)
    ax1.set_title("Hit Rate Comparison")
    ax1.set_ylabel("Hit Rate")
    ax1.tick_params(axis='x', rotation=45)
    
    ax2.bar(policies, outage_rates, color='orange')
    ax2.set_title("Outage Rate Comparison")
    ax2.set_ylabel("Outage Rate")
    ax2.tick_params(axis='x', rotation=45)
    
    # Improvement percentages
    novel_policies = [p for p in improvements.keys()]
    hit_improvements = [improvements[p]["hit_rate_improvement_pct"] for p in novel_policies]
    outage_improvements = [improvements[p]["outage_reduction_pct"] for p in novel_policies]
    
    ax3.bar(novel_policies, hit_improvements, color='green')
    ax3.set_title("Hit Rate Improvement vs Top-K (%)")
    ax3.set_ylabel("Improvement %")
    ax3.tick_params(axis='x', rotation=45)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    ax4.bar(novel_policies, outage_improvements, color='red') 
    ax4.set_title("Outage Reduction vs Top-K (%)")
    ax4.set_ylabel("Reduction %")
    ax4.tick_params(axis='x', rotation=45)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("novel_algorithm_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Learning curves (for adaptive algorithms)
    plt.figure(figsize=(10, 6))
    for policy in ["noma_aware", "joint_opt", "rl"]:
        if policy in results_dict:
            df = results_dict[policy]
            if "learning_episodes" in df and df["learning_episodes"].iloc[0] > 0:
                # Plot learning progression (simplified)
                plt.plot(range(len(df)), df["hit_rate"], label=f"{policy}", marker='o')
    
    plt.title("Learning Performance Over Time")
    plt.xlabel("Simulation Run")
    plt.ylabel("Hit Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig("learning_curves.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    from src import config as cfg
    t0 = time.time()

    MODE = "novel"  # options: "baseline", "novel"

    if MODE == "baseline":
        df = run_mc_runs(cfg)
        print("\nSummary:")
        print(df[["seed", "hit_rate", "outage_rate", "total_requests"]].to_string(index=False))
        print(f"\nMean hit_rate = {df['hit_rate'].mean():.4f} (std={df['hit_rate'].std():.4f})")
        print(f"Mean outage_rate = {df['outage_rate'].mean():.4f} (std={df['outage_rate'].std():.4f})")

    elif MODE == "novel":
        cfg.TIME_SLOTS = 500
        cfg.CACHE_UPDATE_INTERVAL = 50
        results = compare_novel_algorithms(cfg)
        metrics, improvements = analyze_novel_contributions(results)

        print("\n" + "="*60)
        print("NOVEL ALGORITHM PERFORMANCE ANALYSIS")
        print("="*60)
        for policy, imp in improvements.items():
            print(f"\n{policy.upper()}:")
            print(f"  Hit Rate Improvement: {imp['hit_rate_improvement_pct']:+.2f}%")
            print(f"  Outage Reduction: {imp['outage_reduction_pct']:+.2f}%")
            print(f"  Combined Benefit: {imp['joint_metric']:+.2f}")

        plot_novel_algorithm_performance(results, improvements)
        print("\nVisualizations saved: novel_algorithm_comparison.png, learning_curves.png")

    print(f"\nElapsed: {time.time()-t0:.2f}s")
