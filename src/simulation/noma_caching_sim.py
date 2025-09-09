# src/simulation/noma_caching_sim.py
import numpy as np
import pandas as pd
import time
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
            # dynamic caches (LRU, LFU, Random)
            if file_id in cache:         # check
                hits += 1
                cache.request(file_id)   # update usage
            else:                        # miss
                misses.append(user_idx)
                cache.request(file_id)   # insert (may evict)

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


if __name__ == "__main__":
    from src import config as cfg
    t0 = time.time()
    df = run_mc_runs(cfg)
    print("\nSummary:")
    print(df[["seed", "hit_rate", "outage_rate", "total_requests"]].to_string(index=False))
    print(f"\nMean hit_rate = {df['hit_rate'].mean():.4f} (std={df['hit_rate'].std():.4f})")
    print(f"Mean outage_rate = {df['outage_rate'].mean():.4f} (std={df['outage_rate'].std():.4f})")
    print(f"Elapsed: {time.time()-t0:.2f}s")
