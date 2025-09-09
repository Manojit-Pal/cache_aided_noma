# src/experiments/run_experiments.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.simulation import noma_caching_sim
from src import config as cfg

def sweep_cache_sizes(sizes, cfg):
    results = []
    for csize in sizes:
        cfg.CACHE_SIZE = csize
        df = noma_caching_sim.run_mc_runs(cfg)
        results.append({
            "cache_size": csize,
            "hit_rate": df["hit_rate"].mean(),
            "outage": df["outage_rate"].mean()
        })
    return pd.DataFrame(results)

def sweep_zipf_alpha(alphas, cfg):
    results = []
    for a in alphas:
        cfg.ZIPF_ALPHA = a
        df = noma_caching_sim.run_mc_runs(cfg)
        results.append({
            "zipf_alpha": a,
            "hit_rate": df["hit_rate"].mean(),
            "outage": df["outage_rate"].mean()
        })
    return pd.DataFrame(results)

def sweep_target_rate(rates, cfg):
    results = []
    for r in rates:
        cfg.TARGET_RATE_BPS = r
        df = noma_caching_sim.run_mc_runs(cfg)
        results.append({
            "target_rate": r,
            "hit_rate": df["hit_rate"].mean(),
            "outage": df["outage_rate"].mean()
        })
    return pd.DataFrame(results)

def plot_results(df, x, y, ylabel, title, filename):
    plt.figure()
    plt.plot(df[x], df[y], marker="o")
    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")

if __name__ == "__main__":
    # Example sweeps
    cache_sizes = [50, 100, 200, 300, 400, 500]
    zipf_alphas = [0.6, 0.8, 1.0, 1.2, 1.4]
    target_rates = [0.5, 1.0, 1.5, 2.0]

    print("Running cache size sweep...")
    df_cache = sweep_cache_sizes(cache_sizes, cfg)
    plot_results(df_cache, "cache_size", "hit_rate", "Hit Rate", "Cache Size vs Hit Rate", "cache_vs_hit.png")
    plot_results(df_cache, "cache_size", "outage", "Outage Probability", "Cache Size vs Outage", "cache_vs_outage.png")

    print("Running Zipf alpha sweep...")
    df_zipf = sweep_zipf_alpha(zipf_alphas, cfg)
    plot_results(df_zipf, "zipf_alpha", "hit_rate", "Hit Rate", "Zipf Alpha vs Hit Rate", "zipf_vs_hit.png")
    plot_results(df_zipf, "zipf_alpha", "outage", "Outage Probability", "Zipf Alpha vs Outage", "zipf_vs_outage.png")

    print("Running target rate sweep...")
    df_rate = sweep_target_rate(target_rates, cfg)
    plot_results(df_rate, "target_rate", "outage", "Outage Probability", "Target Rate vs Outage", "rate_vs_outage.png")
