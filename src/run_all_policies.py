import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src import config
from src.simulation import noma_caching_sim


def run_all_policies():
    policies = ["topk", "lru", "lfu", "random"]
    all_results = []

    for pol in policies:
        print(f"\n=== Running policy: {pol} ===")
        config.CACHE_POLICY = pol
        df = noma_caching_sim.run_mc_runs(config)
        df["policy"] = pol
        df["run_idx"] = range(1, len(df) + 1)  # add per-policy run index
        df.to_csv(f"results_{pol}.csv", index=False)
        all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv("results_all_policies.csv", index=False)
    print("\nAll results saved to results_all_policies.csv")

    # --- Compute averages ---
    summary = combined.groupby("policy").agg(
        mean_hit_rate=("hit_rate", "mean"),
        std_hit_rate=("hit_rate", "std"),
        mean_outage=("outage_rate", "mean"),
        std_outage=("outage_rate", "std")
    ).reset_index()

    summary = summary.fillna(0)  # handle NaN std
    summary = summary.set_index("policy").loc[policies].reset_index()  # preserve order

    summary.to_csv("policy_summary.csv", index=False)
    print("\nPolicy summary saved to policy_summary.csv")
    print(summary)

    # --- Identify best policies ---
    best_hit = summary.loc[summary["mean_hit_rate"].idxmax()]
    best_outage = summary.loc[summary["mean_outage"].idxmin()]

    print("\n🏆 Best Policies:")
    print(f"   - Highest Hit Rate: {best_hit['policy']} "
          f"(avg={best_hit['mean_hit_rate']:.4f}, std={best_hit['std_hit_rate']:.4f})")
    print(f"   - Lowest Outage Rate: {best_outage['policy']} "
          f"(avg={best_outage['mean_outage']:.4f}, std={best_outage['std_outage']:.4f})")

    # --- Line plots ---
    plt.figure(figsize=(10, 5))
    for pol in policies:
        subset = combined[combined["policy"] == pol]
        plt.plot(subset["run_idx"], subset["hit_rate"], marker="o", label=f"{pol}")
    plt.title("Cache Policies: Hit Rate (per run)")
    plt.xlabel("Simulation Run")
    plt.ylabel("Hit Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig("hit_rate_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    for pol in policies:
        subset = combined[combined["policy"] == pol]
        plt.plot(subset["run_idx"], subset["outage_rate"], marker="s", label=f"{pol}")
    plt.title("Cache Policies: Outage Rate (per run)")
    plt.xlabel("Simulation Run")
    plt.ylabel("Outage Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig("outage_rate_comparison.png", dpi=300)
    plt.close()

    # --- Bar charts (averages) ---
    plt.figure(figsize=(8, 5))
    plt.bar(summary["policy"], summary["mean_hit_rate"],
            yerr=summary["std_hit_rate"], capsize=5)
    plt.title("Average Hit Rate per Policy")
    plt.ylabel("Hit Rate")
    plt.savefig("avg_hit_rate.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(summary["policy"], summary["mean_outage"],
            yerr=summary["std_outage"], capsize=5, color="orange")
    plt.title("Average Outage Rate per Policy")
    plt.ylabel("Outage Rate")
    plt.savefig("avg_outage_rate.png", dpi=300)
    plt.close()

    # --- Boxplots (distributions) ---
    plt.figure(figsize=(8, 5))
    combined.boxplot(column="hit_rate", by="policy")
    plt.title("Hit Rate Distribution per Policy")
    plt.suptitle("")
    plt.ylabel("Hit Rate")
    plt.savefig("hit_rate_boxplot.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    combined.boxplot(column="outage_rate", by="policy")
    plt.title("Outage Rate Distribution per Policy")
    plt.suptitle("")
    plt.ylabel("Outage Rate")
    plt.savefig("outage_rate_boxplot.png", dpi=300)
    plt.close()

    # --- CDF plots ---
    plt.figure(figsize=(8, 5))
    for pol in policies:
        subset = combined[combined["policy"] == pol]["hit_rate"].sort_values()
        yvals = np.arange(1, len(subset)+1) / float(len(subset))
        plt.plot(subset, yvals, label=pol)
    plt.title("CDF of Hit Rates")
    plt.xlabel("Hit Rate")
    plt.ylabel("Cumulative Probability")
    plt.grid(True)
    plt.legend()
    plt.savefig("hit_rate_cdf.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    for pol in policies:
        subset = combined[combined["policy"] == pol]["outage_rate"].sort_values()
        yvals = np.arange(1, len(subset)+1) / float(len(subset))
        plt.plot(subset, yvals, label=pol)
    plt.title("CDF of Outage Rates")
    plt.xlabel("Outage Rate")
    plt.ylabel("Cumulative Probability")
    plt.grid(True)
    plt.legend()
    plt.savefig("outage_rate_cdf.png", dpi=300)
    plt.close()

    print("\nPlots saved:")
    print("   hit_rate_comparison.png, outage_rate_comparison.png")
    print("   avg_hit_rate.png, avg_outage_rate.png")
    print("   hit_rate_boxplot.png, outage_rate_boxplot.png")
    print("   hit_rate_cdf.png, outage_rate_cdf.png")


if __name__ == "__main__":
    run_all_policies()
