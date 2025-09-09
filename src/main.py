# src/main.py
from src.simulation.noma_caching_sim import run_mc_runs
from src import config

def main():
    df = run_mc_runs(config)
    # save results
    df.to_csv("results_phase1.csv", index=False)
    print("Results saved to results_phase1.csv")

if __name__ == "__main__":
    main()
