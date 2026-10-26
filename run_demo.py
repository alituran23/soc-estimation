"""Run the SOC estimation demo and save the results plot.

Usage:
    python run_demo.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from soc_estimation.simulation import ScenarioConfig, error_stats, run_scenario

OUT = Path(__file__).parent / "docs" / "results.png"


def main():
    cfg = ScenarioConfig()
    res = run_scenario(cfg)
    stats = error_stats(res)
    t_min = res["t"] / 60.0

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.4, 1.2]})

    ax = axes[0]
    ax.plot(t_min, 100 * res["soc_true"], color="black", lw=2, label="True SOC")
    ax.plot(t_min, 100 * res["soc_cc"], color="#d62728", lw=1.6, ls="--", label="Coulomb counting")
    ax.plot(t_min, 100 * res["soc_ekf"], color="#1f77b4", lw=1.6, label="Extended Kalman Filter")
    ax.set_ylabel("SOC [%]")
    ax.set_title("SOC estimation with a wrong start SOC, aged cell and biased current sensor")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t_min, 100 * (res["soc_cc"] - res["soc_true"]), color="#d62728", lw=1.4,
            label=f"Coulomb counting (RMSE {stats['cc']['rmse_pct']:.1f} %)")
    ax.plot(t_min, 100 * (res["soc_ekf"] - res["soc_true"]), color="#1f77b4", lw=1.4,
            label=f"EKF (RMSE {stats['ekf']['rmse_pct']:.1f} %)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("SOC error [%-pt]")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(t_min, res["i"], color="#555555", lw=0.8)
    ax.set_ylabel("Current [A]\n(+ discharge)")
    ax.set_xlabel("Time [min]")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=130)

    print("Scenario:")
    print(f"  true start SOC {cfg.true_soc0:.0%}, BMS start estimate {cfg.est_soc0:.0%}")
    print(f"  true capacity {cfg.true_capacity_ah} Ah vs BMS assumption {cfg.bms_params.capacity_ah} Ah")
    print(f"  current-sensor offset {cfg.current_offset_a} A")
    print("Errors after the first 5 minutes (SOC percentage points):")
    for name, label in (("cc", "Coulomb counting"), ("ekf", "EKF")):
        s = stats[name]
        print(f"  {label:17s} RMSE {s['rmse_pct']:5.2f}  max {s['max_abs_pct']:5.2f}  final {s['final_pct']:+5.2f}")
    print(f"Plot saved to {OUT}")


if __name__ == "__main__":
    main()
