"""Drive-cycle generation and a closed-loop scenario runner."""

from dataclasses import dataclass, field

import numpy as np

from .battery_model import CellModel, CellParams
from .estimators import CoulombCounter, ExtendedKalmanFilter


def drive_cycle(duration_s: int = 3600, dt: float = 1.0) -> np.ndarray:
    """Synthetic urban-style current profile [A], positive = discharge.

    A 120 s pattern of acceleration, cruise, regenerative braking and idle,
    repeated for `duration_s`. Average draw is ~3.8 A on a 5 Ah cell.
    """
    pattern = np.concatenate([
        np.full(15, 12.0),   # acceleration
        np.full(60, 4.0),    # cruise
        np.full(10, -6.0),   # regen braking
        np.full(15, 0.0),    # idle
        np.full(10, 9.0),    # second acceleration
        np.full(10, -4.0),   # regen
    ])
    n = int(duration_s / dt)
    reps = int(np.ceil(n / len(pattern)))
    return np.tile(pattern, reps)[:n]


@dataclass
class ScenarioConfig:
    """Real-world error sources the estimators have to cope with."""

    true_soc0: float = 0.90                 # actual starting SOC
    est_soc0: float = 0.70                  # what the BMS believes at start-up
    true_capacity_ah: float = 4.6           # aged cell (~92 % SOH)
    current_offset_a: float = 0.15          # current-sensor bias
    current_noise_a: float = 0.05           # current-sensor noise (1 sigma)
    voltage_noise_v: float = 0.005          # voltage-sensor noise (1 sigma)
    duration_s: int = 3600
    dt: float = 1.0
    seed: int = 7
    bms_params: CellParams = field(default_factory=CellParams)  # what the BMS assumes


def run_scenario(cfg: ScenarioConfig = None) -> dict:
    """Simulate the true cell and run both estimators on noisy sensor data."""
    cfg = cfg or ScenarioConfig()
    rng = np.random.default_rng(cfg.seed)

    true_params = CellParams(
        capacity_ah=cfg.true_capacity_ah,
        r0=cfg.bms_params.r0,
        r1=cfg.bms_params.r1,
        c1=cfg.bms_params.c1,
    )
    cell = CellModel(true_params, cfg.true_soc0, cfg.dt)
    cc = CoulombCounter(cfg.bms_params, cfg.est_soc0, cfg.dt)
    ekf = ExtendedKalmanFilter(cfg.bms_params, cfg.est_soc0, cfg.dt)

    current = drive_cycle(cfg.duration_s, cfg.dt)
    n = len(current)
    out = {k: np.zeros(n) for k in ("t", "i", "v", "soc_true", "soc_cc", "soc_ekf")}

    for k, i_true in enumerate(current):
        out["soc_true"][k] = cell.soc
        v_true = cell.step(i_true)

        i_meas = i_true + cfg.current_offset_a + rng.normal(0.0, cfg.current_noise_a)
        v_meas = v_true + rng.normal(0.0, cfg.voltage_noise_v)

        out["t"][k] = k * cfg.dt
        out["i"][k] = i_true
        out["v"][k] = v_meas
        out["soc_cc"][k] = cc.update(i_meas, v_meas)
        out["soc_ekf"][k] = ekf.update(i_meas, v_meas)

    return out


def error_stats(result: dict, skip_s: float = 300.0) -> dict:
    """RMSE and max |error| in SOC percentage points, ignoring the first `skip_s` seconds."""
    mask = result["t"] >= skip_s
    stats = {}
    for name in ("cc", "ekf"):
        err = 100.0 * (result[f"soc_{name}"][mask] - result["soc_true"][mask])
        stats[name] = {
            "rmse_pct": float(np.sqrt(np.mean(err**2))),
            "max_abs_pct": float(np.max(np.abs(err))),
            "final_pct": float(err[-1]),
        }
    return stats
