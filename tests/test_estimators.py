import numpy as np

from soc_estimation import ScenarioConfig, docv_dsoc, ocv, run_scenario
from soc_estimation.simulation import error_stats


def test_ocv_is_monotonic_and_in_range():
    s = np.linspace(0.01, 0.99, 200)
    v = ocv(s)
    assert np.all(np.diff(v) > 0)
    assert 3.2 < v[0] < 3.6
    assert 4.1 < v[-1] < 4.3
    assert np.all(docv_dsoc(s) > 0)


def test_ekf_recovers_from_wrong_initial_soc():
    res = run_scenario(ScenarioConfig())
    stats = error_stats(res, skip_s=300)
    assert stats["ekf"]["max_abs_pct"] < 2.0


def test_coulomb_counting_keeps_initial_error_and_drifts():
    res = run_scenario(ScenarioConfig())
    stats = error_stats(res, skip_s=300)
    assert stats["cc"]["rmse_pct"] > 10.0
    assert stats["ekf"]["rmse_pct"] < stats["cc"]["rmse_pct"] / 5


def test_perfect_conditions_coulomb_counting_is_accurate():
    cfg = ScenarioConfig(est_soc0=0.90, true_capacity_ah=5.0,
                         current_offset_a=0.0, current_noise_a=0.0)
    stats = error_stats(run_scenario(cfg), skip_s=0)
    assert stats["cc"]["max_abs_pct"] < 0.1
