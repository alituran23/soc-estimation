"""First-order (1RC) equivalent-circuit model of a Li-ion cell.

Sign convention: positive current = discharge.

    SOC[k+1]  = SOC[k] - I[k] * dt / (3600 * Q)
    Vrc[k+1]  = a * Vrc[k] + R1 * (1 - a) * I[k],   a = exp(-dt / (R1 * C1))
    Vt[k]     = OCV(SOC[k]) - R0 * I[k] - Vrc[k]
"""

from dataclasses import dataclass

import numpy as np

# OCV(SOC) uses the "combined model" form (Plett, Battery Management Systems Vol. II):
#   OCV = K0 + K1*s + K2*ln(s) + K3*ln(1 - s)
# Coefficients are chosen to give a plausible NMC-like curve (~3.4 V empty, ~4.2 V full).
OCV_K = (3.60, 0.55, 0.05, -0.03)
SOC_MIN, SOC_MAX = 0.005, 0.995


def ocv(soc):
    """Open-circuit voltage [V] as a function of SOC [0..1]."""
    s = np.clip(soc, SOC_MIN, SOC_MAX)
    k0, k1, k2, k3 = OCV_K
    return k0 + k1 * s + k2 * np.log(s) + k3 * np.log(1.0 - s)


def docv_dsoc(soc):
    """Slope of the OCV curve [V per unit SOC], used in the EKF Jacobian."""
    s = np.clip(soc, SOC_MIN, SOC_MAX)
    _, k1, k2, k3 = OCV_K
    return k1 + k2 / s - k3 / (1.0 - s)


@dataclass
class CellParams:
    capacity_ah: float = 5.0   # usable capacity Q [Ah]
    r0: float = 0.015          # ohmic resistance [ohm]
    r1: float = 0.010          # polarisation resistance [ohm]
    c1: float = 2000.0         # polarisation capacitance [F] (tau = 20 s)

    @property
    def tau(self) -> float:
        return self.r1 * self.c1


class CellModel:
    """Discrete-time 1RC cell used as the 'true' plant in simulation."""

    def __init__(self, params: CellParams, soc0: float, dt: float = 1.0):
        self.p = params
        self.dt = dt
        self.soc = soc0
        self.vrc = 0.0
        self._a = np.exp(-dt / params.tau)

    def terminal_voltage(self, current: float) -> float:
        return float(ocv(self.soc) - self.p.r0 * current - self.vrc)

    def step(self, current: float) -> float:
        """Apply `current` for one time step; return terminal voltage at the start of the step."""
        v = self.terminal_voltage(current)
        self.soc -= current * self.dt / (3600.0 * self.p.capacity_ah)
        self.vrc = self._a * self.vrc + self.p.r1 * (1.0 - self._a) * current
        return v
