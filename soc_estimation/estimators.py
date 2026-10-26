"""SOC estimators: Coulomb counting and an Extended Kalman Filter (EKF)."""

import numpy as np

from .battery_model import CellParams, docv_dsoc, ocv


class CoulombCounter:
    """Open-loop current integration. Simple, but has no way to correct
    a wrong initial SOC, a current-sensor offset or a capacity error."""

    def __init__(self, params: CellParams, soc0: float, dt: float = 1.0):
        self.q_as = params.capacity_ah * 3600.0
        self.dt = dt
        self.soc = soc0

    def update(self, current_meas: float, voltage_meas: float = None) -> float:
        self.soc -= current_meas * self.dt / self.q_as
        return self.soc


class ExtendedKalmanFilter:
    """EKF on the 1RC model. State x = [SOC, Vrc].

    Prediction uses the measured current (same model as Coulomb counting);
    the correction step compares the predicted terminal voltage with the
    measured one and pulls SOC back through the OCV slope.
    """

    def __init__(
        self,
        params: CellParams,
        soc0: float,
        dt: float = 1.0,
        p0=(0.2**2, 1e-4),
        q=(1e-8, 1e-7),
        r=1e-4,
    ):
        self.p = params
        self.dt = dt
        self.a = np.exp(-dt / params.tau)
        self.x = np.array([soc0, 0.0])
        self.P = np.diag(p0)
        self.Q = np.diag(q)
        self.R = r

    @property
    def soc(self) -> float:
        return float(self.x[0])

    def update(self, current_meas: float, voltage_meas: float) -> float:
        i = current_meas

        # 1) Correct with the voltage measured at this step (before current is applied)
        soc, vrc = self.x
        v_pred = ocv(soc) - self.p.r0 * i - vrc
        H = np.array([[docv_dsoc(soc), -1.0]])
        S = (H @ self.P @ H.T).item() + self.R
        K = (self.P @ H.T / S).ravel()
        self.x = self.x + K * (voltage_meas - v_pred)
        self.P = (np.eye(2) - np.outer(K, H)) @ self.P

        # 2) Predict to the next step using the measured current
        F = np.array([[1.0, 0.0], [0.0, self.a]])
        self.x = np.array([
            self.x[0] - i * self.dt / (3600.0 * self.p.capacity_ah),
            self.a * self.x[1] + self.p.r1 * (1.0 - self.a) * i,
        ])
        self.P = F @ self.P @ F.T + self.Q

        self.x[0] = np.clip(self.x[0], 0.0, 1.0)
        return self.soc
