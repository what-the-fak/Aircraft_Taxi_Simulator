"""Table-driven tire cornering model.

Each tire type holds three vertical-load cases (rated-, rated, rated+) and a
cornering-force table Fy(slip) for each, slip 0..18 deg.

Lookup for an arbitrary (slip, Fz):
  * slip  : linear interpolation in the table, symmetric for negative slip,
            held constant beyond the last slip point.
  * Fz    : linear interpolation between load cases, with an implicit
            (Fz = 0, Fy = 0) case below the lowest load and linear
            extrapolation above the highest load.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_SLIPS = np.arange(0.0, 18.0 + 1e-9, 2.0)  # 0, 2, ... 18 deg
LOAD_CASE_NAMES = ("Rated -", "Rated", "Rated +")


def default_curves(loads, slips=DEFAULT_SLIPS, fy_frac_at_ref=0.15, ref_slip_deg=6.0):
    """Placeholder curves: Fy = frac * Fz at ref slip, linear through zero."""
    loads = np.asarray(loads, float)
    slips = np.asarray(slips, float)
    return np.outer(loads, fy_frac_at_ref * slips / ref_slip_deg)


@dataclass
class TireData:
    name: str
    size: str = ""
    od_in: float = 40.0
    width_in: float = 15.0
    pressure_psi: float = 200.0
    loads_lb: np.ndarray = field(default_factory=lambda: np.array([0.75, 1.0, 1.25]) * 40000)
    slip_deg: np.ndarray = field(default_factory=lambda: DEFAULT_SLIPS.copy())
    fy_lb: np.ndarray | None = None  # shape (3, n_slip)

    def __post_init__(self):
        self.loads_lb = np.asarray(self.loads_lb, float)
        self.slip_deg = np.asarray(self.slip_deg, float)
        if self.fy_lb is None:
            self.fy_lb = default_curves(self.loads_lb, self.slip_deg)
        self.fy_lb = np.asarray(self.fy_lb, float)
        self.prepare()

    # ------------------------------------------------------------------ setup
    def prepare(self):
        if self.fy_lb.shape != (len(self.loads_lb), len(self.slip_deg)):
            raise ValueError(f"{self.name}: Fy table shape does not match loads x slips")
        if np.any(np.diff(self.slip_deg) <= 0):
            raise ValueError(f"{self.name}: slip angles must be strictly increasing")
        order = np.argsort(self.loads_lb)
        loads = self.loads_lb[order]
        if loads[0] <= 0 or np.any(np.diff(loads) <= 0):
            raise ValueError(f"{self.name}: load cases must be positive and distinct")
        fy = self.fy_lb[order]
        slips = self.slip_deg
        if slips[0] > 0:  # make sure the curve passes through the origin
            slips = np.concatenate([[0.0], slips])
            fy = np.hstack([np.zeros((fy.shape[0], 1)), fy])
        self._slips = slips
        self._loads = np.concatenate([[0.0], loads])
        self._fy = np.vstack([np.zeros(len(slips)), fy])

    # ------------------------------------------------------------------ lookup
    def fy(self, alpha_deg, fz_lb):
        """Signed lateral force (lb) for slip angle(s) alpha_deg at load(s) fz_lb."""
        alpha = np.atleast_1d(np.asarray(alpha_deg, float))
        fz = np.broadcast_to(np.atleast_1d(np.asarray(fz_lb, float)), alpha.shape)
        a = np.abs(alpha)
        vals = np.array([np.interp(a, self._slips, row) for row in self._fy])  # (n_load, n)
        L = self._loads
        idx = np.clip(np.searchsorted(L, fz, side="right") - 1, 0, len(L) - 2)
        w = (fz - L[idx]) / (L[idx + 1] - L[idx])
        cols = np.arange(len(a))
        f = vals[idx, cols] * (1 - w) + vals[idx + 1, cols] * w
        f = np.where(fz > 0, f, 0.0)
        return np.sign(alpha) * f

    def cornering_stiffness(self, fz_lb):
        """Approximate initial slope dFy/dalpha (lb/rad) at load fz."""
        d = 0.5
        return float(self.fy(d, fz_lb)[0] / np.radians(d))

    @property
    def max_slip(self):
        return float(self._slips[-1])

    # ------------------------------------------------------------------ io
    def to_dict(self):
        return dict(name=self.name, size=self.size, od_in=self.od_in, width_in=self.width_in,
                    pressure_psi=self.pressure_psi, loads_lb=self.loads_lb.tolist(),
                    slip_deg=self.slip_deg.tolist(), fy_lb=self.fy_lb.tolist())

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
