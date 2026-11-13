"""Table-driven tire cornering model.

Each tire type holds three vertical-load cases (rated-, rated, rated+) and a
cornering-force table Fy(slip) for each, slip 0..18 deg.

Blank cells are allowed (stored as NaN): each load case's curve is built only
from the slip points it has a value for.

Each load case also carries a relaxation length (ft). relax(Fz) interpolates
linearly between the load cases and holds the end values outside them.
All three = 0 means no lag (slip follows the geometry instantly).

Lookup for an arbitrary (slip, Fz):
  * slip  : linear interpolation in that load case's points, symmetric for
            negative slip, held constant beyond its last slip point.
  * Fz    : linear interpolation between load cases, with an implicit
            (Fz = 0, Fy = 0) case below the lowest load and linear
            extrapolation above the highest load.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_SLIPS = np.arange(0.0, 18.0 + 1e-9, 3.0)  # 0, 3, ... 18 deg
LOAD_CASE_NAMES = ("Rated -", "Rated", "Rated +")


def default_curves(loads, slips=DEFAULT_SLIPS, fy_frac_at_ref=0.15, ref_slip_deg=6.0):
    """Placeholder curves: Fy = frac * Fz at ref slip, linear through zero."""
    loads = np.asarray(loads, float)
    slips = np.asarray(slips, float)
    return np.outer(loads, fy_frac_at_ref * slips / ref_slip_deg)


def default_relaxation(od_in, n_loads=3, frac_of_od=0.67, spread=0.12):
    """Placeholder relaxation lengths (ft): 0.67 x tire OD at rated load, -/+12 % at rated -/+.

    Only an estimate; replace with measured values when available.
    """
    rated = frac_of_od * od_in / 12.0
    if n_loads == 3:
        return rated * np.array([1 - spread, 1.0, 1 + spread])
    return np.full(n_loads, rated)


@dataclass
class TireData:
    name: str
    size: str = ""
    od_in: float = 40.0
    width_in: float = 15.0
    pressure_psi: float = 200.0
    loads_lb: np.ndarray = field(default_factory=lambda: np.array([0.75, 1.0, 1.25]) * 40000)
    slip_deg: np.ndarray = field(default_factory=lambda: DEFAULT_SLIPS.copy())
    fy_lb: np.ndarray | None = None  # shape (3, n_slip); NaN = blank cell
    relax_ft: np.ndarray | None = None  # relaxation length per load case (ft), same order as loads_lb

    def __post_init__(self):
        self.loads_lb = np.asarray(self.loads_lb, float)
        self.slip_deg = np.asarray(self.slip_deg, float)
        if self.fy_lb is None:
            self.fy_lb = default_curves(self.loads_lb, self.slip_deg)
        self.fy_lb = np.array(self.fy_lb, dtype=float)  # None -> NaN
        if self.relax_ft is None:
            self.relax_ft = default_relaxation(self.od_in, len(self.loads_lb))
        self.relax_ft = np.asarray(self.relax_ft, float)
        self.prepare()

    # ------------------------------------------------------------------ setup
    def prepare(self):
        if self.fy_lb.shape != (len(self.loads_lb), len(self.slip_deg)):
            raise ValueError(f"{self.name}: Fy table shape does not match loads x slips")
        if len(self.slip_deg) == 0:
            raise ValueError(f"{self.name}: no slip points entered")
        if np.any(~np.isfinite(self.slip_deg)) or np.any(np.diff(self.slip_deg) <= 0):
            raise ValueError(f"{self.name}: slip angles must be distinct and increasing")
        order = np.argsort(self.loads_lb)
        loads = self.loads_lb[order]
        if loads[0] <= 0 or np.any(np.diff(loads) <= 0):
            raise ValueError(f"{self.name}: load cases must be positive and distinct")
        curves = [(np.array([0.0, 1.0]), np.array([0.0, 0.0]))]  # implicit Fz = 0 case
        for j in order:
            ok = np.isfinite(self.fy_lb[j])
            if not ok.any():
                raise ValueError(f"{self.name}: load case {self.loads_lb[j]:,.0f} lb has no Fy values")
            s, f = self.slip_deg[ok], self.fy_lb[j][ok]
            if s[0] > 0:  # make sure the curve passes through the origin
                s, f = np.concatenate([[0.0], s]), np.concatenate([[0.0], f])
            curves.append((s, f))
        self._curves = curves
        self._loads = np.concatenate([[0.0], loads])
        if self.relax_ft.shape != self.loads_lb.shape or np.any(~np.isfinite(self.relax_ft)):
            raise ValueError(f"{self.name}: enter a relaxation length for every load case")
        if np.any(self.relax_ft < 0):
            raise ValueError(f"{self.name}: relaxation lengths cannot be negative")
        self.has_lag = bool(np.any(self.relax_ft > 0))
        if self.has_lag and np.any(self.relax_ft <= 0):
            raise ValueError(f"{self.name}: relaxation lengths must be all positive, or all 0 for no lag")
        self._relax_loads = loads
        self._relax = self.relax_ft[order]

    def curve(self, j):
        """(slips, Fy) points actually used for load case j (in entered order)."""
        ok = np.isfinite(self.fy_lb[j])
        return self.slip_deg[ok], self.fy_lb[j][ok]

    # ------------------------------------------------------------------ lookup
    def fy(self, alpha_deg, fz_lb):
        """Signed lateral force (lb) for slip angle(s) alpha_deg at load(s) fz_lb."""
        alpha = np.atleast_1d(np.asarray(alpha_deg, float))
        fz = np.broadcast_to(np.atleast_1d(np.asarray(fz_lb, float)), alpha.shape)
        a = np.abs(alpha)
        vals = np.array([np.interp(a, s, f) for s, f in self._curves])  # (n_load, n)
        L = self._loads
        idx = np.clip(np.searchsorted(L, fz, side="right") - 1, 0, len(L) - 2)
        w = (fz - L[idx]) / (L[idx + 1] - L[idx])
        cols = np.arange(len(a))
        f = vals[idx, cols] * (1 - w) + vals[idx + 1, cols] * w
        f = np.where(fz > 0, f, 0.0)
        return np.sign(alpha) * f

    def relax(self, fz_lb):
        """Relaxation length (ft) at vertical load(s) fz: linear between load cases, ends held."""
        return np.interp(fz_lb, self._relax_loads, self._relax)

    @property
    def min_relax(self):
        return float(self._relax.min())

    def cornering_stiffness(self, fz_lb):
        """Approximate initial slope dFy/dalpha (lb/rad) at load fz."""
        d = 0.5
        return float(self.fy(d, fz_lb)[0] / np.radians(d))

    @property
    def max_slip(self):
        """Largest slip covered by every load case (forces are held constant beyond each curve's end)."""
        return float(min(s[-1] for s, _ in self._curves[1:]))

    # ------------------------------------------------------------------ io
    def to_dict(self):
        fy = [[None if not np.isfinite(v) else float(v) for v in row] for row in self.fy_lb]
        return dict(name=self.name, size=self.size, od_in=self.od_in, width_in=self.width_in,
                    pressure_psi=self.pressure_psi, loads_lb=self.loads_lb.tolist(),
                    slip_deg=self.slip_deg.tolist(), fy_lb=fy, relax_ft=self.relax_ft.tolist())

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
