"""Planar rigid-body ground-handling model (ft, lb, slug, s).

Body axes at the CG: x forward, y left, yaw rate r positive = nose left (CCW).
States: X, Y, psi (earth), v (body lateral velocity), r, distance s, and one
lagged slip angle per wheel (first-order relaxation length).
Forward speed u is held constant (thrust assumed to balance tire drag).

Slip angle convention (per wheel, in the wheel's own frame):
    alpha = delta_wheel - atan2(v_lat, v_long)   (SAE-like)
    alpha > 0  ->  lateral tire force to the LEFT (+y).
In a steady LEFT turn the nose and main tires therefore show positive slip.

Vertical loads: rigid airframe on equal per-wheel tire springs (a plane of
deflection). Exact statics for a 3-strut aircraft; shares load sensibly for
statically indeterminate layouts (bogies, 4-post). Includes lateral load
transfer from centripetal acceleration (CG height). No longitudinal transfer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .layouts import Strut
from .tire import TireData

G = 32.174           # ft/s^2
KT_TO_FPS = 1.6878098571


@dataclass
class Wheel:
    name: str
    strut: str
    tire: str
    px: float  # strut pivot, gear frame (ft)
    py: float
    ox: float  # offset about the steering pivot (ft)
    oy: float
    steer_ratio: float
    sx: float = 0.0  # steering pivot, gear frame (ft): strut pivot, or axle centre for a steered axle
    sy: float = 0.0
    axle_steer: object = None  # optional callable: nose command (rad) -> axle steer angle (rad)


class AircraftModel:
    def __init__(self, struts: list[Strut], weight_lb, x_cg, y_cg, h_cg, kz,
                 tires: dict[str, TireData], sigma: dict[str, float]):
        self.struts = struts
        self.W = float(weight_lb)
        self.m = self.W / G
        self.Iz = self.m * kz ** 2
        self.x_cg, self.y_cg, self.h_cg = float(x_cg), float(y_cg), float(h_cg)
        self.tires = tires
        self.wheels: list[Wheel] = []
        for s in struts:
            for (ox, oy), nm in zip(s.offsets, s.wheel_names):
                fn = None
                for dx_axle, f in (s.axle_steer or {}).items():
                    if abs(ox - dx_axle) < 1e-6:
                        fn = f
                if fn is None:   # rotates about the strut pivot (nose gear) or is fixed
                    self.wheels.append(Wheel(nm, s.name, s.tire, s.x, s.y, ox, oy, s.steer_ratio, s.x, s.y))
                else:            # steered axle: rotates about the axle centre
                    self.wheels.append(Wheel(nm, s.name, s.tire, s.x, s.y, 0.0, oy, s.steer_ratio,
                                             s.x + ox, s.y, fn))
        w = self.wheels
        self.n = len(w)
        self.names = [wh.name for wh in w]
        self.px = np.array([wh.px for wh in w]) - self.x_cg   # CG frame
        self.py = np.array([wh.py for wh in w]) - self.y_cg
        self.sx = np.array([wh.sx for wh in w]) - self.x_cg   # steering pivots, CG frame
        self.sy = np.array([wh.sy for wh in w]) - self.y_cg
        self.ox = np.array([wh.ox for wh in w])
        self.oy = np.array([wh.oy for wh in w])
        self.ratio = np.array([wh.steer_ratio for wh in w])
        self.tire_key = [wh.tire for wh in w]
        self.masks = {k: np.array([t == k for t in self.tire_key]) for k in set(self.tire_key)}
        groups = {}
        for i, wh in enumerate(w):
            if wh.axle_steer is not None:
                groups.setdefault(id(wh.axle_steer), [wh.axle_steer, np.zeros(self.n, bool)])[1][i] = True
        self._axle_groups = [tuple(g) for g in groups.values()]
        self.sigma = np.array([sigma.get(t, 0.0) for t in self.tire_key])
        for k in self.masks:
            if k not in tires:
                raise KeyError(f"No tire data for tire type {k!r}")

    # ------------------------------------------------------------------ geometry
    def geometry(self, delta):
        d = self.ratio * delta
        if self._axle_groups:
            d = d.copy()
            for fn, mask in self._axle_groups:
                d[mask] = fn(delta)
        c, s = np.cos(d), np.sin(d)
        x = self.sx + c * self.ox - s * self.oy
        y = self.sy + s * self.ox + c * self.oy
        return x, y, d

    def axle_steer(self, delta):
        """Steer angles (rad) of steered main axles for nose command delta, keyed by first wheel name."""
        return {self.names[int(np.argmax(m))]: fn(delta) for fn, m in self._axle_groups}

    def strut_points(self):
        """Strut pivot positions in CG frame, keyed by strut name."""
        return {s.name: (s.x - self.x_cg, s.y - self.y_cg) for s in self.struts}

    def ref_point(self, which: str):
        if which == "CG":
            return 0.0, 0.0
        if which == "Nose gear":
            steer = [s for s in self.struts if s.steer_ratio != 0] or self.struts[:1]
            return steer[0].x - self.x_cg, steer[0].y - self.y_cg
        if which == "Main gear centroid":
            mm = ~self._steer_mask()
            return float(np.mean(self.px[mm])), float(np.mean(self.py[mm]))
        raise ValueError(which)

    def _steer_mask(self):
        return self.ratio != 0

    # ------------------------------------------------------------------ loads
    def vertical_loads(self, x, y, ay):
        """Per-wheel Fz (lb). ay = lateral acceleration at CG (ft/s^2, +left).

        Bogies are free to pitch about their pivot, so every wheel's fore/aft
        position for load sharing is its strut pivot x (load per axle equal
        within a bogie, resultant at the pivot). Lateral positions are actual.
        """
        x = self.px
        active = np.ones(self.n, bool)
        target = np.array([self.W, 0.0, -self.h_cg * self.m * ay])
        fz = np.zeros(self.n)
        for _ in range(self.n):
            A = np.vstack([np.ones(active.sum()), x[active], y[active]]).T  # (k,3)
            coef, *_ = np.linalg.lstsq(A.T @ A, target, rcond=None)
            fz[:] = 0.0
            fz[active] = A @ coef
            neg = active & (fz < 0)
            if not neg.any():
                break
            active &= ~neg
        return np.maximum(fz, 0.0)

    # ------------------------------------------------------------------ kinematics / forces
    def kinematics(self, u, v, r, delta):
        x, y, dw = self.geometry(delta)
        vx = u - r * y
        vy = v + r * x
        c, s = np.cos(dw), np.sin(dw)
        vxw = c * vx + s * vy
        vyw = -s * vx + c * vy
        alpha = -np.arctan2(vyw, vxw)
        speed = np.hypot(vx, vy)
        return x, y, dw, alpha, speed

    def tire_fy(self, alpha_rad, fz):
        fy = np.zeros(self.n)
        a_deg = np.degrees(alpha_rad)
        for k, mk in self.masks.items():
            fy[mk] = self.tires[k].fy(a_deg[mk], fz[mk])
        return fy

    def body_loads(self, x, y, dw, fyw):
        fxb = -np.sin(dw) * fyw
        fyb = np.cos(dw) * fyw
        return fxb.sum(), fyb.sum(), float(np.sum(x * fyb - y * fxb)), fxb, fyb

    def steady_eval(self, u, v, r, delta):
        x, y, dw, alpha, speed = self.kinematics(u, v, r, delta)
        fz = self.vertical_loads(x, y, u * r)
        fyw = self.tire_fy(alpha, fz)
        Fx, Fy, Mz, fxb, fyb = self.body_loads(x, y, dw, fyw)
        return dict(x=x, y=y, dw=dw, alpha=alpha, fz=fz, fy=fyw, Fx=Fx, Fy=Fy, Mz=Mz)

    def total_cornering_stiffness(self):
        fz = self.vertical_loads(*self.geometry(0.0)[:2], 0.0)
        return sum(self.tires[k].cornering_stiffness(f) for k, f in zip(self.tire_key, fz))


# ============================================================================ steady state
@dataclass
class SteadyState:
    ok: bool
    message: str
    u: float
    v: float
    r: float
    delta: float          # nose steer command (rad)
    alpha: np.ndarray     # per wheel (rad)
    fz: np.ndarray
    fy: np.ndarray
    x: np.ndarray
    y: np.ndarray
    dw: np.ndarray
    Fx: float
    turn_center: tuple    # CG frame (ft)
    thrust_lb: float      # longitudinal force needed to hold speed
    ay: float


def kinematic_guess(model: AircraftModel, u, R, direction, ref):
    """Zero-slip mains guess: turn center on the main-gear centroid line."""
    xm, _ = model.ref_point("Main gear centroid")
    xp, yp = ref
    rc = np.sqrt(max(R ** 2 - (xp - xm) ** 2, (0.05 * R) ** 2))
    yc = direction * rc
    r = u / yc
    v = -r * xm
    xn, yn = model.ref_point("Nose gear")
    delta = np.arctan2(xn - xm, yc - yn) if direction > 0 else -np.arctan2(xn - xm, -(yc - yn))
    return np.array([v, r, delta])


def solve_steady(model: AircraftModel, u, R, direction=+1, ref_name="CG", tol=1e-9, max_iter=80):
    ref = model.ref_point(ref_name)
    xp, yp = ref
    W, L = model.W, max(1.0, float(np.ptp(model.px)))

    def resid(z):
        v, r, d = z
        e = model.steady_eval(u, v, r, d)
        vp = np.hypot(u - r * yp, v + r * xp)
        return np.array([(e["Fy"] - model.m * u * r) / W,
                         e["Mz"] / (W * L),
                         (direction * r * R - vp) / u])

    z = kinematic_guess(model, u, R, direction, ref)
    f = resid(z)
    steps = np.array([1e-4 * u, 1e-6 + 1e-4 * abs(z[1]), 1e-6])
    ok = False
    for _ in range(max_iter):
        J = np.empty((3, 3))
        for j in range(3):
            dz = np.zeros(3)
            dz[j] = steps[j]
            J[:, j] = (resid(z + dz) - resid(z - dz)) / (2 * steps[j])
        try:
            step = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(J, -f, rcond=None)[0]
        lam, n0 = 1.0, np.linalg.norm(f)
        while lam > 1e-4:
            z_new = z + lam * step
            f_new = resid(z_new)
            if np.linalg.norm(f_new) < n0 or lam < 2e-4:
                break
            lam *= 0.5
        z, f = z_new, f_new
        if np.linalg.norm(f) < tol:
            ok = True
            break
    v, r, d = z
    e = model.steady_eval(u, v, r, d)
    xc, yc = -v / r, u / r
    thrust = -e["Fx"] - model.m * v * r
    msg = "converged" if ok else f"did not fully converge (residual {np.linalg.norm(f):.2e})"
    return SteadyState(ok, msg, u, v, r, d, e["alpha"], e["fz"], e["fy"], e["x"], e["y"], e["dw"],
                       e["Fx"], (xc, yc), thrust, u * r)


# ============================================================================ transient
@dataclass
class SimResult:
    t: np.ndarray
    s: np.ndarray            # CG distance travelled (ft)
    delta: np.ndarray        # nose steer command (rad)
    alpha: np.ndarray        # (N, n) lagged slip used for force (rad)
    alpha_kin: np.ndarray    # (N, n) kinematic (geometric) slip (rad)
    fz: np.ndarray
    fy: np.ndarray
    v: np.ndarray
    r: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    psi: np.ndarray
    radius_ref: np.ndarray   # instantaneous path radius of the reference point (ft)
    steer_start_s: float
    steer_end_s: float
    steady_s: float | None   # distance where the whole aircraft reached steady state
    settle_s: np.ndarray     # per-wheel distance to settle within band
    names: list = field(default_factory=list)


def simulate(model: AircraftModel, u, ss: SteadyState, steer_rate_rad, lead_in_ft, max_dist_ft,
             ref_name="CG", psi0=np.pi / 2, band_frac=0.02, band_abs_deg=0.02):
    n = model.n
    xp, yp = model.ref_point(ref_name)
    sig = model.sigma
    lag = sig > 1e-6
    d_target = ss.delta
    t0 = lead_in_ft / u
    t_ramp = abs(d_target) / max(steer_rate_rad, 1e-9)

    # time step: resolve tire relaxation and lateral (v) time constant
    ctot = max(model.total_cornering_stiffness(), 1.0)
    dt = min(0.01, 0.1 * model.m * u / ctot)
    if lag.any():
        dt = min(dt, 0.2 * sig[lag].min() / u)
    dt = max(dt, 1e-4)

    def steer(t):
        if t <= t0:
            return 0.0
        return np.sign(d_target) * min(steer_rate_rad * (t - t0), abs(d_target))

    def deriv(t, z):
        X, Y, psi, v, r, s = z[:6]
        al = z[6:]
        d = steer(t)
        x, y, dw, ak, spd = model.kinematics(u, v, r, d)
        a_used = np.where(lag, al, ak)
        fz = model.vertical_loads(x, y, u * r)
        fyw = model.tire_fy(a_used, fz)
        Fx, Fy, Mz, _, _ = model.body_loads(x, y, dw, fyw)
        dal = np.where(lag, spd / np.where(lag, sig, 1.0) * (ak - al), 0.0)
        c, sn = np.cos(psi), np.sin(psi)
        dz = np.empty_like(z)
        dz[0] = u * c - v * sn
        dz[1] = u * sn + v * c
        dz[2] = r
        dz[3] = Fy / model.m - u * r
        dz[4] = Mz / model.Iz
        dz[5] = np.hypot(u, v)
        dz[6:] = dal
        return dz, (d, a_used, ak, fz, fyw)

    z = np.zeros(6 + n)
    z[2] = psi0
    t = 0.0
    rec = {k: [] for k in ("t", "s", "d", "a", "ak", "fz", "fy", "v", "r", "X", "Y", "psi")}
    steady_s = None
    stop_s = max_dist_ft
    a_ss = ss.alpha
    a_tol = np.maximum(np.radians(0.005), 0.002 * np.abs(a_ss))
    max_steps = int(2e6)
    for _ in range(max_steps):
        k1, aux = deriv(t, z)
        d, a_used, ak, fz, fyw = aux
        for key, val in (("t", t), ("s", z[5]), ("d", d), ("a", a_used), ("ak", ak), ("fz", fz),
                         ("fy", fyw), ("v", z[3]), ("r", z[4]), ("X", z[0]), ("Y", z[1]), ("psi", z[2])):
            rec[key].append(val.copy() if isinstance(val, np.ndarray) else val)
        if z[5] >= stop_s:
            break
        if steady_s is None and t > t0 + t_ramp:
            if np.all(np.abs(a_used - a_ss) < a_tol) and abs(z[4] - ss.r) < 1e-3 * abs(ss.r):
                steady_s = z[5]
                stop_s = min(max_dist_ft, steady_s + max(40.0, 0.25 * steady_s))
        k2, _ = deriv(t + dt / 2, z + dt / 2 * k1)
        k3, _ = deriv(t + dt / 2, z + dt / 2 * k2)
        k4, _ = deriv(t + dt, z + dt * k3)
        z = z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt

    A = {k: np.array(v) for k, v in rec.items()}
    v_arr, r_arr = A["v"], A["r"]
    vp = np.hypot(u - r_arr * yp, v_arr + r_arr * xp)
    with np.errstate(divide="ignore", invalid="ignore"):
        rad = np.where(np.abs(r_arr) > 1e-9, vp / np.abs(r_arr), np.inf)

    # per-wheel settle distance: last time outside the band around steady state
    band = np.maximum(np.radians(band_abs_deg), band_frac * np.abs(a_ss))
    out = np.abs(A["a"] - a_ss) > band
    settle = np.empty(n)
    for j in range(n):
        idx = np.nonzero(out[:, j])[0]
        if len(idx) == 0:
            settle[j] = 0.0
        elif idx[-1] + 1 < len(A["s"]):
            settle[j] = A["s"][idx[-1] + 1]
        else:
            settle[j] = np.nan  # never settled within simulated distance
    s_arr = A["s"]
    t_arr = A["t"]
    steer_start_s = float(np.interp(t0, t_arr, s_arr))
    steer_end_s = float(np.interp(t0 + t_ramp, t_arr, s_arr)) if t0 + t_ramp <= t_arr[-1] else np.nan
    return SimResult(t_arr, s_arr, A["d"], A["a"], A["ak"], A["fz"], A["fy"], v_arr, r_arr,
                     A["X"], A["Y"], A["psi"], rad, steer_start_s, steer_end_s, steady_s, settle,
                     names=list(model.names))
