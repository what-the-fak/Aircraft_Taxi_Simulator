"""Landing-gear layout definitions (the "gear configuration" dropdown).

Every layout is a GearLayout: a list of user-editable dimensions (Param) plus a
builder function that turns those dimensions into a list of Strut objects.

Gear-frame coordinates (used everywhere a layout is defined):
    x  : forward (+) along the aircraft centerline, ft
    y  : left (+), ft
    origin : on the centerline, on the "main-gear reference line" described by
             each layout's ``origin_note`` (e.g. the main axle line for a 737).
The CG location the user enters is measured forward of this same origin.

HOW TO ADD A NEW GEAR CONFIGURATION
-----------------------------------
1.  Write a builder ``def _my_layout(p): ...`` that receives a dict of the
    layout's params (already converted to ft) and returns a list of Strut.
    Use the helpers ``side_by_side`` and ``bogie`` for wheel patterns.
2.  Call ``register(GearLayout(name=..., params=[Param(...), ...], builder=_my_layout))``.
    It appears in the dropdown automatically (in registration order).
3.  Optionally add an aircraft preset that uses it in presets.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Param:
    key: str
    label: str
    unit: str = "ft"  # "ft" or "in" (unit the user types the value in)
    default: float = 0.0

    def to_ft(self, value: float) -> float:
        """Convert to model units: inches -> ft; ft, deg and dimensionless pass through."""
        return value / 12.0 if self.unit == "in" else float(value)


@dataclass
class Strut:
    """One gear leg. Wheels are placed at ``offsets`` (ft) about the strut pivot
    and rotate about that pivot when the strut is steered."""
    name: str
    x: float
    y: float
    tire: str  # key into the tire data set: "nose" or "main"
    offsets: list[tuple[float, float]]
    steer_ratio: float = 0.0  # wheel steer angle = steer_ratio * nose steer command
    wheel_names: list[str] | None = None
    # Steered axles inside this strut (e.g. 777 aft axle): {axle dx offset: fn(nose cmd rad) -> axle angle rad}.
    # Those wheels rotate about the axle centre instead of the strut pivot.
    axle_steer: dict | None = None

    def __post_init__(self):
        if self.wheel_names is None:
            self.wheel_names = _auto_wheel_names(self)


@dataclass
class GearLayout:
    name: str
    description: str
    origin_note: str
    params: list[Param]
    builder: Callable[[dict[str, float]], list[Strut]]

    def defaults(self) -> dict[str, float]:
        return {p.key: p.default for p in self.params}

    def build(self, values: dict[str, float]) -> list[Strut]:
        ft = {p.key: p.to_ft(values.get(p.key, p.default)) for p in self.params}
        return self.builder(ft)


LAYOUTS: dict[str, GearLayout] = {}


def register(layout: GearLayout) -> GearLayout:
    LAYOUTS[layout.name] = layout
    return layout


# --------------------------------------------------------------------------- helpers
def side_by_side(spacing: float) -> list[tuple[float, float]]:
    """Two wheels on one axle, ``spacing`` = wheel centerline to centerline (ft)."""
    return [(0.0, spacing / 2.0), (0.0, -spacing / 2.0)]


def bogie(n_axles: int, axle_spacing: float, wheel_spacing: float) -> list[tuple[float, float]]:
    """Bogie with ``n_axles`` tandem axles (2 wheels each), centred on the pivot."""
    offs = []
    for i in range(n_axles):
        dx = (n_axles - 1) / 2.0 * axle_spacing - i * axle_spacing
        offs += [(dx, wheel_spacing / 2.0), (dx, -wheel_spacing / 2.0)]
    return offs


def steer_schedule(start_deg: float, full_deg: float, max_deg: float, opposite: bool = True):
    """Piecewise-linear axle steering vs nose steer command.

    0 until |nose| > start_deg, then linear up to max_deg at |nose| = full_deg, held beyond.
    opposite=True steers the axle against the nose direction (777-style aft axle).
    """
    start, full, amax = math.radians(start_deg), math.radians(max(full_deg, start_deg + 1e-3)), math.radians(max_deg)
    sgn = -1.0 if opposite else 1.0

    def fn(delta: float) -> float:
        a = abs(delta)
        if amax == 0 or a <= start:
            return 0.0
        return sgn * math.copysign(amax * min((a - start) / (full - start), 1.0), delta)
    return fn


def _auto_wheel_names(strut: Strut) -> list[str]:
    xs = sorted({round(o[0], 4) for o in strut.offsets}, reverse=True)
    ys = sorted({round(o[1], 4) for o in strut.offsets}, reverse=True)  # left first
    axle = {1: [""], 2: ["Fwd", "Aft"], 3: ["Fwd", "Mid", "Aft"]}.get(
        len(xs), [f"A{i + 1}" for i in range(len(xs))])
    if len(ys) == 1:
        lat = [""]
    elif len(ys) == 2:
        if strut.y > 1e-6:
            lat = ["Outbd", "Inbd"]
        elif strut.y < -1e-6:
            lat = ["Inbd", "Outbd"]
        else:
            lat = ["L", "R"]
    else:
        lat = [f"W{i + 1}" for i in range(len(ys))]
    names = []
    for dx, dy in strut.offsets:
        a = axle[xs.index(round(dx, 4))]
        b = lat[ys.index(round(dy, 4))]
        names.append(" ".join(s for s in (strut.name, a, b) if s))
    return names


# --------------------------------------------------------------------------- layouts
def _twin_nose_twin_mains(p):
    return [
        Strut("Nose", p["wheelbase"], 0.0, "nose", side_by_side(p["nose_wheel_spacing"]), steer_ratio=1.0),
        Strut("L Main", 0.0, p["main_track"] / 2, "main", side_by_side(p["main_wheel_spacing"])),
        Strut("R Main", 0.0, -p["main_track"] / 2, "main", side_by_side(p["main_wheel_spacing"])),
    ]


register(GearLayout(
    name="Twin nose + 2 twin-wheel mains (737 / A320 type)",
    description="Steerable 2-wheel nose gear, two 2-wheel main struts.",
    origin_note="x = 0 on the main axle line",
    params=[
        Param("wheelbase", "Wheelbase (nose axle to main axle)", "ft", 51 + 2 / 12),
        Param("main_track", "Main gear track (strut CL to CL)", "ft", 18.75),
        Param("main_wheel_spacing", "Main wheel spacing (tire CL to CL)", "in", 34.0),
        Param("nose_wheel_spacing", "Nose wheel spacing (tire CL to CL)", "in", 16.0),
    ],
    builder=_twin_nose_twin_mains,
))


def _twin_nose_bogies(n_axles):
    def build(p):
        offs = bogie(n_axles, p["axle_spacing"], p["main_wheel_spacing"])
        return [
            Strut("Nose", p["wheelbase"], 0.0, "nose", side_by_side(p["nose_wheel_spacing"]), steer_ratio=1.0),
            Strut("L Main", 0.0, p["main_track"] / 2, "main", offs),
            Strut("R Main", 0.0, -p["main_track"] / 2, "main", offs),
        ]
    return build


register(GearLayout(
    name="Twin nose + 2 four-wheel bogies (767 / A330 type)",
    description="Steerable 2-wheel nose gear, two 4-wheel (2x2) main bogies, no aft-axle steering.",
    origin_note="x = 0 on the main bogie pivot line",
    params=[
        Param("wheelbase", "Wheelbase (nose axle to bogie pivot)", "ft", 72.0),
        Param("main_track", "Main gear track (bogie CL to CL)", "ft", 31.0),
        Param("main_wheel_spacing", "Main wheel spacing (lateral, tire CL)", "in", 54.0),
        Param("axle_spacing", "Bogie axle spacing (fwd to aft axle)", "in", 64.0),
        Param("nose_wheel_spacing", "Nose wheel spacing (tire CL to CL)", "in", 24.0),
    ],
    builder=_twin_nose_bogies(2),
))

def _twin_nose_six_wheel(p):
    s1, s2, w = p["axle_spacing"], p["aft_axle_spacing"], p["main_wheel_spacing"]
    offs = [(dx, dy) for dx in (s1, 0.0, -s2) for dy in (w / 2, -w / 2)]
    fn = steer_schedule(p["aft_steer_start"], p["aft_steer_full"], p["aft_steer_max"])
    aft = {-s2: fn} if p["aft_steer_max"] > 0 else None
    return [
        Strut("Nose", p["wheelbase"], 0.0, "nose", side_by_side(p["nose_wheel_spacing"]), steer_ratio=1.0),
        Strut("L Main", 0.0, p["main_track"] / 2, "main", offs, axle_steer=aft),
        Strut("R Main", 0.0, -p["main_track"] / 2, "main", offs, axle_steer=aft),
    ]


register(GearLayout(
    name="Twin nose + 2 six-wheel bogies (777 type)",
    description="Steerable 2-wheel nose gear, two 6-wheel (3x2) main bogies with optional aft-axle steering "
                "(steers opposite to the nose; set max = 0 to lock the aft axle).",
    origin_note="x = 0 on the middle axle of the main bogies",
    params=[
        Param("wheelbase", "Wheelbase (nose axle to middle main axle)", "ft", 85.0),
        Param("main_track", "Main gear track (bogie CL to CL)", "ft", 36.0),
        Param("main_wheel_spacing", "Main wheel spacing (lateral, tire CL)", "in", 55.0),
        Param("axle_spacing", "Fwd axle to middle axle", "in", 57.0),
        Param("aft_axle_spacing", "Middle axle to aft axle", "in", 57.0),
        Param("nose_wheel_spacing", "Nose wheel spacing (tire CL to CL)", "in", 26.0),
        Param("aft_steer_max", "Aft-axle steer: max angle (0 = locked)", "deg", 0.0),
        Param("aft_steer_start", "Aft-axle steer: starts above nose angle", "deg", 13.0),
        Param("aft_steer_full", "Aft-axle steer: max reached at nose angle", "deg", 70.0),
    ],
    builder=_twin_nose_six_wheel,
))


def _four_post(p):
    offs_w = bogie(2, p["axle_spacing"], p["main_wheel_spacing"])
    xb = -p["body_aft"]
    return [
        Strut("Nose", p["wheelbase"], 0.0, "nose", side_by_side(p["nose_wheel_spacing"]), steer_ratio=1.0),
        Strut("L Wing", 0.0, p["wing_track"] / 2, "main", offs_w),
        Strut("R Wing", 0.0, -p["wing_track"] / 2, "main", offs_w),
        Strut("L Body", xb, p["body_track"] / 2, "main", offs_w),
        Strut("R Body", xb, -p["body_track"] / 2, "main", offs_w),
    ]


register(GearLayout(
    name="Twin nose + 2 wing + 2 body bogies (747 type)",
    description="Four 4-wheel main bogies (statically indeterminate: loads shared via equal tire stiffness).",
    origin_note="x = 0 on the wing-gear bogie pivot line",
    params=[
        Param("wheelbase", "Wheelbase (nose axle to wing-gear pivot)", "ft", 84.0),
        Param("wing_track", "Wing gear track (CL to CL)", "ft", 36.0),
        Param("body_track", "Body gear track (CL to CL)", "ft", 12.75),
        Param("body_aft", "Body gear pivot aft of wing gear pivot", "ft", 10.0),
        Param("main_wheel_spacing", "Main wheel spacing (lateral, tire CL)", "in", 44.0),
        Param("axle_spacing", "Bogie axle spacing", "in", 58.0),
        Param("nose_wheel_spacing", "Nose wheel spacing (tire CL to CL)", "in", 26.0),
    ],
    builder=_four_post,
))
