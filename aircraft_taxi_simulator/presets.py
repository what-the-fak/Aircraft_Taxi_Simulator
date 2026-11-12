"""Aircraft presets (the "aircraft" dropdown).

A preset picks a gear layout (layouts.py), fills its dimensions, and supplies
weight / CG / tire / outline defaults. Everything remains editable in the GUI.

HOW TO ADD AN AIRCRAFT
----------------------
Append an AircraftPreset to PRESETS (or call ``add_preset``). ``layout`` must be
the exact name of a registered GearLayout and ``layout_params`` uses that
layout's Param keys (in the Param's own unit, ft or in).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .layouts import LAYOUTS


@dataclass
class TireDefaults:
    size: str
    od_in: float
    width_in: float
    pressure_psi: float
    rated_lb: float
    minus_frac: float = 0.75  # rated- = minus_frac * rated
    plus_frac: float = 1.25   # rated+ = plus_frac * rated
    fy_frac_at_6deg: float = 0.15

    @property
    def loads(self):
        return [self.minus_frac * self.rated_lb, self.rated_lb, self.plus_frac * self.rated_lb]


@dataclass
class Outline:
    """Planform used only for the animation / wingtip-radius output (gear-frame ft)."""
    length_ft: float
    span_ft: float
    fus_width_ft: float
    nose_ahead_ft: float      # fuselage nose tip ahead of the nose gear axle
    wing_root_le_ft: float    # wing root leading edge, ahead of the main-gear reference
    wing_root_chord_ft: float
    wing_sweep_deg: float = 28.0
    n_engines_per_side: int = 1


@dataclass
class AircraftPreset:
    name: str
    layout: str
    layout_params: dict
    weight_lb: float
    cg_fwd_ft: float          # CG ahead of main-gear reference at the most-forward-CG case
    cg_aft_ft: float          # ... at the most-aft-CG case
    cg_height_ft: float
    kz_ft: float              # yaw radius of gyration
    nose_tire: TireDefaults
    main_tire: TireDefaults
    outline: Outline
    max_steer_deg: float = 78.0
    notes: str = ""

    def __post_init__(self):
        if self.layout not in LAYOUTS:
            raise KeyError(f"Preset {self.name!r} refers to unknown layout {self.layout!r}")


PRESETS: dict[str, AircraftPreset] = {}


def add_preset(p: AircraftPreset):
    PRESETS[p.name] = p
    return p


# --------------------------------------------------------------------------- 737 MAX family
# Source: Boeing 737 MAX Airport Planning doc (images supplied by user):
#   gear footprint drawing + sec 7.3.1 maximum pavement loads.
# CG positions are back-calculated from the static loads in that table:
#   fwd CG:  x_cg = V_NG(static, fwd CG) * WB / W
#   aft CG:  x_cg = (W - 2 * V_MG(per strut, aft CG)) * WB / W
# Rated tire loads default to half the max static strut load (replace with the
# tire manufacturer's rated load). CG height and kz are ESTIMATES.

_737_LAYOUT = "Twin nose + 2 twin-wheel mains (737 / A320 type)"


def _737(name, wb_ft, W, vng_fwd, vmg_aft, nose_psi, main_psi, main_ply, length_ft, kz_ft):
    return AircraftPreset(
        name=name,
        layout=_737_LAYOUT,
        layout_params=dict(wheelbase=wb_ft, main_track=18.75, main_wheel_spacing=34.0, nose_wheel_spacing=16.0),
        weight_lb=W,
        cg_fwd_ft=vng_fwd * wb_ft / W,
        cg_aft_ft=(W - 2 * vmg_aft) * wb_ft / W,
        cg_height_ft=10.0,
        kz_ft=kz_ft,
        nose_tire=TireDefaults("27x7.75R15, 12PR", 27.0, 7.75, nose_psi,
                               rated_lb=math.ceil(vng_fwd / 2 / 100) * 100),
        main_tire=TireDefaults(f"H44.5x16.5R21, {main_ply}PR", 44.5, 16.5, main_psi,
                               rated_lb=math.ceil(vmg_aft / 2 / 100) * 100),
        outline=Outline(length_ft=length_ft, span_ft=117 + 10 / 12, fus_width_ft=12.3,
                        nose_ahead_ft=13.0, wing_root_le_ft=18.0, wing_root_chord_ft=24.0),
        max_steer_deg=78.0,
        notes="Gear geometry, weights and pavement loads from Boeing 737 MAX ACAP; "
              "CG height, kz, outline and tire rated loads are estimates.",
    )


add_preset(_737("737-7", 43 + 10 / 12, 177_500, 18_918, 82_866, 200, 200, 30, 116 + 8 / 12, 24.0))
add_preset(_737("737-8", 51 + 2 / 12, 182_700, 15_894, 85_258, 185, 207, 30, 129 + 8 / 12, 26.0))
add_preset(_737("737-8-200 / BBJ MAX 8", 51 + 2 / 12, 181_700, 15_807, 84_791, 185, 205, 30, 129 + 8 / 12, 26.0))
add_preset(_737("737-9 / BBJ MAX 9", 56 + 4 / 12, 195_200, 15_514, 91_868, 165, 225, 32, 138 + 4 / 12, 27.5))
add_preset(_737("737-10", 60 + 2 / 12, 198_400, 13_613, 93_679, 185, 230, 32, 143 + 8 / 12, 28.5))

# --------------------------------------------------------------------------- 777X
# Source: Boeing 777-9 Airport Planning doc (images supplied by user): fig 7-1 footprint,
# table 7-3 max pavement loads. Wheelbase 106 ft 1 in to the middle main axle; main track 35 ft 6 in
# (41 ft 10 in over tire outer edges); lateral wheel spacing 55 in; axles 57.2 in fwd-mid and
# 58.0 in mid-aft (115.2 in total); nose wheels 30.8 in apart.
# Span uses the FOLDED wingtip (212 ft 9 in) since that is the taxi configuration.
# CG height, kz, the planform and rated tire loads are ESTIMATES (same method as 737).
# Aft-axle steering: public descriptions give up to 8 deg, active above ~13 deg nose steer and
# opposite to the nose. The ramp between (linear to full at 70 deg nose) is an ASSUMED schedule.
_W9, _WB9, _VNG9, _VMG9 = 777_000, 106 + 1 / 12, 68_796, 366_337
add_preset(AircraftPreset(
    name="777-9",
    layout="Twin nose + 2 six-wheel bogies (777 type)",
    layout_params=dict(wheelbase=_WB9, main_track=35.5, main_wheel_spacing=55.0, axle_spacing=57.2,
                       aft_axle_spacing=58.0, nose_wheel_spacing=30.8,
                       aft_steer_max=8.0, aft_steer_start=13.0, aft_steer_full=70.0),
    weight_lb=_W9,
    cg_fwd_ft=_VNG9 * _WB9 / _W9,
    cg_aft_ft=(_W9 - 2 * _VMG9) * _WB9 / _W9,
    cg_height_ft=17.0, kz_ft=52.0,
    nose_tire=TireDefaults("43x17.5R17, 32PR", 43.0, 17.5, 218, rated_lb=math.ceil(_VNG9 / 2 / 100) * 100),
    main_tire=TireDefaults("52x21.0R22, 38PR", 52.0, 21.0, 224, rated_lb=math.ceil(_VMG9 / 6 / 100) * 100),
    outline=Outline(length_ft=251 + 9 / 12, span_ft=212 + 9 / 12, fus_width_ft=20.3, nose_ahead_ft=20.0,
                    wing_root_le_ft=34.0, wing_root_chord_ft=46.0, wing_sweep_deg=32.0),
    max_steer_deg=70.0,
    notes="Gear geometry, weight and pavement loads from Boeing 777-9 ACAP; folded-wingtip span; "
          "CG height, kz, outline and tire rated loads are estimates. Aft-axle steering schedule "
          "(8 deg max above 13 deg nose, linear to 70 deg) is an assumed linear ramp.",
))

# --------------------------------------------------------------------------- illustrative examples
# NOT real aircraft data: placeholder values so the other layouts can be exercised.
add_preset(AircraftPreset(
    name="Example: twin-aisle, 4-wheel bogies (illustrative values)",
    layout="Twin nose + 2 four-wheel bogies (767 / A330 type)",
    layout_params=dict(wheelbase=72.0, main_track=31.0, main_wheel_spacing=54.0, axle_spacing=64.0,
                       nose_wheel_spacing=24.0),
    weight_lb=400_000, cg_fwd_ft=7.2, cg_aft_ft=3.6, cg_height_ft=14.0, kz_ft=40.0,
    nose_tire=TireDefaults("40x15.5R16 (example)", 40.0, 15.5, 180, 20_000),
    main_tire=TireDefaults("46x17R20 (example)", 46.0, 17.0, 200, 46_000),
    outline=Outline(190, 190, 18.5, 18.0, 26.0, 36.0, 31.0),
    max_steer_deg=70.0, notes="Illustrative placeholder values only.",
))
add_preset(AircraftPreset(
    name="Example: twin-aisle, 6-wheel bogies (illustrative values)",
    layout="Twin nose + 2 six-wheel bogies (777 type)",
    layout_params=dict(wheelbase=85.0, main_track=36.0, main_wheel_spacing=55.0, axle_spacing=57.0,
                       aft_axle_spacing=57.0, nose_wheel_spacing=26.0),
    weight_lb=650_000, cg_fwd_ft=8.5, cg_aft_ft=4.2, cg_height_ft=16.0, kz_ft=50.0,
    nose_tire=TireDefaults("43x17.5R17 (example)", 43.0, 17.5, 200, 32_500),
    main_tire=TireDefaults("52x21R22 (example)", 52.0, 21.0, 220, 50_000),
    outline=Outline(240, 210, 20.3, 20.0, 30.0, 45.0, 31.5),
    max_steer_deg=70.0, notes="Illustrative placeholder values only.",
))
add_preset(AircraftPreset(
    name="Example: four-post widebody (illustrative values)",
    layout="Twin nose + 2 wing + 2 body bogies (747 type)",
    layout_params=dict(wheelbase=84.0, wing_track=36.0, body_track=12.75, body_aft=10.0,
                       main_wheel_spacing=44.0, axle_spacing=58.0, nose_wheel_spacing=26.0),
    weight_lb=800_000, cg_fwd_ft=3.0, cg_aft_ft=0.0, cg_height_ft=17.0, kz_ft=55.0,
    nose_tire=TireDefaults("49x19R22 (example)", 49.0, 19.0, 200, 32_000),
    main_tire=TireDefaults("52x21R22 (example)", 52.0, 21.0, 220, 46_000),
    outline=Outline(230, 210, 21.0, 22.0, 32.0, 48.0, 37.5, n_engines_per_side=2),
    max_steer_deg=70.0, notes="Illustrative placeholder values only.",
))
