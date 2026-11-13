# Aircraft Taxi Simulator

Aircraft ground-turn tire slip model.

A simplified "vehicle dynamics" model for aircraft taxiing. Pick a gear configuration, enter the
geometry, weight/CG and tire cornering data, then give a turn radius. Aircraft Taxi Simulator finds the nose steer
angle you need and simulates the run from straight rolling to steady state. You get the slip angle,
Fy and Fz history for every tire, plus an animation.

## Run

* **macOS:** double-click `Aircraft_Taxi_Simulator.command`.
* **Windows:** double-click `Aircraft_Taxi_Simulator.bat`.

The first launch creates a private Python environment (`.venv` on Mac, `.venv-win` on Windows) and
installs numpy and matplotlib, so it needs internet once. It needs Python 3.10+ from python.org,
which includes tkinter. On Windows, tick "Add python.exe to PATH" during install. If macOS blocks
the `.command` file after it was downloaded, right-click it and choose Open, or run
`xattr -d com.apple.quarantine Aircraft_Taxi_Simulator.command`.

From a terminal: `.venv/bin/python main.py`.

In PyCharm, set the project interpreter to `.venv` and run `main.py`. Dependencies are listed in
`requirements.txt` (numpy, matplotlib; tkinter ships with python.org Python).

## GUI tabs

| Tab | Content |
|---|---|
| **Gear geometry** | Plan view of the struts and wheels, CG, and static load per wheel (redraws live as you type). |
| **Tire data** | Nose and main tires: 3 vertical loads (rated −, rated, rated +), each with an Fy vs slip table (defaults 0, 3, 6 … 18°, plus 3 spare rows), and a plot. Blank rows are ignored, and a blank cell only drops that point, so each curve is built from the points it has. Past a curve's last point the force is held constant. *Fill table* writes the placeholder curve (15 % Fz at 6°, linear). After a run, the steady-state operating points are overlaid. |
| **Results summary** | Required nose steer angle, yaw rate, lateral g, turn center, path radii (struts, wingtips, nose, tail), the force needed to hold speed, and a per-wheel table (slip, Fy, Fz, settle distance). |
| **Steer & slip vs distance** | Steer command, slip, Fy and Fz for all tires, and reference-point radius, plotted against CG distance. |
| **Slip per tire** | One subplot per tire: lagged tire slip, geometric slip, the steady-state line, and where it settles. |
| **Animation** | Top-down view with gear, CG and wingtip trails. Tires are colored by slip. |

Inputs can be saved and loaded as JSON. Results can be exported to CSV (every time step, every tire).

## Model

* Planar rigid body with 3 DOF (lateral velocity v, yaw rate r, heading/position). Forward speed is
  held constant, so thrust balances the tire scrub drag; that drag is reported.
* The airframe and struts are rigid. Vertical loads come from a rigid plane on equal per-wheel tire
  springs. Bogies pitch freely about their pivot, so a bogie's load acts at the pivot and is shared
  equally between its axles. This gives exact statics for 3-strut aircraft and a reasonable load share for bogies and
  4-post gear. Lateral load transfer from centripetal acceleration (CG height) is included.
  Longitudinal load transfer is not.
* The nose gear steers as one unit: both wheels rotate about the strut pivot through the same angle.
  Each strut has a `steer_ratio` (linear, used by the nose gear) and an optional `axle_steer`
  `{axle offset: schedule}` for individually steered axles; `steer_schedule()` in `layouts.py`
  builds the piecewise-linear schedule.
* Slip angle (per wheel, in the wheel frame): `alpha = delta_wheel − atan2(v_lat, v_long)`.
  **+ slip means force to the left.** In a left turn, positive slip pulls toward the center.
* Tire force comes from the tables: linear in slip (held beyond 18°) and linear in Fz between load
  cases, with an implicit Fz = 0 → Fy = 0 case and linear extrapolation above rated +.
* Tire lag uses a first-order relaxation length: `dα/dt = V/σ (α_geometric − α)`. Setting σ = 0
  gives pure geometric slip.
* Steady state is solved directly with Newton's method on [v, r, δ] for the target radius at the
  chosen reference point (CG, nose gear, or main-gear centroid). The transient is RK4 integration
  with the nose steer ramped to that angle at the given rate, after a straight lead-in.
* "Settled @ s" is the CG distance after which that tire's slip stays within ±2 % (or ±0.02°) of
  its steady-state value.

## Boeing ACAP data used

Source: https://www.boeing.com/commercial/airports/plan-manuals. The static loads the model computes
match the section 7.3.1 tables exactly at both CG limits.

**777-9:** wheelbase 106 ft 1 in to the middle main axle, main track 35 ft 6 in, lateral wheel
spacing 55 in, axles 57.2 in (fwd to middle) and 58.0 in (middle to aft), nose wheels 30.8 in
apart, 777,000 lb, V_NG 68,796 lb and V_MG 366,337 lb per strut (CG 9.39 ft or 6.05 ft ahead of
the middle axle). Tires: nose 43x17.5R17 at 218 psi, main 52x21.0R22 at 224 psi. The animation
uses the folded-wingtip span (212 ft 9 in).

**Aft-axle steering (6-wheel layout):** the aft axle of each main bogie turns about its own centre,
opposite to the nose. Three layout inputs set the schedule: max angle (8°; 0 locks the axle), the
nose angle where it starts (13°), and the nose angle where it reaches max (70°). The 8° and 13° come
from public descriptions. The linear ramp between them is an **assumption**, so replace it if you
have Boeing's actual schedule. The axle angle is plotted on the steer chart and listed in the
summary. Example, 777-9 at 120 ft radius and 10 kt: nose steer 43.1° and aft axle −4.2° give peak
main slip 5.2° and scrub drag 3,680 lb. With the axle locked it is 45.1°, 6.2° and 4,690 lb.

**737 MAX:**

Wheelbases, main track 18 ft 9 in, main wheel spacing 2 ft 10 in, nose wheel spacing 1 ft 4 in,
max design taxi weights, tire sizes and pressures. The CG positions are back-calculated from the
section 7.3.1 static loads:

* Most-forward CG: `x_cg = V_NG · WB / W` (737-8: 4.45 ft ahead of the main axle)
* Most-aft CG: `x_cg = (W − 2·V_MG) · WB / W` (737-8: 3.41 ft)

**Estimates you should replace:** CG height (10 ft), yaw radius of gyration (24–28.5 ft), tire
rated loads (defaulted to half the max static strut load, with rated ± at ±25 %), relaxation lengths
(nose 1.5 ft, main 2.5 ft), the planform outline, and the default cornering curves. The three
"Example" widebody presets use made-up illustrative values that only exercise the other layouts.

## Adding configurations

**New gear layout** (`aircraft_taxi_simulator/layouts.py`): write a builder that returns `Strut`s and register it.

```python
def _my_layout(p):  # p = params already converted to ft
    return [
        Strut("Nose", p["wheelbase"], 0, "nose", side_by_side(p["nose_wheel_spacing"]), steer_ratio=1.0),
        Strut("L Main", 0, p["track"] / 2, "main", bogie(2, p["axle_spacing"], p["wheel_spacing"])),
        Strut("R Main", 0, -p["track"] / 2, "main", bogie(2, p["axle_spacing"], p["wheel_spacing"])),
    ]

register(GearLayout(
    name="My layout", description="...", origin_note="x = 0 on ...",
    params=[Param("wheelbase", "Wheelbase", "ft", 60.0), Param("track", "Track", "ft", 25.0),
            Param("axle_spacing", "Axle spacing", "in", 60.0), Param("wheel_spacing", "Wheel spacing", "in", 50.0),
            Param("nose_wheel_spacing", "Nose wheel spacing", "in", 20.0)],
    builder=_my_layout))
```

The GUI builds the input fields from `params` automatically. Wheel names are generated
automatically (Fwd/Mid/Aft, Inbd/Outbd).

**New aircraft** (`aircraft_taxi_simulator/presets.py`): `add_preset(AircraftPreset(...))` with the layout name,
its parameter values, weight, CG cases, tire defaults and outline.

**Different tire types per strut** (for example body gear tires that differ from wing gear tires):
give the strut a new `tire` key and add a matching `TirePanel` in `App._build_tabs`.

## Files

```
main.py                              launcher (the .command / .bat files wrap it)
aircraft_taxi_simulator/layouts.py   gear-configuration registry (dropdown 1)
aircraft_taxi_simulator/presets.py   aircraft presets (dropdown 2)
aircraft_taxi_simulator/tire.py      table-driven cornering-force model
aircraft_taxi_simulator/model.py     loads, forces, steady-state solver, transient simulation
aircraft_taxi_simulator/outline.py   planform drawing
aircraft_taxi_simulator/gui.py       Tkinter + matplotlib GUI
```

## License

MIT, see [LICENSE](LICENSE). Aircraft data comes from Boeing's public airport planning documents
(https://www.boeing.com/commercial/airports/plan-manuals). This is a simplified engineering model,
not certified data: verify any results before using them operationally.
