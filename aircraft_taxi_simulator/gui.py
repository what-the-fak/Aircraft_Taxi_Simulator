"""Tkinter GUI for Aircraft Taxi Simulator."""
from __future__ import annotations

import csv
import json
import math
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

from .layouts import LAYOUTS
from .model import KT_TO_FPS, AircraftModel, simulate, solve_steady
from .outline import planform
from .presets import PRESETS, TireDefaults
from .tire import DEFAULT_SLIPS, LOAD_CASE_NAMES, TireData, default_curves

CG_CASES = ("Most FWD (from loads table)", "Most AFT (from loads table)", "Custom")
REF_POINTS = ("CG", "Nose gear", "Main gear centroid")
DIRECTIONS = ("Left", "Right")

MASS_FIELDS = [
    ("weight", "Taxi weight", "lb"),
    ("cg_x", "CG ahead of main-gear reference", "ft"),
    ("cg_y", "CG lateral offset (+ left)", "ft"),
    ("cg_h", "CG height above ground", "ft"),
    ("kz", "Yaw radius of gyration (Iz = m kz²)", "ft"),
]
MANEUVER_FIELDS = [
    ("radius", "Turn radius", "ft"),
    ("speed", "Taxi speed (constant)", "kt"),
    ("steer_rate", "Nose steer rate", "deg/s"),
    ("lead_in", "Straight roll before steering", "ft"),
    ("max_dist", "Max simulated distance", "ft"),
    ("max_steer", "Max nose steer angle", "deg"),
]
TIRE_DYN_FIELDS = [
    ("sigma_nose", "Nose tire relaxation length", "ft"),
    ("sigma_main", "Main tire relaxation length", "ft"),
]
OUTLINE_FIELDS = [
    ("length", "Fuselage length", "ft"),
    ("span", "Wing span", "ft"),
    ("fus_width", "Fuselage width", "ft"),
    ("nose_ahead", "Nose tip ahead of nose gear", "ft"),
    ("wing_le", "Wing root LE ahead of main ref", "ft"),
    ("wing_root_chord", "Wing root chord", "ft"),
    ("wing_sweep", "Wing LE sweep", "deg"),
    ("n_engines", "Engines per side", "-"),
]
DEFAULTS = dict(cg_y=0.0, radius=150.0, speed=10.0, steer_rate=10.0, lead_in=20.0, max_dist=1500.0,
                sigma_nose=1.5, sigma_main=2.5)


class InputError(Exception):
    pass


def fmt(v):
    if isinstance(v, str):
        return v
    return f"{v:.6g}"


def parse(var: tk.StringVar, label: str) -> float:
    try:
        return float(var.get().replace(",", ""))
    except ValueError:
        raise InputError(f"'{label}' is not a number: {var.get()!r}")


# =============================================================================== widgets
class ScrollFrame(ttk.Frame):
    def __init__(self, master, width=430):
        super().__init__(master)
        self.canvas = tk.Canvas(self, width=width, highlightthickness=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.inner.bind("<Enter>", lambda e: self.bind_all("<MouseWheel>", self._wheel))
        self.inner.bind("<Leave>", lambda e: self.unbind_all("<MouseWheel>"))

    def _wheel(self, e):
        d = e.delta if abs(e.delta) < 20 else e.delta / 120
        self.canvas.yview_scroll(int(-d), "units")


class PlotFrame(ttk.Frame):
    def __init__(self, master, figsize=(9, 6), toolbar=True):
        super().__init__(master)
        self.fig = Figure(figsize=figsize, dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        if toolbar:
            tb = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
            tb.update()
            tb.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    def draw(self):
        self.canvas.draw_idle()


class TirePanel(ttk.LabelFrame):
    """Editable tire data: 3 load cases x Fy(slip) table + plot."""

    def __init__(self, master, app, key, title):
        super().__init__(master, text=title, padding=6)
        self.app, self.key = app, key
        self.v = {k: tk.StringVar() for k in ("size", "od", "width", "psi", "frac")}
        self.loads = [tk.StringVar() for _ in range(3)]
        self.slips = [tk.StringVar(value=fmt(s)) for s in DEFAULT_SLIPS]
        self.fy = [[tk.StringVar() for _ in DEFAULT_SLIPS] for _ in range(3)]
        self.points = None

        top = ttk.Frame(self)
        top.pack(fill="x")
        for i, (k, lab, w) in enumerate((("size", "Size", 20), ("od", "OD (in)", 6),
                                          ("width", "Width (in)", 6), ("psi", "Pressure (psi)", 6))):
            ttk.Label(top, text=lab).grid(row=0, column=2 * i, sticky="e", padx=(6, 2))
            ttk.Entry(top, textvariable=self.v[k], width=w).grid(row=0, column=2 * i + 1, sticky="w")

        tbl = ttk.Frame(self)
        tbl.pack(fill="x", pady=4)
        ttk.Label(tbl, text="Vertical load (lb) →", font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0)
        for j in range(3):
            ttk.Label(tbl, text=LOAD_CASE_NAMES[j]).grid(row=1, column=j + 1)
            ttk.Entry(tbl, textvariable=self.loads[j], width=11, justify="right").grid(row=0, column=j + 1, padx=2)
        ttk.Label(tbl, text="Slip (deg)").grid(row=1, column=0)
        for i in range(len(DEFAULT_SLIPS)):
            ttk.Entry(tbl, textvariable=self.slips[i], width=7, justify="right").grid(row=i + 2, column=0, padx=2)
            for j in range(3):
                ttk.Entry(tbl, textvariable=self.fy[j][i], width=11, justify="right").grid(row=i + 2, column=j + 1,
                                                                                          padx=2)
        ttk.Label(tbl, text="Cornering force Fy (lb)").grid(row=len(DEFAULT_SLIPS) + 2, column=1, columnspan=3)

        bot = ttk.Frame(self)
        bot.pack(fill="x")
        ttk.Label(bot, text="Default curves: Fy =").pack(side="left")
        ttk.Entry(bot, textvariable=self.v["frac"], width=5).pack(side="left")
        ttk.Label(bot, text="% of Fz at 6°, linear").pack(side="left")
        ttk.Button(bot, text="Fill table", command=self.fill_default).pack(side="left", padx=6)
        ttk.Button(bot, text="Re-plot", command=self.replot).pack(side="left")

        self.plot = PlotFrame(self, figsize=(5, 3.0), toolbar=False)
        self.plot.pack(fill="both", expand=True)
        self.ax = self.plot.fig.add_subplot(111)

    # --------------------------------------------------------------
    def set_defaults(self, td: TireDefaults):
        self.v["size"].set(td.size)
        self.v["od"].set(fmt(td.od_in))
        self.v["width"].set(fmt(td.width_in))
        self.v["psi"].set(fmt(td.pressure_psi))
        self.v["frac"].set(fmt(100 * td.fy_frac_at_6deg))
        for j, L in enumerate(td.loads):
            self.loads[j].set(fmt(round(L)))
        for i, s in enumerate(DEFAULT_SLIPS):
            self.slips[i].set(fmt(s))
        self.fill_default()

    def fill_default(self):
        try:
            loads = [parse(v, f"{self.key} load") for v in self.loads]
            slips = [parse(v, f"{self.key} slip") for v in self.slips]
            frac = parse(self.v["frac"], "default Fy %") / 100
        except InputError as e:
            messagebox.showerror("Tire data", str(e))
            return
        tbl = default_curves(loads, slips, frac, 6.0)
        for j in range(3):
            for i in range(len(slips)):
                self.fy[j][i].set(fmt(round(tbl[j, i], 1)))
        self.replot()

    def get(self) -> TireData:
        lab = "Nose" if self.key == "nose" else "Main"
        return TireData(
            name=self.key, size=self.v["size"].get(),
            od_in=parse(self.v["od"], f"{lab} tire OD"), width_in=parse(self.v["width"], f"{lab} tire width"),
            pressure_psi=parse(self.v["psi"], f"{lab} tire pressure"),
            loads_lb=[parse(v, f"{lab} tire load case") for v in self.loads],
            slip_deg=[parse(v, f"{lab} tire slip angle") for v in self.slips],
            fy_lb=[[parse(v, f"{lab} tire Fy") for v in row] for row in self.fy])

    def state(self):
        return dict(v={k: x.get() for k, x in self.v.items()}, loads=[x.get() for x in self.loads],
                    slips=[x.get() for x in self.slips], fy=[[x.get() for x in r] for r in self.fy])

    def set_state(self, st):
        for k, x in st["v"].items():
            self.v[k].set(x)
        for x, s in zip(self.loads, st["loads"]):
            x.set(s)
        for x, s in zip(self.slips, st["slips"]):
            x.set(s)
        for row, srow in zip(self.fy, st["fy"]):
            for x, s in zip(row, srow):
                x.set(s)
        self.replot()

    def replot(self, points=None):
        if points is not None:
            self.points = points
        ax = self.ax
        ax.clear()
        try:
            td = self.get()
        except (InputError, ValueError) as e:
            ax.text(0.5, 0.5, str(e), ha="center", va="center", transform=ax.transAxes, wrap=True)
            self.plot.draw()
            return
        sl = np.linspace(0, td.slip_deg.max(), 100)
        cols = ["#4C72B0", "#222222", "#C44E52"]
        for j in range(3):
            ax.plot(td.slip_deg, td.fy_lb[j], "o", ms=3, color=cols[j])
            ax.plot(sl, td.fy(sl, td.loads_lb[j]), "-", color=cols[j],
                    label=f"{LOAD_CASE_NAMES[j]}: {td.loads_lb[j]:,.0f} lb")
        if self.points is not None:
            a, f, names = self.points
            ax.plot(np.abs(a), np.abs(f), "D", color="#DD8452", ms=6, label="Steady-state operating pts")
        ax.set_xlabel("Slip angle (deg)")
        ax.set_ylabel("Cornering force Fy (lb)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
        self.plot.draw()


# =============================================================================== animation
class AnimPanel(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        ctl = ttk.Frame(self)
        ctl.pack(side="top", fill="x", pady=3)
        self.play_btn = ttk.Button(ctl, text="▶ Play", command=self.toggle)
        self.play_btn.pack(side="left", padx=4)
        ttk.Button(ctl, text="⟲ Restart", command=self.restart).pack(side="left")
        ttk.Label(ctl, text="  Speed ×").pack(side="left")
        self.speed = tk.StringVar(value="2")
        ttk.Combobox(ctl, textvariable=self.speed, values=["0.25", "0.5", "1", "2", "4", "8", "16"],
                     width=5, state="readonly").pack(side="left")
        self.follow = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctl, text="Follow aircraft", variable=self.follow,
                        command=lambda: self.show(self.t)).pack(side="left", padx=8)
        self.tvar = tk.DoubleVar(value=0)
        self.slider = ttk.Scale(ctl, from_=0, to=1, variable=self.tvar, orient="horizontal",
                                command=lambda v: self.show(float(v)) if not self.playing else None)
        self.slider.pack(side="left", fill="x", expand=True, padx=8)
        self.info = ttk.Label(self, text="Run a simulation to animate.", font=("Menlo", 11))
        self.info.pack(side="top", fill="x")
        self.plot = PlotFrame(self, figsize=(9, 7))
        self.plot.pack(fill="both", expand=True)
        self.ax = self.plot.fig.add_subplot(111)
        self.playing = False
        self.t = 0.0
        self.data = None

    def load(self, model: AircraftModel, res, polys_cg, pts_cg, tire_dims, ref_name):
        self.playing = False
        self.play_btn.config(text="▶ Play")
        self.model, self.res = model, res
        self.polys = polys_cg
        ax = self.ax
        ax.clear()
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        ax.set_xlabel("X (ft)")
        ax.set_ylabel("Y (ft)")
        X, Y, psi = res.X, res.Y, res.psi
        c, s = np.cos(psi), np.sin(psi)

        def world(px, py):
            return X + c * px - s * py, Y + s * px + c * py

        # trails
        trails = {"CG": (0.0, 0.0)}
        trails.update(model.strut_points())
        trails.update({k: v for k, v in pts_cg.items() if "wingtip" in k})
        self.trail_lines = []
        cmap = colormaps["tab10"]
        allx, ally = [], []
        for i, (nm, (px, py)) in enumerate(trails.items()):
            wx, wy = world(px, py)
            allx += [wx.min(), wx.max()]
            ally += [wy.min(), wy.max()]
            ls = ":" if "wingtip" in nm else "-"
            ax.plot(wx, wy, ls, color=cmap(i % 10), lw=0.8, alpha=0.35)
            ln, = ax.plot([], [], ls, color=cmap(i % 10), lw=1.6, label=nm)
            self.trail_lines.append((ln, wx, wy))
        ax.legend(loc="upper right", fontsize=8)
        span = max(np.ptp(np.concatenate([p[:, 1] for p in polys_cg])), 50)
        self.full_lim = (min(allx) - 0.3 * span, max(allx) + 0.3 * span, min(ally) - 0.3 * span,
                         max(ally) + 0.3 * span)
        # aircraft artists
        self.body_patches = []
        for k, p in enumerate(polys_cg):
            fc = "#c9d3e0" if k == 0 else "#dfe5ee"
            patch = Polygon(p, closed=True, fc=fc, ec="#44556a", lw=0.8, alpha=0.85, zorder=3)
            ax.add_patch(patch)
            self.body_patches.append(patch)
        self.tire_dims = tire_dims  # per wheel (length ft, width ft)
        amax = max(np.abs(np.degrees(res.alpha)).max(), 0.1)
        self.norm = Normalize(0, amax)
        self.cmap = colormaps["plasma"]
        self.wheel_patches = []
        for j in range(model.n):
            patch = Polygon(np.zeros((4, 2)), closed=True, fc="k", ec="k", lw=0.4, zorder=5)
            ax.add_patch(patch)
            self.wheel_patches.append(patch)
        sm = ScalarMappable(norm=self.norm, cmap=self.cmap)
        if getattr(self, "cbar", None) is not None:
            try:
                self.cbar.remove()
            except Exception:
                pass
        self.cbar = self.plot.fig.colorbar(sm, ax=ax, shrink=0.6, label="|tire slip| (deg)")
        self.slider.configure(to=float(res.t[-1]))
        self.data = True
        self.show(0.0)

    def _tf(self, pts, X, Y, psi):
        c, s = math.cos(psi), math.sin(psi)
        return np.column_stack([X + c * pts[:, 0] - s * pts[:, 1], Y + s * pts[:, 0] + c * pts[:, 1]])

    def show(self, t):
        if not self.data:
            return
        res, model = self.res, self.model
        self.t = t
        i = int(np.clip(np.searchsorted(res.t, t), 0, len(res.t) - 1))
        X, Y, psi = res.X[i], res.Y[i], res.psi[i]
        for patch, p in zip(self.body_patches, self.polys):
            patch.set_xy(self._tf(p, X, Y, psi))
        x, y, dw = model.geometry(res.delta[i])
        for j, patch in enumerate(self.wheel_patches):
            L, W = self.tire_dims[j]
            ang = psi + dw[j]
            c, s = math.cos(ang), math.sin(ang)
            cx, cy = X + math.cos(psi) * x[j] - math.sin(psi) * y[j], Y + math.sin(psi) * x[j] + math.cos(psi) * y[j]
            corners = np.array([(L / 2, W / 2), (-L / 2, W / 2), (-L / 2, -W / 2), (L / 2, -W / 2)])
            patch.set_xy(np.column_stack([cx + c * corners[:, 0] - s * corners[:, 1],
                                          cy + s * corners[:, 0] + c * corners[:, 1]]))
            patch.set_facecolor(self.cmap(self.norm(abs(math.degrees(res.alpha[i, j])))))
        for ln, wx, wy in self.trail_lines:
            ln.set_data(wx[:i + 1], wy[:i + 1])
        if self.follow.get():
            h = 0.5 * max(np.ptp(np.concatenate([p[:, 1] for p in self.polys])), 60) * 1.4
            self.ax.set_xlim(X - h, X + h)
            self.ax.set_ylim(Y - h, Y + h)
        else:
            self.ax.set_xlim(*self.full_lim[:2])
            self.ax.set_ylim(*self.full_lim[2:])
        sl = np.degrees(res.alpha[i])
        rad = res.radius_ref[i]
        self.info.config(text=f"t = {res.t[i]:6.2f} s   s = {res.s[i]:7.1f} ft   steer = "
                              f"{math.degrees(res.delta[i]):6.2f}°   yaw rate = {math.degrees(res.r[i]):5.2f}°/s   "
                              f"ref radius = {rad if np.isfinite(rad) and rad < 1e5 else float('inf'):8.1f} ft   "
                              f"max |slip| = {np.abs(sl).max():5.2f}°")
        if self.playing:
            self.tvar.set(t)
        self.plot.canvas.draw_idle()

    def toggle(self):
        if not self.data:
            return
        self.playing = not self.playing
        self.play_btn.config(text="⏸ Pause" if self.playing else "▶ Play")
        if self.playing:
            if self.t >= self.res.t[-1]:
                self.t = 0.0
            self._tick()

    def restart(self):
        self.t = 0.0
        self.tvar.set(0)
        self.show(0.0)

    def _tick(self):
        if not self.playing:
            return
        if not self.winfo_viewable():
            self.toggle()
            return
        t = self.t + 0.04 * float(self.speed.get())
        if t >= self.res.t[-1]:
            t = float(self.res.t[-1])
            self.show(t)
            self.toggle()
            return
        self.show(t)
        self.after(40, self._tick)


# =============================================================================== main app
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Aircraft Taxi Simulator – ground-turn tire slip model")
        self.geometry("1560x980")
        self.vars: dict[str, tk.StringVar] = {}
        self.labels: dict[str, str] = {}
        self.layout_vars: dict[str, tk.StringVar] = {}
        self._suspend = False
        self._redraw_job = None
        self.result = None

        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True)
        left = ScrollFrame(paned)
        paned.add(left, weight=0)
        self.nb = ttk.Notebook(paned)
        paned.add(self.nb, weight=1)
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.status, anchor="w", relief="sunken").pack(side="bottom", fill="x")

        self._build_inputs(left.inner)
        self._build_tabs()
        self.vars["preset"].set(next(iter(PRESETS)))
        self.apply_preset()

    # ------------------------------------------------------------------ input panel
    def _entry(self, parent, key, label, unit, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=1)
        var = self.vars.setdefault(key, tk.StringVar(value=fmt(DEFAULTS.get(key, ""))))
        ttk.Entry(parent, textvariable=var, width=11, justify="right").grid(row=row, column=1, padx=3)
        ttk.Label(parent, text=unit).grid(row=row, column=2, sticky="w")
        self.labels[key] = label
        var.trace_add("write", lambda *a, k=key: self._changed(k))

    def _combo(self, parent, key, label, values, row, width=22, cmd=None):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=1)
        var = self.vars.setdefault(key, tk.StringVar(value=values[0]))
        cb = ttk.Combobox(parent, textvariable=var, values=list(values), state="readonly", width=width)
        cb.grid(row=row, column=1, columnspan=2, sticky="w", padx=3)
        if cmd:
            cb.bind("<<ComboboxSelected>>", lambda e: cmd())
        else:
            var.trace_add("write", lambda *a, k=key: self._changed(k))
        return cb

    def _build_inputs(self, root):
        pad = dict(fill="x", padx=6, pady=4)
        f = ttk.LabelFrame(root, text="Aircraft / gear configuration", padding=6)
        f.pack(**pad)
        ttk.Label(f, text="Aircraft preset").grid(row=0, column=0, sticky="w")
        self.vars["preset"] = tk.StringVar()
        cb = ttk.Combobox(f, textvariable=self.vars["preset"], values=list(PRESETS), state="readonly", width=40)
        cb.grid(row=1, column=0, columnspan=3, sticky="w")
        cb.bind("<<ComboboxSelected>>", lambda e: self.apply_preset())
        self.vars["preset_tires"] = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Preset also loads its default tire data",
                        variable=self.vars["preset_tires"]).grid(row=2, column=0, columnspan=3, sticky="w")
        ttk.Label(f, text="Gear layout").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.vars["layout"] = tk.StringVar()
        cb = ttk.Combobox(f, textvariable=self.vars["layout"], values=list(LAYOUTS), state="readonly", width=40)
        cb.grid(row=4, column=0, columnspan=3, sticky="w")
        cb.bind("<<ComboboxSelected>>", lambda e: self._build_layout_fields(self.vars["layout"].get(), None))
        self.layout_note = ttk.Label(f, text="", foreground="#555", wraplength=380)
        self.layout_note.grid(row=5, column=0, columnspan=3, sticky="w")
        self.layout_frame = ttk.Frame(f)
        self.layout_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=3)

        f = ttk.LabelFrame(root, text="Weight & CG", padding=6)
        f.pack(**pad)
        self._combo(f, "cg_case", "CG case", CG_CASES, 0, cmd=self._cg_case)
        for i, (k, lab, u) in enumerate(MASS_FIELDS):
            self._entry(f, k, lab, u, i + 1)
        self.derived = ttk.Label(f, text="", foreground="#245", wraplength=380, justify="left")
        self.derived.grid(row=len(MASS_FIELDS) + 1, column=0, columnspan=3, sticky="w", pady=(4, 0))

        f = ttk.LabelFrame(root, text="Maneuver", padding=6)
        f.pack(**pad)
        self._combo(f, "direction", "Turn direction", DIRECTIONS, 0, width=10)
        self._combo(f, "ref_point", "Radius measured at", REF_POINTS, 1, width=18)
        for i, (k, lab, u) in enumerate(MANEUVER_FIELDS):
            self._entry(f, k, lab, u, i + 2)

        f = ttk.LabelFrame(root, text="Tire dynamics", padding=6)
        f.pack(**pad)
        for i, (k, lab, u) in enumerate(TIRE_DYN_FIELDS):
            self._entry(f, k, lab, u, i)
        ttk.Label(f, text="(0 = no lag: slip follows geometry instantly)", foreground="#555").grid(
            row=3, column=0, columnspan=3, sticky="w")

        f = ttk.LabelFrame(root, text="Planform (animation & wingtip radius only)", padding=6)
        f.pack(**pad)
        for i, (k, lab, u) in enumerate(OUTLINE_FIELDS):
            self._entry(f, k, lab, u, i)

        b = ttk.Frame(root, padding=6)
        b.pack(fill="x")
        ttk.Button(b, text="▶  RUN SIMULATION", command=self.run).pack(fill="x", ipady=6)
        r2 = ttk.Frame(b)
        r2.pack(fill="x", pady=4)
        ttk.Button(r2, text="Save inputs…", command=self.save_inputs).pack(side="left", expand=True, fill="x")
        ttk.Button(r2, text="Load inputs…", command=self.load_inputs).pack(side="left", expand=True, fill="x")
        ttk.Button(b, text="Export results CSV…", command=self.export_csv).pack(fill="x")

    def _build_layout_fields(self, layout_name, values):
        lay = LAYOUTS[layout_name]
        for w in self.layout_frame.winfo_children():
            w.destroy()
        self.layout_vars = {}
        vals = lay.defaults() if values is None else {**lay.defaults(), **values}
        for i, p in enumerate(lay.params):
            ttk.Label(self.layout_frame, text=p.label).grid(row=i, column=0, sticky="w", pady=1)
            var = tk.StringVar(value=fmt(round(vals[p.key], 4)))
            ttk.Entry(self.layout_frame, textvariable=var, width=11, justify="right").grid(row=i, column=1, padx=3)
            ttk.Label(self.layout_frame, text=p.unit).grid(row=i, column=2, sticky="w")
            var.trace_add("write", lambda *a: self._changed("layout_param"))
            self.layout_vars[p.key] = var
        self.vars["layout"].set(layout_name)
        self.layout_note.config(text=f"{lay.description}\nOrigin: {lay.origin_note}.")
        self._changed("layout")

    def _changed(self, key):
        if self._suspend:
            return
        if key == "cg_x" and self.vars["cg_case"].get() != CG_CASES[2]:
            self._suspend = True
            self.vars["cg_case"].set(CG_CASES[2])
            self._suspend = False
        if self._redraw_job:
            self.after_cancel(self._redraw_job)
        self._redraw_job = self.after(350, self.draw_geometry)

    def _cg_case(self):
        p = PRESETS.get(self.vars["preset"].get())
        case = self.vars["cg_case"].get()
        if p is None or case == CG_CASES[2]:
            return
        self._suspend = True
        self.vars["cg_x"].set(fmt(round(p.cg_fwd_ft if case == CG_CASES[0] else p.cg_aft_ft, 3)))
        self._suspend = False
        self._changed("cg_case")

    def apply_preset(self):
        p = PRESETS[self.vars["preset"].get()]
        self._suspend = True
        try:
            self._build_layout_fields(p.layout, p.layout_params)
            self.vars["weight"].set(fmt(p.weight_lb))
            self.vars["cg_case"].set(CG_CASES[0])
            self.vars["cg_x"].set(fmt(round(p.cg_fwd_ft, 3)))
            self.vars["cg_h"].set(fmt(p.cg_height_ft))
            self.vars["kz"].set(fmt(p.kz_ft))
            self.vars["max_steer"].set(fmt(p.max_steer_deg))
            o = p.outline
            for k, v in dict(length=o.length_ft, span=o.span_ft, fus_width=o.fus_width_ft,
                             nose_ahead=o.nose_ahead_ft, wing_le=o.wing_root_le_ft,
                             wing_root_chord=o.wing_root_chord_ft, wing_sweep=o.wing_sweep_deg,
                             n_engines=o.n_engines_per_side).items():
                self.vars[k].set(fmt(round(v, 3)))
        finally:
            self._suspend = False
        if self.vars["preset_tires"].get():
            self.tire_panels["nose"].set_defaults(p.nose_tire)
            self.tire_panels["main"].set_defaults(p.main_tire)
        self.status.set(f"Loaded preset {p.name}. {p.notes}")
        self.draw_geometry()

    # ------------------------------------------------------------------ tabs
    def _build_tabs(self):
        self.geo = PlotFrame(self.nb, figsize=(9, 7))
        self.nb.add(self.geo, text="Gear geometry")

        tf = ttk.Frame(self.nb)
        self.nb.add(tf, text="Tire data")
        self.tire_panels = {
            "nose": TirePanel(tf, self, "nose", "Nose gear tire"),
            "main": TirePanel(tf, self, "main", "Main gear tire"),
        }
        self.tire_panels["nose"].pack(side="left", fill="both", expand=True, padx=4, pady=4)
        self.tire_panels["main"].pack(side="left", fill="both", expand=True, padx=4, pady=4)

        sf = ttk.Frame(self.nb)
        self.nb.add(sf, text="Results summary")
        self.summary = tk.Text(sf, height=22, font=("Menlo", 11), wrap="none")
        self.summary.pack(fill="x", padx=4, pady=4)
        cols = ("tire", "x", "y", "fz0", "fz", "steer", "slip", "fy", "mu", "settle", "settle2", "radius")
        heads = ("Tire", "x fwd CG (ft)", "y left (ft)", "Fz static (lb)", "Fz in turn (lb)", "Wheel steer (°)",
                 "Slip SS (°)", "Fy SS (lb)", "Fy/Fz", "Settled @ s (ft)", "…after steer start (ft)",
                 "Path radius (ft)")
        tvf = ttk.Frame(sf)
        tvf.pack(fill="both", expand=True, padx=4, pady=4)
        self.table = ttk.Treeview(tvf, columns=cols, show="tree headings", height=14)
        self.table.heading("#0", text="Wheel")
        self.table.column("#0", width=170)
        for c, h in zip(cols, heads):
            self.table.heading(c, text=h)
            self.table.column(c, width=105, anchor="e")
        vs = ttk.Scrollbar(tvf, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=vs.set)
        self.table.pack(side="left", fill="both", expand=True)
        vs.pack(side="right", fill="y")

        self.hist = PlotFrame(self.nb, figsize=(10, 9))
        self.nb.add(self.hist, text="Steer & slip vs distance")
        self.grid_plot = PlotFrame(self.nb, figsize=(10, 9))
        self.nb.add(self.grid_plot, text="Slip per tire")
        self.anim = AnimPanel(self.nb)
        self.nb.add(self.anim, text="Animation")

    # ------------------------------------------------------------------ build model
    def _layout_values(self):
        lay = LAYOUTS[self.vars["layout"].get()]
        return lay, {p.key: parse(self.layout_vars[p.key], p.label) for p in lay.params}

    def _f(self, key):
        return parse(self.vars[key], self.labels.get(key, key))

    def outline_dict(self):
        return {k: self._f(k) for k, _, _ in OUTLINE_FIELDS}

    def build_model(self, tires=None):
        lay, vals = self._layout_values()
        struts = lay.build(vals)
        if tires is None:
            tires = {k: p.get() for k, p in self.tire_panels.items()}
        W = self._f("weight")
        if W <= 0:
            raise InputError("Weight must be positive")
        model = AircraftModel(struts, W, self._f("cg_x"), self._f("cg_y"), self._f("cg_h"), self._f("kz"),
                              tires, {"nose": self._f("sigma_nose"), "main": self._f("sigma_main")})
        return model, struts

    def _planform_cg(self, model, struts):
        steer = [s for s in struts if s.steer_ratio != 0] or struts[:1]
        polys, pts = planform(self.outline_dict(), steer[0].x)
        sh = np.array([model.x_cg, model.y_cg])
        return [p - sh for p in polys], {k: (v[0] - sh[0], v[1] - sh[1]) for k, v in pts.items()}

    # ------------------------------------------------------------------ geometry drawing
    def draw_geometry(self):
        self._redraw_job = None
        fig = self.geo.fig
        fig.clear()
        try:
            dummy = {k: TireData(k) for k in ("nose", "main")}
            model, struts = self.build_model(tires=dummy)
            fz = model.vertical_loads(*model.geometry(0.0)[:2], 0.0)
            tires = {}
            for k, p in self.tire_panels.items():
                try:
                    tires[k] = (parse(p.v["od"], "od") / 12, parse(p.v["width"], "w") / 12)
                except InputError:
                    tires[k] = (3.0, 1.0)
        except (InputError, KeyError, ValueError) as e:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Input error:\n{e}", ha="center", va="center", transform=ax.transAxes)
            self.geo.draw()
            return
        ax0, ax = fig.subplots(1, 2, width_ratios=[1, 2.2])
        xcg, ycg = model.x_cg, model.y_cg
        nose = [s for s in struts if s.steer_ratio != 0] or struts[:1]
        try:
            polys, _ = planform(self.outline_dict(), nose[0].x)
        except InputError:
            polys = []
        x, y, _ = model.geometry(0.0)
        x, y = x + xcg, y + ycg
        # plot coordinates: horizontal = -y (left of aircraft on left of screen), vertical = x (fwd up)
        for a, detail in ((ax0, False), (ax, True)):
            for p in polys:
                a.fill(-p[:, 1], p[:, 0], fc="#eef1f5", ec="#b8c2cf", lw=0.8, zorder=0)
            for j, wh in enumerate(model.wheels):
                L, Wd = tires[wh.tire]
                a.add_patch(Polygon([(-y[j] - Wd / 2, x[j] - L / 2), (-y[j] + Wd / 2, x[j] - L / 2),
                                     (-y[j] + Wd / 2, x[j] + L / 2), (-y[j] - Wd / 2, x[j] + L / 2)],
                                    fc="#333", ec="k", zorder=3))
            a.plot(-ycg, xcg, "o", ms=11 if detail else 6, mfc="w", mec="k", zorder=5)
            a.plot(-ycg, xcg, "o", ms=4 if detail else 2, color="k", zorder=6)
            a.axhline(0, color="C3", lw=0.6, ls="--")
            a.axvline(0, color="#999", lw=0.6, ls="-.")
            a.set_aspect("equal")
            a.grid(alpha=0.25)
        ax0.set_title("Overview", fontsize=9)
        ax0.set_ylabel("x fwd of main-gear reference (ft)")
        # detail view: label each strut with its wheel loads
        for s in struts:
            ax.plot(-s.y, s.x, "+", color="C3", ms=10, mew=1.5, zorder=4)
            idx = [j for j, wh in enumerate(model.wheels) if wh.strut == s.name]
            lines = [f"{s.name}  (x {s.x:.2f}, y {s.y:+.2f} ft)"]
            lines += [f"{model.wheels[j].name[len(s.name):].strip() or 'wheel'}: {fz[j]:,.0f} lb" for j in idx]
            lines.append(f"strut total {fz[idx].sum():,.0f} lb")
            if s.axle_steer:
                lines.append("aft axle steerable")
            above = s.steer_ratio != 0
            half = tires[s.tire][0] / 2
            anchor = (x[idx].max() + half) if above else (x[idx].min() - half)
            ax.annotate("\n".join(lines), (-s.y, anchor), xytext=(0, 6 if above else -6),
                        textcoords="offset points", ha="center", va="bottom" if above else "top",
                        fontsize=7.5, color="#203050",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#c0c8d4", alpha=0.9), zorder=7)
        ax.annotate(f"CG\n{xcg:.2f} ft fwd", (-ycg, xcg), xytext=(12, 0), textcoords="offset points",
                    fontsize=8, weight="bold", va="center")
        if nose and nose[0].steer_ratio != 0:
            ax.annotate("", (0.0, nose[0].x), (0.0, 0.0), arrowprops=dict(arrowstyle="<->", color="C0"))
            ax.text(-0.6, nose[0].x * 0.6, f"WB {nose[0].x:.2f} ft", color="C0", fontsize=8, rotation=90,
                    va="center", ha="right")
        ys = np.concatenate([[-model.y_cg], -y])
        xs = np.concatenate([x, [xcg]])
        span_y = max(np.ptp(ys), 10.0)
        span_x = max(np.ptp(xs), 10.0)
        ax.set_xlim(ys.min() - 0.45 * span_y - 4, ys.max() + 0.45 * span_y + 4)
        ax.set_ylim(xs.min() - 0.12 * span_x - 6, xs.max() + 0.12 * span_x + 6)
        ax.set_xlabel("lateral position (ft)   ← left  |  right →")
        ax0.set_xlabel("lateral (ft)")
        ax.set_title(f"{self.vars['layout'].get()} — static wheel loads\n"
                     f"red + = strut pivots, origin: {LAYOUTS[self.vars['layout'].get()].origin_note}", fontsize=9)
        nz = fz[model._steer_mask()].sum()
        mains = {}
        for wh, f_ in zip(model.wheels, fz):
            if wh.steer_ratio == 0:
                mains[wh.strut] = mains.get(wh.strut, 0) + f_
        self.derived.config(text=f"Static loads: nose {nz:,.0f} lb ({100 * nz / model.W:.1f}% W); "
                                 + ", ".join(f"{k} {v:,.0f}" for k, v in mains.items())
                                 + f" lb.\nIz = {model.Iz:,.3e} slug·ft²")
        self.geo.draw()

    # ------------------------------------------------------------------ run
    def run(self):
        try:
            self.config(cursor="watch")
            self.update_idletasks()
            self._run()
        except InputError as e:
            messagebox.showerror("Input error", str(e))
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Simulation error", f"{type(e).__name__}: {e}")
        finally:
            self.config(cursor="")

    def _run(self):
        model, struts = self.build_model()
        u = self._f("speed") * KT_TO_FPS
        R = self._f("radius")
        if u <= 0 or R <= 0:
            raise InputError("Speed and turn radius must be positive")
        direction = 1 if self.vars["direction"].get() == "Left" else -1
        ref = self.vars["ref_point"].get()
        ss = solve_steady(model, u, R, direction, ref)
        res = simulate(model, u, ss, math.radians(self._f("steer_rate")), self._f("lead_in"), self._f("max_dist"),
                       ref)
        polys, pts = self._planform_cg(model, struts)
        self.result = dict(model=model, ss=ss, res=res, u=u, R=R, ref=ref, direction=direction, pts=pts)
        self._fill_summary()
        self._plot_histories()
        self._plot_grid()
        dims = []
        for wh in model.wheels:
            p = self.tire_panels[wh.tire]
            dims.append((self._safe(p.v["od"], 40) / 12, self._safe(p.v["width"], 15) / 12))
        self.anim.load(model, res, polys, pts, dims, ref)
        for k, p in self.tire_panels.items():
            mk = model.masks.get(k)
            if mk is not None:
                p.replot((np.degrees(ss.alpha[mk]), ss.fy[mk], None))
        self.nb.select(2)
        self.status.set(f"Done: steer {math.degrees(ss.delta):.2f}°, solver {ss.message}, "
                        f"{len(res.t)} time steps.")

    @staticmethod
    def _safe(var, default):
        try:
            return float(var.get())
        except ValueError:
            return default

    # ------------------------------------------------------------------ outputs
    def _fill_summary(self):
        r = self.result
        model, ss, res, u, R = r["model"], r["ss"], r["res"], r["u"], r["R"]
        xc, yc = ss.turn_center
        rad = lambda px, py: math.hypot(px - xc, py - yc)
        d_ss = math.degrees(ss.delta)
        max_steer = self._f("max_steer")
        L = []
        L.append(f"Aircraft preset : {self.vars['preset'].get()}")
        L.append(f"Gear layout     : {self.vars['layout'].get()}")
        L.append(f"Weight {model.W:,.0f} lb   CG {model.x_cg:.2f} ft fwd of main ref, {model.y_cg:+.2f} ft lat, "
                 f"{model.h_cg:.1f} ft high   Iz {model.Iz:,.3e} slug·ft²")
        L.append(f"Speed {u / KT_TO_FPS:.1f} kt ({u:.2f} ft/s)   {self.vars['direction'].get()} turn, "
                 f"R = {R:.1f} ft at {r['ref']}")
        L.append("")
        L.append(f"STEADY STATE  (solver {ss.message})")
        L.append(f"  Nose steer angle required : {d_ss:8.3f} deg" +
                 (f"   *** EXCEEDS max steer {max_steer:.1f} deg ***" if abs(d_ss) > max_steer else
                  f"   (limit {max_steer:.1f})"))
        for wn, ang in model.axle_steer(ss.delta).items():
            L.append(f"  Steered main axles        : {math.degrees(ang):8.3f} deg  (opposite to nose = negative "
                     f"in a left turn)")
        L.append(f"  Yaw rate                  : {math.degrees(ss.r):8.3f} deg/s")
        L.append(f"  Lateral accel at CG       : {ss.ay / 32.174:8.4f} g")
        L.append(f"  CG sideslip (v/u)         : {math.degrees(math.atan2(ss.v, ss.u)):8.3f} deg")
        L.append(f"  Turn center (rel. CG)     : {xc:8.2f} ft fwd, {yc:8.2f} ft left")
        L.append(f"  Fwd force to hold speed   : {ss.thrust_lb:8.0f} lb  (tire scrub drag)")
        radii = {"CG": rad(0, 0)}
        for nm, (px, py) in model.strut_points().items():
            radii[nm] = rad(px, py)
        for nm, (px, py) in r["pts"].items():
            radii[nm] = rad(px, py)
        L.append("  Path radii (ft)           : " + ",  ".join(f"{k} {v:.1f}" for k, v in radii.items()))
        L.append("")
        L.append("TRANSIENT (distance = CG path length from start of roll)")
        L.append(f"  Steering starts at s = {res.steer_start_s:.1f} ft, reaches {d_ss:.2f}° at s = "
                 f"{res.steer_end_s:.1f} ft")
        if res.steady_s is not None:
            L.append(f"  Whole aircraft at steady state (slips within 0.2%) at s = {res.steady_s:.1f} ft")
        else:
            L.append("  *** Steady state NOT reached within max simulated distance ***")
        L.append(f"  Simulated {r['ref']} radius at end: {res.radius_ref[-1]:.2f} ft (target {R:.2f})")
        warn = []
        amax = np.abs(np.degrees(res.alpha)).max()
        lim = min(t.max_slip for t in model.tires.values())
        if amax > lim:
            warn.append(f"max |slip| {amax:.2f}° exceeds tire table range ({lim:.0f}°): force held at last value")
        if np.any(res.fz < 1):
            warn.append("one or more wheels unloaded (Fz = 0) during the run")
        if not ss.ok:
            warn.append("steady-state solver did not converge; check inputs")
        if warn:
            L.append("")
            L += ["WARNING: " + w for w in warn]
        self.summary.delete("1.0", "end")
        self.summary.insert("1.0", "\n".join(L))

        fz0 = model.vertical_loads(*model.geometry(0.0)[:2], 0.0)
        self.table.delete(*self.table.get_children())
        for j, nm in enumerate(model.names):
            st = res.settle_s[j]
            vals = (model.tire_key[j], f"{ss.x[j]:.2f}", f"{ss.y[j]:.2f}", f"{fz0[j]:,.0f}", f"{ss.fz[j]:,.0f}",
                    f"{math.degrees(ss.dw[j]):.2f}", f"{math.degrees(ss.alpha[j]):.3f}", f"{ss.fy[j]:,.0f}",
                    f"{ss.fy[j] / ss.fz[j]:.4f}" if ss.fz[j] > 0 else "-",
                    "not settled" if np.isnan(st) else f"{st:.1f}",
                    "-" if np.isnan(st) else f"{max(st - res.steer_start_s, 0):.1f}",
                    f"{rad(ss.x[j], ss.y[j]):.1f}")
            self.table.insert("", "end", text=nm, values=vals)

    def _wheel_colors(self, n):
        cmap = colormaps["tab20" if n > 10 else "tab10"]
        return [cmap(i % cmap.N) for i in range(n)]

    def _plot_histories(self):
        r = self.result
        model, ss, res, R = r["model"], r["ss"], r["res"], r["R"]
        fig = self.hist.fig
        fig.clear()
        axs = fig.subplots(5, 1, sharex=True)
        s = res.s
        cols = self._wheel_colors(model.n)
        a = axs[0]
        a.plot(s, np.degrees(res.delta), color="k", lw=1.8, label="Nose steer command")
        a.axhline(math.degrees(ss.delta), color="C3", ls="--", lw=0.8, label=f"Required {math.degrees(ss.delta):.2f}°")
        for k, (fn, mask) in enumerate(model._axle_groups):
            wn = model.names[int(np.argmax(mask))]
            label = "Aft-axle steer" if len(model._axle_groups) == 1 else f"Steered axle ({wn})"
            a.plot(s, np.degrees([fn(d) for d in res.delta]), color=f"C{k + 4}", lw=1.4, ls="-.",
                   label=f"{label} (SS {math.degrees(fn(ss.delta)):.2f}°)")
        a.set_ylabel("Steer (deg)")
        a.legend(fontsize=8, loc="lower right")
        for j, nm in enumerate(model.names):
            axs[1].plot(s, np.degrees(res.alpha[:, j]), color=cols[j], lw=1.2, label=nm)
            axs[2].plot(s, res.fy[:, j], color=cols[j], lw=1.2)
            axs[3].plot(s, res.fz[:, j], color=cols[j], lw=1.2)
        axs[1].set_ylabel("Slip angle (deg)")
        axs[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1 if model.n <= 12 else 2)
        axs[2].set_ylabel("Fy (lb)")
        axs[3].set_ylabel("Fz (lb)")
        rr = np.where(np.isfinite(res.radius_ref), res.radius_ref, np.nan)
        rr = np.where(rr < 6 * R, rr, np.nan)
        axs[4].plot(s, rr, color="C0", lw=1.5, label=f"{r['ref']} path radius")
        axs[4].axhline(R, color="C3", ls="--", lw=0.8, label=f"Target {R:.0f} ft")
        axs[4].set_ylabel("Radius (ft)")
        axs[4].set_xlabel("Distance travelled by CG (ft)")
        axs[4].legend(fontsize=8)
        for a in axs:
            a.grid(alpha=0.3)
            a.axvline(res.steer_start_s, color="#888", lw=0.7, ls=":")
            if np.isfinite(res.steer_end_s):
                a.axvline(res.steer_end_s, color="#888", lw=0.7, ls=":")
            if res.steady_s is not None:
                a.axvline(res.steady_s, color="C2", lw=0.9, ls="--")
        axs[0].set_title("Dotted: steer start / end.  Green dashed: whole aircraft at steady state", fontsize=9)
        self.hist.draw()

    def _plot_grid(self):
        r = self.result
        model, ss, res = r["model"], r["ss"], r["res"]
        fig = self.grid_plot.fig
        fig.clear()
        n = model.n
        ncol = 2 if n <= 4 else 3 if n <= 9 else 4
        nrow = math.ceil(n / ncol)
        axs = np.atleast_1d(fig.subplots(nrow, ncol, sharex=True, sharey=True)).ravel()
        s = res.s
        for j in range(n):
            a = axs[j]
            a.plot(s, np.degrees(res.alpha_kin[:, j]), color="#999", lw=0.9, ls="--", label="geometric")
            a.plot(s, np.degrees(res.alpha[:, j]), color="C0", lw=1.6, label="tire (lagged)")
            a.axhline(math.degrees(ss.alpha[j]), color="C3", lw=0.7, ls=":")
            st = res.settle_s[j]
            if np.isfinite(st):
                a.axvline(st, color="C2", lw=0.8, ls="--")
            a.set_title(f"{model.names[j]}\nSS {math.degrees(ss.alpha[j]):.3f}°, "
                        f"settled @ {'—' if np.isnan(st) else f'{st:.0f} ft'}", fontsize=8)
            a.grid(alpha=0.3)
            if j % ncol == 0:
                a.set_ylabel("Slip (deg)")
            if j >= n - ncol:
                a.set_xlabel("CG distance (ft)")
            a.tick_params(labelbottom=True)
        for a in axs[n:]:
            a.set_visible(False)
        axs[0].legend(fontsize=7)
        fig.suptitle("Slip angle per tire  (red dotted = steady state, green dashed = within 2% / 0.02° band)",
                     fontsize=9)
        self.grid_plot.draw()

    # ------------------------------------------------------------------ io
    def _collect(self):
        return dict(
            preset=self.vars["preset"].get(), layout=self.vars["layout"].get(),
            layout_params={k: v.get() for k, v in self.layout_vars.items()},
            vars={k: v.get() for k, v in self.vars.items() if k not in ("preset", "layout")},
            tires={k: p.state() for k, p in self.tire_panels.items()})

    def save_inputs(self):
        fn = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Aircraft Taxi Simulator inputs", "*.json")])
        if not fn:
            return
        with open(fn, "w") as f:
            json.dump(self._collect(), f, indent=2)
        self.status.set(f"Saved inputs to {fn}")

    def load_inputs(self, fn=None):
        fn = fn or filedialog.askopenfilename(filetypes=[("Aircraft Taxi Simulator inputs", "*.json")])
        if not fn:
            return
        with open(fn) as f:
            d = json.load(f)
        self._suspend = True
        try:
            self.vars["preset"].set(d.get("preset", ""))
            self._build_layout_fields(d["layout"], {k: float(v) for k, v in d["layout_params"].items()})
            for k, v in d["vars"].items():
                if k in self.vars:
                    self.vars[k].set(v)
        finally:
            self._suspend = False
        for k, st in d.get("tires", {}).items():
            self.tire_panels[k].set_state(st)
        self.draw_geometry()
        self.status.set(f"Loaded inputs from {fn}")

    def export_csv(self):
        if not self.result:
            messagebox.showinfo("Export", "Run a simulation first.")
            return
        fn = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not fn:
            return
        self.write_csv(fn)
        self.status.set(f"Results written to {fn}")

    def write_csv(self, fn):
        r = self.result
        model, res = r["model"], r["res"]
        head = ["t_s", "s_ft", "nose_steer_deg", "yaw_rate_deg_s", "v_lat_ft_s", "X_ft", "Y_ft", "heading_deg",
                f"radius_{r['ref'].replace(' ', '_')}_ft"]
        for nm in model.names:
            k = nm.replace(" ", "_")
            head += [f"{k}_slip_deg", f"{k}_geom_slip_deg", f"{k}_Fz_lb", f"{k}_Fy_lb"]
        with open(fn, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(head)
            for i in range(len(res.t)):
                row = [res.t[i], res.s[i], math.degrees(res.delta[i]), math.degrees(res.r[i]), res.v[i],
                       res.X[i], res.Y[i], math.degrees(res.psi[i]), res.radius_ref[i]]
                for j in range(model.n):
                    row += [math.degrees(res.alpha[i, j]), math.degrees(res.alpha_kin[i, j]), res.fz[i, j],
                            res.fy[i, j]]
                w.writerow([f"{x:.6g}" for x in row])


def main():
    App().mainloop()
