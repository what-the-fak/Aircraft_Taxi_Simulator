"""Simple aircraft planform for drawing (gear-frame ft, x fwd, y left)."""
from __future__ import annotations

import numpy as np


def planform(o: dict, x_nose_gear: float):
    """Return (polys, points). polys: list of (N,2) arrays; points: dict of named (x,y)."""
    L, b, w = o["length"], o["span"], o["fus_width"]
    xn = x_nose_gear + o["nose_ahead"]
    xt = xn - L
    nose_len, tail_len = 1.6 * w, 3.0 * w
    th = np.linspace(0, np.pi / 2, 10)
    left = [(xn - nose_len + nose_len * np.cos(t), w / 2 * np.sin(t)) for t in th]
    left += [(xt + tail_len, w / 2), (xt, 0.12 * w)]
    left = np.array(left)
    right = left[::-1] * [1, -1]
    fus = np.vstack([left, right])

    sweep = np.radians(o["wing_sweep"])
    cr = o["wing_root_chord"]
    ct = 0.22 * cr
    xle = o["wing_le"]
    xtip = xle - (b / 2 - w / 2) * np.tan(sweep)
    wing_l = np.array([(xle, w / 2), (xtip, b / 2), (xtip - ct, b / 2), (xle - cr, w / 2)])

    hs = 0.2 * b
    hcr = 0.45 * cr
    hx = xt + 0.17 * L
    htip = hx - (hs - w / 4) * np.tan(np.radians(33))
    stab_l = np.array([(hx, w / 4), (htip, hs), (htip - 0.35 * hcr, hs), (hx - hcr, w / 4)])

    polys = [fus, wing_l, wing_l * [1, -1], stab_l, stab_l * [1, -1]]

    n_eng = int(o.get("n_engines", 1))
    fracs = [0.34] if n_eng == 1 else [0.40, 0.68][:n_eng]
    nl, nd = 0.12 * L if L < 160 else 0.08 * L, 0.57 * w
    for f in fracs:
        ye = f * b / 2
        xle_e = xle - (ye - w / 2) * np.tan(sweep)
        xf = xle_e + 0.55 * nl
        eng = np.array([(xf, ye + nd / 2), (xf, ye - nd / 2), (xf - nl, ye - nd / 2), (xf - nl, ye + nd / 2)])
        polys += [eng, eng * [1, -1]]

    pts = {"Nose tip": (xn, 0.0), "Tail": (xt, 0.0),
           "L wingtip": (xtip - ct / 2, b / 2), "R wingtip": (xtip - ct / 2, -b / 2)}
    return polys, pts
