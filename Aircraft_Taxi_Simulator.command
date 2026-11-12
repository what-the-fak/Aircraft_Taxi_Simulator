#!/bin/bash
# Aircraft Taxi Simulator launcher for macOS - double-click in Finder.
# First run creates a private Python environment (.venv) and installs numpy + matplotlib.
cd "$(dirname "$0")" || exit 1

fail() { echo; echo "ERROR: $1"; echo "Press any key to close."; read -n 1; exit 1; }

if [ ! -f .venv/.ats_ready ]; then
    PY=""
    for c in python3.13 python3.12 python3.11 python3.10 python3; do
        if command -v "$c" >/dev/null 2>&1 && "$c" -c "import tkinter" >/dev/null 2>&1; then PY="$c"; break; fi
    done
    [ -n "$PY" ] || fail "Python 3 with tkinter not found. Install Python from https://www.python.org/downloads/"
    echo "First run: setting up Python environment with $PY (one minute)..."
    [ -x .venv/bin/python ] || "$PY" -m venv .venv || fail "could not create .venv"
    .venv/bin/python -m pip install --quiet --upgrade pip || fail "pip upgrade failed"
    .venv/bin/python -m pip install --quiet -r requirements.txt || fail "package install failed (internet connection?)"
    touch .venv/.ats_ready
fi

.venv/bin/python main.py || fail "Aircraft Taxi Simulator exited with an error (see above)."
