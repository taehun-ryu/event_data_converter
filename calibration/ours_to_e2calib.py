#!/usr/bin/env python3
# ours_to_e2calib_us_int.py
# /events/{ps,ts,xs,ys} -> root {p,t,x,y}
# ts(seconds float) -> t(microseconds int64)

import argparse
from pathlib import Path
import sys
import numpy as np
import h5py

def _get_1d(ds):
    a = np.asarray(ds[...])
    if a.ndim == 2 and 1 in a.shape:
        a = a.ravel()
    if a.ndim != 1:
        raise ValueError(f"{ds.name} must be 1-D, got {a.shape}")
    return a

def _pick_key(obj, *names):
    low = {k.lower(): k for k in obj.keys()}
    for n in names:
        if n in obj:
            return n
        if n.lower() in low:
            return low[n.lower()]
    raise KeyError(f"Missing any of {names}; found {list(obj.keys())}")

def _find_events_group(h5):
    if "events" in h5:
        return h5["events"]
    low = {k.lower(): k for k in h5.keys()}
    if "events" in low:
        return h5[low["events"]]
    raise KeyError("Missing /events group")

def convert(inp: Path, out: Path):
    if not inp.exists():
        raise FileNotFoundError(inp)
    if out.exists():
        raise FileExistsError(f"Output exists: {out}")

    with h5py.File(inp, "r") as src:
        ge = _find_events_group(src)
        ps = _get_1d(ge[_pick_key(ge, "ps", "p")])
        ts = _get_1d(ge[_pick_key(ge, "ts", "time")])
        xs = _get_1d(ge[_pick_key(ge, "xs", "x")])
        ys = _get_1d(ge[_pick_key(ge, "ys", "y")])

        n = len(ts)
        if not (len(ps) == len(xs) == len(ys) == n):
            raise ValueError(f"Length mismatch: |t|={len(ts)}, |x|={len(xs)}, |y|={len(ys)}, |p|={len(ps)}")

        # seconds(float) -> microseconds(int64)
        t_us = np.rint(ts.astype(np.float64)).astype(np.int64)

        with h5py.File(out, "w") as dst:
            dst.create_dataset("p", data=ps)
            dst.create_dataset("t", data=t_us)
            dst.create_dataset("x", data=xs)
            dst.create_dataset("y", data=ys)

    print(f"✓ {inp} → {out}  (N={n})  [ts(seconds)->t(int64 microseconds)]")

def main():
    ap = argparse.ArgumentParser(description="Convert /events to root AND seconds->microseconds(int)")
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path,
                    help="Output .h5 (default: <input>_e2calib.h5)")
    args = ap.parse_args()

    inp = args.input.expanduser().resolve()
    out = args.output or inp.with_name(f"{inp.stem}_e2calib.h5")

    try:
        convert(inp, out)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
