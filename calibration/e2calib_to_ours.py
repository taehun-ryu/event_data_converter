#!/usr/bin/env python3
# h5_convert_events.py
# Root {p,t,x,y}  ->  /events/{ps,ts,xs,ys} with robust time-unit conversion.

import argparse
from pathlib import Path
import sys
import numpy as np
import h5py

UNIT_SEC = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}

def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_events.h5")

def _get_1d(ds) -> np.ndarray:
    a = np.asarray(ds[...])
    if a.ndim == 2 and 1 in a.shape:
        a = a.ravel()
    if a.ndim != 1:
        raise ValueError(f"{ds.name} must be 1-D, got {a.shape}")
    return a

def _pick_key(h5: h5py.File, *names: str) -> str:
    low = {k.lower(): k for k in h5.keys()}
    for n in names:
        if n in h5: return n
        if n.lower() in low: return low[n.lower()]
    raise KeyError(f"Missing any of {names}; found {list(h5.keys())}")

def _sec_per_in(in_unit: str, tick_hz: float | None) -> float:
    if in_unit in UNIT_SEC:
        return UNIT_SEC[in_unit]
    if in_unit == "ticks":
        if not tick_hz or tick_hz <= 0:
            raise ValueError("--tick-hz must be > 0 when --in-unit ticks")
        return 1.0 / float(tick_hz)
    raise ValueError(f"Unsupported in-unit: {in_unit}")

def convert(inp: Path, out: Path, shift_t0: bool, in_unit: str, out_unit: str, tick_hz: float | None):
    if not inp.exists():
        raise FileNotFoundError(inp)
    if out.exists():
        raise FileExistsError(f"Output exists: {out}")

    with h5py.File(inp, "r") as src:
        k_p = _pick_key(src, "p", "ps")
        k_t = _pick_key(src, "ts", "time", "ts")
        k_x = _pick_key(src, "x", "xs")
        k_y = _pick_key(src, "y", "ys")

        p = _get_1d(src[k_p])
        t = _get_1d(src[k_t]).astype(np.float64)  # float64 for safe math
        x = _get_1d(src[k_x])
        y = _get_1d(src[k_y])

        n = len(t)
        if not (len(p) == len(x) == len(y) == n):
            raise ValueError(f"Length mismatch: |t|={len(t)}, |x|={len(x)}, |y|={len(y)}, |p|={len(p)}")

        # optional t0 shift (NaN-safe)
        if shift_t0:
            if np.all(np.isnan(t)):
                raise ValueError("All times are NaN")
            t -= np.nanmin(t)

        # unit conversion: t_in_seconds -> t_out_unit
        sec_per_in = _sec_per_in(in_unit, tick_hz)
        sec_per_out = UNIT_SEC[out_unit]
        t = (t * sec_per_in) / sec_per_out  # stays float64

        # basic monotonicity check (allow equality)
        if np.any(np.diff(t) < -1e-12):
            print("WARN: ts not non-decreasing after conversion", file=sys.stderr)

        with h5py.File(out, "w") as dst:
            for k, v in src.attrs.items():
                dst.attrs[k] = v
            g = dst.create_group("events")
            ds_ps = g.create_dataset("ps", data=p, compression="gzip")
            ds_ts = g.create_dataset("ts", data=t, compression="gzip")
            ds_xs = g.create_dataset("xs", data=x, compression="gzip")
            ds_ys = g.create_dataset("ys", data=y, compression="gzip")

            # annotate time unit & t0 shift info
            g.attrs["time_unit"] = out_unit
            ds_ts.attrs["unit"] = out_unit
            ds_ts.attrs["source_time_unit"] = in_unit
            ds_ts.attrs["t0_shifted"] = bool(shift_t0)

    print(f"✓ {inp} → {out}  (N={len(t)}, ts unit={out_unit})")

def main():
    ap = argparse.ArgumentParser(description="Convert {p,t,x,y}@root → /events/{ps,ts,xs,ys} with time scaling.")
    ap.add_argument("input", help="Input .h5 path containing p,t,x,y at root")
    ap.add_argument("-o", "--output", help="Output .h5 path (default: <input>_events.h5)")
    ap.add_argument("--t0", action="store_true", help="Shift time so min(ts)=0")
    ap.add_argument("--in-unit", choices=["s","ms","us","ns","ticks"], default="us",
                    help="Unit of input t (default: us)")
    ap.add_argument("--out-unit", choices=["s","ms","us","ns"], default="s",
                    help="Unit of output ts (default: s)")
    ap.add_argument("--tick-hz", type=float, help="Clock frequency (Hz) when --in-unit ticks")
    args = ap.parse_args()

    inp = Path(args.input).expanduser().resolve()
    out = Path(args.output).expanduser().resolve() if args.output else _default_output_path(inp)
    try:
        convert(inp, out, args.t0, args.in_unit, args.out_unit, args.tick_hz)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
