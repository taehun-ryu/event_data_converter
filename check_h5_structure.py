#!/usr/bin/env python3
import argparse
from pathlib import Path
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
    return None

def _human(x):
    return f"{x:.3g}"

def main():
    ap = argparse.ArgumentParser(description="Quickly inspect t and p (root or /events)")
    ap.add_argument("h5", type=Path, help="Input .h5 path")
    ap.add_argument("--events", action="store_true", help="Prefer /events/* if present")
    ap.add_argument("--max", type=int, default=2_000_000, help="Max samples to read")
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as f:
        ds_t = None
        ds_p = None

        if args.events:
            ge = _find_events_group(f)
            if ge is not None:
                try:
                    ds_t = ge[_pick_key(ge, "ts", "time")]
                    ds_p = ge[_pick_key(ge, "ps", "p")]
                except Exception:
                    ds_t = None
                    ds_p = None

        if ds_t is None or ds_p is None:
            if "t" in f and "p" in f:
                ds_t = f["t"]
                ds_p = f["p"]
            else:
                raise KeyError("Cannot find timestamps/polarity: tried /events and root.")

        # timestamps
        n_t = len(ds_t)
        m_t = min(n_t, args.max)
        t = _get_1d(ds_t)[:m_t]
        t_dtype = t.dtype
        t_is_int = np.issubdtype(t_dtype, np.integer)
        unit_attr = ds_t.attrs.get("unit", None)

        tmin = float(np.min(t)) if t.size else float("nan")
        tmax = float(np.max(t)) if t.size else float("nan")
        span = tmax - tmin
        if t.size >= 2:
            dt = np.diff(t.astype(np.float64))
            nz = dt[np.abs(dt) > 0]
            q = np.quantile(nz, [0.1, 0.5, 0.9]) if nz.size else [0.0, 0.0, 0.0]
        else:
            q = [np.nan, np.nan, np.nan]

        # crude unit guess
        if t_is_int:
            if 1e8 <= tmax <= 1e12:
                guess = "likely microseconds (us)"
            elif 1e11 <= tmax <= 1e15:
                guess = "possibly nanoseconds (ns)"
            elif tmax < 1e6:
                guess = "small integer range; maybe s/ms"
            else:
                guess = "unknown integer scale"
        else:
            guess = "likely seconds (float)" if tmax < 1e6 else "float with large magnitude (maybe ms)"

        print(f"[t] N={m_t}/{n_t} dtype={t_dtype} integer={t_is_int} unit_attr={unit_attr}")
        print(f"    min={_human(tmin)}  max={_human(tmax)}  span={_human(span)}")
        print(f"    nonzero Δt quantiles[0.1,0.5,0.9]= {q}")
        print(f"    -> unit_guess: {guess}")
        print(f"    first {min(10, m_t)} samples:", t[:min(10, m_t)])

        # polarity
        n_p = len(ds_p)
        m_p = min(n_p, args.max)
        p = _get_1d(ds_p)[:m_p]
        uniq = np.unique(p)
        print(f"[p] N={m_p}/{n_p} dtype={p.dtype} unique={uniq}")
        print(f"    first {min(50, m_p)} samples:", p[:min(50, m_p)])

if __name__ == "__main__":
    main()
