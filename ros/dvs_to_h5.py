#!/usr/bin/env python3
import os
import glob
import argparse

import h5py
import numpy as np
from tqdm import tqdm

import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

from dvs_msgs.msg import EventArray

def timestamp_float(ros_time):
    return ros_time.sec + ros_time.nanosec * 1e-9


def append_to_dataset(ds, data):
    if len(data) == 0:
        return
    old = ds.shape[0]
    ds.resize((old + len(data),))
    ds[old:] = data


def resolve_output_path(input_path, output_dir=None, output_name=None):
    base = output_name or os.path.splitext(os.path.basename(input_path))[0]
    if base.endswith(".h5"):
        base = os.path.splitext(base)[0]
    out_dir = output_dir or os.path.dirname(os.path.abspath(input_path))
    return os.path.join(out_dir, f"{base}.h5")


def extract_events_ros2(bag_path, h5_path, event_topic, zero_ts=False):
    # --- open the bag
    storage_opts = StorageOptions(uri=bag_path, storage_id='sqlite3')
    conv_opts    = ConverterOptions('', '')
    reader = SequentialReader()
    reader.open(storage_opts, conv_opts)

    # --- prepare HDF5 file
    os.makedirs(os.path.dirname(h5_path) or ".", exist_ok=True)
    f = h5py.File(h5_path, 'w')
    grp = f.create_group('events')
    ds_x = grp.create_dataset('xs', shape=(0,), maxshape=(None,), dtype='uint16', compression="gzip", chunks=True)
    ds_y = grp.create_dataset('ys', shape=(0,), maxshape=(None,), dtype='uint16', compression="gzip", chunks=True)
    ds_t = grp.create_dataset('ts', shape=(0,), maxshape=(None,), dtype='float64', compression="gzip", chunks=True)
    ds_p = grp.create_dataset('ps', shape=(0,), maxshape=(None,), dtype='uint8', compression="gzip", chunks=True)

    # count how many messages on the event topic
    meta = reader.get_metadata()
    total_msgs = sum(e.message_count
                     for e in meta.topics_with_message_count
                     if e.topic_metadata.name == event_topic)

    first_event_ts = None
    collected = 0

    with tqdm(total=total_msgs, desc="Reading events") as pbar:
        while reader.has_next():
            topic_name, ser_data, _ = reader.read_next()
            if topic_name != event_topic:
                continue

            msg = deserialize_message(ser_data, EventArray)
            if len(msg.events) == 0:
                pbar.update(1)
                continue

            if zero_ts and first_event_ts is None:
                first_event_ts = timestamp_float(msg.events[0].ts)

            xs, ys, ts, ps = [], [], [], []
            for e in msg.events:
                raw_t = timestamp_float(e.ts)
                if zero_ts and first_event_ts is not None:
                    raw_t -= first_event_ts
                xs.append(e.x)
                ys.append(e.y)
                ts.append(raw_t)
                ps.append(1 if e.polarity else 0)

            append_to_dataset(ds_x, np.array(xs, dtype='uint16'))
            append_to_dataset(ds_y, np.array(ys, dtype='uint16'))
            append_to_dataset(ds_t, np.array(ts, dtype='float64'))
            append_to_dataset(ds_p, np.array(ps, dtype='uint8'))

            collected += len(xs)
            pbar.update(1)

    f.close()
    print(f"✓ Wrote {collected} events to {h5_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROS2 .db3 → HDF5 (events only)")
    parser.add_argument("input", help="rosbag2 .db3 file or directory")
    parser.add_argument(
        "--output_dir", "-o", default=None,
        help="output directory for .h5 (default: same directory as each input .db3)"
    )
    parser.add_argument(
        "--output_name", "-n", default=None,
        help="output filename without extension (single input only)"
    )
    parser.add_argument(
        "--event_topic", "--topic", "-t", dest="event_topic",
        default="/dvs/events", help="event topic name"
    )
    parser.add_argument(
        "--zero_ts", "-z", action="store_true",
        help="subtract first timestamp (i.e. make first event t=0)"
    )
    args = parser.parse_args()

    rclpy.init()
    try:
        if os.path.isdir(args.input):
            bags = sorted(glob.glob(os.path.join(args.input, "*.db3")))
        else:
            bags = [args.input]

        if len(bags) == 0:
            print(f"[-] No .db3 files found in: {args.input}")
            raise SystemExit(0)

        is_single_input = len(bags) == 1
        if args.output_name and not is_single_input:
            print("[!] --output_name is only applied for a single input .db3; ignoring it.")

        for bag in bags:
            output_name = args.output_name if is_single_input else None
            out_path = resolve_output_path(
                bag,
                output_dir=args.output_dir,
                output_name=output_name
            )

            print(f"\n→ Extracting events from {bag} → {out_path}")
            extract_events_ros2(bag, out_path, args.event_topic, zero_ts=args.zero_ts)
    finally:
        rclpy.shutdown()
