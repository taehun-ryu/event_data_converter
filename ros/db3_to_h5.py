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

# TODO: need to set time to microseconds
def timestamp_float(ros_time):
    return ros_time.sec + ros_time.nanosec * 1e-9


def append_to_dataset(ds, data):
    if len(data) == 0:
        return
    old = ds.shape[0]
    ds.resize((old + len(data),))
    ds[old:] = data


def extract_events_ros2(bag_path, h5_path, event_topic, zero_ts=False):
    # --- open the bag
    storage_opts = StorageOptions(uri=bag_path, storage_id='sqlite3')
    conv_opts    = ConverterOptions('', '')
    reader = SequentialReader()
    reader.open(storage_opts, conv_opts)

    # --- prepare HDF5 file
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    f = h5py.File(h5_path, 'w')
    grp = f.create_group('events')
    ds_x = grp.create_dataset('xs', shape=(0,), maxshape=(None,), dtype='uint16')
    ds_y = grp.create_dataset('ys', shape=(0,), maxshape=(None,), dtype='uint16')
    ds_t = grp.create_dataset('ts', shape=(0,), maxshape=(None,), dtype='float64')
    ds_p = grp.create_dataset('ps', shape=(0,), maxshape=(None,), dtype='uint8')

    # count how many messages on the event topic
    meta = reader.get_metadata()
    total_msgs = sum(e.message_count
                     for e in meta.topics_with_message_count
                     if e.topic_metadata.name == event_topic)

    first_ts = None
    collected = 0

    for _ in tqdm(range(total_msgs), desc="Reading events"):
        if not reader.has_next():
            break

        # unpack the tuple (topic, serialized_data, time_stamp)
        topic_name, ser_data, t_ns = reader.read_next()
        if topic_name != event_topic:
            continue

        # deserialize the EventArray message
        msg = deserialize_message(ser_data, EventArray)

        # bag timestamp is in nanoseconds
        t0 = t_ns * 1e-9
        if first_ts is None:
            first_ts = t0

        # collect per-event fields
        xs, ys, ts, ps = [], [], [], []
        for e in msg.events:
            raw_t = timestamp_float(e.ts)
            rel_t = raw_t - (first_ts if zero_ts else 0.0)
            xs.append(e.x)
            ys.append(e.y)
            ts.append(rel_t)
            ps.append(1 if e.polarity else 0)

        # append into the datasets
        append_to_dataset(ds_x, np.array(xs, dtype='uint16'))
        append_to_dataset(ds_y, np.array(ys, dtype='uint16'))
        append_to_dataset(ds_t, np.array(ts, dtype='float64'))
        append_to_dataset(ds_p, np.array(ps, dtype='uint8'))

        collected += len(xs)

    f.close()
    print(f"✓ Wrote {collected} events to {h5_path}")


if __name__ == "__main__":
    rclpy.init()
    parser = argparse.ArgumentParser(description="ROS2 .db3 → HDF5 (events only)")
    parser.add_argument("input", help="rosbag2 folder or .db3 file")
    parser.add_argument(
        "output_dir", nargs="?", default=None,
        help="where to write *.h5 (default: same folder as each .db3)"
    )
    parser.add_argument("--topic",   default="/dvs/events", help="event topic name")
    parser.add_argument(
        "--zero_ts", action="store_true",
        help="subtract first timestamp (i.e. make first event t=0)"
    )
    args = parser.parse_args()

    # gather .db3 paths
    if os.path.isdir(args.input):
        bags = sorted(glob.glob(os.path.join(args.input, "*.db3")))
    else:
        bags = [args.input]

    for bag in bags:
        base = os.path.splitext(os.path.basename(bag))[0]
        if args.output_dir:
            out_dir = args.output_dir
        else:
            bag_folder = os.path.dirname(bag)
            out_dir    = os.path.dirname(bag_folder)

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{base}.h5")

        print(f"\n→ Extracting events from {bag} → {out_path}")
        extract_events_ros2(bag, out_path, args.topic, zero_ts=args.zero_ts)

    rclpy.shutdown()