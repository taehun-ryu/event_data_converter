#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

rclpy = None
deserialize_message = None
ConverterOptions = None
SequentialReader = None
StorageOptions = None
EventPacket = None


@dataclass
class BagInput:
    uri: Path
    label: str


def load_ros2_dependencies() -> None:
    global rclpy, deserialize_message, ConverterOptions
    global SequentialReader, StorageOptions
    global EventPacket

    if rclpy is not None:
        return

    try:
        import rclpy as _rclpy
        from rclpy.serialization import deserialize_message as _deserialize_message
        from rosbag2_py import ConverterOptions as _ConverterOptions
        from rosbag2_py import SequentialReader as _SequentialReader
        from rosbag2_py import StorageOptions as _StorageOptions
    except ImportError as exc:
        raise RuntimeError(
            "Missing ROS2 dependencies. Please source your ROS2 workspace "
            "and ensure rclpy and rosbag2_py are installed."
        ) from exc

    try:
        from event_camera_msgs.msg import EventPacket as _EventPacket
    except ImportError as exc:
        raise RuntimeError(
            "Missing event_camera_msgs. Please install/build event_camera_msgs "
            "in your ROS2 workspace."
        ) from exc

    rclpy = _rclpy
    deserialize_message = _deserialize_message
    ConverterOptions = _ConverterOptions
    SequentialReader = _SequentialReader
    StorageOptions = _StorageOptions
    EventPacket = _EventPacket


def discover_inputs(input_path: Path) -> list[BagInput]:
    path = input_path.expanduser().resolve()

    if path.is_file():
        if path.suffix != ".db3":
            raise ValueError(f"Expected a .db3 file, got: {path}")
        return [BagInput(uri=path, label=path.stem)]

    if path.is_dir():
        if (path / "metadata.yaml").is_file():
            return [BagInput(uri=path, label=path.name)]

        db3_files = sorted(path.glob("*.db3"))
        if db3_files:
            return [BagInput(uri=p, label=p.stem) for p in db3_files]

        bag_dirs = sorted(
            p for p in path.iterdir()
            if p.is_dir() and (p / "metadata.yaml").is_file()
        )
        if bag_dirs:
            return [BagInput(uri=p, label=p.name) for p in bag_dirs]

    raise FileNotFoundError(
        f"No rosbag2 input found at {path}. "
        "Use a .db3 file, a rosbag2 folder(with metadata.yaml), "
        "or a directory containing those."
    )


def open_reader(uri: Path) -> tuple[object, Path]:
    candidates = [uri]
    if uri.is_file() and uri.suffix == ".db3" and (uri.parent / "metadata.yaml").is_file():
        candidates.append(uri.parent)

    errors = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)

        reader = SequentialReader()
        storage_opts = StorageOptions(uri=str(candidate), storage_id="sqlite3")
        conv_opts = ConverterOptions("", "")
        try:
            reader.open(storage_opts, conv_opts)
            return reader, candidate
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    raise RuntimeError("Failed to open rosbag2 source.\n" + "\n".join(errors))


def topic_to_file_stem(topic_name: str) -> str:
    stem = str(topic_name).strip().strip("/")
    if not stem:
        stem = "events"
    stem = stem.replace("/", "_")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    stem = stem.strip("._-")
    if not stem:
        stem = "events"
    return stem


def default_bag_output_dir(source_uri: Path) -> Path:
    if source_uri.is_dir():
        return source_uri
    return source_uri.parent


def append_to_dataset(ds: h5py.Dataset, data: np.ndarray) -> None:
    if data.size == 0:
        return
    old_size = ds.shape[0]
    ds.resize((old_size + data.shape[0],))
    ds[old_size:] = data


def _decode_mono(events_bytes: bytes, time_base_ns: int, is_bigendian: bool):
    if len(events_bytes) % 8 != 0:
        raise ValueError(f"mono packet byte size is not a multiple of 8: {len(events_bytes)}")
    if len(events_bytes) == 0:
        return (
            np.array([], dtype=np.uint16),
            np.array([], dtype=np.uint16),
            np.array([], dtype=np.uint8),
            np.array([], dtype=np.int64),
        )

    byte_order = ">" if is_bigendian else "<"
    words = np.frombuffer(events_bytes, dtype=f"{byte_order}u8")
    dt_ns = (words & np.uint64(0xFFFFFFFF)).astype(np.int64)
    xs = ((words >> np.uint64(32)) & np.uint64(0xFFFF)).astype(np.uint16)
    ys = ((words >> np.uint64(48)) & np.uint64(0x7FFF)).astype(np.uint16)
    ps = ((words >> np.uint64(63)) & np.uint64(0x1)).astype(np.uint8)
    ts_ns = dt_ns + np.int64(time_base_ns)
    return xs, ys, ps, ts_ns


def _decode_libcaer(events_bytes: bytes, time_base_ns: int, is_bigendian: bool):
    if len(events_bytes) % 8 != 0:
        raise ValueError(f"libcaer packet byte size is not a multiple of 8: {len(events_bytes)}")
    if len(events_bytes) == 0:
        return (
            np.array([], dtype=np.uint16),
            np.array([], dtype=np.uint16),
            np.array([], dtype=np.uint8),
            np.array([], dtype=np.int64),
        )

    byte_order = ">" if is_bigendian else "<"
    words = np.frombuffer(events_bytes, dtype=f"{byte_order}u4").reshape(-1, 2)
    data = words[:, 0]
    ts_low = words[:, 1].astype(np.uint64)

    xs = ((data >> np.uint32(17)) & np.uint32(0x7FFF)).astype(np.uint16)
    ys = ((data >> np.uint32(2)) & np.uint32(0x7FFF)).astype(np.uint16)
    ps = ((data >> np.uint32(1)) & np.uint32(0x1)).astype(np.uint8)

    rollover = np.zeros(ts_low.shape[0], dtype=np.uint64)
    if ts_low.shape[0] > 1:
        rollover[1:] = np.cumsum(ts_low[1:] < ts_low[:-1], dtype=np.uint64)
    ts_high = np.uint64(time_base_ns) + rollover * np.uint64((1 << 31) * 1000)
    ts_ns = (ts_high + ts_low * np.uint64(1000)).astype(np.int64)

    return xs, ys, ps, ts_ns


def _decode_libcaer_cmp(
    events_bytes: bytes,
    time_base_ns: int,
    width: int,
    height: int,
    is_bigendian: bool,
):
    if len(events_bytes) % 2 != 0:
        raise ValueError(f"libcaer_cmp packet byte size is not a multiple of 2: {len(events_bytes)}")
    if len(events_bytes) == 0:
        return (
            np.array([], dtype=np.uint16),
            np.array([], dtype=np.uint16),
            np.array([], dtype=np.uint8),
            np.array([], dtype=np.int64),
        )

    byte_order = ">" if is_bigendian else "<"
    tokens = np.frombuffer(events_bytes, dtype=f"{byte_order}u2")

    code_time_high = 0
    code_time_low = 1
    code_addr_x = 2
    code_addr_y = 3
    code_vect_base_y = 4
    code_vect_8 = 5

    x_list = []
    y_list = []
    p_list = []
    t_list = []

    ex = 0
    time_low = 0
    time_high = 0
    current_polarity = 0
    current_base_y = 0

    for token in tokens:
        token = int(token)
        code = token >> 12
        rest = token & 0x0FFF

        if code == code_addr_x:
            ex = rest & 0x07FF
        elif code == code_time_high:
            time_high = (rest & 0x0FFF) << 12
        elif code == code_time_low:
            time_low = rest & 0x0FFF
        elif code == code_addr_y:
            y = rest & 0x07FF
            polarity = (rest >> 11) & 0x1
            if ex < width and y < height:
                x_list.append(ex)
                y_list.append(y)
                p_list.append(polarity)
                t_list.append((time_high | time_low) * 1000 + time_base_ns)
        elif code == code_vect_base_y:
            current_base_y = rest & 0x07FF
            current_polarity = (rest >> 11) & 0x1
        elif code == code_vect_8:
            valid = rest & 0x00FF
            if valid:
                t_ns = (time_high | time_low) * 1000 + time_base_ns
                for i in range(8):
                    if valid & (1 << i):
                        y = current_base_y + i
                        if ex < width and y < height:
                            x_list.append(ex)
                            y_list.append(y)
                            p_list.append(current_polarity)
                            t_list.append(t_ns)
            current_base_y += 8

    return (
        np.asarray(x_list, dtype=np.uint16),
        np.asarray(y_list, dtype=np.uint16),
        np.asarray(p_list, dtype=np.uint8),
        np.asarray(t_list, dtype=np.int64),
    )


def decode_event_packet(msg):
    encoding = str(msg.encoding).lower()
    payload = bytes(msg.events)
    time_base_ns = int(msg.time_base)
    width = int(msg.width)
    height = int(msg.height)
    is_bigendian = bool(msg.is_bigendian)

    if encoding == "mono":
        return _decode_mono(payload, time_base_ns, is_bigendian)
    if encoding == "libcaer":
        return _decode_libcaer(payload, time_base_ns, is_bigendian)
    if encoding == "libcaer_cmp":
        return _decode_libcaer_cmp(payload, time_base_ns, width, height, is_bigendian)

    if encoding == "trigger":
        raise ValueError(
            "trigger encoding is not an image event stream. "
            "Use an event topic with CD events (e.g. /event_camera/events)."
        )
    raise ValueError(
        f"Unsupported EventPacket encoding: {msg.encoding}. "
        "Supported encodings: mono, libcaer, libcaer_cmp."
    )


def extract_events_ros2_to_ours(
    bag_uri: Path,
    out_h5: Path,
    event_topic: str,
    zero_ts: bool = False,
    overwrite: bool = False,
    buffer_events: int = 2_000_000,
) -> int:
    if out_h5.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {out_h5} (use --overwrite)")
        out_h5.unlink()
    out_h5.parent.mkdir(parents=True, exist_ok=True)

    reader, opened_uri = open_reader(bag_uri)
    metadata = reader.get_metadata()

    total_msgs = 0
    topic_type = None
    for topic_info in metadata.topics_with_message_count:
        if topic_info.topic_metadata.name == event_topic:
            total_msgs = topic_info.message_count
            topic_type = topic_info.topic_metadata.type
            break

    if topic_type is None:
        available = ", ".join(t.topic_metadata.name for t in metadata.topics_with_message_count)
        raise ValueError(f"Topic not found: {event_topic}. Available topics: {available}")

    accepted_types = {
        "event_camera_msgs/msg/EventPacket",
        "event_camera_msgs/EventPacket",
    }
    if topic_type not in accepted_types:
        raise RuntimeError(
            f"Unsupported topic type for this converter: {topic_type}. "
            "Expected event_camera_msgs/msg/EventPacket."
        )

    first_ts_us = None
    last_ts_us = None
    total_events = 0

    with h5py.File(out_h5, "w") as hf:
        grp = hf.create_group("events")
        ds_ts = grp.create_dataset(
            "ts", shape=(0,), maxshape=(None,), dtype=np.uint64,
            compression="gzip", chunks=True
        )
        ds_xs = grp.create_dataset(
            "xs", shape=(0,), maxshape=(None,), dtype=np.uint16,
            compression="gzip", chunks=True
        )
        ds_ys = grp.create_dataset(
            "ys", shape=(0,), maxshape=(None,), dtype=np.uint16,
            compression="gzip", chunks=True
        )
        ds_ps = grp.create_dataset(
            "ps", shape=(0,), maxshape=(None,), dtype=np.uint8,
            compression="gzip", chunks=True
        )

        buf_xs = []
        buf_ys = []
        buf_ts = []
        buf_ps = []
        buffered = 0

        def flush() -> None:
            nonlocal buf_xs, buf_ys, buf_ts, buf_ps, buffered
            if buffered == 0:
                return
            xs = np.concatenate(buf_xs, axis=0)
            ys = np.concatenate(buf_ys, axis=0)
            ts = np.concatenate(buf_ts, axis=0)
            ps = np.concatenate(buf_ps, axis=0)

            append_to_dataset(ds_xs, xs)
            append_to_dataset(ds_ys, ys)
            append_to_dataset(ds_ts, ts)
            append_to_dataset(ds_ps, ps)

            buf_xs = []
            buf_ys = []
            buf_ts = []
            buf_ps = []
            buffered = 0

        pbar = tqdm(total=total_msgs, desc=f"Reading {opened_uri.name}", unit="msg")
        while reader.has_next():
            topic_name, ser_data, _ = reader.read_next()
            if topic_name != event_topic:
                continue

            msg = deserialize_message(ser_data, EventPacket)
            xs, ys, ps, ts_ns = decode_event_packet(msg)
            if ts_ns.size == 0:
                pbar.update(1)
                continue

            ts_us_i64 = (ts_ns // np.int64(1000)).astype(np.int64)
            if first_ts_us is None:
                first_ts_us = int(ts_us_i64[0])
            if zero_ts:
                ts_us_i64 = ts_us_i64 - np.int64(first_ts_us)

            if ts_us_i64.shape[0] > 1 and np.any(ts_us_i64[1:] < ts_us_i64[:-1]):
                raise ValueError("Decoded event timestamps are not monotonic inside one packet.")
            if last_ts_us is not None and int(ts_us_i64[0]) < last_ts_us:
                raise ValueError(
                    "Decoded event timestamps are not monotonic across packets: "
                    f"{int(ts_us_i64[0])} < {last_ts_us}"
                )
            if np.any(ts_us_i64 < 0):
                raise ValueError(
                    "Negative timestamps detected after conversion. "
                    "Check packet timestamps and --zero_ts usage."
                )
            last_ts_us = int(ts_us_i64[-1])

            ts_us = ts_us_i64.astype(np.uint64, copy=False)

            buf_xs.append(xs)
            buf_ys.append(ys)
            buf_ts.append(ts_us)
            buf_ps.append(ps)
            buffered += xs.shape[0]
            total_events += xs.shape[0]

            if buffered >= buffer_events:
                flush()

            pbar.update(1)
        pbar.close()
        flush()

    return total_events


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ROS2 db3/rosbag2(EventPacket) -> ours HDF5 (/events/{ts,xs,ys,ps})"
    )
    parser.add_argument(
        "input",
        help="a .db3 file, a rosbag2 folder(with metadata.yaml), "
        "or a directory containing those",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="output directory (default: each bag folder)",
    )
    parser.add_argument("--topic", default="/event_camera/events", help="EventPacket topic name")
    parser.add_argument(
        "--zero_ts",
        action="store_true",
        help="subtract first event timestamp so first ts becomes zero",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite output file if it already exists",
    )
    parser.add_argument(
        "--buffer_events",
        type=int,
        default=2_000_000,
        help="number of buffered events before writing to disk",
    )
    args = parser.parse_args()

    sources = discover_inputs(Path(args.input))
    out_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else None

    load_ros2_dependencies()

    rclpy.init()
    try:
        topic_stem = topic_to_file_stem(args.topic)
        used_output_paths = set()

        for source in sources:
            out_dir = out_root if out_root else default_bag_output_dir(source.uri)
            out_path = out_dir / f"{topic_stem}.h5"
            if out_path in used_output_paths:
                out_path = out_dir / f"{topic_stem}_{source.label}.h5"
            used_output_paths.add(out_path)

            print(f"\n→ Extracting EventPacket from {source.uri} → {out_path}")
            count = extract_events_ros2_to_ours(
                bag_uri=source.uri,
                out_h5=out_path,
                event_topic=args.topic,
                zero_ts=args.zero_ts,
                overwrite=args.overwrite,
                buffer_events=args.buffer_events,
            )
            print(f"✓ Wrote {count} events to {out_path}")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
