import argparse
import dv_processing as dv
import cv2 as cv
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("path", help="aedat4 file path")
args = parser.parse_args()

aedat_path = Path(args.path)
filename_stem = aedat_path.stem

output_dir = aedat_path.parent / f"frame_{filename_stem}"
output_dir.mkdir(parents=True, exist_ok=True)

ts_txt = output_dir / "timestamps.txt"
f_ts = open(ts_txt, "w")

reader = dv.io.MonoCameraRecording(str(aedat_path))

lastTimestamp = None
i = 0
while reader.isRunning():
    frame = reader.getNextFrame()
    if frame is not None:
        print(f"Received a frame at time [{frame.timestamp}]")

        # Save the frame
        out_img_path = output_dir / f"{frame.timestamp}.png"
        cv.imwrite(str(out_img_path), frame.image)

        # Save timestamp
        f_ts.write(f"{frame.timestamp}\n")

        lastTimestamp = frame.timestamp
        i += 1

f_ts.close()
print(f"[INFO] #Images is {i}")
print(f"[INFO] timestamps saved to {ts_txt}")
