import argparse
import csv
import time
from pathlib import Path
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("label", choices=["forward", "left", "right", "stop"])
parser.add_argument("--camera", type=int, default=0)
parser.add_argument("--session", default="session01")
parser.add_argument("--condition", default="normal")
parser.add_argument("--interval", type=float, default=0.5)
args = parser.parse_args()

out = Path("data/dataset_raw") / args.label
out.mkdir(parents=True, exist_ok=True)

metadata = Path("data/dataset_raw/metadata.csv")
new = not metadata.exists()
cap = cv2.VideoCapture(args.camera)

if not cap.isOpened():
    raise RuntimeError("Kamera tidak dapat dibuka.")

with metadata.open("a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if new:
        writer.writerow(["nama_file", "kelas", "session", "condition", "timestamp"])

    count = 0
    last = 0
    print("Tekan Q untuk keluar. Foto diambil otomatis sesuai interval.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cv2.putText(frame, f"class={args.label} session={args.session}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Capture - Maqueen", frame)

        now = time.time()
        if now - last >= args.interval:
            filename = f"{args.session}_{args.condition}_{args.label}_{count:04d}.jpg"
            path = out / filename
            cv2.imwrite(str(path), frame)
            writer.writerow([filename, args.label, args.session,
                             args.condition, time.strftime("%Y-%m-%d %H:%M:%S")])
            count += 1
            last = now

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print(f"Tersimpan {count} gambar pada {out}")
