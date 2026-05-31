"""Vantage perception loop — webcam -> YOLOE + Florence-2 -> RoomState.

Standalone, NO voice. Captures the webcam, runs YOLOE detection ~every 1s and a
Florence-2 scene caption ~every 4s (in a background thread so the video stays
smooth), and writes everything into ``room_state.json``.

Run:
    python -m vantage.vision.vision_loop

A window opens with live bounding boxes + labels (green = confident,
orange = low confidence). Press ``q`` in the window to quit. Detections also
print to the terminal every cycle.
"""

from __future__ import annotations

import argparse
import threading
import time

import cv2

from vantage.memory.room_state import CONFIDENCE_THRESHOLD, RoomState, region_of_bbox
from vantage.vision.detector import Detector


class _CaptionWorker(threading.Thread):
    """Runs Florence-2 captioning off the main thread to avoid stutter."""

    def __init__(self, captioner, room: RoomState, interval: float) -> None:
        super().__init__(daemon=True)
        self._captioner = captioner
        self._room = room
        self._interval = interval
        self._frame = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def submit(self, frame) -> None:
        with self._lock:
            self._frame = frame.copy()

    def run(self) -> None:
        while not self._stop.wait(self._interval):
            with self._lock:
                frame = None if self._frame is None else self._frame.copy()
            if frame is None:
                continue
            try:
                text = self._captioner.caption(frame)
                self._room.set_caption(text)
                print(f"  scene: {text}")
            except Exception as e:  # never let captioning kill the loop
                print(f"  caption error: {e}")

    def stop(self) -> None:
        self._stop.set()


def _draw(frame, detections, width, caption: str) -> None:
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d["bbox"])
        low = d["confidence"] < CONFIDENCE_THRESHOLD
        color = (0, 165, 255) if low else (0, 220, 0)  # BGR: orange / green
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f'{d["label"]} {d["confidence"]:.2f}'
        cv2.putText(
            frame, text, (x1, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
    if caption:
        h = frame.shape[0]
        cv2.putText(
            frame, caption[:90], (8, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Vantage vision loop (standalone).")
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    parser.add_argument("--detect-interval", type=float, default=1.0)
    parser.add_argument("--caption-interval", type=float, default=4.0)
    parser.add_argument("--model-size", default="s", help="YOLOE-26 size: n/s/m/l/x")
    parser.add_argument("--no-window", action="store_true", help="headless (no OpenCV window)")
    parser.add_argument("--no-caption", action="store_true", help="skip Florence-2 captioning")
    args = parser.parse_args()

    print("Loading models (first run downloads weights — can take a minute)...")
    detector = Detector(model_size=args.model_size)
    print(f"  YOLOE-26{args.model_size} ready on: {detector.device}")

    room = RoomState()
    print(f"  RoomState -> {room.path}")

    caption_worker = None
    if not args.no_caption:
        from vantage.vision.scene import SceneCaptioner

        captioner = SceneCaptioner()
        print(f"  Florence-2 ready on: {captioner.device}")
        caption_worker = _CaptionWorker(captioner, room, args.caption_interval)
        caption_worker.start()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera {args.camera}. Try a different --camera index, "
            "and grant Terminal camera permission in System Settings > Privacy & Security."
        )

    print("Running. Press 'q' in the window (or Ctrl+C) to quit.\n")
    last_detect = 0.0
    detections: list[dict] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera read failed")
                break
            now = time.time()
            width = frame.shape[1]

            if now - last_detect >= args.detect_interval:
                detections = detector.detect(frame)
                room.upsert(detections, frame_width=width)
                last_detect = now
                stamp = time.strftime("%H:%M:%S")
                if detections:
                    summary = ", ".join(
                        f'{d["label"]}({d["confidence"]:.2f},{region_of_bbox(d["bbox"], width)})'
                        for d in detections
                    )
                    print(f"[{stamp}] {len(detections)} det: {summary}")
                else:
                    print(f"[{stamp}] no detections")
                if caption_worker is not None:
                    caption_worker.submit(frame)

            if not args.no_window:
                _draw(frame, detections, width, room.get_caption())
                cv2.imshow("Vantage — vision loop (press q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Headless: don't busy-spin.
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        if caption_worker is not None:
            caption_worker.stop()
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nStopped. room_state written to {room.path}")


if __name__ == "__main__":
    main()
