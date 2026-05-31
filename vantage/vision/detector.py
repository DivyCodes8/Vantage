"""YOLOE open-vocabulary object detector (ultralytics YOLOE-26).

Defaults to PROMPT-FREE mode: the ``-pf`` checkpoints ship a large built-in
vocabulary, so a general room scan needs no prompts. Call :meth:`set_prompts`
to switch to text-prompt mode for specific objects (e.g. "charger", "keys").

Returns plain dicts: ``{"label", "bbox": [x1,y1,x2,y2], "confidence"}``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("vantage.detector")


def pick_device() -> str:
    """Prefer Apple Silicon MPS, else CPU."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # torch missing or MPS probe failed
        pass
    return "cpu"


class Detector:
    """Thin wrapper over ultralytics YOLOE with prompt-free + text-prompt modes."""

    def __init__(
        self,
        model_size: str = "s",
        device: str | None = None,
        conf: float = 0.20,
        imgsz: int = 640,
    ) -> None:
        # conf is intentionally low so low-confidence items still surface; the
        # RoomState marks anything under its own threshold as low_confidence.
        self.model_size = model_size
        self.device = device or pick_device()
        self.conf = conf
        self.imgsz = imgsz
        self.prompts: list[str] | None = None

        self._pf_name = f"yoloe-26{model_size}-seg-pf.pt"  # prompt-free checkpoint
        self._prompt_name = f"yoloe-26{model_size}-seg.pt"  # text/visual-prompt checkpoint
        self.model = self._load(self._pf_name)

    # ------------------------------------------------------------------ load
    @staticmethod
    def _load(name: str):
        from ultralytics import YOLOE

        logger.info("Loading YOLOE checkpoint %s (downloads on first use)", name)
        return YOLOE(name)

    # --------------------------------------------------------------- prompts
    def set_prompts(self, words) -> None:
        """Switch to open-vocabulary TEXT-prompt mode for the given words."""
        words = [w.strip() for w in words if w and w.strip()]
        if not words:
            return self.clear_prompts()

        model = self._load(self._prompt_name)
        try:
            # Newer ultralytics: set_classes(names)
            model.set_classes(words)
        except TypeError:
            # Older ultralytics: set_classes(names, text_pe)
            model.set_classes(words, model.get_text_pe(words))
        self.model = model
        self.prompts = words
        logger.info("YOLOE text prompts set: %s", words)

    def clear_prompts(self) -> None:
        """Return to prompt-free general-scan mode."""
        self.model = self._load(self._pf_name)
        self.prompts = None

    # --------------------------------------------------------------- predict
    def detect(self, frame) -> list[dict]:
        """Run detection on a BGR frame (numpy array) and return detection dicts."""
        try:
            results = self.model.predict(
                frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False
            )
        except (RuntimeError, NotImplementedError) as e:
            # An MPS op gap can blow up mid-inference — fall back to CPU for good.
            if self.device != "cpu":
                logger.warning("YOLOE failed on %s (%s); falling back to CPU", self.device, e)
                self.device = "cpu"
                results = self.model.predict(
                    frame, conf=self.conf, imgsz=self.imgsz, device="cpu", verbose=False
                )
            else:
                raise

        if not results:
            return []
        r = results[0]
        names = r.names
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            return []

        detections: list[dict] = []
        for b in boxes:
            cls = int(b.cls[0])
            if isinstance(names, dict):
                label = names.get(cls, str(cls))
            else:
                label = names[cls] if cls < len(names) else str(cls)
            detections.append(
                {
                    "label": str(label),
                    "bbox": [float(v) for v in b.xyxy[0].tolist()],
                    "confidence": float(b.conf[0]),
                }
            )
        return detections
