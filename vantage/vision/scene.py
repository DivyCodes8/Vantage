"""Scene captioning with microsoft/Florence-2-base (transformers, remote code).

Produces a short natural-language caption for the current frame, used as ambient
context for the voice agent ("looks like a desk with a laptop and a coffee mug").

Florence-2 has known gaps in its MPS kernel coverage, so we try MPS first (when
available) and fall back to CPU automatically — both at load time and if an
individual generate() call raises on MPS.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("vantage.scene")

MODEL_ID = "microsoft/Florence-2-base"
# "<CAPTION>" = short caption. "<DETAILED_CAPTION>"/"<MORE_DETAILED_CAPTION>" exist too.
DEFAULT_TASK = "<CAPTION>"


class SceneCaptioner:
    def __init__(self, prefer_mps: bool = True, task: str = DEFAULT_TASK) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._torch = torch
        self.task = task
        # float32 everywhere — Florence-2 + MPS is flaky in half precision.
        self.dtype = torch.float32
        self.device = "mps" if (prefer_mps and torch.backends.mps.is_available()) else "cpu"

        logger.info("Loading %s (downloads on first use)...", MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            attn_implementation="eager",  # avoid sdpa/flash paths that break on MPS
        )
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------ api
    def caption(self, image) -> str:
        """Return a short caption for a BGR numpy frame or a PIL image."""
        pil = self._to_pil(image)
        try:
            return self._run(pil, self.device)
        except (RuntimeError, NotImplementedError) as e:
            if self.device != "cpu":
                logger.warning("Florence-2 failed on %s (%s); falling back to CPU", self.device, e)
                self.device = "cpu"
                self.model.to("cpu")
                return self._run(pil, "cpu")
            raise

    # --------------------------------------------------------------- helpers
    def _run(self, pil, device: str) -> str:
        torch = self._torch
        inputs = self.processor(text=self.task, images=pil, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        pixel_values = inputs["pixel_values"].to(device, self.dtype)
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=128,
                num_beams=3,
                do_sample=False,
            )
        text = self.processor.batch_decode(generated, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            text, task=self.task, image_size=(pil.width, pil.height)
        )
        result = parsed.get(self.task, parsed)
        return result.strip() if isinstance(result, str) else str(result).strip()

    @staticmethod
    def _to_pil(image):
        from PIL import Image

        if isinstance(image, Image.Image):
            return image.convert("RGB")
        # Assume an OpenCV BGR ndarray.
        import numpy as np

        rgb = np.ascontiguousarray(image[:, :, ::-1])
        return Image.fromarray(rgb)
