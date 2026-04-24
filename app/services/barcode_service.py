from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

try:
    from pyzbar.pyzbar import decode as zbar_decode
except Exception:  # pragma: no cover - optional dependency
    zbar_decode = None


logger = logging.getLogger(__name__)


@dataclass
class BarcodeDetection:
    code: str
    format_type: str


class BarcodeService:
    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled

    def get_readiness(self) -> tuple[bool, str, dict[str, bool]]:
        backends = {
            "enabled": self._enabled,
            "opencv_qr": True,
            "opencv_barcode": hasattr(cv2, "barcode_BarcodeDetector"),
            "pyzbar": zbar_decode is not None,
        }

        if not self._enabled:
            return False, "disabled_by_config", backends

        ready = backends["opencv_barcode"] or backends["pyzbar"]
        reason = "ready" if ready else "no_1d_barcode_backend"
        return ready, reason, backends

    def extract(self, image_path: str) -> BarcodeDetection | None:
        if not self._enabled:
            logger.info("[BARCODE] Skipped: disabled_by_config")
            return None

        image = cv2.imread(image_path)
        if image is None:
            return None

        for candidate in self._iter_candidates(image):
            qr = self._decode_qr(candidate)
            if qr:
                logger.info("[BARCODE] QR decoded EAN=%s", qr)
                return BarcodeDetection(code=qr, format_type="qr")

            barcode = self._decode_barcode(candidate)
            if barcode:
                code, code_type = barcode
                logger.info("[BARCODE] Barcode decoded type=%s", code_type)
                return BarcodeDetection(code=code, format_type=code_type)

            zbar_barcode = self._decode_with_pyzbar(candidate)
            if zbar_barcode:
                code, code_type = zbar_barcode
                logger.info("[BARCODE] pyzbar decoded type=%s", code_type)
                return BarcodeDetection(code=code, format_type=code_type)

        logger.info("[BARCODE] No barcode/QR detected")
        return None

    def _iter_candidates(self, image: np.ndarray) -> Iterable[np.ndarray]:
        h, w = image.shape[:2]
        rois: list[np.ndarray] = [
            image,
            image[int(h * 0.40) :, :],
            image[int(h * 0.50) :, :],
            image[int(h * 0.60) :, int(w * 0.05) : int(w * 0.95)],
        ]

        seen_shapes: set[tuple[int, int]] = set()
        for roi in rois:
            if roi.size == 0:
                continue

            roi_shape = roi.shape[:2]
            if roi_shape in seen_shapes and roi_shape != image.shape[:2]:
                continue
            seen_shapes.add(roi_shape)

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            sharpen = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
            adaptive = cv2.adaptiveThreshold(
                sharpen,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                8,
            )

            variants = [
                roi,
                cv2.cvtColor(sharpen, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR),
            ]

            for variant in variants:
                for scale in (1.0, 1.8, 2.4):
                    if scale == 1.0:
                        scaled = variant
                    else:
                        scaled = cv2.resize(
                            variant,
                            None,
                            fx=scale,
                            fy=scale,
                            interpolation=cv2.INTER_CUBIC,
                        )

                    yield scaled
                    yield cv2.rotate(scaled, cv2.ROTATE_90_CLOCKWISE)
                    yield cv2.rotate(scaled, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def _decode_qr(self, image: np.ndarray) -> str | None:
        try:
            qr_detector = cv2.QRCodeDetector()
            decoded, _, _ = qr_detector.detectAndDecode(image)
        except Exception:
            return None
        return self._normalize_numeric_code(decoded)

    def _decode_barcode(self, image: np.ndarray) -> tuple[str, str] | None:
        if not hasattr(cv2, "barcode_BarcodeDetector"):
            return None

        try:
            detector = cv2.barcode_BarcodeDetector()
            ok, decoded_info, decoded_types, _ = detector.detectAndDecode(image)
            if not ok or not decoded_info:
                return None

            for value, code_type in zip(decoded_info, decoded_types):
                code = self._normalize_numeric_code(value)
                if code:
                    return code, str(code_type)
        except Exception as exc:
            logger.debug("[BARCODE] OpenCV barcode detector failed: %s", exc)

        return None

    def _decode_with_pyzbar(self, image: np.ndarray) -> tuple[str, str] | None:
        if zbar_decode is None:
            return None

        try:
            decoded_items = zbar_decode(image)
        except Exception as exc:
            logger.debug("[BARCODE] pyzbar decode failed: %s", exc)
            return None

        for item in decoded_items:
            raw = item.data.decode("utf-8", errors="ignore") if hasattr(item, "data") else ""
            code = self._normalize_numeric_code(raw)
            if code:
                code_type = str(getattr(item, "type", "pyzbar"))
                return code, code_type

        return None

    def _normalize_numeric_code(self, raw: str | None) -> str | None:
        if not raw:
            return None

        text = str(raw).strip()
        if not text:
            return None

        compact = re.sub(r"\D", "", text)
        candidates = [compact] if compact else []
        if not candidates:
            candidates = re.findall(r"\d{8,14}", text)

        for candidate in candidates:
            if len(candidate) == 12:
                # Normalize UPC-A to EAN-13 for OpenFoodFacts compatibility.
                candidate = f"0{candidate}"

            if len(candidate) not in {8, 13, 14}:
                continue

            if len(candidate) == 8 and not self._is_valid_ean8(candidate):
                continue
            if len(candidate) == 13 and not self._is_valid_ean13(candidate):
                continue
            if len(candidate) == 14 and not self._is_valid_ean14(candidate):
                continue

            return candidate

        return None

    def _is_valid_ean8(self, code: str) -> bool:
        if not re.fullmatch(r"\d{8}", code):
            return False
        digits = [int(ch) for ch in code]
        checksum = (10 - ((3 * sum(digits[0:7:2]) + sum(digits[1:7:2])) % 10)) % 10
        return checksum == digits[7]

    def _is_valid_ean13(self, code: str) -> bool:
        if not re.fullmatch(r"\d{13}", code):
            return False
        digits = [int(ch) for ch in code]
        checksum = (10 - ((sum(digits[0:12:2]) + 3 * sum(digits[1:12:2])) % 10)) % 10
        return checksum == digits[12]

    def _is_valid_ean14(self, code: str) -> bool:
        if not re.fullmatch(r"\d{14}", code):
            return False
        digits = [int(ch) for ch in code]
        checksum = (10 - ((3 * sum(digits[0:13:2]) + sum(digits[1:13:2])) % 10)) % 10
        return checksum == digits[13]
