from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass

import cv2

from app.core.config import Settings


logger = logging.getLogger(__name__)


@dataclass
class OCRExtraction:
    name: str | None
    category: str
    ingredients: list[str]
    raw_text: str


class OCRService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._enabled = bool(getattr(settings, "enable_ocr", True))
        self._pytesseract = None
        self._runtime_available = False

        if not self._enabled:
            logger.info("[OCR] Disabled: disabled_by_config")
            return

        try:
            import pytesseract

            self._pytesseract = pytesseract
            configured_cmd = self._settings.tesseract_cmd
            if configured_cmd:
                pytesseract.pytesseract.tesseract_cmd = configured_cmd

            resolved = configured_cmd or shutil.which("tesseract")
            self._runtime_available = bool(resolved)
            if not self._runtime_available:
                logger.info("[OCR] Disabled: tesseract binary not found in PATH and TESSERACT_CMD is empty")
            else:
                logger.info("[OCR] Enabled with tesseract executable")

        except Exception:
            self._pytesseract = None
            self._runtime_available = False
            logger.info("[OCR] Disabled: pytesseract package not available")

    def get_readiness(self) -> tuple[bool, str]:
        if not self._enabled:
            return False, "disabled_by_config"
        if self._pytesseract is None:
            return False, "missing_pytesseract_package"
        if not self._runtime_available:
            return False, "missing_tesseract_binary"
        return True, "ready"

    def extract(self, image_path: str) -> OCRExtraction | None:
        if not self._enabled:
            logger.info("[OCR] Skipped: disabled_by_config")
            return None

        image = cv2.imread(image_path)
        if image is None:
            return None

        if self._pytesseract is None or not self._runtime_available:
            logger.info("[OCR] Skipped: OCR runtime unavailable")
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            8,
        )

        try:
            text = self._pytesseract.image_to_string(processed, lang="eng+fra")
        except Exception as exc:
            logger.error("[OCR] Runtime failure: %s", exc)
            return None

        if not text or len(text.strip()) < 8:
            logger.info("[OCR] No meaningful text extracted")
            return None

        ingredients = self._extract_ingredients(text)
        category = self._infer_category(text)
        name = self._extract_product_name(text)

        logger.info("[OCR] Extracted chars=%s ingredients=%s", len(text), len(ingredients))
        return OCRExtraction(name=name, category=category, ingredients=ingredients, raw_text=text)

    def _extract_product_name(self, text: str) -> str | None:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None

        for line in lines[:4]:
            if len(line) >= 4 and not line.lower().startswith("ingredients"):
                return line[:80]
        return None

    def _extract_ingredients(self, text: str) -> list[str]:
        match = re.search(r"ingredients?\s*[:\-]\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
        payload = match.group(1) if match else text
        payload = re.sub(r"\n+", " ", payload)
        parts = [part.strip(" .") for part in re.split(r"[,;]", payload)]
        cleaned: list[str] = []
        for item in parts:
            if len(item) < 2:
                continue
            if item.lower() in {"ingredients", "composition"}:
                continue
            cleaned.append(item)

        deduped = list(dict.fromkeys(cleaned))
        return deduped[:50]

    def _infer_category(self, text: str) -> str:
        t = text.lower()
        cosmetic_keywords = ["parfum", "linalool", "glycerin", "cream", "shampoo", "lotion", "inci"]
        food_keywords = ["sugar", "sucre", "salt", "milk", "flour", "chocolate", "huile"]

        cosmetic_score = sum(1 for kw in cosmetic_keywords if kw in t)
        food_score = sum(1 for kw in food_keywords if kw in t)

        if cosmetic_score > food_score:
            return "cosmetic"
        if food_score > cosmetic_score:
            return "food"
        return "unknown"
