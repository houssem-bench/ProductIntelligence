from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from fastapi import UploadFile

from app.core.config import PROJECT_ROOT, Settings
from app.models.schemas import BBox, SegmentedProduct
from app.services.yolo_world_code import YOLOWorldProductWorkflow


logger = logging.getLogger(__name__)


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    confidence: float
    label: str = "product"
    mask: np.ndarray | None = None


@dataclass
class SessionProduct:
    product_id: str
    label: str
    confidence: float
    bbox: BBox
    crop_path: str
    source_image_path: str
    crop_url: str


class SegmentationService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._sessions: dict[str, dict[str, SessionProduct]] = {}
        self._workflow: YOLOWorldProductWorkflow | None = None

        if self._settings.enable_yolo:
            model_path = str(self._resolve_model_path(self._settings.yolo_model_path))
            self._workflow = YOLOWorldProductWorkflow(
                model_name=model_path,
                world_classes=[
                    "food product",
                    "package",
                    "box",
                    "bottle",
                    "can",
                    "carton",
                    "jar",
                    "pouch",
                ],
            )
        else:
            logger.info("YOLO disabled by config (ENABLE_YOLO=false)")

    async def segment_upload(
        self,
        image: UploadFile,
        *,
        segmentation_mode: str = "auto",
        expected_products: int | None = None,
    ) -> tuple[str, list[SegmentedProduct]]:
        image_bytes = await image.read()
        if not image_bytes:
            raise ValueError("Uploaded file is empty")

        np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if cv_img is None:
            raise ValueError("Unsupported or corrupted image")

        session_id = str(uuid.uuid4())
        incoming_path = self._settings.incoming_dir / f"{session_id}_{image.filename or 'upload.jpg'}"
        incoming_path.write_bytes(image_bytes)

        detections = self._detect_products(
            cv_img,
            segmentation_mode=segmentation_mode,
            expected_products=expected_products,
        )

        crops_dir = self._settings.crops_dir / session_id
        crops_dir.mkdir(parents=True, exist_ok=True)
        image_height, image_width = cv_img.shape[:2]

        session_products: dict[str, SessionProduct] = {}
        response_products: list[SegmentedProduct] = []

        for idx, detection in enumerate(detections, start=1):
            x, y, w, h = detection.bbox
            x, y, w, h = self._pad_and_clamp_bbox(
                x,
                y,
                x + w,
                y + h,
                image_width=image_width,
                image_height=image_height,
            )

            crop = cv_img[y : y + h, x : x + w]
            if crop.size == 0:
                continue

            product_id = str(idx)
            crop_path = crops_dir / f"{product_id}.jpg"
            cv2.imwrite(str(crop_path), crop)

            bbox_model = BBox(x=x, y=y, width=w, height=h)
            crop_url = f"/uploads/crops/{session_id}/{product_id}.jpg"
            item = SessionProduct(
                product_id=product_id,
                label=detection.label,
                confidence=round(detection.confidence, 4),
                bbox=bbox_model,
                crop_path=str(crop_path),
                source_image_path=str(incoming_path),
                crop_url=crop_url,
            )
            session_products[product_id] = item
            response_products.append(
                SegmentedProduct(
                    product_id=product_id,
                    label=item.label,
                    confidence=item.confidence,
                    bbox=bbox_model,
                    crop_url=crop_url,
                )
            )

        self._sessions[session_id] = session_products
        logger.info(
            "Segmentation completed session=%s mode=%s expected=%s products=%s",
            session_id,
            segmentation_mode,
            expected_products,
            len(response_products),
        )
        return session_id, response_products

    def get_session_products(self, session_id: str) -> dict[str, SessionProduct] | None:
        return self._sessions.get(session_id)

    def _detect_products(
        self,
        image: np.ndarray,
        segmentation_mode: str = "auto",
        expected_products: int | None = None,
    ) -> list[Detection]:
        mode = (segmentation_mode or "auto").strip().lower()

        if mode == "single":
            height, width = image.shape[:2]
            return [Detection(bbox=(0, 0, width, height), confidence=1.0, label="product")]

        detections = self._detect_with_yolo(image)
        detections.sort(key=lambda det: float(det.confidence), reverse=True)
        detections = self._limit_for_expected(detections, expected_products)
        return detections[:12]

    def _detect_with_yolo(self, image: np.ndarray) -> list[Detection]:
        if self._workflow is None or self._workflow.model is None:
            return []

        raw_detections = self._workflow._detect_with_yolo_world(image)
        detections: list[Detection] = []

        for det in raw_detections:
            xyxy = det.get("xyxy")
            if not isinstance(xyxy, tuple) or len(xyxy) != 4:
                continue

            x1, y1, x2, y2 = [int(v) for v in xyxy]
            detections.append(
                Detection(
                    bbox=(x1, y1, max(1, x2 - x1), max(1, y2 - y1)),
                    confidence=float(det.get("confidence", 0.0)),
                    label=str(det.get("label", "product")),
                    mask=det.get("mask") if isinstance(det.get("mask"), np.ndarray) else None,
                )
            )

        return detections

    def _pad_and_clamp_bbox(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int, int, int]:
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)

        pad_x = int(width * 0.08)
        pad_y = int(height * 0.08)

        nx1 = max(0, x1 - pad_x)
        ny1 = max(0, y1 - pad_y)
        nx2 = min(image_width, x2 + pad_x)
        ny2 = min(image_height, y2 + pad_y)

        return nx1, ny1, max(1, nx2 - nx1), max(1, ny2 - ny1)

    def _limit_for_expected(
        self,
        detections: list[Detection],
        expected_products: int | None,
    ) -> list[Detection]:
        if expected_products is None:
            return detections

        expected = max(1, min(12, expected_products))
        if len(detections) <= expected:
            return detections
        return detections[:expected]

    def _resolve_model_path(self, raw_model_path: str) -> Path:
        configured = Path(raw_model_path)
        if configured.is_absolute():
            return configured

        candidates = [
            (Path.cwd() / configured).resolve(),
            (PROJECT_ROOT / configured).resolve(),
            (PROJECT_ROOT.parent / configured).resolve(),
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return candidates[0]
