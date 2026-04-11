from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import UploadFile

from app.core.config import PROJECT_ROOT, Settings
from app.models.schemas import BBox, SegmentedProduct


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
        self._yolo = None

        # Tuned from legacy ProductWorkflow for better product-level segmentation.
        self._yolo_iou_threshold = 0.10
        self._padding_ratio = 0.08
        self._min_area_ratio = 0.008
        self._min_area_floor = 1200
        self._nested_containment_threshold = 0.90
        self._nested_size_ratio_threshold = 0.35
        self._mask_iou_threshold = 0.50
        self._mask_conf_margin = 0.08
        self._max_yolo_area_ratio = 0.25
        self._max_yolo_dim_ratio = 0.70
        self._edge_threshold = 20

        if self._settings.enable_yolo:
            self._init_yolo_model()
        else:
            logger.info("YOLO disabled by config (ENABLE_YOLO=false), using CV segmentation only")

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

        session_products: dict[str, SessionProduct] = {}
        response_products: list[SegmentedProduct] = []

        for idx, detection in enumerate(detections, start=1):
            x, y, w, h = detection.bbox

            if isinstance(detection.mask, np.ndarray):
                mask_bool = detection.mask > 0
                white_bg = np.full_like(cv_img, 255)
                white_bg[mask_bool] = cv_img[mask_bool]
                crop = white_bg[y : y + h, x : x + w]
            else:
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
            h, w = image.shape[:2]
            return [self._full_image_detection(w, h, with_margin=False)]

        if mode == "multi":
            return self._detect_multi_products(image, expected_products=expected_products)

        multi = self._detect_multi_products(image, expected_products=expected_products)
        if multi:
            return multi

        return self._detect_single_product(image)

    def _detect_multi_products(self, image: np.ndarray, expected_products: int | None) -> list[Detection]:
        h, w = image.shape[:2]

        yolo_detections = self._detect_with_yolo(image) if self._yolo is not None else []
        contour_detections = self._detect_with_contours(image)
        watershed_detections = self._detect_with_watershed(image)

        merged = self._merge_candidates([*yolo_detections, *contour_detections, *watershed_detections], w, h)

        if expected_products is not None:
            merged = self._limit_for_expected(merged, expected_products)

        if self._should_force_grid_fallback(merged, w, h):
            logger.info("Segmentation fallback: switching to grid proposals")
            if expected_products is None:
                return self._grid_fallback(w, h)
            return self._grid_fallback(w, h, expected_products=expected_products)

        return merged[:12]

    def _detect_single_product(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]

        yolo_detections = self._detect_with_yolo(image) if self._yolo is not None else []
        contour_detections = self._detect_with_contours(image)
        watershed_detections = self._detect_with_watershed(image)

        candidates = self._merge_candidates([*yolo_detections, *contour_detections, *watershed_detections], w, h)
        if not candidates:
            return [self._full_image_detection(w, h)]

        best = max(candidates, key=lambda det: self._single_score(det, w, h))
        bx, by, bw, bh = best.bbox
        area_ratio = (bw * bh) / max(1.0, float(w * h))
        if area_ratio < 0.03:
            return [self._full_image_detection(w, h)]

        return [best]

    def _single_score(self, detection: Detection, width: int, height: int) -> float:
        x, y, w, h = detection.bbox
        cx = x + (w / 2.0)
        cy = y + (h / 2.0)

        center_x = width / 2.0
        center_y = height / 2.0
        max_dist = math.sqrt((center_x**2) + (center_y**2))
        dist = math.sqrt(((cx - center_x) ** 2) + ((cy - center_y) ** 2))
        center_score = 1.0 - min(1.0, dist / max(1.0, max_dist))

        area_ratio = (w * h) / max(1.0, float(width * height))
        target_ratio = 0.40
        size_score = 1.0 - min(1.0, abs(area_ratio - target_ratio) / target_ratio)

        touches_edges = x <= 3 or y <= 3 or (x + w) >= (width - 3) or (y + h) >= (height - 3)
        edge_penalty = 0.12 if touches_edges else 0.0

        return (detection.confidence * 0.60) + (center_score * 0.25) + (size_score * 0.15) - edge_penalty

    def _full_image_detection(self, width: int, height: int, *, with_margin: bool = True) -> Detection:
        if with_margin:
            margin_x = max(1, int(width * 0.03))
            margin_y = max(1, int(height * 0.03))
        else:
            margin_x = 0
            margin_y = 0

        x = margin_x
        y = margin_y
        w = max(1, width - (2 * margin_x))
        h = max(1, height - (2 * margin_y))
        confidence = 1.0 if not with_margin else 0.30
        return Detection(bbox=(x, y, w, h), confidence=confidence, label="product")

    def _init_yolo_model(self) -> None:
        model_path = self._resolve_model_path(self._settings.yolo_model_path)

        if not model_path.exists():
            logger.info("YOLO disabled: model file not found at %s", model_path)
            return

        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]

            self._yolo = YOLO(str(model_path))
            logger.info("YOLO enabled with model=%s", model_path)
        except Exception as exc:
            self._yolo = None
            logger.info("YOLO unavailable, using contour segmentation only: %s", exc)

    def _detect_with_yolo(self, image: np.ndarray) -> list[Detection]:
        try:
            result = self._yolo.predict(
                source=image,
                conf=self._settings.yolo_conf_threshold,
                iou=self._yolo_iou_threshold,
                verbose=False,
            )[0]
        except Exception as exc:
            logger.error("YOLO inference failed: %s", exc)
            return []

        h, w = image.shape[:2]
        image_area = float(h * w)
        names = result.names if hasattr(result, "names") else {}
        masks = result.masks.data if getattr(result, "masks", None) is not None else None
        detections: list[Detection] = []

        for idx, box in enumerate(result.boxes):
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(x1 + 1, min(w, x2))
            y2 = max(y1 + 1, min(h, y2))

            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            area = bw * bh
            area_ratio = area / max(1.0, image_area)

            reject_reason = self._yolo_box_filter_reason(x1, y1, x2, y2, h, w)
            if reject_reason is not None:
                continue

            class_id = int(box.cls[0]) if hasattr(box, "cls") else -1
            label = str(names.get(class_id, "product")) if isinstance(names, dict) else "product"
            conf = float(box.conf[0])

            mask = None
            if masks is not None and idx < len(masks):
                mask = masks[idx].cpu().numpy()
                if mask.shape != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                mask = (mask > 0.5).astype(np.uint8)

            px, py, pw, ph = self._pad_and_clamp_bbox(x1, y1, x2, y2, image_width=w, image_height=h)
            detections.append(Detection(bbox=(px, py, pw, ph), confidence=conf, label=label, mask=mask))

        detections = self._suppress_nested_detections(detections)
        detections = self._suppress_high_overlap_conflicts(detections)
        detections = self._suppress_mask_overlap_conflicts(detections)
        return detections

    def _detect_with_contours(self, image: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        thresh = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = image.shape[:2]
        image_area = float(h * w)
        min_area = self._dynamic_min_area(image_width=w, image_height=h)
        max_area = image_area * 0.15
        detections: list[Detection] = []

        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            area = bw * bh
            if area < min_area:
                continue

            if area > max_area:
                continue

            aspect_ratio = bw / max(1, bh)
            if aspect_ratio > 5.0 or aspect_ratio < 0.2:
                continue

            width_ratio = bw / max(1, w)
            height_ratio = bh / max(1, h)
            if width_ratio > 0.7 or height_ratio > 0.7:
                continue

            touches_top = y < self._edge_threshold
            touches_bottom = (y + bh) > (h - self._edge_threshold)
            touches_left = x < self._edge_threshold
            touches_right = (x + bw) > (w - self._edge_threshold)
            if touches_top and touches_bottom and touches_left and touches_right:
                continue

            confidence = min(0.95, 0.4 + (area / max(1.0, image_area)) * 0.5)
            px, py, pw, ph = self._pad_and_clamp_bbox(x, y, x + bw, y + bh, image_width=w, image_height=h)
            detections.append(Detection(bbox=(px, py, pw, ph), confidence=confidence, label="product"))

        return detections

    def _detect_with_watershed(self, image: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        sure_bg = cv2.dilate(opened, kernel, iterations=3)
        dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
        max_dist = float(dist.max())
        if max_dist <= 0.0:
            return []

        _, sure_fg = cv2.threshold(dist, 0.35 * max_dist, 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        components, markers = cv2.connectedComponents(sure_fg)
        if components <= 1:
            return []

        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(image.copy(), markers)

        h, w = image.shape[:2]
        image_area = float(h * w)
        min_area = max(1100, int(0.006 * image_area))

        detections: list[Detection] = []
        for marker_id in range(2, int(markers.max()) + 1):
            mask = np.uint8(markers == marker_id) * 255
            if cv2.countNonZero(mask) == 0:
                continue

            x, y, bw, bh = cv2.boundingRect(mask)
            area = bw * bh
            if area < min_area:
                continue

            area_ratio = area / max(1.0, image_area)
            touches_edges = x <= 3 or y <= 3 or (x + bw) >= (w - 3) or (y + bh) >= (h - 3)
            if area_ratio > 0.65 and touches_edges:
                continue

            aspect_ratio = bw / max(1, bh)
            if aspect_ratio > 5.0 or aspect_ratio < 0.2:
                continue

            confidence = min(0.82, 0.33 + (area_ratio * 1.2))
            px, py, pw, ph = self._pad_and_clamp_bbox(x, y, x + bw, y + bh, image_width=w, image_height=h)
            detections.append(Detection(bbox=(px, py, pw, ph), confidence=confidence, label="product"))

        return detections

    def _yolo_box_filter_reason(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        image_height: int,
        image_width: int,
    ) -> Optional[str]:
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        area = w * h
        image_area = image_height * image_width

        if area > image_area * self._max_yolo_area_ratio:
            return "area"

        width_ratio = w / max(1, image_width)
        height_ratio = h / max(1, image_height)
        if width_ratio > self._max_yolo_dim_ratio or height_ratio > self._max_yolo_dim_ratio:
            return "dimensions"

        touches_top = y1 < self._edge_threshold
        touches_bottom = y2 > (image_height - self._edge_threshold)
        touches_left = x1 < self._edge_threshold
        touches_right = x2 > (image_width - self._edge_threshold)
        if touches_top and touches_bottom and touches_left and touches_right:
            return "full_frame"

        return None

    def _dynamic_min_area(self, image_width: int, image_height: int) -> int:
        image_area = image_width * image_height
        ratio_area = int(image_area * self._min_area_ratio)
        return max(self._min_area_floor, ratio_area)

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

        pad_x = int(width * self._padding_ratio)
        pad_y = int(height * self._padding_ratio)

        nx1 = max(0, x1 - pad_x)
        ny1 = max(0, y1 - pad_y)
        nx2 = min(image_width, x2 + pad_x)
        ny2 = min(image_height, y2 + pad_y)

        return nx1, ny1, max(1, nx2 - nx1), max(1, ny2 - ny1)

    def _merge_confidence(self, base_conf: float, nested_conf: float) -> float:
        base = min(1.0, max(0.0, float(base_conf)))
        nested = min(1.0, max(0.0, float(nested_conf)))
        return min(1.0, base + nested - (base * nested))

    def _merge_detection_geometry(self, container: Detection, inner: Detection) -> None:
        x1c, y1c, w1, h1 = container.bbox
        x2c = x1c + w1
        y2c = y1c + h1

        x1i, y1i, w2, h2 = inner.bbox
        x2i = x1i + w2
        y2i = y1i + h2

        nx1 = min(x1c, x1i)
        ny1 = min(y1c, y1i)
        nx2 = max(x2c, x2i)
        ny2 = max(y2c, y2i)
        container.bbox = (nx1, ny1, max(1, nx2 - nx1), max(1, ny2 - ny1))

        if isinstance(container.mask, np.ndarray) and isinstance(inner.mask, np.ndarray):
            container.mask = np.logical_or(container.mask > 0, inner.mask > 0).astype(np.uint8)
        elif not isinstance(container.mask, np.ndarray) and isinstance(inner.mask, np.ndarray):
            container.mask = inner.mask.copy()

    def _suppress_nested_detections(self, detections: list[Detection]) -> list[Detection]:
        if len(detections) < 2:
            return detections

        merged = [Detection(det.bbox, det.confidence, det.label, det.mask.copy() if isinstance(det.mask, np.ndarray) else None) for det in detections]
        suppressed_indices: set[int] = set()

        for i, det_i in enumerate(merged):
            if i in suppressed_indices:
                continue

            x1i, y1i, wi, hi = det_i.bbox
            x2i = x1i + wi
            y2i = y1i + hi
            area_i = max(1, wi * hi)

            for j, det_j in enumerate(merged):
                if i == j or j in suppressed_indices:
                    continue

                x1j, y1j, wj, hj = det_j.bbox
                x2j = x1j + wj
                y2j = y1j + hj
                area_j = max(1, wj * hj)

                ix1 = max(x1i, x1j)
                ix2 = min(x2i, x2j)
                iy1 = max(y1i, y1j)
                iy2 = min(y2i, y2j)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue

                inter_area = (ix2 - ix1) * (iy2 - iy1)
                containment = inter_area / float(area_i)
                size_ratio = area_i / float(area_j)

                if (
                    containment >= self._nested_containment_threshold
                    and size_ratio <= self._nested_size_ratio_threshold
                ):
                    det_j.confidence = self._merge_confidence(det_j.confidence, det_i.confidence)
                    self._merge_detection_geometry(det_j, det_i)
                    suppressed_indices.add(i)
                    break

        return [det for idx, det in enumerate(merged) if idx not in suppressed_indices]

    def _suppress_high_overlap_conflicts(self, detections: list[Detection]) -> list[Detection]:
        if len(detections) < 2:
            return detections

        sorted_detections = sorted(detections, key=lambda d: float(d.confidence), reverse=True)
        kept: list[Detection] = []

        for candidate in sorted_detections:
            x1c, y1c, wc, hc = candidate.bbox
            x2c = x1c + wc
            y2c = y1c + hc
            area_c = max(1, wc * hc)
            skip_candidate = False

            for existing in kept:
                x1e, y1e, we, he = existing.bbox
                x2e = x1e + we
                y2e = y1e + he
                area_e = max(1, we * he)

                ix1 = max(x1c, x1e)
                ix2 = min(x2c, x2e)
                iy1 = max(y1c, y1e)
                iy2 = min(y2c, y2e)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue

                inter_area = (ix2 - ix1) * (iy2 - iy1)
                iou = inter_area / max(1, area_c + area_e - inter_area)
                containment_c = inter_area / float(area_c)

                if iou >= 0.8 or containment_c >= 0.9:
                    skip_candidate = True
                    break

            if not skip_candidate:
                kept.append(candidate)

        return kept

    def _mask_iou(self, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
        if inter == 0:
            return 0.0
        union = np.logical_or(mask_a > 0, mask_b > 0).sum()
        if union == 0:
            return 0.0
        return float(inter / union)

    def _suppress_mask_overlap_conflicts(self, detections: list[Detection]) -> list[Detection]:
        if len(detections) < 2:
            return detections

        sorted_detections = sorted(detections, key=lambda d: float(d.confidence), reverse=True)
        kept: list[Detection] = []

        for candidate in sorted_detections:
            candidate_mask = candidate.mask
            skip_candidate = False

            for existing in kept:
                existing_mask = existing.mask
                if not isinstance(candidate_mask, np.ndarray) or not isinstance(existing_mask, np.ndarray):
                    continue

                mask_iou = self._mask_iou(candidate_mask, existing_mask)
                if mask_iou < self._mask_iou_threshold:
                    continue

                candidate_conf = float(candidate.confidence)
                existing_conf = float(existing.confidence)
                if candidate_conf <= existing_conf + self._mask_conf_margin:
                    skip_candidate = True
                    break

            if not skip_candidate:
                kept.append(candidate)

        return kept

    def _merge_candidates(self, detections: list[Detection], width: int, height: int) -> list[Detection]:
        if not detections:
            return []

        merged = self._non_max_suppression(detections, iou_threshold=0.45)
        filtered = [det for det in merged if self._is_reasonable_bbox(det.bbox, width, height)]
        filtered.sort(key=lambda det: det.confidence, reverse=True)
        return filtered

    def _non_max_suppression(self, detections: list[Detection], iou_threshold: float) -> list[Detection]:
        ordered = sorted(detections, key=lambda det: det.confidence, reverse=True)
        selected: list[Detection] = []

        for candidate in ordered:
            keep = True
            for picked in selected:
                if self._bbox_iou(candidate.bbox, picked.bbox) >= iou_threshold:
                    keep = False
                    break
            if keep:
                selected.append(candidate)

        return selected

    def _is_reasonable_bbox(self, bbox: tuple[int, int, int, int], width: int, height: int) -> bool:
        x, y, w, h = bbox
        area_ratio = (w * h) / max(1.0, float(width * height))
        if area_ratio < 0.004 or area_ratio > 0.80:
            return False

        aspect_ratio = w / max(1, h)
        if aspect_ratio > 6.0 or aspect_ratio < 0.16:
            return False

        touches_edges = x <= 3 or y <= 3 or (x + w) >= (width - 3) or (y + h) >= (height - 3)
        if touches_edges and area_ratio > 0.60:
            return False

        return True

    def _bbox_iou(self, a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b

        ax2 = ax + aw
        ay2 = ay + ah
        bx2 = bx + bw
        by2 = by + bh

        inter_x1 = max(ax, bx)
        inter_y1 = max(ay, by)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = aw * ah
        area_b = bw * bh
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / float(union)

    def _should_force_grid_fallback(self, detections: list[Detection], width: int, height: int) -> bool:
        if not detections:
            return True

        if len(detections) >= 2:
            return False

        x, y, w, h = detections[0].bbox
        area_ratio = (w * h) / max(1.0, float(width * height))
        touches_edges = x <= 3 or y <= 3 or (x + w) >= (width - 3) or (y + h) >= (height - 3)
        return area_ratio > 0.65 and touches_edges

    def _limit_for_expected(self, detections: list[Detection], expected_products: int) -> list[Detection]:
        expected = max(1, min(12, expected_products))
        if len(detections) <= expected:
            return detections

        max_keep = min(12, max(expected + 2, expected * 2))
        return detections[:max_keep]

    def _grid_fallback(self, width: int, height: int, expected_products: int | None = None) -> list[Detection]:
        if expected_products is not None:
            expected = max(1, min(9, expected_products))
            if expected <= 2:
                cols, rows = (2, 1) if width >= height else (1, 2)
            elif expected <= 4:
                cols, rows = 2, 2
            elif expected <= 6:
                cols, rows = 3, 2
            else:
                cols, rows = 3, 3
        elif width >= int(height * 1.3):
            cols, rows = 3, 2
        elif height >= int(width * 1.3):
            cols, rows = 2, 3
        else:
            cols, rows = 2, 2

        cell_w = width // cols
        cell_h = height // rows
        pad_w = int(cell_w * 0.07)
        pad_h = int(cell_h * 0.07)

        detections: list[Detection] = []
        for row in range(rows):
            for col in range(cols):
                x = max(0, col * cell_w + pad_w)
                y = max(0, row * cell_h + pad_h)
                w = max(1, cell_w - (2 * pad_w))
                h = max(1, cell_h - (2 * pad_h))
                detections.append(Detection(bbox=(x, y, w, h), confidence=0.35, label="product"))

        if expected_products is not None:
            return detections[: max(1, min(len(detections), expected_products + 1))]
        return detections

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
