from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

try:
	from ultralytics import YOLO  # type: ignore[import-not-found]

	ULTRALYTICS_AVAILABLE = True
except ImportError:
	ULTRALYTICS_AVAILABLE = False


@dataclass
class BoundingBox:
	x: int
	y: int
	width: int
	height: int


@dataclass
class ProductObject:
	product_id: str
	source_image_id: str
	label: str
	confidence: float
	bbox: BoundingBox
	crop_path: str
	created_at: str
	extra: Dict[str, str]

	def to_dict(self) -> Dict[str, object]:
		data = asdict(self)
		data["bbox"] = asdict(self.bbox)
		return data


@dataclass
class ImageAnalysisResult:
	image_id: str
	original_path: str
	products: List[ProductObject]
	total_products: int
	status: str

	def to_dict(self) -> Dict[str, object]:
		return {
			"image_id": self.image_id,
			"original_path": self.original_path,
			"products": [product.to_dict() for product in self.products],
			"total_products": self.total_products,
			"status": self.status,
		}


def yolo_box_filter_reason(
	x1: int,
	y1: int,
	x2: int,
	y2: int,
	image_height: int,
	image_width: int,
	max_area_ratio: float = 0.25,
	max_dim_ratio: float = 0.7,
	edge_threshold: int = 20,
) -> Optional[str]:
	"""Return rejection reason for a YOLO box, or None if it passes filters."""
	w = max(1, x2 - x1)
	h = max(1, y2 - y1)
	area = w * h
	image_area = image_height * image_width

	if area > image_area * max_area_ratio:
		return "area"

	width_ratio = w / max(1, image_width)
	height_ratio = h / max(1, image_height)
	if width_ratio > max_dim_ratio or height_ratio > max_dim_ratio:
		return "dimensions"

	touches_top = y1 < edge_threshold
	touches_bottom = y2 > (image_height - edge_threshold)
	touches_left = x1 < edge_threshold
	touches_right = x2 > (image_width - edge_threshold)
	if touches_top and touches_bottom and touches_left and touches_right:
		return "full_frame"

	return None


class ProductWorkflow:
	"""Detect products in an image, crop each detection, and keep product objects in a list."""

	def __init__(
		self,
		model_name: str = "yolov8n-seg.pt",
		conf_threshold: float = 0.01,
		iou_threshold: float = 0.1,
		padding_ratio: float = 0.08,
		min_area_ratio: float = 0.008,
		min_area_floor: int = 1200,
		nested_containment_threshold: float = 0.9,
		nested_size_ratio_threshold: float = 0.35,
		mask_iou_threshold: float = 0.5,
		mask_conf_margin: float = 0.08,
	) -> None:
		root_dir = Path(__file__).resolve().parents[1]
		self.uploads_dir = root_dir / "uploads"
		self.originals_dir = self.uploads_dir / "originals"
		self.crops_dir = self.uploads_dir / "crops"
		self.metadata_dir = self.uploads_dir / "metadata"

		self.model_name = model_name
		self.conf_threshold = conf_threshold
		self.iou_threshold = iou_threshold
		self.padding_ratio = padding_ratio
		self.min_area_ratio = min_area_ratio
		self.min_area_floor = min_area_floor
		self.nested_containment_threshold = nested_containment_threshold
		self.nested_size_ratio_threshold = nested_size_ratio_threshold
		self.mask_iou_threshold = mask_iou_threshold
		self.mask_conf_margin = mask_conf_margin

		self.product_objects: List[ProductObject] = []
		self.image_results: Dict[str, ImageAnalysisResult] = {}
		self.image_counter = 0

		self._ensure_storage_dirs()
		self._load_existing_counters()
		self.model = YOLO(model_name) if ULTRALYTICS_AVAILABLE else None

	def _ensure_storage_dirs(self) -> None:
		self.uploads_dir.mkdir(parents=True, exist_ok=True)
		self.originals_dir.mkdir(parents=True, exist_ok=True)
		self.crops_dir.mkdir(parents=True, exist_ok=True)
		self.metadata_dir.mkdir(parents=True, exist_ok=True)

	def _load_existing_counters(self) -> None:
		"""Load counters from existing images to avoid overwriting."""
		# Find max image ID
		if self.originals_dir.exists():
			existing_images = list(self.originals_dir.glob("*.*"))
			if existing_images:
				# Extract numeric IDs from filenames like "1.jpg", "2.jpg"
				ids = []
				for img_file in existing_images:
					stem = img_file.stem
					try:
						ids.append(int(stem))
					except ValueError:
						pass
				if ids:
					self.image_counter = max(ids)

	def _timestamp(self) -> str:
		return datetime.now(timezone.utc).isoformat()

	def _pad_and_clamp_bbox(
		self, x1: int, y1: int, x2: int, y2: int, image_width: int, image_height: int
	) -> BoundingBox:
		width = max(1, x2 - x1)
		height = max(1, y2 - y1)

		pad_x = int(width * self.padding_ratio)
		pad_y = int(height * self.padding_ratio)

		nx1 = max(0, x1 - pad_x)
		ny1 = max(0, y1 - pad_y)
		nx2 = min(image_width, x2 + pad_x)
		ny2 = min(image_height, y2 + pad_y)

		return BoundingBox(
			x=nx1,
			y=ny1,
			width=max(1, nx2 - nx1),
			height=max(1, ny2 - ny1),
		)

	def _dynamic_min_area(self, image_width: int, image_height: int) -> int:
		"""Compute a size-adaptive minimum area threshold from image dimensions."""
		image_area = image_width * image_height
		ratio_area = int(image_area * self.min_area_ratio)
		return max(self.min_area_floor, ratio_area)

	def _detect_with_yolo(self, image) -> List[Dict[str, object]]:
		print("🔍 Using YOLOv8-seg detection")
		detections: List[Dict[str, object]] = []
		result = self.model.predict(
			source=image,
			conf=self.conf_threshold,
			iou=self.iou_threshold,
			verbose=False,
		)[0]

		names = result.names if hasattr(result, "names") else {}
		image_height = image.shape[0]
		image_width = image.shape[1]
		masks = result.masks.data if result.masks is not None else None

		for idx, box in enumerate(result.boxes):
			confidence = float(box.conf[0])
			class_idx = int(box.cls[0])
			x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
			reason = yolo_box_filter_reason(x1, y1, x2, y2, image_height, image_width)
			if reason is not None:
				continue

			mask = None
			if masks is not None and idx < len(masks):
				mask = masks[idx].cpu().numpy()
				if mask.shape != (image_height, image_width):
					mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
				mask = (mask > 0.5).astype(np.uint8)

			detections.append(
				{
					"label": names.get(class_idx, "unknown_product"),
					"confidence": confidence,
					"xyxy": (x1, y1, x2, y2),
					"mask": mask,
				}
			)

		detections = self._suppress_nested_detections(detections)
		detections = self._suppress_high_overlap_conflicts(detections)
		return self._suppress_mask_overlap_conflicts(detections)

	def _merge_confidence(self, base_conf: float, nested_conf: float) -> float:
		"""Merge confidence for two detections represented as one object.

		Using probabilistic union keeps the merged score in [0, 1] and increases
		confidence when two detections support the same physical product.
		"""
		base = min(1.0, max(0.0, float(base_conf)))
		nested = min(1.0, max(0.0, float(nested_conf)))
		return min(1.0, base + nested - (base * nested))

	def _merge_detection_geometry(self, container: Dict[str, object], inner: Dict[str, object]) -> None:
		"""Merge bbox and mask from two detections into a single unified detection."""
		x1c, y1c, x2c, y2c = container["xyxy"]
		x1i, y1i, x2i, y2i = inner["xyxy"]

		container["xyxy"] = (
			min(x1c, x1i),
			min(y1c, y1i),
			max(x2c, x2i),
			max(y2c, y2i),
		)

		container_mask = container.get("mask")
		inner_mask = inner.get("mask")
		if isinstance(container_mask, np.ndarray) and isinstance(inner_mask, np.ndarray):
			container["mask"] = np.logical_or(container_mask > 0, inner_mask > 0).astype(np.uint8)
		elif not isinstance(container_mask, np.ndarray) and isinstance(inner_mask, np.ndarray):
			container["mask"] = inner_mask.copy()

	def _suppress_nested_detections(self, detections: List[Dict[str, object]]) -> List[Dict[str, object]]:
		"""Merge detections when one is mostly contained inside another.

		If a candidate box is >=90% inside another box, we treat both as the same
		physical product and merge confidence into the containing detection.
		"""
		if len(detections) < 2:
			return detections

		merged = [dict(det) for det in detections]
		suppressed_indices: set[int] = set()

		for i, det_i in enumerate(merged):
			if i in suppressed_indices:
				continue

			x1i, y1i, x2i, y2i = det_i["xyxy"]
			area_i = max(1, (x2i - x1i) * (y2i - y1i))

			for j, det_j in enumerate(merged):
				if i == j:
					continue
				if j in suppressed_indices:
					continue

				x1j, y1j, x2j, y2j = det_j["xyxy"]

				ix1 = max(x1i, x1j)
				ix2 = min(x2i, x2j)
				iy1 = max(y1i, y1j)
				iy2 = min(y2i, y2j)
				if ix2 <= ix1 or iy2 <= iy1:
					continue

				inter_area = (ix2 - ix1) * (iy2 - iy1)
				containment = inter_area / area_i

				if containment >= self.nested_containment_threshold:
					merged_conf = self._merge_confidence(
						float(det_j["confidence"]),
						float(det_i["confidence"]),
					)
					det_j["confidence"] = merged_conf
					self._merge_detection_geometry(det_j, det_i)
					suppressed_indices.add(i)
					break

		return [det for idx, det in enumerate(merged) if idx not in suppressed_indices]

	def _suppress_high_overlap_conflicts(self, detections: List[Dict[str, object]]) -> List[Dict[str, object]]:
		"""For near-identical overlapping boxes, keep only the higher-confidence detection."""
		if len(detections) < 2:
			return detections

		sorted_detections = sorted(detections, key=lambda d: float(d["confidence"]), reverse=True)
		kept: List[Dict[str, object]] = []

		for candidate in sorted_detections:
			x1c, y1c, x2c, y2c = candidate["xyxy"]
			area_c = max(1, (x2c - x1c) * (y2c - y1c))
			skip_candidate = False

			for existing in kept:
				x1e, y1e, x2e, y2e = existing["xyxy"]
				area_e = max(1, (x2e - x1e) * (y2e - y1e))

				ix1 = max(x1c, x1e)
				ix2 = min(x2c, x2e)
				iy1 = max(y1c, y1e)
				iy2 = min(y2c, y2e)
				if ix2 <= ix1 or iy2 <= iy1:
					continue

				inter_area = (ix2 - ix1) * (iy2 - iy1)
				iou = inter_area / max(1, area_c + area_e - inter_area)
				containment_c = inter_area / area_c

				if iou >= 0.8 or containment_c >= 0.9:
					skip_candidate = True
					break

			if not skip_candidate:
				kept.append(candidate)

		return kept

	def _mask_iou(self, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
		"""Compute IoU between two binary masks."""
		inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
		if inter == 0:
			return 0.0
		union = np.logical_or(mask_a > 0, mask_b > 0).sum()
		if union == 0:
			return 0.0
		return float(inter / union)

	def _suppress_mask_overlap_conflicts(self, detections: List[Dict[str, object]]) -> List[Dict[str, object]]:
		"""Suppress overlapping segmentation masks using mask IoU and confidence margin."""
		if len(detections) < 2:
			return detections

		sorted_detections = sorted(detections, key=lambda d: float(d["confidence"]), reverse=True)
		kept: List[Dict[str, object]] = []

		for candidate in sorted_detections:
			candidate_mask = candidate.get("mask")
			skip_candidate = False

			for existing in kept:
				existing_mask = existing.get("mask")
				if not isinstance(candidate_mask, np.ndarray) or not isinstance(existing_mask, np.ndarray):
					continue

				mask_iou = self._mask_iou(candidate_mask, existing_mask)
				if mask_iou < self.mask_iou_threshold:
					continue

				candidate_conf = float(candidate["confidence"])
				existing_conf = float(existing["confidence"])

				# If masks strongly overlap, keep only the stronger detection.
				if candidate_conf <= existing_conf + self.mask_conf_margin:
					skip_candidate = True
					break

			if not skip_candidate:
				kept.append(candidate)

		return kept

	def _detect_with_contours(self, image) -> List[Dict[str, object]]:
		print("🔍 Using CONTOUR-based detection (YOLO not available)")
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (7, 7), 0)
		
		# Use adaptive thresholding to handle varying lighting
		thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
		
		# Morphological operations to clean up
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
		closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
		opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

		contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		raw_boxes: List[List[int]] = []
		confidences: List[float] = []
		image_area = image.shape[0] * image.shape[1]
		image_height = image.shape[0]
		image_width = image.shape[1]
		min_area = self._dynamic_min_area(image_width=image_width, image_height=image_height)
		max_product_area = image_area * 0.15  # Box cannot be more than 15% of image

		for contour in contours:
			x, y, w, h = cv2.boundingRect(contour)
			area = w * h
			if area < min_area:
				continue
			
			# Filter by aspect ratio to avoid too thin/wide boxes
			aspect_ratio = w / max(1, h)
			if aspect_ratio > 5 or aspect_ratio < 0.2:
				continue
			
			# Reject boxes that cover too much of the image (likely noise/background)
			if area > max_product_area:
				continue
			
			# Reject boxes covering too much of width or height
			width_ratio = w / image_width
			height_ratio = h / image_height
			if width_ratio > 0.7 or height_ratio > 0.7:
				continue
			
			# Reject boxes touching all 4 edges (likely whole image)
			edge_threshold = 20  # pixels from edge
			touches_top = y < edge_threshold
			touches_bottom = (y + h) > (image_height - edge_threshold)
			touches_left = x < edge_threshold
			touches_right = (x + w) > (image_width - edge_threshold)
			
			if touches_top and touches_bottom and touches_left and touches_right:
				continue
				
			raw_boxes.append([x, y, w, h])
			# Confidence based on area (larger products = higher confidence)
			confidence = min(0.95, 0.4 + (area / max(1, image_area)) * 0.5)
			confidences.append(confidence)

		if len(raw_boxes) == 0:
			return []

		indices = cv2.dnn.NMSBoxes(raw_boxes, confidences, 0.3, self.iou_threshold)
		if len(indices) == 0:
			return []

		detections: List[Dict[str, object]] = []
		for idx in indices.flatten().tolist():
			x, y, w, h = raw_boxes[idx]
			detections.append(
				{
					"label": "product",
					"confidence": float(confidences[idx]),
					"xyxy": (x, y, x + w, y + h),
				}
			)

		return detections

	def detect_products(self, image) -> List[Dict[str, object]]:
		if self.model is not None:
			return self._detect_with_yolo(image)
		return self._detect_with_contours(image)

	def _persist_result(self, result: ImageAnalysisResult) -> None:
		output_file = self.metadata_dir / f"{result.image_id}.json"
		output_file.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

	def analyze_image(self, image_path: str) -> ImageAnalysisResult:
		source = Path(image_path)
		if not source.exists():
			raise FileNotFoundError(f"Image not found: {image_path}")

		image = cv2.imread(str(source))
		if image is None:
			raise ValueError(f"Unable to read image: {image_path}")

		self.image_counter += 1
		image_id = str(self.image_counter)
		file_ext = source.suffix.lower() if source.suffix else ".jpg"
		stored_original_path = self.originals_dir / f"{image_id}{file_ext}"
		cv2.imwrite(str(stored_original_path), image)

		detections = self.detect_products(image)
		products: List[ProductObject] = []
		min_area = self._dynamic_min_area(image_width=image.shape[1], image_height=image.shape[0])

		# Use image filename (without extension) as folder name
		image_folder_name = source.stem
		image_crop_dir = self.crops_dir / image_folder_name
		image_crop_dir.mkdir(parents=True, exist_ok=True)
		
		# Reset product counter for this image
		product_counter = 0

		for detection in detections:
			x1, y1, x2, y2 = detection["xyxy"]
			bbox = self._pad_and_clamp_bbox(
				x1,
				y1,
				x2,
				y2,
				image_width=image.shape[1],
				image_height=image.shape[0],
			)

			if bbox.width * bbox.height < min_area:
				continue

			# YOLOv8-seg: precise mask crop (img * mask) before bbox crop
			mask = detection.get("mask")
			if isinstance(mask, np.ndarray):
				mask_uint8 = (mask * 255).astype(np.uint8)
				mask_bool = mask_uint8 > 0
				white_bg = np.full_like(image, 255)
				white_bg[mask_bool] = image[mask_bool]
				crop = white_bg[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]
			else:
				crop = image[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]

			if crop.size == 0:
				continue

			product_counter += 1
			product_id = str(product_counter)
			crop_path = image_crop_dir / f"{product_id}.jpg"
			cv2.imwrite(str(crop_path), crop)

			product = ProductObject(
				product_id=product_id,
				source_image_id=image_id,
				label=str(detection["label"]),
				confidence=round(float(detection["confidence"]), 4),
				bbox=bbox,
				crop_path=str(crop_path),
				created_at=self._timestamp(),
				extra={},
			)
			products.append(product)

		status = "success" if products else "no_products_detected"
		result = ImageAnalysisResult(
			image_id=image_id,
			original_path=str(stored_original_path),
			products=products,
			total_products=len(products),
			status=status,
		)

		self.product_objects.extend(products)
		self.image_results[image_id] = result
		self._persist_result(result)

		return result

	def get_all_products(self) -> List[Dict[str, object]]:
		return [item.to_dict() for item in self.product_objects]

	def get_products_by_image(self, image_id: str) -> List[Dict[str, object]]:
		result = self.image_results.get(image_id)
		if result is None:
			return []
		return [item.to_dict() for item in result.products]


def benchmark_yolo_models(
	image_path: str,
	model_names: List[str],
	conf_threshold: float,
	iou_threshold: float,
) -> List[Dict[str, object]]:
	"""Benchmark several YOLO models on the same image and filtering rules."""
	source = Path(image_path)
	if not source.exists():
		raise FileNotFoundError(f"Image not found: {image_path}")

	image = cv2.imread(str(source))
	if image is None:
		raise ValueError(f"Unable to read image: {image_path}")

	image_height = image.shape[0]
	image_width = image.shape[1]
	reports: List[Dict[str, object]] = []

	for model_name in model_names:
		start = time.perf_counter()
		model = YOLO(model_name)
		result = model.predict(
			source=image,
			conf=conf_threshold,
			iou=iou_threshold,
			verbose=False,
		)[0]
		elapsed_ms = (time.perf_counter() - start) * 1000

		raw_count = len(result.boxes)
		kept_count = 0
		rejected_area = 0
		rejected_dimensions = 0
		rejected_full_frame = 0

		for box in result.boxes:
			x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
			reason = yolo_box_filter_reason(x1, y1, x2, y2, image_height, image_width)
			if reason is None:
				kept_count += 1
			elif reason == "area":
				rejected_area += 1
			elif reason == "dimensions":
				rejected_dimensions += 1
			elif reason == "full_frame":
				rejected_full_frame += 1

		reports.append(
			{
				"model": model_name,
				"time_ms": round(elapsed_ms, 2),
				"raw_detections": raw_count,
				"kept_detections": kept_count,
				"rejected_area": rejected_area,
				"rejected_dimensions": rejected_dimensions,
				"rejected_full_frame": rejected_full_frame,
			}
		)

	reports.sort(key=lambda item: (-int(item["kept_detections"]), float(item["time_ms"])))
	return reports


def main() -> None:
	# Hardcoded image path
	image_path = "ImageC/test1.jpg"
	model_candidates = ["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt"]
	
	
	selected_model = "yolov8l-seg.pt"

	workflow = ProductWorkflow(model_name=selected_model)
	result = workflow.analyze_image(image_path)
	print(f"✅ Found {result.total_products} products")
	print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
	main()
