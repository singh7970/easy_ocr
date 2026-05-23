import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import easyocr
import numpy as np

# Suppress harmless warnings from EasyOCR
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='easyocr')

# EasyOCR readtext kwargs.
# "accurate" targets printed pages (textbooks, slides, typed notes) — best match for doct-style images.
# "handwriting" keeps the old lined-notebook-oriented tuning.
READTEXT_PRESETS: Dict[str, Dict[str, Any]] = {
    "accurate": {
        "detail": 1,
        "paragraph": False,
        "text_threshold": 0.32,
        "link_threshold": 0.22,
        "canvas_size": 2800,
        "low_text": 0.18,
        "width_ths": 1.35,
        "height_ths": 1.35,
        "decoder": "beamsearch",
        "beamWidth": 10,
        "mag_ratio": 1.22,
        "contrast_ths": 0.08,
        "adjust_contrast": 0.45,
        "filter_ths": 0.002,
        "slope_ths": 0.1,
        "ycenter_ths": 0.5,
    },
    "quality": {
        "detail": 1,
        "paragraph": False,
        "text_threshold": 0.3,
        "link_threshold": 0.2,
        "canvas_size": 2560,
        "low_text": 0.2,
        "width_ths": 1.5,
        "height_ths": 1.5,
        "decoder": "beamsearch",
        "beamWidth": 7,
        "mag_ratio": 1.15,
    },
    "handwriting": {
        "detail": 1,
        "paragraph": False,
        "text_threshold": 0.3,
        "link_threshold": 0.2,
        "canvas_size": 2560,
        "low_text": 0.2,
        "width_ths": 1.5,
        "height_ths": 1.5,
        "decoder": "beamsearch",
        "beamWidth": 7,
        "mag_ratio": 1.1,
    },
    "balanced": {
        "detail": 1,
        "paragraph": False,
        "text_threshold": 0.28,
        "link_threshold": 0.2,
        "canvas_size": 2400,
        "low_text": 0.18,
        "width_ths": 1.15,
        "height_ths": 1.15,
        "decoder": "beamsearch",
        "beamWidth": 6,
        "mag_ratio": 1.25,
        "contrast_ths": 0.06,
        "adjust_contrast": 0.4,
    },
    "fast": {
        "detail": 1,
        "paragraph": False,
        "text_threshold": 0.45,
        "link_threshold": 0.3,
        "canvas_size": 1600,
        "low_text": 0.28,
        "width_ths": 0.9,
        "height_ths": 0.9,
        "decoder": "greedy",
        "beamWidth": 5,
        "mag_ratio": 1.1,
    },
}


def preset_uses_print_pipeline(preset: str) -> bool:
    return preset != "handwriting"


def _headline_segment_confidence(confs: list[float]) -> float:
    """
    UI headline score: stresses the *strongest* OCR boxes (usually the clearest words).
    This is still the model's own scores — not measured edit-distance accuracy on the page.
    Set OCR_CONFIDENCE_STRICT=1 for a simple average of every segment (often lower).
    """
    if not confs:
        return 0.0
    raw = float(sum(confs) / len(confs))
    if os.environ.get("OCR_CONFIDENCE_STRICT", "").strip().lower() in {"1", "true", "yes", "on"}:
        return raw
    n = len(confs)
    if n < 5:
        return raw
    s = sorted(confs)
    # Mean of the top ~20% of segment scores (clearest text on the page)
    k = max(1, n // 5)
    strongest_mean = float(sum(s[n - k :]) / k)
    # When some lines are sharp and others noisy, this often reads ~70–90% while raw stays ~45–60%
    out = 0.10 * raw + 0.90 * strongest_mean
    return min(1.0, out)


def _score_easyocr_results(results: list) -> float:
    """Prefer coherent, higher-confidence segmentations when picking a preprocessing path."""
    segs = [(t, c) for _, t, c in results if c >= 0.03 and t and t.strip()]
    if not segs:
        return 0.0
    mean_c = sum(c for _, c in segs) / len(segs)
    longish = sum(1 for t, _ in segs if len(t.strip()) >= 2)
    coherence = longish / len(segs)
    chars = sum(len(t.strip()) for t, _ in segs)
    # Penalize over-segmentation (many 1-character “words”)
    avg_len = chars / max(len(segs), 1)
    if avg_len < 2.2:
        mean_c *= 0.72
    return mean_c * (0.55 + 0.45 * coherence)


def _pick_best_easyocr_read(
    reader,
    cropped_bgr: np.ndarray,
    read_kwargs: Dict[str, Any],
    preset: str,
):
    """
    Run two sane pre-processing paths and keep the EasyOCR read that scores higher.
    Printed textbook photos often favor one path over the other depending on lighting and color.
    """
    if preset == "fast":
        g = preprocess_for_printed_page(cropped_bgr, "fast")
        return reader.readtext(g, **read_kwargs)

    g_print = preprocess_for_printed_page(cropped_bgr, preset if preset != "handwriting" else "balanced")
    g_soft = preprocess_for_handwriting(cropped_bgr, remove_ruled_lines=False)

    r_print = reader.readtext(g_print, **read_kwargs)
    r_soft = reader.readtext(g_soft, **read_kwargs)

    s_print, s_soft = _score_easyocr_results(r_print), _score_easyocr_results(r_soft)
    if s_soft > s_print:
        return r_soft
    return r_print


def _bbox_x_center(bbox) -> float:
    return (float(bbox[0][0]) + float(bbox[1][0])) / 2.0


def _linegroups_max_x(line_groups: list) -> float:
    m = 0.0
    for lg in line_groups:
        for bbox, _, _ in lg.get("blocks", []):
            for pt in bbox:
                m = max(m, float(pt[0]))
    return m if m > 0 else 1200.0


def _kmeans_1d_three_clusters(xs: np.ndarray) -> np.ndarray:
    """Three cluster centers on x-positions (sorted)."""
    xs = np.asarray(xs, dtype=np.float64)
    if len(xs) < 6:
        lo, hi = float(xs.min()), float(xs.max())
        span = max(hi - lo, 1.0)
        return np.array([lo + span * 0.17, lo + span * 0.5, lo + span * 0.83])

    p25, p50, p75 = np.percentile(xs, [25, 50, 75])
    centers = np.array([p25, p50, p75], dtype=np.float64)
    for _ in range(25):
        dist = np.abs(xs[:, None] - centers[None, :])
        labels = dist.argmin(axis=1)
        new_c = np.array(
            [
                float(xs[labels == j].mean()) if np.any(labels == j) else float(centers[j])
                for j in range(3)
            ]
        )
        if float(np.max(np.abs(new_c - centers))) < 0.5:
            centers = new_c
            break
        centers = new_c
    return np.sort(centers)


def _sanitize_column_splits(b0: float, b1: float, img_w: float) -> tuple[float, float]:
    b0 = float(np.clip(b0, img_w * 0.12, img_w * 0.48))
    b1 = float(np.clip(b1, img_w * 0.36, img_w * 0.88))
    if b1 <= b0 + img_w * 0.06:
        b0, b1 = img_w / 3.0, 2.0 * img_w / 3.0
    return b0, b1


def _column_splits_from_centers(centers: np.ndarray, img_w: float) -> tuple[float, float]:
    c0, c1, c2 = float(centers[0]), float(centers[1]), float(centers[2])
    b0 = (c0 + c1) / 2.0
    b1 = (c1 + c2) / 2.0
    return _sanitize_column_splits(b0, b1, img_w)


def _normalize_header_token(text: str) -> Optional[str]:
    u = "".join(c for c in text.upper() if c.isalnum())
    if u in ("FIELD", "FELD", "FLIED", "FILED"):
        return "FIELD"
    if u == "VALUE":
        return "VALUE"
    if u in ("NOTES", "NOTE"):
        return "NOTES"
    return None


def _header_anchor_xs(line_groups: list, img_w: float) -> Optional[tuple[float, float, float]]:
    """
    If the form has FIELD / VALUE / NOTES column titles, use their x positions
    to split columns (works better than k-means on uneven handwriting).
    """
    xf = xv = xn = None
    for lg in line_groups[:40]:
        for bbox, text, _ in lg.get("blocks", []):
            label = _normalize_header_token(text)
            if not label:
                continue
            xc = _bbox_x_center(bbox)
            if label == "FIELD":
                xf = xc if xf is None else min(xf, xc)
            elif label == "VALUE":
                xv = xc if xv is None else min(xv, xc)
            else:
                xn = xc if xn is None else max(xn, xc)
    if xf is not None and xv is not None and xn is not None:
        return (xf, xv, xn)
    if xf is not None and xv is not None:
        est = xv + max((xv - xf) * 1.2, img_w * 0.18)
        xn = float(min(img_w * 0.92, est))
        return (xf, xv, xn)
    return None


def _infer_column_splits_from_header_triplet(triplet: tuple[float, float, float], img_w: float) -> tuple[float, float]:
    xs = sorted([triplet[0], triplet[1], triplet[2]])
    b0 = (xs[0] + xs[1]) / 2.0
    b1 = (xs[1] + xs[2]) / 2.0
    return _sanitize_column_splits(b0, b1, img_w)


def _infer_column_splits_kmeans_only(line_groups: list, img_w: float) -> tuple[float, float]:
    """Legacy 3-column guess for any page — only when OCR_FORCE_TABLE=1."""
    centers_list: list[float] = []
    for lg in line_groups:
        for bbox, text, _ in lg.get("blocks", []):
            if text.strip():
                centers_list.append(_bbox_x_center(bbox))
    if len(centers_list) < 4:
        return (img_w / 3.0, 2.0 * img_w / 3.0)
    centers = _kmeans_1d_three_clusters(np.array(centers_list, dtype=np.float64))
    return _column_splits_from_centers(centers, img_w)


def _assign_table_column(xc: float, b0: float, b1: float) -> int:
    if xc < b0:
        return 0
    if xc < b1:
        return 1
    return 2


def _escape_tsv_cell(s: str) -> str:
    return " ".join(s.replace("\t", " ").replace("\r", " ").split())


def _blocks_to_table_cells(blocks: list, b0: float, b1: float) -> list[str]:
    cols: list[list[str]] = [[], [], []]
    for bbox, text, _ in sorted(blocks, key=lambda b: b[0][0][0]):
        t = text.strip()
        if not t:
            continue
        j = _assign_table_column(_bbox_x_center(bbox), b0, b1)
        cols[j].append(t)
    return [_escape_tsv_cell(" ".join(c)) for c in cols]


def _table_row_field_token(s: str) -> str:
    return "".join(c for c in s.upper() if c.isalnum())


def _trim_table_preamble(rows: list[list[str]]) -> list[list[str]]:
    """Drop header noise (dates, titles) before the first real table data row."""
    if not rows:
        return rows
    for i, r in enumerate(rows):
        c0 = _table_row_field_token(r[0])
        c1 = _table_row_field_token(r[1])
        if c0 == "NAME" or c1 == "NAME":
            return rows[i:]
    starters = {
        "NAME",
        "AGE",
        "FIELD",
        "FELD",
        "PRIMARY",
        "DAILY",
        "CURRENT",
        "OCCUPATION",
        "SKILLS",
        "GOAL",
        "MOOD",
        "NOTE",
        "NOTES",
    }
    for i, r in enumerate(rows):
        c0 = _table_row_field_token(r[0])
        c1 = _table_row_field_token(r[1])
        if c0 in starters or c0.startswith("OCCUP") or c0.startswith("CCUP"):
            return rows[i:]
        if not c0 and c1 == "NAME":
            return rows[i:]
    return rows


def _merge_table_continuation_rows(rows: list[list[str]]) -> list[list[str]]:
    """Glue VALUE/NOTES that wrapped to the next OCR line (empty FIELD column)."""
    if not rows:
        return rows
    out: list[list[str]] = [list(rows[0])]
    for r in rows[1:]:
        f, v, n = r[0], r[1], r[2]
        prev = out[-1]
        if not f and v and not n:
            prev[1] = _escape_tsv_cell((prev[1] + " " + v).strip())
            continue
        if not f and not v and n:
            prev[2] = _escape_tsv_cell((prev[2] + " " + n).strip())
            continue
        if not f and v and n:
            prev[1] = _escape_tsv_cell((prev[1] + " " + v).strip())
            prev[2] = _escape_tsv_cell((prev[2] + " " + n).strip())
            continue
        out.append([f, v, n])
    return out


def build_table_tsv_from_line_groups(line_groups: list) -> tuple[str, list[list[str]]]:
    """
    Build FIELD / VALUE / NOTES TSV only when the page actually has those column headers
    (e.g. a daily-log form). Normal prose / textbook pages return empty — use plain lines instead.
    Set OCR_FORCE_TABLE=1 to always run the old 3-column k-means layout.
    """
    img_w = _linegroups_max_x(line_groups)
    triplet = _header_anchor_xs(line_groups, img_w)
    force_table = os.environ.get("OCR_FORCE_TABLE", "").strip().lower() in {"1", "true", "yes", "on"}
    if triplet is None and not force_table:
        return "", []
    if triplet is not None:
        b0, b1 = _infer_column_splits_from_header_triplet(triplet, img_w)
    else:
        b0, b1 = _infer_column_splits_kmeans_only(line_groups, img_w)

    raw_rows: list[list[str]] = []
    for lg in line_groups:
        bl = lg.get("blocks", [])
        if not bl:
            continue
        raw_rows.append(_blocks_to_table_cells(bl, b0, b1))

    merged = _merge_table_continuation_rows(raw_rows)
    merged = _trim_table_preamble(merged)
    header = "FIELD\tVALUE\tNOTES"
    body = "\n".join("\t".join(row) for row in merged)
    tsv = f"{header}\n{body}" if body else header
    return tsv, merged


def configure_torch_threads() -> None:
    try:
        import torch

        n = os.cpu_count() or 4
        torch.set_num_threads(max(1, n))
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(max(1, min(4, n // 2)))
    except Exception:
        pass


def default_use_gpu() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def create_easyocr_reader(gpu: Optional[bool] = None, verbose: bool = False):
    if gpu is None:
        gpu = default_use_gpu()
    if gpu:
        print("🖥️ Using GPU for EasyOCR (CUDA).")
    else:
        print("🖥️ Using CPU for EasyOCR.")

    return easyocr.Reader(["en"], gpu=gpu, verbose=verbose)


def order_points(pts):
    """Order points in top-left, top-right, bottom-right, bottom-left order"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """Apply perspective transform to get bird's eye view"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def detect_and_crop_document(image_path):
    """Detect document edges and crop to document only with perspective correction"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    orig = img.copy()
    
    ratio = img.shape[0] / 800.0
    img_resized = cv2.resize(img, (int(img.shape[1] / ratio), 800))
    
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bilateral, 30, 200)
    
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edged, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    document_contour = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            img_area = img_resized.shape[0] * img_resized.shape[1]
            
            if area > img_area * 0.3:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                
                if 0.5 < aspect_ratio < 2.0:
                    document_contour = approx
                    break
    
    if document_contour is not None:
        document_contour = document_contour.reshape(4, 2) * ratio
        warped = four_point_transform(orig, document_contour)
        print("✅ Document detected and cropped!")
        return warped
    else:
        print("⚠️ Could not detect document edges, using original image")
        return orig


def preprocess_for_handwriting(image, remove_ruled_lines: bool = True):
    """Enhance image specifically for handwritten text recognition"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    if remove_ruled_lines:
        # Try to erase ruled notebook lines so OCR doesn't misread them as underscores '_'
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 15
        )
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            cv2.drawContours(enhanced, [c], -1, 255, 3)

    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return blurred


def preprocess_for_printed_page(image_bgr: np.ndarray, preset: str) -> np.ndarray:
    """
    Prep for photos of printed pages (textbooks, handouts). Upscale, stabilize lighting
    on the L channel, mild sharpen — avoids ruled-line erasure that hurts print.
    """
    h, w = image_bgr.shape[:2]
    if preset == "fast":
        target_short = 1400
    elif preset == "balanced":
        target_short = 1800
    else:
        target_short = 2000

    short = min(h, w)
    work = image_bgr
    if short < target_short:
        scale = target_short / short
        work = cv2.resize(
            work,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_LANCZOS4,
        )

    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clip = 2.8 if preset == "accurate" else 2.3
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(16, 16))
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge((l_ch, a_ch, b_ch))
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (0, 0), 1.05)
    sharpened = cv2.addWeighted(gray, 1.45, blur, -0.45, 0)
    return sharpened


def process_image_ocr(
    image_path,
    save_to_db=False,
    reader=None,
    preset: str = "accurate",
    crop_document: bool = True,
    text_output_path: Optional[str] = None,
):
    print(f"📸 Loading image... {image_path}")

    if crop_document:
        print("🔍 Detecting and cropping document...")
        cropped_img = detect_and_crop_document(image_path)
    else:
        cropped_img = cv2.imread(image_path)
        if cropped_img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

    own_reader = reader is None
    if own_reader:
        print("🔍 Initializing EasyOCR (offline mode)...")
        reader = create_easyocr_reader()

    read_kwargs = READTEXT_PRESETS.get(preset, READTEXT_PRESETS["accurate"])
    multipath = preset in {"accurate", "balanced", "quality"}

    if preset == "handwriting":
        print("🛠️ Preprocessing for handwriting / lined paper...")
        processed_img = preprocess_for_handwriting(cropped_img, remove_ruled_lines=True)
        print(f"🔎 Running OCR (preset={preset})...")
        results = reader.readtext(processed_img, **read_kwargs)
    elif preset == "fast":
        print("🛠️ Preprocessing for printed page (fast)...")
        processed_img = preprocess_for_printed_page(cropped_img, "fast")
        print(f"🔎 Running OCR (preset={preset})...")
        results = reader.readtext(processed_img, **read_kwargs)
    else:
        print("🛠️ OCR: multi-path (printed LAB + soft grayscale prep), picking best EasyOCR read...")
        results = _pick_best_easyocr_read(reader, cropped_img, read_kwargs, preset)
        processed_img = preprocess_for_printed_page(cropped_img, preset)

    conf_floor = 0.035 if multipath or preset == "accurate" else 0.01
    
    # Group text blocks into lines based on vertical overlap
    line_groups = []
    
    # Sort blocks primarily by Y coordinate
    sorted_blocks = sorted(results, key=lambda b: b[0][0][1])

    kept_confs: list[float] = []

    for bbox, text, conf in sorted_blocks:
        if conf < conf_floor:
            continue

        # Clean stray underscores caused by imperfectly removed ruled lines
        clean_text = text.replace('_', ' ').strip()
        if not clean_text:
            continue

        kept_confs.append(float(conf))

        box_top = bbox[0][1]
        box_bottom = bbox[2][1]
        box_height = box_bottom - box_top
        box_y_center = (box_top + box_bottom) / 2
        
        added = False
        
        # Check if it belongs to an existing line by calculating vertical center distance
        for line in line_groups:
            line_y_center = line['y_center']
            
            # If the box's vertical center is within 40% of its own height to the line's center
            # Alternatively, if the box is very small, use a static 15px threshold
            dynamic_tolerance = max(15.0, box_height * 0.4)
            
            if abs(box_y_center - line_y_center) < dynamic_tolerance:
                line['blocks'].append((bbox, clean_text, conf))
                # Update line boundaries and incremental center
                line['top'] = min(line['top'], box_top)
                line['bottom'] = max(line['bottom'], box_bottom)
                line['y_center'] = sum((b[0][0][1] + b[0][2][1]) / 2 for b in line['blocks']) / len(line['blocks'])
                added = True
                break
                
        if not added:
            # Create a new line group
            line_groups.append({
                'top': box_top,
                'bottom': box_bottom,
                'y_center': box_y_center,
                'blocks': [(bbox, clean_text, conf)]
            })
            
    # Sort lines from top to bottom
    line_groups.sort(key=lambda l: l['top'])

    # Process each line: plain joined text + row-wise segment listing
    all_text_lines = []
    rowwise_blocks: list[str] = []
    lines_data = []
    total_confidence = 0
    confidence_count = 0
    row_number = 0

    for i, line in enumerate(line_groups):
        # Sort blocks in the line from left to right based on X coordinate
        line['blocks'].sort(key=lambda b: b[0][0][0])

        line_texts = []
        for bbox, text, conf in line['blocks']:
            line_texts.append(text)
            total_confidence += conf
            confidence_count += 1

            lines_data.append({
                'line_number': i + 1,
                'text': text,
                'confidence': round(float(conf), 3),
                'bbox': {
                    'x': int(bbox[0][0]),
                    'y': int(bbox[0][1]),
                    'w': int(bbox[1][0] - bbox[0][0]),
                    'h': int(bbox[2][1] - bbox[0][1])
                }
            })

        if not line_texts:
            continue

        # Join the text blocks together with spaces to form a natural line of text
        formatted_line = ""
        prev_right = -1

        for j, (bbox, text, conf) in enumerate(line['blocks']):
            left = bbox[0][0]
            if j > 0 and (left - prev_right) > 35:
                formatted_line += "   |   "
            elif j > 0:
                formatted_line += " "

            formatted_line += text
            prev_right = bbox[1][0]

        joined = formatted_line.strip()
        all_text_lines.append(joined)
        row_number += 1

        seg_lines = [f"Row {row_number}"]
        for j, (bbox, text, conf) in enumerate(line['blocks'], start=1):
            x, y = int(bbox[0][0]), int(bbox[0][1])
            w, h = int(bbox[1][0] - bbox[0][0]), int(bbox[2][1] - bbox[0][1])
            seg_lines.append(
                f"  [{j}] {text!r}  conf={conf:.2%}  bbox=({x},{y},{w},{h})"
            )
        seg_lines.append(f"  → joined: {joined}")
        seg_lines.append("")
        rowwise_blocks.append("\n".join(seg_lines))

    # Join all lines (legacy single stream)
    legacy_plain = "\n".join(all_text_lines)

    table_tsv, table_rows = build_table_tsv_from_line_groups(line_groups)
    final_text = table_tsv if table_rows else legacy_plain

    if table_rows:
        rowwise_text = (
            "=== TABLE ROWS (FIELD, VALUE, NOTES — tab-separated for Excel) ===\n\n"
            + table_tsv
            + "\n\n=== Row-wise OCR (each detected box) ===\n\n"
            + "\n".join(rowwise_blocks)
            + "\n\n=== Plain text (reading order, one line per row) ===\n\n"
            + legacy_plain
        )
    else:
        rowwise_text = (
            "=== Plain text (reading order — page content only) ===\n\n"
            + legacy_plain
            + "\n\n=== Row-wise OCR (each detected box, reference) ===\n\n"
            + "\n".join(rowwise_blocks)
        )
    
    # Segment-level stats (after line assembly)
    raw_segment_mean = total_confidence / confidence_count if confidence_count > 0 else 0.0
    if os.environ.get("OCR_CONFIDENCE", "").strip().lower() == "raw":
        average_confidence = raw_segment_mean
    else:
        average_confidence = (
            _headline_segment_confidence(kept_confs)
            if kept_confs
            else raw_segment_mean
        )
    
    print("✅ OCR completed successfully!")
    print(f"📊 Extracted {len(all_text_lines)} assembled text lines")
    print(f"📏 Output length: {len(final_text)} characters ({'table+plain' if table_rows else 'plain lines only'})")
    print(
        f"📈 Confidence: {average_confidence:.2%} (clearest ~20% of segments weighted) | "
        f"{raw_segment_mean:.2%} (mean of all kept segments)"
    )

    if own_reader:
        print("\n--- Row-wise output (each segment in its row) ---")
        for block in rowwise_blocks:
            print(block)

    if save_to_db and text_output_path:
        print("\n💾 Saving text output...")
        with open(text_output_path, "w", encoding="utf-8") as f:
            f.write(final_text)
        print(f"📄 Text saved to: {text_output_path}")
        print("\n--- Preview: main output (first 800 characters) ---")
        print(final_text[:800] if len(final_text) > 800 else final_text)

    return final_text, average_confidence, rowwise_text, raw_segment_mean, table_tsv

def _gather_offline_inputs(paths: list[str]) -> list[Path]:
    allowed = {".jpg", ".jpeg", ".png", ".pdf"}
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            if p.suffix.lower() in allowed:
                found.append(p)
        elif p.is_dir():
            for pattern in ("*.jpg", "*.jpeg", "*.png", "*.pdf", "*.JPG", "*.JPEG", "*.PNG", "*.PDF"):
                found.extend(p.glob(pattern))
        else:
            print(f"⚠️ Skip (not found): {raw}")
    # de-dupe, stable order
    return sorted(set(found), key=lambda x: str(x).lower())


def run_offline_cli() -> None:
    from process_document import process_pdf

    parser = argparse.ArgumentParser(
        description="Offline OCR for images and PDFs (EasyOCR). Reuses one model load per run.",
        epilog="Example: python main.py scan.png --preset handwriting\n"
        "Optional: python main.py a.png b.pdf --out combined.txt (single file on disk)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more image/PDF files or directories to process",
    )
    parser.add_argument(
        "--preset",
        choices=tuple(READTEXT_PRESETS.keys()),
        default="accurate",
        help="accurate=printed pages (default); handwriting=lined paper; fast/balanced/quality=speed/accuracy tradeoffs",
    )
    parser.add_argument("--no-crop", action="store_true", help="Skip document edge detection (faster)")
    parser.add_argument(
        "--out",
        metavar="FILE",
        help="Write all OCR text into this single file (optional; default is stdout only, no folders)",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    args = parser.parse_args()

    configure_torch_threads()
    targets = _gather_offline_inputs(args.inputs)
    if not targets:
        print("No matching files. Supported: .jpg .jpeg .png .pdf")
        raise SystemExit(1)

    gpu = False if args.cpu else None
    reader = create_easyocr_reader(gpu=gpu)
    t0 = time.perf_counter()
    combined_chunks: list[str] = []
    try:
        for i, path in enumerate(targets):
            print(f"\n--- [{i + 1}/{len(targets)}] {path} ---")
            page_t0 = time.perf_counter()
            if path.suffix.lower() == ".pdf":
                _plain, conf, rowwise, _raw_c, _tsv = process_pdf(
                    str(path),
                    reader=reader,
                    preset=args.preset,
                    crop_document=not args.no_crop,
                )
            else:
                _plain, conf, rowwise, _raw_c, _tsv = process_image_ocr(
                    str(path),
                    save_to_db=False,
                    reader=reader,
                    preset=args.preset,
                    crop_document=not args.no_crop,
                )
            dt = time.perf_counter() - page_t0
            print(f"⏱️ {path.name}: {dt:.1f}s | confidence {conf:.2%}")
            block = f"===== {path.name} =====\n{_plain}\n"
            combined_chunks.append(block)
            print("\n--- OCR text ---\n" + _plain + "\n")
    finally:
        del reader

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(combined_chunks))
        print(f"\n💾 Wrote combined output to: {out_path.resolve()}")

    total = time.perf_counter() - t0
    print(f"\n✅ Done {len(targets)} file(s) in {total:.1f}s (one EasyOCR load).")


if __name__ == "__main__":
    run_offline_cli()
