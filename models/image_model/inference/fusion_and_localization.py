"""
Fusion & Localization Engine for Image Deepfake & Tampering Forensics.
Blends multi-modal signals (Spatial CAM, Noise SRM, Frequency ELA, Illumination Physics),
generates overlaid forensic heatmaps, extracts suspicious bounding boxes,
and synthesizes natural forensic explanations conforming to TruthLens standards.
"""

import os
from pathlib import Path
import numpy as np
import cv2


class ImageForensicsFusionEngine:
    """
    Consolidates spatial, frequency, and illumination signals into a single
    calibrated forensic assessment with heatmap visualizations and localized regions.
    """

    def __init__(self, heatmap_output_dir: str = "storage/heatmaps"):
        self.heatmap_output_dir = Path(heatmap_output_dir)
        self.heatmap_output_dir.mkdir(parents=True, exist_ok=True)

    def fuse_signals(
        self,
        spatial_score: float,
        srm_score: float,
        frequency_score: float,
        illum_score: float,
        angle_deg: float
    ) -> tuple[float, float, float]:
        """
        Calibrates and computes visual_score, frequency_score, and composite manipulation_score.
        
        Returns:
            (visual_score, frequency_score, manipulation_score)
        """
        # Primary anchor is deep neural patch representation + sensor noise
        if spatial_score > 0.45:
            visual_score = float(0.55 * spatial_score + 0.25 * srm_score + 0.20 * illum_score)
        elif angle_deg > 45.0 and srm_score > 0.45 and spatial_score > 0.38:
            visual_score = float(0.45 * spatial_score + 0.30 * srm_score + 0.25 * illum_score)
        else:
            # Authentic photo/video regime: anchor primarily on deep neural patch features
            visual_score = float(0.70 * spatial_score + 0.20 * srm_score + 0.10 * illum_score)
            
        visual_score = float(np.clip(visual_score, 0.0, 1.0))
        
        # Unified Frequency / SRM Inconsistency
        # Frequency/ELA is corroborated with SRM sensor noise to separate real compression from tampering
        if srm_score > 0.40 and frequency_score > 0.40:
            freq_score = float(np.clip(0.60 * srm_score + 0.40 * frequency_score, 0.0, 1.0))
        elif spatial_score < 0.35 and srm_score < 0.35:
            # Standard video/JPEG compression without spatial tampering
            freq_score = float(np.clip(0.40 * srm_score + 0.30 * frequency_score, 0.0, 0.45))
        else:
            freq_score = float(np.clip(0.50 * srm_score + 0.50 * frequency_score, 0.0, 1.0))

        # Balanced fusion: anchor on visual spatial neural evidence
        if visual_score > 0.50:
            manipulation_score = float(np.clip(0.70 * visual_score + 0.30 * freq_score, 0.0, 1.0))
        elif visual_score < 0.35:
            manipulation_score = float(np.clip(0.75 * visual_score + 0.25 * freq_score, 0.0, 1.0))
        else:
            manipulation_score = float(np.clip(0.60 * visual_score + 0.40 * freq_score, 0.0, 1.0))

        return visual_score, freq_score, manipulation_score

    def generate_heatmap(
        self,
        image_bgr: np.ndarray,
        cam_map: np.ndarray,
        srm_map: np.ndarray,
        freq_map: np.ndarray,
        illum_map: np.ndarray,
        evidence_id: str
    ) -> tuple[str, np.ndarray]:
        """
        Synthesizes a combined forensic heatmap, overlays it onto the original image,
        and saves it to storage/heatmaps/{evidence_id}_heatmap.png.
        
        Returns:
            (heatmap_filename, blended_anomaly_map)
        """
        h, w = image_bgr.shape[:2]

        # Normalize and align dimensions
        maps = [cam_map, srm_map, freq_map, illum_map]
        resized_maps = []
        for m in maps:
            if m.shape[:2] != (h, w):
                m_res = cv2.resize(m, (w, h), interpolation=cv2.INTER_CUBIC)
            else:
                m_res = m
            resized_maps.append(np.clip(m_res, 0.0, 1.0))

        # Weighted blend of anomaly maps: prioritize neural mask
        combined = (
            0.45 * resized_maps[0] +
            0.20 * resized_maps[1] +
            0.15 * resized_maps[2] +
            0.20 * resized_maps[3]
        )
        
        combined = cv2.GaussianBlur(combined, (11, 11), 2.0)
        p98 = np.percentile(combined, 98.0)
        if p98 > 1e-4:
            combined_norm = np.clip(combined / p98, 0.0, 1.0)
        else:
            combined_norm = combined

        heatmap_uint8 = np.uint8(255 * combined_norm)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        alpha = 0.45
        overlay = cv2.addWeighted(heatmap_color, alpha, image_bgr, 1.0 - alpha, 0)

        filename = f"{evidence_id}_heatmap.png"
        filepath = self.heatmap_output_dir / filename
        cv2.imwrite(str(filepath), overlay)

        return filename, combined_norm

    def extract_suspicious_regions(
        self,
        anomaly_map: np.ndarray,
        illum_score: float,
        freq_score: float,
        angle_deg: float,
        manipulation_score: float = 0.5,
        max_regions: int = 4
    ) -> list[dict]:
        """
        Extracts bounding boxes [ymin, xmin, ymax, xmax] for high-anomaly clusters.
        """
        # If overall manipulation score indicates high authenticity, do not emit false regions
        if manipulation_score < 0.35:
            return []

        h, w = anomaly_map.shape[:2]
        img_area = h * w

        # Threshold top activations
        threshold = max(0.50, float(np.percentile(anomaly_map, 88.0)))
        binary_mask = (anomaly_map >= threshold).astype(np.uint8) * 255

        # Morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter out tiny noise (< 1.5% of image)
            if area < (0.015 * img_area):
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            # Bounding box as [ymin, xmin, ymax, xmax]
            box = [int(y), int(x), int(y + bh), int(x + bw)]

            # Compute regional anomaly score
            region_patch = anomaly_map[y:y+bh, x:x+bw]
            score = float(np.mean(region_patch))

            # Classify label based on dominant forensic signals
            if illum_score > 0.70 and angle_deg > 35.0:
                label = "illumination_vector_mismatch"
            elif freq_score > 0.70:
                label = "frequency_dct_anomaly"
            else:
                label = "facial_boundary_distortion" if (y < h*0.6 and x > w*0.2 and x < w*0.8) else "splicing_boundary_distortion"

            regions.append({
                "frame_index": 0,
                "box": box,
                "label": label,
                "anomaly_score": round(min(1.0, score * 1.2), 3)
            })

        # Sort by anomaly score descending
        regions = sorted(regions, key=lambda r: r["anomaly_score"], reverse=True)[:max_regions]
        return regions

    def generate_explanations(
        self,
        visual_score: float,
        frequency_score: float,
        manipulation_score: float,
        illum_score: float,
        angle_deg: float,
        regions: list[dict]
    ) -> list[str]:
        """
        Synthesizes precise, human-readable forensic explanations.
        """
        explanations = []

        if manipulation_score > 0.65:
            explanations.append(
                f"High-confidence image manipulation detected across spatial and physics signals (Score: {round(manipulation_score * 100, 1)}%)"
            )
        elif manipulation_score > 0.40:
            explanations.append(
                f"Moderate forensic anomalies detected with partial signal agreement (Score: {round(manipulation_score * 100, 1)}%)"
            )
        else:
            explanations.append(
                f"Image demonstrates authentic physical illumination and homogeneous spatial noise (Authenticity confidence: {round((1.0 - manipulation_score) * 100, 1)}%)"
            )

        # Illumination explanation
        if angle_deg > 35.0:
            explanations.append(
                f"Physical illumination direction discrepancy of {round(angle_deg, 1)}° detected between foreground subject and background lighting vectors."
            )
        
        # Frequency / ELA explanation
        if frequency_score > 0.60:
            explanations.append(
                f"Discrete Cosine Transform (DCT) & Error Level Analysis anomaly indicates localized recompression and high-frequency edge tampering ({round(frequency_score * 100, 1)}%)."
            )

        # Region explanations
        if regions:
            labels = set([r["label"].replace("_", " ") for r in regions])
            labels_str = ", ".join(labels)
            explanations.append(f"Localized anomalous regions identified: {labels_str}.")

        return explanations
