"""
Physical Illumination & Lighting Consistency Forensics.
Inspired by the MIT Multi-Illumination physical light direction modeling (phi, theta).
Estimates 3D dominant lighting vectors across segmented regions (subject vs background)
and evaluates angular divergence and shadow/shading physics consistency.
"""

import numpy as np
import cv2


class IlluminationForensicsAnalyzer:
    """
    Estimates 3D scene lighting vectors and detects physical illumination inconsistencies.
    """

    def __init__(self, angle_threshold_deg: float = 45.0):
        self.angle_threshold_deg = angle_threshold_deg

    def estimate_surface_normals(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Approximates surface normal vectors using Shape-from-Shading / intensity gradient fields.
        
        Args:
            gray_image: Grayscale float32 image normalized to [0, 1].
            
        Returns:
            normals: (H, W, 3) unit normal vector field.
        """
        # Smooth image to suppress noise
        blurred = cv2.GaussianBlur(gray_image, (9, 9), 2.5)

        # Spatial derivatives
        gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)

        # Shape from shading: normal vector (-gx, -gy, 1) normalized
        nx = -gx
        ny = -gy
        nz = np.ones_like(gx, dtype=np.float32)

        norm = np.sqrt(nx**2 + ny**2 + nz**2 + 1e-8)
        nx /= norm
        ny /= norm
        nz /= norm

        normals = np.stack([nx, ny, nz], axis=2)
        return normals

    def segment_saliency_and_background(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Extracts salient foreground subject vs background context using
        spectral residual saliency + Otsu clustering.
        """
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Spectral Saliency
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create() if hasattr(cv2, 'saliency') else None
        if saliency is not None:
            success, sal_map = saliency.computeSaliency(image_bgr)
            sal_uint = (sal_map * 255).astype(np.uint8)
            _, fg_mask = cv2.threshold(sal_uint, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # Fallback: Otsu + adaptive gradient mask
            grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            _, fg_mask = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Ensure minimum foreground presence
        if np.sum(fg_mask > 0) < 0.05 * (h * w):
            # Center region fallback
            cy, cx = h // 2, w // 2
            ry, rx = int(h * 0.3), int(w * 0.3)
            fg_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(fg_mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

        # Morphological refinement
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        bg_mask = cv2.bitwise_not(fg_mask)

        return fg_mask, bg_mask

    def estimate_dominant_light_vector(self, normals: np.ndarray, intensities: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Solves for the optimal 3D light vector L = (Lx, Ly, Lz) in the masked region
        using linear least squares: Intensity ≈ N · L + c.
        
        Returns:
            (unit_light_vector, fit_confidence)
        """
        valid = (mask > 0) & (intensities > 0.08) & (intensities < 0.92)
        if np.sum(valid) < 200:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32), 0.0

        n_pts = normals[valid]  # (K, 3)
        i_pts = intensities[valid]  # (K,)

        # Design matrix: [Nx, Ny, Nz, 1]
        A = np.hstack([n_pts, np.ones((len(n_pts), 1), dtype=np.float32)])
        
        try:
            params, _, _, _ = np.linalg.lstsq(A, i_pts, rcond=None)
            L = params[:3]
            mag = np.linalg.norm(L)
            if mag > 1e-5:
                L_unit = L / mag
            else:
                L_unit = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                
            # Confidence based on correlation between predicted and observed intensity
            pred = np.clip(np.dot(n_pts, params[:3]) + params[3], 0.0, 1.0)
            corr = np.corrcoef(pred, i_pts)[0, 1]
            conf = float(np.clip(corr if not np.isnan(corr) else 0.0, 0.0, 1.0))
            return L_unit, conf
        except Exception:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32), 0.0

    def compute_illumination_inconsistency(self, image_bgr: np.ndarray) -> tuple[float, float, np.ndarray, dict]:
        """
        Segments foreground and background, estimates lighting vectors,
        and computes confidence-weighted angular divergence.
        
        Returns:
            (illumination_anomaly_score, angular_discrepancy_deg, illum_map, details)
        """
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        normals = self.estimate_surface_normals(gray)

        fg_mask, bg_mask = self.segment_saliency_and_background(image_bgr)

        # Estimate lighting vectors
        L_fg, conf_fg = self.estimate_dominant_light_vector(normals, gray, fg_mask)
        L_bg, conf_bg = self.estimate_dominant_light_vector(normals, gray, bg_mask)

        # Raw angular divergence
        cos_sim = np.clip(np.dot(L_fg, L_bg), -1.0, 1.0)
        raw_angle_deg = float(np.degrees(np.arccos(cos_sim)))

        # Effective angular discrepancy weighted by confidence of both estimates
        joint_conf = float(min(conf_fg, conf_bg))
        if joint_conf < 0.25:
            # Low confidence in shading model -> suppress false angular alarm
            effective_angle_deg = raw_angle_deg * (joint_conf / 0.25)
        else:
            effective_angle_deg = raw_angle_deg

        # Non-linear mapping:
        # Authentic images have low effective divergence (< 20 deg -> score < 0.15)
        # Spliced/lighting mismatched images have divergence (> 32 deg -> score > 0.50, > 40 deg -> score > 0.75)
        illum_score = float(1.0 / (1.0 + np.exp(-0.16 * (effective_angle_deg - 32.0))))

        # Generate pixel-wise illumination residual map
        pred_shading = np.clip(np.dot(normals, L_bg), 0.0, 1.0)
        shading_error = np.abs(gray - pred_shading)
        shading_error = cv2.GaussianBlur(shading_error, (15, 15), 3.0)
        
        p98 = np.percentile(shading_error, 98.0)
        if p98 > 1e-4:
            illum_map = np.clip(shading_error / p98, 0.0, 1.0)
        else:
            illum_map = np.zeros_like(shading_error)

        details = {
            "subject_light_vector": [round(float(v), 3) for v in L_fg],
            "background_light_vector": [round(float(v), 3) for v in L_bg],
            "raw_angular_discrepancy_deg": round(raw_angle_deg, 2),
            "effective_angular_discrepancy_deg": round(effective_angle_deg, 2),
            "confidence_joint": round(joint_conf, 3),
            "illumination_score": round(illum_score, 4)
        }

        return illum_score, effective_angle_deg, illum_map, details
