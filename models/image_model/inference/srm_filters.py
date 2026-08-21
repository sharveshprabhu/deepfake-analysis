"""
Spatial Rich Model (SRM) and Noise Inconsistency Extractor.
Extracts high-pass spatial noise residuals and PRNU (Photo-Response Non-Uniformity) variance
with Poisson-Gaussian noise stabilization to detect spliced boundaries and noise inconsistencies.
"""

import numpy as np
import cv2


class SRMNoiseExtractor:
    """
    Applies Steganalysis / Forensics Spatial Rich Model (SRM) linear filter banks
    to capture micro-texture inconsistencies and boundary artifacts.
    """

    def __init__(self):
        self.filters = self._build_srm_filters()

    def _build_srm_filters(self):
        """Builds standard 3x3 and 5x5 SRM high-pass residual filter kernels."""
        filters = []

        # 1st order horizontal & vertical edge
        f1 = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32)
        f2 = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=np.float32)

        # 2nd order horizontal & vertical
        f3 = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32) / 2.0
        f4 = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=np.float32) / 2.0

        # Diagonal 2nd order
        f5 = np.array([[0, 0, 1], [0, -2, 0], [1, 0, 0]], dtype=np.float32) / 2.0
        f6 = np.array([[1, 0, 0], [0, -2, 0], [0, 0, 1]], dtype=np.float32) / 2.0

        # 3x3 Laplacian / High-pass
        f7 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32) / 4.0
        f8 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32) / 8.0

        # 5x5 EDGE residual filter
        f9 = np.array([
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1]
        ], dtype=np.float32) / 12.0

        # 5x5 Square residual filter
        f10 = np.array([
            [-1, -1, -1, -1, -1],
            [-1, 2, 2, 2, -1],
            [-1, 2, 8, 2, -1],
            [-1, 2, 2, 2, -1],
            [-1, -1, -1, -1, -1]
        ], dtype=np.float32) / 8.0

        filters.extend([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10])
        return filters

    def extract_residuals(self, gray_image: np.ndarray, clip_val: float = 3.0) -> np.ndarray:
        """
        Convolves grayscale image with SRM filter bank and computes combined residual energy.
        Applies standard high-pass clipping (T=3.0) to suppress dominant scene edges.
        
        Args:
            gray_image: Grayscale float32 image in range [0, 255].
            clip_val: High-pass truncation threshold.
            
        Returns:
            residual_map: 2D float32 numpy array representing normalized noise residual energy [0, 1].
        """
        if gray_image.ndim == 3:
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
        
        gray = gray_image.astype(np.float32)
        h, w = gray.shape
        combined_energy = np.zeros((h, w), dtype=np.float32)

        for filt in self.filters:
            res = cv2.filter2D(gray, -1, filt)
            res = np.abs(res)
            res = np.clip(res, 0.0, clip_val)
            combined_energy += res

        combined_energy /= len(self.filters)

        p99 = np.percentile(combined_energy, 99.0)
        if p99 > 1e-5:
            normalized = np.clip(combined_energy / p99, 0.0, 1.0)
        else:
            normalized = np.zeros_like(combined_energy)

        return normalized

    def compute_noise_variance_inconsistency(self, gray_image: np.ndarray, patch_size: int = 32) -> tuple[float, np.ndarray]:
        """
        Computes Poisson-Gaussian stabilized and texture-normalized local noise variance across patches.
        Accurately differentiates uniform camera sensor noise from spliced / deepfake boundary inconsistency.
        
        Returns:
            (inconsistency_score, variance_map)
        """
        if gray_image.ndim == 3:
            gray = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = gray_image.astype(np.float32)

        residuals = self.extract_residuals(gray, clip_val=3.0)
        h, w = residuals.shape
        
        nh = h // patch_size
        nw = w // patch_size
        if nh < 2 or nw < 2:
            return 0.15, residuals

        # Compute local gradient for texture normalization
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)

        stabilized_variances = []
        var_map = np.zeros((nh, nw), dtype=np.float32)

        for i in range(nh):
            for j in range(nw):
                patch_res = residuals[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                patch_lum = gray[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                patch_grad = grad_mag[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                
                raw_var = float(np.var(patch_res))
                mean_lum = float(np.mean(patch_lum))
                mean_grad = float(np.mean(patch_grad))
                
                # Shot noise factor with texture scaling
                stab_factor = np.sqrt((mean_lum / 255.0) + 0.15) * (1.0 + 0.05 * mean_grad)
                stab_var = raw_var / stab_factor
                
                var_map[i, j] = stab_var
                stabilized_variances.append(stab_var)

        stabilized_variances = np.array(stabilized_variances)
        med_v = float(np.median(stabilized_variances))
        mad_v = float(np.median(np.abs(stabilized_variances - med_v))) + 1e-6
        std_v = float(np.std(stabilized_variances))
        mean_v = float(np.mean(stabilized_variances)) + 1e-6

        rel_dispersion = float(mad_v / (med_v + 1e-5))
        cv_val = float(std_v / mean_v)
        
        p90 = float(np.percentile(stabilized_variances, 90.0))
        p10 = float(np.percentile(stabilized_variances, 10.0)) + 1e-6
        dyn_ratio = float(p90 / p10)

        # Calibrated Sigmoids for authentic single-camera vs multi-source/spliced noise
        s_mad = 1.0 / (1.0 + np.exp(-10.0 * (rel_dispersion - 0.40)))
        s_cv = 1.0 / (1.0 + np.exp(-8.0 * (cv_val - 0.50)))
        s_dyn = 1.0 / (1.0 + np.exp(-0.7 * (dyn_ratio - 3.5)))

        inconsistency_score = float(0.40 * s_mad + 0.35 * s_cv + 0.25 * s_dyn)
        inconsistency_score = float(np.clip(inconsistency_score, 0.02, 0.98))
        
        var_map_full = cv2.resize(var_map, (w, h), interpolation=cv2.INTER_CUBIC)
        var_map_full = np.clip(var_map_full / (np.percentile(var_map_full, 98.0) + 1e-6), 0.0, 1.0)

        return inconsistency_score, var_map_full
