import math
import logging
from typing import Dict, Any, Optional, Callable, List, Tuple
from backend.config import (
    THRESHOLD_MANIPULATED,
    THRESHOLD_AUTHENTIC,
    THRESHOLD_INCONCLUSIVE_SPREAD,
    FUSION_WEIGHTS
)
from backend.schemas.contracts import VerdictEnum

logger = logging.getLogger("TruthLens.Fusion")


def _np_clip(val: Any, low: float, high: float) -> float:
    """Clamps scalar float safely into [low, high]."""
    try:
        f = float(val)
        return max(low, min(high, f))
    except Exception:
        return 0.50


class ForensicFusionAdapter:
    """
    State-of-the-Art Multimodal Forensic Fusion Adapter for TruthLens Platform.
    Implements:
    1. Dempster-Shafer Evidential Mass Pooling over frame of discernment {Manipulated, Authentic, Uncertainty}
    2. Asymmetric Risk-Aware Soft-OR Anomaly Gating (prevents single-modality spoofing from being averaged out)
    3. Orthogonal Inter-Modality Conflict Metric (K) & Epistemic Uncertainty Estimation
    4. Automated Forensic Attack Vector Classification (Voice Clone, Face Swap, Lip Desync, Frame Splice, Full Synthesis)
    5. Courtroom-Grade Structured Explainability Breakdown
    """

    def __init__(self, custom_fusion_fn: Optional[Callable] = None):
        self.custom_fusion_fn = custom_fusion_fn

    def set_real_fusion(self, fusion_fn: Callable):
        """Allows Person 2B to plug in their calibrated fusion algorithm."""
        self.custom_fusion_fn = fusion_fn

    @staticmethod
    def _compute_entropy(p: float) -> float:
        """Computes binary Shannon entropy normalized in [0, 1]."""
        p = max(1e-6, min(1.0 - 1e-6, p))
        return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))

    @staticmethod
    def _dempster_shafer_combine(
        bba_list: List[Tuple[str, float, float, float]]
    ) -> Tuple[float, float, float, float]:
        """
        Sequentially combines Basic Belief Assignments (BBAs) using Dempster's Rule of Combination.
        Each BBA has: (modality_name, m({Manipulated}), m({Authentic}), m(Theta_Uncertainty))
        Returns: (m_final_manip, m_final_auth, m_final_uncertainty, total_conflict_K)
        """
        if not bba_list:
            return 0.0, 0.0, 1.0, 0.0

        # Initialize with first modality BBA
        _, m_m, m_a, m_u = bba_list[0]
        cumulative_conflict = 0.0

        for _, next_m, next_a, next_u in bba_list[1:]:
            # Orthogonal conflict metric between current combined state and next evidence
            k = (m_m * next_a) + (m_a * next_m)
            cumulative_conflict = max(cumulative_conflict, k)

            denom = max(1e-6, 1.0 - k)
            new_m = ((m_m * next_m) + (m_m * next_u) + (m_u * next_m)) / denom
            new_a = ((m_a * next_a) + (m_a * next_u) + (m_u * next_a)) / denom
            new_u = (m_u * next_u) / denom

            # Renormalize
            total = new_m + new_a + new_u
            m_m = new_m / total
            m_a = new_a / total
            m_u = new_u / total

        return m_m, m_a, m_u, cumulative_conflict

    def classify_attack_vector(
        self,
        active_scores: Dict[str, float],
        visual_res: Dict[str, Any],
        temporal_res: Dict[str, Any],
        audio_res: Dict[str, Any],
        fusion_score: float,
        verdict: VerdictEnum
    ) -> str:
        """
        Classifies the precise forensic attack vector based on multi-stream profiles.
        """
        if verdict == VerdictEnum.AUTHENTIC:
            return "VERIFIED_AUTHENTIC_MEDIA"

        if verdict == VerdictEnum.INCONCLUSIVE:
            return "AMBIGUOUS_FORENSIC_SIGNAL_OR_COMPRESSION"

        v = active_scores.get("visual", 0.0)
        f = active_scores.get("frequency", 0.0)
        t = active_scores.get("temporal", 0.0)
        a = active_scores.get("audio", 0.0)

        high_signals = [k for k, s in active_scores.items() if s >= 0.65]

        if len(high_signals) >= 3:
            return "MULTIMODAL_FULL_SYNTHESIS"

        # Check for isolated single-modality attacks
        if a >= 0.70 and v <= 0.45 and t <= 0.45:
            av_offset = audio_res.get("av_sync_offset_ms")
            if av_offset and abs(av_offset) > 80.0:
                return "AUDIO_VISUAL_LIP_DESYNC_OR_DUBBING"
            return "AUDIO_DEEPFAKE_VOICE_CLONE"

        if t >= 0.70 and v <= 0.45 and a <= 0.45:
            return "TEMPORAL_INTERFRAME_SPLICE"

        if f >= 0.70 and v <= 0.45:
            return "FREQUENCY_DIFFUSION_RECOMPRESSION"

        if v >= 0.65:
            vis_details = visual_res.get("details", {})
            illum_angle = vis_details.get("illumination_angle_discrepancy_deg", 0.0)
            if illum_angle > 35.0:
                return "PHYSICS_ILLUMINATION_INCONSISTENCY"
            return "VISUAL_FACE_SWAP_OR_SPLICING"

        return "COMPOSITE_DIGITAL_TAMPERING"

    def fuse(
        self,
        evidence_id: str,
        visual_res: Dict[str, Any],
        temporal_res: Dict[str, Any],
        audio_res: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fuses multimodal evidence signals into a calibrated verdict and confidence score.
        Combines Dempster-Shafer evidential mass pooling with asymmetric Soft-OR anomaly gating.
        """
        if self.custom_fusion_fn is not None:
            try:
                return self.custom_fusion_fn(evidence_id, visual_res, temporal_res, audio_res)
            except Exception as e:
                logger.warning(f"Custom fusion function error ({e}); falling back to HERM-Fusion.")

        # 1. Extract valid active forensic scores
        v_score = visual_res.get("visual_score") if visual_res.get("status") != "ERROR" else None
        f_score = visual_res.get("frequency_score") if visual_res.get("status") != "ERROR" else None
        t_score = temporal_res.get("temporal_score") if temporal_res.get("status") != "ERROR" else None
        a_score = audio_res.get("audio_score") if audio_res.get("status") != "ERROR" else None

        active_scores: Dict[str, float] = {}
        if v_score is not None:
            active_scores["visual"] = float(_np_clip(v_score, 0.0, 1.0))
        if f_score is not None:
            active_scores["frequency"] = float(_np_clip(f_score, 0.0, 1.0))
        if t_score is not None:
            active_scores["temporal"] = float(_np_clip(t_score, 0.0, 1.0))
        if a_score is not None:
            active_scores["audio"] = float(_np_clip(a_score, 0.0, 1.0))

        if not active_scores:
            return {
                "module": "fusion",
                "evidence_id": evidence_id,
                "fusion_score": 0.50,
                "confidence": 0.50,
                "verdict": VerdictEnum.INCONCLUSIVE,
                "attack_vector": "INSUFFICIENT_SIGNALS",
                "weights_used": {},
                "conflict_index": 0.0,
                "uncertainty_index": 1.0,
                "modality_scores": {},
                "verdict_reasoning": "Insufficient forensic signals available to compute verdict.",
                "status": "INSUFFICIENT_SIGNALS"
            }

        # 2. Modality Reliability Factors (conditioned on model depth, confidence, and speech clarity)
        audio_conf = float(audio_res.get("confidence", 0.85) if audio_res.get("confidence") is not None else 0.85)
        base_reliabilities = {
            "visual": 0.90,
            "frequency": 0.80,
            "temporal": 0.88,
            "audio": float(_np_clip(0.85 * audio_conf + 0.10, 0.20, 0.88))
        }
        if not audio_res.get("has_audio", False) and "audio" in active_scores:
            base_reliabilities["audio"] = 0.20

        # 3. Construct Dempster-Shafer Basic Belief Assignments (BBAs)
        bba_list = []
        for mod_name, score in active_scores.items():
            alpha = base_reliabilities.get(mod_name, 0.80)
            if score >= 0.50:
                m_m = alpha * (2.0 * (score - 0.50))
                m_a = 0.0
            else:
                m_m = 0.0
                m_a = alpha * (2.0 * (0.50 - score))
            m_u = max(0.0, 1.0 - (m_m + m_a))
            bba_list.append((mod_name, m_m, m_a, m_u))

        m_manip, m_auth, m_unc, conflict_k = self._dempster_shafer_combine(bba_list)

        # 4. Asymmetric Risk-Aware Soft-OR Anomaly Gating
        max_single_score = max(active_scores.values())
        min_single_score = min(active_scores.values())
        score_spread = max_single_score - min_single_score if len(active_scores) > 1 else 0.0

        # Normalized weights strictly derived from empirical validation ROC-AUCs of trained models
        total_raw_w = sum(FUSION_WEIGHTS.get(k, 0.25) for k in active_scores)
        norm_weights = {k: FUSION_WEIGHTS.get(k, 0.25) / total_raw_w for k in active_scores}

        soft_or_prod = 1.0
        for mod_name, score in active_scores.items():
            w_i = norm_weights[mod_name]
            soft_or_prod *= (1.0 - w_i * (score ** 1.8))
        soft_or_score = 1.0 - soft_or_prod

        # Evidential posterior score
        if (m_manip + m_auth) > 1e-5:
            evidential_score = m_manip / (m_manip + m_auth)
        else:
            evidential_score = 0.50

        # Standard weighted average strictly using trained model weights
        weighted_avg = sum(s * norm_weights[k] for k, s in active_scores.items())

        # Synthesize composite fusion score:
        top_mod = [k for k, s in active_scores.items() if s == max_single_score][0]
        
        # When physical video continuity is pristine (temporal <= 0.35 and visual <= 0.40),
        # an unconstrained audio or frequency anomaly is treated as ambient/codec noise
        physical_continuity_pristine = (active_scores.get("temporal", 1.0) <= 0.35 and active_scores.get("visual", 1.0) <= 0.40)
        is_ambient_audio_anomaly = (
            top_mod == "audio" and 
            max_single_score < 0.85 and 
            audio_conf < 0.65 and 
            physical_continuity_pristine
        )
        is_isolated_frequency_artifact = (
            top_mod == "frequency" and
            physical_continuity_pristine
        )

        if max_single_score >= 0.60 and not is_ambient_audio_anomaly and not is_isolated_frequency_artifact:
            fusion_score = max(evidential_score, soft_or_score, max_single_score * 0.85)
        elif is_ambient_audio_anomaly or is_isolated_frequency_artifact:
            # Physical camera recordings with video codec or ambient acoustic artifacts
            fusion_score = 0.70 * evidential_score + 0.30 * weighted_avg
        elif min_single_score <= 0.25 and max_single_score <= 0.48:
            fusion_score = min(evidential_score, weighted_avg)
        else:
            fusion_score = 0.55 * evidential_score + 0.45 * weighted_avg

        fusion_score = round(max(0.0, min(1.0, float(fusion_score))), 3)

        # 5. Multi-Signal Consensus & Decision Rules
        manip_signals = [k for k, score in active_scores.items() if score >= 0.55]
        auth_signals = [k for k, score in active_scores.items() if score <= 0.38]

        if is_ambient_audio_anomaly or is_isolated_frequency_artifact:
            verdict = VerdictEnum.AUTHENTIC
            entropy = self._compute_entropy(fusion_score)
            confidence = round(0.50 + 0.45 * (1.0 - entropy) + 0.05, 3)
            confidence = max(0.50, min(0.99, confidence))
            reasoning = (
                "Visual spatial features and temporal inter-frame dynamics verify authentic physical video recording. "
                "Auxiliary signals exhibit benign compression or ambient environmental variance."
            )
        elif len(manip_signals) >= 2 or fusion_score >= THRESHOLD_MANIPULATED or (active_scores.get("temporal", 0.0) >= 0.60) or (active_scores.get("visual", 0.0) >= 0.60):
            verdict = VerdictEnum.MANIPULATED
            eff_score = max(fusion_score, max_single_score)
            entropy = self._compute_entropy(eff_score)
            confidence = round(0.50 + 0.48 * (1.0 - entropy) * (1.0 - 0.15 * conflict_k) + 0.05, 3)
            confidence = max(0.50, min(0.99, confidence))
            if len(manip_signals) >= 2:
                reasoning = (
                    f"High-confidence manipulation detected by consensus across {len(manip_signals)} "
                    f"forensic signals ({', '.join(manip_signals).upper()}). Evidence score: {fusion_score*100:.1f}%."
                )
            else:
                top_mod = [k for k, s in active_scores.items() if s == max_single_score][0]
                reasoning = (
                    f"Targeted single-modality manipulation detected in {top_mod.upper()} stream "
                    f"(Score: {max_single_score*100:.1f}%). Anomaly threshold exceeded."
                )
        elif fusion_score <= THRESHOLD_AUTHENTIC or (active_scores.get("temporal", 1.0) <= 0.40 and active_scores.get("visual", 1.0) <= 0.40):
            verdict = VerdictEnum.AUTHENTIC
            entropy = self._compute_entropy(fusion_score)
            confidence = round(0.50 + 0.48 * (1.0 - entropy) * (1.0 - 0.15 * conflict_k) + 0.05, 3)
            confidence = max(0.50, min(0.99, confidence))
            reasoning = f"Media evaluated as authentic with low forensic anomaly levels (Score: {fusion_score*100:.1f}%)."
        elif score_spread > THRESHOLD_INCONCLUSIVE_SPREAD and 0.40 < fusion_score < 0.60:
            verdict = VerdictEnum.INCONCLUSIVE
            confidence = round(0.50 + abs(fusion_score - 0.50) * 0.45, 3)
            confidence = max(0.50, min(0.68, confidence))
            reasoning = (
                f"Significant inter-signal disagreement detected across modalities (spread: {score_spread:.2f}, "
                f"conflict K: {conflict_k:.2f}). System marked INCONCLUSIVE for human forensic review."
            )
        elif fusion_score < 0.50:
            verdict = VerdictEnum.AUTHENTIC
            entropy = self._compute_entropy(fusion_score)
            confidence = round(0.50 + 0.48 * (1.0 - entropy) + 0.05, 3)
            confidence = max(0.50, min(0.99, confidence))
            reasoning = f"Media evaluated as authentic (Composite anomaly score: {fusion_score*100:.1f}%)."
        else:
            verdict = VerdictEnum.MANIPULATED
            confidence = round(0.50 + abs(fusion_score - 0.50) * 0.45, 3)
            confidence = max(0.50, min(0.85, confidence))
            reasoning = f"Forensic score ({fusion_score*100:.1f}%) exceeds authentic threshold."

        # 6. Classify Attack Vector
        attack_vector = self.classify_attack_vector(
            active_scores=active_scores,
            visual_res=visual_res,
            temporal_res=temporal_res,
            audio_res=audio_res,
            fusion_score=fusion_score,
            verdict=verdict
        )

        return {
            "module": "fusion",
            "evidence_id": evidence_id,
            "fusion_score": float(fusion_score),
            "confidence": float(confidence),
            "verdict": verdict,
            "attack_vector": attack_vector,
            "weights_used": {k: round(v, 3) for k, v in norm_weights.items()},
            "conflict_index": round(float(conflict_k), 3),
            "uncertainty_index": round(float(m_unc), 3),
            "modality_scores": {k: round(v, 3) for k, v in active_scores.items()},
            "verdict_reasoning": reasoning,
            "status": "SUCCESS"
        }

