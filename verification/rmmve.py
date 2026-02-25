import time
from .modules import (
    LexicalOntologyVerification, ProtocolOntologyVerification,
    MotionAwareVerification, WebSourceVerification,
    EmbeddingSimilarityVerification
)
from models.constants import (
    EPSILON_REVIEW, VETO_THRESHOLDS, CRITICAL_MODULES,
)


def _classify_fact_type(candidate_fact):
    """
    Classify fact as 'ST' (spatiotemporal) or 'SEM' (semantic-only).

    ST facts have resolvable spatiotemporal coordinates.
    SEM facts lack complete coordinates (T(d) = ∅).
    """
    from datetime import datetime

    def _valid_timestamp(ts):
        if not ts or not isinstance(ts, str):
            return False
        try:
            datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return True
        except (ValueError, TypeError):
            return False

    def _valid_coords(c):
        if not isinstance(c, dict):
            return False
        return all(
            k in c and isinstance(c[k], (int, float))
            for k in ("x_coord", "y_coord", "z_coord")
        )

    st = candidate_fact.get("spatiotemporal", {})
    if st and _valid_timestamp(st.get("timestamp")) and _valid_coords(st):
        return "ST"

    inner = candidate_fact.get("spatiotemporal_inspection_data", {})
    spatial = inner.get("spatial_data", [])
    temporal = inner.get("temporal_data", [])
    if spatial and temporal:
        coords = spatial[0].get("coordinates", {})
        ts = temporal[0].get("timestamp")
        if _valid_coords(coords) and _valid_timestamp(ts):
            return "ST"

    return "SEM"


class RMMVeProcess:
    """
    Ranked Multi-Modal Verification (RMMVe) — Algorithm 1.

    Implements sequential module execution with:
    - ST / SEM fact type classification
    - Critical module veto check (τ_veto)
    - Module activation criterion: S_i ≥ θ_i
    - Early termination suspended until M3 for ST facts
    - Three-way decision: Accept / Review / Reject (Eq. 23)
    """

    def __init__(self, global_threshold=0.65, domain="aerospace"):
        self.modules = [
            LexicalOntologyVerification(0.25, 0.7, 0.70),    # M1 – LOV
            ProtocolOntologyVerification(0.25, 0.65, 0.70),   # M2 – POV
            MotionAwareVerification(0.20, 0.75, 0.65),        # M3 – MAV
            WebSourceVerification(0.15, 0.65, 0.60),          # M4 – WSV
            EmbeddingSimilarityVerification(0.15, 0.7, 0.65), # M5 – ESV
        ]
        self.global_threshold = global_threshold
        self.domain = domain
        self.veto_threshold = VETO_THRESHOLDS.get(domain, VETO_THRESHOLDS["default"])

    def verify(self, candidate_fact, knowledge_graph, fact_quality=None,
               llm_confidence=1.0):
        """
        Verify a candidate fact using Algorithm 1.

        Returns dict with decision ('Accept', 'Review', or 'Reject'),
        module results, and cumulative confidence.
        """
        fact_type = _classify_fact_type(candidate_fact)
        is_st = (fact_type == "ST")

        verification_results = {
            "activated_modules": [],
            "module_results": [],
            "confidence_scores": {},
            "metrics": {},
            "total_confidence": 0.0,
            "decision": False,
            "decision_label": "Reject",
            "early_termination": False,
            "verification_time": 0.0,
            "fact_quality": fact_quality,
            "fact_type": fact_type,
            "contains_shift": candidate_fact.get("contains_performance_shift", None),
        }

        start_time = time.time()
        activated_modules = []
        total_weighted_conf = 0.0
        total_weights = 0.0
        vetoed = False
        mav_index = 2  # M3 is at index 2

        for idx, module in enumerate(self.modules):
            # For SEM facts, MAV assigns neutral S_3 = 1.0 (prevents false
            # rejection but does not positively confirm the fact).
            mav_neutral = False
            if idx == mav_index and not is_st:
                confidence, m1, m2 = 1.0, 1.0, 1.0
                mav_neutral = True
            else:
                confidence, m1, m2 = module.compute_confidence(
                    candidate_fact, knowledge_graph, fact_quality, llm_confidence
                )

            module_result = {
                "module_name": module.name,
                "confidence": confidence,
                "threshold": module.threshold,
                "activated": False,
                "metric1": m1,
                "metric2": m2,
                "weight": module.weight,
                "alpha": module.alpha,
            }
            verification_results["module_results"].append(module_result)
            verification_results["confidence_scores"][module.name] = confidence
            verification_results["metrics"][module.name] = {"metric1": m1, "metric2": m2}

            # --- Critical Module Veto Check ---
            if module.name in CRITICAL_MODULES and is_st:
                if confidence < self.veto_threshold:
                    vetoed = True
                    verification_results["early_termination"] = True
                    verification_results["early_termination_module"] = module.name
                    break

            # --- Module Activation Criterion: S_i ≥ θ_i ---
            # Neutral MAV on SEM facts does not activate (no positive signal).
            if mav_neutral:
                continue
            if confidence >= module.threshold:
                activated_modules.append(module.name)
                total_weighted_conf += confidence * module.weight
                total_weights += module.weight
                module_result["activated"] = True

                current_cumulative = (
                    total_weighted_conf / total_weights if total_weights > 0 else 0.0
                )

                # --- Early Termination Check ---
                if current_cumulative >= self.global_threshold:
                    if is_st and idx < mav_index:
                        continue  # Force M3 execution for ST facts
                    verification_results["early_termination"] = True
                    verification_results["early_termination_module"] = module.name
                    verification_results["early_termination_confidence"] = confidence
                    verification_results["early_termination_cumulative"] = current_cumulative
                    break

        verification_results["activated_modules"] = activated_modules
        cumulative = (
            total_weighted_conf / total_weights if total_weights > 0 else 0.0
        )
        verification_results["total_confidence"] = cumulative

        # --- Final Decision Rule (Eq. 23) ---
        if vetoed:
            decision_label = "Reject"
            decision_bool = False
        elif cumulative >= self.global_threshold:
            decision_label = "Accept"
            decision_bool = True
        elif cumulative >= self.global_threshold - EPSILON_REVIEW:
            decision_label = "Review"
            decision_bool = False
        else:
            decision_label = "Reject"
            decision_bool = False

        verification_results["decision"] = decision_bool
        verification_results["decision_label"] = decision_label

        end_time = time.time()
        verification_results["verification_time"] = end_time - start_time

        return verification_results
