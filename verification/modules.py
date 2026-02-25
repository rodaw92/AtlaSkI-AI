import random
import math
import re
import numpy as np
from .base import VerificationModule
from models.constants import (
    STANDARD_TERMINOLOGIES, MIN_PROCESS_DURATIONS,
    ENTITY_CLASSES, RELATIONSHIP_TYPES,
)


class LexicalOntologyVerification(VerificationModule):
    """
    Lexical-Ontological Verification (LOV) – Module M1.

    Targets: Semantic Drift (Definition 6)

    Metric 1 – Structural Compliance (Eq. 8):
        (1/3)[I(type(s) ∈ C) + I(type(o) ∈ C) + I(r ∈ R_o)]

    Metric 2 – Attribute Compliance (Eq. 9):
        Validates hard (type, format) and soft (typical range) constraints
        on the attribute set A_k of the candidate fact.
    """

    def __init__(self, weight, alpha, threshold):
        super().__init__("LOV", weight, alpha, threshold)

    def compute_metrics(self, candidate_fact, knowledge_graph, fact_quality):
        ontology = knowledge_graph.ontology

        # --- Metric 1: Structural Compliance ---
        subject_class = self._extract_subject_class(candidate_fact)
        object_class = self._extract_object_class(candidate_fact)
        relation = self._extract_relation(candidate_fact)

        subject_valid = 1.0 if subject_class and subject_class in ontology.entity_classes else 0.0
        object_valid = 1.0 if object_class and object_class in ontology.entity_classes else 0.0
        relation_valid = 1.0 if relation and relation in ontology.relationship_types else 0.0

        metric1 = (subject_valid + object_valid + relation_valid) / 3.0

        # --- Metric 2: Attribute Compliance ---
        entity_class_def = ontology.get_entity_class(subject_class) if subject_class else None
        if entity_class_def and entity_class_def.required_attributes:
            attrs = self._collect_attributes(candidate_fact)
            n_attrs = len(entity_class_def.required_attributes)
            hard_sum = 0.0
            soft_sum = 0.0
            for attr in entity_class_def.required_attributes:
                attr_lower = attr.lower().replace("_", "")
                present = any(
                    attr_lower in k.lower().replace("_", "")
                    for k in attrs
                )
                hard_score = 1.0 if present else 0.0
                soft_score = self._check_soft_constraint(attr, attrs) if present else 0.0
                hard_sum += hard_score
                soft_sum += soft_score
            metric2 = (hard_sum + 0.5 * soft_sum) / (1.5 * n_attrs) if n_attrs > 0 else 1.0
        else:
            metric2 = 1.0

        return metric1, metric2

    # ------------------------------------------------------------------
    def _extract_subject_class(self, fact):
        if "entity_class" in fact:
            return fact["entity_class"]
        sid = fact.get("subject_entity_id", "")
        if "Blade" in sid or "Turbine" in sid:
            return "Blade"
        if "Engine" in sid:
            return "EngineSet"
        if "Measurement" in sid or "Inspection" in sid:
            return "InspectionMeasurement"
        if "Patient" in sid:
            return "Patient"
        inner = fact.get("spatiotemporal_inspection_data", {})
        measurements = inner.get("inspection_data", {}).get("inspection_measurements", [])
        if measurements:
            cid = measurements[0].get("component_id", "")
            if "Blade" in cid or "Turbine" in cid:
                return "Blade"
        return None

    def _extract_object_class(self, fact):
        oid = fact.get("object_entity_id", "")
        if "Measurement" in oid or "Inspection" in oid:
            return "InspectionMeasurement"
        if "Blade" in oid or "Turbine" in oid:
            return "Blade"
        if "Engine" in oid:
            return "EngineSet"
        inner = fact.get("spatiotemporal_inspection_data", {})
        rels = inner.get("relationships", [])
        if rels:
            oid2 = rels[0].get("object_entity_id", "")
            if "Blade" in oid2 or "Turbine" in oid2:
                return "Blade"
            if "Measurement" in oid2:
                return "InspectionMeasurement"
        return None

    def _extract_relation(self, fact):
        if "relationship_type" in fact:
            return fact["relationship_type"]
        inner = fact.get("spatiotemporal_inspection_data", {})
        rels = inner.get("relationships", [])
        if rels:
            return rels[0].get("relationship_type")
        return None

    def _collect_attributes(self, fact):
        attrs = dict(fact)
        attrs.update(fact.get("attributes", {}))
        inner = fact.get("spatiotemporal_inspection_data", {})
        measurements = inner.get("inspection_data", {}).get("inspection_measurements", [])
        if measurements:
            attrs.update(measurements[0])
        return attrs

    def _check_soft_constraint(self, attr_name, attrs):
        al = attr_name.lower()
        if "value" in al or "deviation" in al or "tolerance" in al:
            for k, v in attrs.items():
                if attr_name.lower().replace("_", "") in k.lower().replace("_", ""):
                    if isinstance(v, (int, float)):
                        return 1.0 if -1000 < v < 1000 else 0.0
            return 0.5
        return 1.0


# Backward-compatible alias
LocalOntologyVerification = LexicalOntologyVerification


class ProtocolOntologyVerification(VerificationModule):
    """
    Protocol-Ontology Verification (POV) – Module M2.

    Targets: Content Hallucination (Definition 4)

    Metric 1 – Standard Terminology Match (Eq. 10):
        Fraction of terms in fact that appear in standard terminology T_std.

    Metric 2 – Cross-Standard Consistency (Eq. 11):
        1 − |conflicts(d_k, S)| / |mappings(d_k, S)|
    """

    def __init__(self, weight, alpha, threshold):
        super().__init__("POV", weight, alpha, threshold)

    def compute_metrics(self, candidate_fact, knowledge_graph, fact_quality):
        domain = self._infer_domain(candidate_fact)
        std_terms = STANDARD_TERMINOLOGIES.get(domain, STANDARD_TERMINOLOGIES.get("aerospace", {}))

        all_std = set()
        for v in std_terms.values():
            all_std.update(t.lower() for t in v)

        fact_terms = self._extract_terms(candidate_fact)

        # --- Metric 1: Standard Terminology Match ---
        if fact_terms:
            matches = sum(1 for t in fact_terms if t.lower() in all_std)
            metric1 = matches / len(fact_terms)
        else:
            metric1 = 0.0

        # --- Metric 2: Cross-Standard Consistency ---
        mappings, conflicts = self._cross_standard_check(candidate_fact, std_terms)
        if mappings > 0:
            metric2 = 1.0 - (conflicts / mappings)
        else:
            metric2 = 1.0

        return metric1, metric2

    def _infer_domain(self, fact):
        text = str(fact).lower()
        if any(w in text for w in ["blade", "engine", "turbine", "inspection", "measurement", "aerospace"]):
            return "aerospace"
        if any(w in text for w in ["patient", "micu", "sicu", "transfer", "healthcare"]):
            return "healthcare"
        return "aerospace"

    def _extract_terms(self, fact):
        terms = []
        if "entity_class" in fact:
            terms.append(fact["entity_class"])
        if "relationship_type" in fact:
            terms.append(fact["relationship_type"])
        if "subject_entity_id" in fact:
            parts = re.split(r'[_\s]+', fact["subject_entity_id"])
            terms.extend(p for p in parts if len(p) > 2 and not p.isdigit())
        if "object_entity_id" in fact:
            parts = re.split(r'[_\s]+', fact["object_entity_id"])
            terms.extend(p for p in parts if len(p) > 2 and not p.isdigit())

        inner = fact.get("spatiotemporal_inspection_data", {})
        rels = inner.get("relationships", [])
        for r in rels:
            terms.append(r.get("relationship_type", ""))
            sid = r.get("subject_entity_id", "")
            terms.extend(p for p in re.split(r'[_\s]+', sid) if len(p) > 2 and not p.isdigit())
            oid = r.get("object_entity_id", "")
            terms.extend(p for p in re.split(r'[_\s]+', oid) if len(p) > 2 and not p.isdigit())

        measurements = inner.get("inspection_data", {}).get("inspection_measurements", [])
        for m in measurements:
            tool = m.get("inspection_tool", "")
            if tool:
                terms.append(tool)
            feature = m.get("feature_name", "")
            if feature:
                terms.extend(feature.split())

        return [t for t in terms if t]

    def _cross_standard_check(self, fact, std_terms):
        mappings = 0
        conflicts = 0

        inner = fact.get("spatiotemporal_inspection_data", {})
        rels = inner.get("relationships", [])
        valid_rels = set(t.lower() for t in std_terms.get("relations", set()))

        for r in rels:
            rt = r.get("relationship_type", "")
            if rt:
                mappings += 1
                if rt.lower() not in valid_rels:
                    conflicts += 1

        measurements = inner.get("inspection_data", {}).get("inspection_measurements", [])
        valid_tools = set(t.lower() for t in std_terms.get("tools", set()))
        for m in measurements:
            tool = m.get("inspection_tool", "")
            if tool:
                mappings += 1
                if tool.lower() not in valid_tools and tool != "":
                    conflicts += 1

        if "relationship_type" in fact:
            mappings += 1
            if fact["relationship_type"].lower() not in valid_rels:
                conflicts += 1

        return mappings, conflicts


# Backward-compatible alias
PublicOntologyVerification = ProtocolOntologyVerification


class MotionAwareVerification(VerificationModule):
    """
    Motion-Aware Verification (MAV) – Module M3.

    Targets: ST-Inconsistency (Definition 5)

    Metric 1 – Temporal-Spatial Validity (Eq. 12):
        (ψ_s(d_k) + ψ_t(d_k)) / 2

    Metric 2 – Physical Feasibility (Eq. 13–16):
        min(Kinematic(d_k), Process(d_k))
    """

    def __init__(self, weight, alpha, threshold):
        super().__init__("MAV", weight, alpha, threshold)

    def compute_metrics(self, candidate_fact, knowledge_graph, fact_quality):
        # --- Metric 1: Temporal-Spatial Validity ---
        psi_s = knowledge_graph.psi_s(candidate_fact)
        psi_t = knowledge_graph.psi_t(candidate_fact, transport_mode="default")
        metric1 = (psi_s + psi_t) / 2.0

        # --- Metric 2: Physical Feasibility = min(Kinematic, Process) ---
        kinematic = self._kinematic_check(candidate_fact, knowledge_graph)
        process = self._process_check(candidate_fact, knowledge_graph)
        metric2 = min(kinematic, process)

        return metric1, metric2

    def _kinematic_check(self, candidate_fact, knowledge_graph):
        coord = knowledge_graph.get_spatiotemporal_coord(candidate_fact)
        if not coord:
            return 1.0

        entity_id = candidate_fact.get("subject_entity_id") or candidate_fact.get("entity_id")
        if not entity_id:
            return 1.0

        _, _, _, t_current = coord
        previous_coord = None

        for other_fact in knowledge_graph.get_all_facts():
            other_id = other_fact.get("subject_entity_id") or other_fact.get("entity_id")
            if other_id != entity_id:
                continue
            other_coord = knowledge_graph.get_spatiotemporal_coord(other_fact)
            if not other_coord:
                continue
            _, _, _, t_other = other_coord
            if isinstance(t_other, type(t_current)):
                if t_other < t_current:
                    if previous_coord is None or t_other > previous_coord[3]:
                        previous_coord = other_coord

        if not previous_coord:
            return 1.0

        distance = knowledge_graph.euclidean_distance(previous_coord, coord)
        dt = knowledge_graph.time_difference(previous_coord, coord)

        if dt <= 0:
            return 0.0

        v_req = distance / dt
        v_max = knowledge_graph.v_max.get("default", 2.0)

        if v_req <= v_max:
            return 1.0
        return math.exp(-(v_req - v_max) / v_max)

    def _process_check(self, candidate_fact, knowledge_graph):
        domain = self._infer_domain(candidate_fact)
        durations = MIN_PROCESS_DURATIONS.get(domain, MIN_PROCESS_DURATIONS["default"])
        t_min = durations.get("default", 300)

        coord = knowledge_graph.get_spatiotemporal_coord(candidate_fact)
        if not coord:
            return 1.0

        entity_id = candidate_fact.get("subject_entity_id") or candidate_fact.get("entity_id")
        if not entity_id:
            return 1.0

        _, _, _, t_current = coord
        previous_coord = None
        for other_fact in knowledge_graph.get_all_facts():
            other_id = other_fact.get("subject_entity_id") or other_fact.get("entity_id")
            if other_id != entity_id:
                continue
            other_coord = knowledge_graph.get_spatiotemporal_coord(other_fact)
            if not other_coord:
                continue
            _, _, _, t_other = other_coord
            if isinstance(t_other, type(t_current)):
                if t_other < t_current:
                    if previous_coord is None or t_other > previous_coord[3]:
                        previous_coord = other_coord

        if not previous_coord:
            return 1.0

        dt = knowledge_graph.time_difference(previous_coord, coord)
        if dt >= t_min:
            return 1.0
        if dt <= 0:
            return 0.0
        return math.exp(-(t_min - dt) / dt)

    def _infer_domain(self, fact):
        text = str(fact).lower()
        if any(w in text for w in ["patient", "micu", "transfer", "healthcare"]):
            return "healthcare"
        return "aerospace"


# Backward-compatible alias
MultiAgentVerification = MotionAwareVerification


class WebSourceVerification(VerificationModule):
    """
    Web-Source Verification (WSV) – Module M4.

    Targets: Content Hallucination (Definition 4)

    Metric 1 – Source Credibility (Eq. 17):
        Σ(w_cred,i · sim(d_k, result_i)) / Σ(w_cred,i)

    Metric 2 – Cross-Source Agreement (Eq. 18):
        1 / (1 + CV({sim_1, …, sim_N}))
    """

    CREDIBILITY_WEIGHTS = {
        "government": 1.0,
        "regulatory": 0.95,
        "manufacturer": 0.85,
        "academic": 0.75,
        "industry_standard": 0.90,
        "news": 0.50,
        "forums": 0.30,
    }

    def __init__(self, weight, alpha, threshold):
        super().__init__("WSV", weight, alpha, threshold)

    def compute_metrics(self, candidate_fact, knowledge_graph, fact_quality):
        sim_scores, cred_weights = self._simulate_source_results(candidate_fact, knowledge_graph)

        # --- Metric 1: Source Credibility ---
        if sim_scores and cred_weights:
            numerator = sum(w * s for w, s in zip(cred_weights, sim_scores))
            denominator = sum(cred_weights)
            metric1 = numerator / denominator if denominator > 0 else 0.0
        else:
            metric1 = 0.0

        # --- Metric 2: Cross-Source Agreement ---
        if len(sim_scores) >= 2:
            arr = np.array(sim_scores)
            mu = np.mean(arr)
            sigma = np.std(arr)
            cv = (sigma / mu) if mu > 0 else 0.0
            metric2 = 1.0 / (1.0 + cv)
        elif len(sim_scores) == 1:
            metric2 = 1.0
        else:
            metric2 = 0.0

        return metric1, metric2

    def _simulate_source_results(self, candidate_fact, knowledge_graph):
        """Compute similarity of fact against KG facts (proxy for web sources)."""
        fact_terms = self._fact_to_terms(candidate_fact)
        if not fact_terms:
            return [], []

        kg_facts = knowledge_graph.get_all_facts()
        sim_scores = []
        cred_weights = []
        source_types = list(self.CREDIBILITY_WEIGHTS.keys())

        for i, kf in enumerate(kg_facts[:10]):
            kg_terms = self._fact_to_terms(kf)
            if not kg_terms:
                continue
            sim = self._term_overlap_similarity(fact_terms, kg_terms)
            sim_scores.append(sim)
            src = source_types[i % len(source_types)]
            cred_weights.append(self.CREDIBILITY_WEIGHTS[src])

        if not sim_scores:
            sim_scores = [0.3]
            cred_weights = [0.5]

        return sim_scores, cred_weights

    def _fact_to_terms(self, fact):
        text = ""
        for key in ["entity_class", "relationship_type", "subject_entity_id",
                     "object_entity_id", "feature_name"]:
            v = fact.get(key, "")
            if v:
                text += " " + str(v)
        inner = fact.get("spatiotemporal_inspection_data", {})
        measurements = inner.get("inspection_data", {}).get("inspection_measurements", [])
        for m in measurements:
            for k2 in ["component_id", "feature_name", "inspection_tool"]:
                v2 = m.get(k2, "")
                if v2:
                    text += " " + str(v2)
        attrs = fact.get("attributes", {})
        text += " " + " ".join(str(v) for v in attrs.values() if isinstance(v, str))
        tokens = re.findall(r'[A-Za-z]{2,}', text)
        return [t.lower() for t in tokens]

    def _term_overlap_similarity(self, terms_a, terms_b):
        set_a = set(terms_a)
        set_b = set(terms_b)
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0


# Backward-compatible alias
WebSearchVerification = WebSourceVerification


class EmbeddingSimilarityVerification(VerificationModule):
    """
    Embedding Similarity Verification (ESV) – Module M5.

    Targets: Semantic Drift (Definition 6) and Content Hallucination (Definition 4)

    Metric 1 – Nearest Neighbor Similarity (Eq. 19):
        (1/K) Σ (1 + cos(e_k, e_ni)) / 2

    Metric 2 – Cluster Membership (Eq. 20):
        max_c P(c | e_k)  via simplified Gaussian distance
    """

    K_NEIGHBORS = 5

    def __init__(self, weight, alpha, threshold):
        super().__init__("ESV", weight, alpha, threshold)

    def compute_metrics(self, candidate_fact, knowledge_graph, fact_quality):
        fact_vec = self._fact_to_vector(candidate_fact)
        kg_facts = knowledge_graph.get_all_facts()
        kg_vecs = [self._fact_to_vector(f) for f in kg_facts]
        kg_vecs = [v for v in kg_vecs if v is not None and np.linalg.norm(v) > 0]

        if fact_vec is None or np.linalg.norm(fact_vec) == 0 or not kg_vecs:
            return 0.5, 0.5

        # --- Metric 1: K-NN Cosine Similarity ---
        sims = []
        for kv in kg_vecs:
            cos_sim = np.dot(fact_vec, kv) / (np.linalg.norm(fact_vec) * np.linalg.norm(kv))
            sims.append(cos_sim)

        sims_sorted = sorted(sims, reverse=True)
        k = min(self.K_NEIGHBORS, len(sims_sorted))
        top_k = sims_sorted[:k]
        metric1 = sum((1.0 + s) / 2.0 for s in top_k) / k

        # --- Metric 2: Cluster Membership (simplified GMM) ---
        metric2 = self._cluster_membership(fact_vec, kg_vecs)

        return metric1, metric2

    def _fact_to_vector(self, fact):
        vocab = self._get_vocab()
        text = self._fact_to_text(fact)
        tokens = re.findall(r'[a-z]+', text.lower())
        vec = np.zeros(len(vocab))
        for t in tokens:
            if t in vocab:
                vec[vocab[t]] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def _fact_to_text(self, fact):
        parts = []
        for key in ["entity_class", "relationship_type", "subject_entity_id",
                     "object_entity_id", "feature_name"]:
            v = fact.get(key, "")
            if v:
                parts.append(str(v))
        inner = fact.get("spatiotemporal_inspection_data", {})
        measurements = inner.get("inspection_data", {}).get("inspection_measurements", [])
        for m in measurements:
            for k2 in ["component_id", "feature_name", "inspection_tool"]:
                v2 = m.get(k2, "")
                if v2:
                    parts.append(str(v2))
        rels = inner.get("relationships", [])
        for r in rels:
            parts.append(r.get("relationship_type", ""))
            parts.append(r.get("subject_entity_id", ""))
            parts.append(r.get("object_entity_id", ""))
        attrs = fact.get("attributes", {})
        parts.extend(str(v) for v in attrs.values() if isinstance(v, str))
        return " ".join(parts)

    def _get_vocab(self):
        base_terms = [
            "blade", "turbine", "engine", "set", "measurement", "inspection",
            "leading", "trailing", "edge", "root", "tip", "clearance",
            "pressure", "suction", "side", "pitch", "distance",
            "contains", "has", "located", "installed", "alpha", "beta",
            "gamma", "delta", "epsilon", "zeta", "eta", "theta",
            "iota", "kappa", "scanner", "unit", "aerospace",
            "patient", "transfer", "micu", "sicu", "ccu", "healthcare",
            "high", "simple", "mount", "gap", "unknown", "fabricated",
            "invalid", "component", "feature", "deviation", "tolerance",
            "pass", "fail", "nominal", "actual", "value",
        ]
        return {t: i for i, t in enumerate(base_terms)}

    def _cluster_membership(self, fact_vec, kg_vecs):
        if not kg_vecs:
            return 0.5
        mat = np.array(kg_vecs)
        centroid = np.mean(mat, axis=0)
        dists = np.array([np.linalg.norm(v - centroid) for v in kg_vecs])
        sigma_c = np.std(dists) if len(dists) > 1 else 1.0
        if sigma_c < 1e-10:
            sigma_c = 1.0
        fact_dist = np.linalg.norm(fact_vec - centroid)
        membership = math.exp(-(fact_dist ** 2) / (2 * sigma_c ** 2))
        return float(membership)
