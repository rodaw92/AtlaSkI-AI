import time

class VerificationModule:
    """Base class for verification modules in the RMMVe process."""
    def __init__(self, name, weight, alpha, threshold):
        self.name = name
        self.weight = weight  # wi in Eq.1
        self.alpha = alpha    # αi in Eq.1
        self.threshold = threshold  # θi for early termination
        self.performance_history = []
    
    def compute_metrics(self, candidate_fact, knowledge_graph, fact_quality):
        """Compute the two metrics used in confidence calculation."""
        raise NotImplementedError("Subclasses must implement compute_metrics")
    
    def compute_confidence(self, candidate_fact, knowledge_graph, fact_quality=None, llm_confidence=1.0):
        """
        Compute confidence score using the methodology formula:
        S_i(d_k) = conf_k × [α_i · Metric1^(i)(d_k) + (1-α_i) · Metric2^(i)(d_k)]

        Where conf_k is the LLM confidence score that weights the module's raw score.
        """
        metric1, metric2 = self.compute_metrics(candidate_fact, knowledge_graph, fact_quality)

        # Raw module score (before LLM confidence weighting)
        raw_score = self.alpha * metric1 + (1 - self.alpha) * metric2

        # Apply LLM confidence weighting as per methodology
        confidence = llm_confidence * raw_score

        # Record metrics for performance history
        self.performance_history.append({
            "metric1": metric1,
            "metric2": metric2,
            "raw_score": raw_score,
            "llm_confidence": llm_confidence,
            "weighted_confidence": confidence,
            "timestamp": time.time()
        })

        return confidence, metric1, metric2 