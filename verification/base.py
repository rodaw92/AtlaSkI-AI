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
    
    def compute_confidence(self, candidate_fact, knowledge_graph, fact_quality=None):
        """
        Compute confidence score using Eq.1 from RMMVe:
        C_module_i(dk) = wi(αi·Metric1^(i)(dk) + (1-αi)·Metric2^(i)(dk))
        """
        metric1, metric2 = self.compute_metrics(candidate_fact, knowledge_graph, fact_quality)
        confidence = self.weight * (self.alpha * metric1 + (1 - self.alpha) * metric2)
        
        # Record metrics for performance history
        self.performance_history.append({
            "metric1": metric1,
            "metric2": metric2,
            "confidence": confidence,
            "timestamp": time.time()
        })
        
        return confidence, metric1, metric2 