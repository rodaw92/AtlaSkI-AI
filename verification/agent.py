import random

class Agent:
    """Agent for Multi-Agent Verification (MAV)."""
    def __init__(self, name, reliability):
        self.name = name  # Agent name (Temporal, Spatial, or 4D Consistency)
        self.reliability = reliability  # Reliability factor rᵢ ∈ [0,1]
    
    def validate(self, candidate_fact, fact_quality):
        """
        Validate a candidate fact.
        Returns 1 if valid, 0 otherwise.
        Uses fact_quality to determine validation outcome.
        """
        # Check if this fact contains a performance shift for MAV
        has_shift = candidate_fact.get("contains_performance_shift") == "MAV"
        
        if has_shift:
            # Lower validation probability with shift
            return random.choices([0, 1], weights=[0.7, 0.3], k=1)[0]
        
        if self.name == "Temporal":
            # Temporal agent validation
            if fact_quality in ["high_quality", "medium_quality"]:
                return 1
            elif fact_quality == "spatial_issue":
                return 1  # Temporal aspects are fine even with spatial issues
            elif fact_quality == "external_ref":
                return random.choices([0, 1], weights=[0.2, 0.8], k=1)[0]
            elif fact_quality == "semantic_issue":
                return random.choices([0, 1], weights=[0.3, 0.7], k=1)[0]
            else:  # low_quality
                return 0
        
        elif self.name == "Spatial":
            # Spatial agent validation
            if fact_quality in ["high_quality", "medium_quality"]:
                return 1
            elif fact_quality == "spatial_issue":
                return 0  # Specifically fails on spatial issues
            elif fact_quality == "external_ref":
                return random.choices([0, 1], weights=[0.3, 0.7], k=1)[0]
            elif fact_quality == "semantic_issue":
                return random.choices([0, 1], weights=[0.4, 0.6], k=1)[0]
            else:  # low_quality
                return 0
        
        elif self.name == "4D Consistency":
            # 4D Consistency agent validation
            if fact_quality == "high_quality":
                return 1
            elif fact_quality == "medium_quality":
                return random.choices([0, 1], weights=[0.1, 0.9], k=1)[0]
            elif fact_quality == "spatial_issue":
                return 0  # Fails on 4D consistency with spatial issues
            elif fact_quality == "external_ref":
                return random.choices([0, 1], weights=[0.4, 0.6], k=1)[0]
            elif fact_quality == "semantic_issue":
                return random.choices([0, 1], weights=[0.5, 0.5], k=1)[0]
            else:  # low_quality
                return 0
        
        return 0  # Default case 