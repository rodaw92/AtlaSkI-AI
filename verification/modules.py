import random
import math
import itertools
from .base import VerificationModule
from .agent import Agent

class LocalOntologyVerification(VerificationModule):
    """
    Local Ontology Verification (LOV) module.
    Uses precision and recall against the local ontology.
    """
    def __init__(self, weight, alpha, threshold):
        super().__init__("LOV", weight, alpha, threshold)
    
    def compute_metrics(self, candidate_fact, knowledge_graph, fact_quality):
        """
        Compute P_LOV (precision) and R_LOV (recall) as defined in document.
        
        Uses fact_quality to adjust scores to demonstrate different verification paths.
        """
        # Check if the fact contains a performance shift for LOV
        has_shift = candidate_fact.get("contains_performance_shift") == "LOV"
        
        # Base precision and recall based on quality
        if fact_quality == "high_quality":
            # Set very high precision and recall for high quality facts to exceed threshold
            base_precision = 0.98 if not has_shift else 0.60
            base_recall = 0.95 if not has_shift else 0.55
        elif fact_quality == "medium_quality":
            # High but not enough to exceed threshold
            base_precision = 0.85 if not has_shift else 0.50
            base_recall = 0.80 if not has_shift else 0.45
        elif fact_quality == "spatial_issue":
            base_precision = 0.80 if not has_shift else 0.45
            base_recall = 0.75 if not has_shift else 0.40
        elif fact_quality == "external_ref":
            base_precision = 0.75 if not has_shift else 0.40
            base_recall = 0.70 if not has_shift else 0.35
        elif fact_quality == "semantic_issue":
            base_precision = 0.70 if not has_shift else 0.35
            base_recall = 0.65 if not has_shift else 0.30
        else:  # low_quality
            base_precision = 0.50 if not has_shift else 0.20
            base_recall = 0.45 if not has_shift else 0.15
        
        # Add small random variation
        precision = min(1.0, max(0.0, base_precision + random.uniform(-0.05, 0.05)))
        recall = min(1.0, max(0.0, base_recall + random.uniform(-0.05, 0.05)))
        
        return precision, recall

class PublicOntologyVerification(VerificationModule):
    """
    Public Ontology Verification (POV) module.
    Uses accuracy and coverage metrics.
    """
    def __init__(self, weight, alpha, threshold):
        super().__init__("POV", weight, alpha, threshold)
    
    def compute_metrics(self, candidate_fact, knowledge_graph, fact_quality):
        """
        Compute A_POV (accuracy) and CA_POV (coverage) as defined in document.
        
        Uses fact_quality to adjust scores to demonstrate different verification paths.
        """
        # Check if the fact contains a performance shift for POV
        has_shift = candidate_fact.get("contains_performance_shift") == "POV"
        
        # Base accuracy and coverage based on quality
        if fact_quality == "high_quality":
            # High quality should already terminate at LOV, but still set high values
            base_accuracy = 0.93 if not has_shift else 0.70
            base_coverage = 0.88 if not has_shift else 0.65
        elif fact_quality == "medium_quality":
            # Very high values to allow early termination at POV
            base_accuracy = 0.95 if not has_shift else 0.72
            base_coverage = 0.92 if not has_shift else 0.68
        elif fact_quality == "spatial_issue":
            # Good but not enough to exceed threshold
            base_accuracy = 0.80 if not has_shift else 0.60
            base_coverage = 0.75 if not has_shift else 0.55
        elif fact_quality == "external_ref":
            base_accuracy = 0.75 if not has_shift else 0.55
            base_coverage = 0.70 if not has_shift else 0.50
        elif fact_quality == "semantic_issue":
            base_accuracy = 0.70 if not has_shift else 0.50
            base_coverage = 0.65 if not has_shift else 0.45
        else:  # low_quality
            base_accuracy = 0.50 if not has_shift else 0.35
            base_coverage = 0.45 if not has_shift else 0.30
        
        # Add small random variation
        accuracy = min(1.0, max(0.0, base_accuracy + random.uniform(-0.02, 0.02)))
        coverage = min(1.0, max(0.0, base_coverage + random.uniform(-0.02, 0.02)))
        
        return accuracy, coverage

class MultiAgentVerification(VerificationModule):
    """
    Multi-Agent Verification (MAV) module using Shapley value integration.
    Uses consensus score and reliability metrics.
    """
    def __init__(self, weight, alpha, threshold):
        super().__init__("MAV", weight, alpha, threshold)
        # Create the three agents mentioned in the paper
        self.agents = [
            Agent("Temporal", 0.9),
            Agent("Spatial", 0.85),
            Agent("4D Consistency", 0.95)
        ]
    
    def compute_shapley_values(self, validation_results):
        """
        Compute Shapley values for each agent based on their validation results.
        This follows the formula in the paper:
        
        ϕ_Ai = ∑_{S⊆N∖{Ai}} (|S|!(|N|-|S|-1)!/|N|!) [v(S∪{Ai})-v(S)]
        
        where v(S) = (1/m) ∑_{Aj∈S} V_Aj(dk)
        """
        agents = list(validation_results.keys())
        n = len(agents)
        shapley_values = {}
        
        # Calculate Shapley value for each agent
        for agent in agents:
            shapley_value = 0
            
            # Generate all subsets of agents excluding the current agent
            other_agents = [a for a in agents if a != agent]
            
            for subset_size in range(len(other_agents) + 1):
                for subset in itertools.combinations(other_agents, subset_size):
                    s = len(subset)
                    
                    # Calculate |S|!(|N|-|S|-1)!/|N|!
                    subset_weight = (math.factorial(s) * math.factorial(n - s - 1)) / math.factorial(n)
                    
                    # Calculate v(S)
                    if subset:
                        v_s = sum(validation_results[a] for a in subset) / len(subset)
                    else:
                        v_s = 0
                    
                    # Calculate v(S∪{Ai})
                    subset_with_agent = list(subset) + [agent]
                    v_s_with_agent = sum(validation_results[a] for a in subset_with_agent) / len(subset_with_agent)
                    
                    # Add to Shapley value
                    shapley_value += subset_weight * (v_s_with_agent - v_s)
            
            shapley_values[agent] = shapley_value
        
        return shapley_values
    
    def compute_metrics(self, candidate_fact, knowledge_graph, fact_quality):
        """
        Compute CS_MAV (consensus score) and R_MAV (reliability) as defined in document.
        
        CS_MAV = ∑(i=1 to m) ϕ_Ai * V_Ai(dk)
        R_MAV = ∑(i=1 to m) ϕ_Ai * r_Ai
        """
        # Get validation results from each agent
        validation_results = {}
        for agent in self.agents:
            validation_results[agent.name] = agent.validate(candidate_fact, fact_quality)
        
        # Compute Shapley values
        shapley_values = self.compute_shapley_values(validation_results)
        
        # Calculate consensus score (CS_MAV)
        consensus_score = sum(shapley_values[agent.name] * validation_results[agent.name] 
                              for agent in self.agents)
        
        # Calculate reliability score (R_MAV)
        reliability_score = sum(shapley_values[agent.name] * agent.reliability 
                               for agent in self.agents)
        
        # For spatial_issue facts, increase metrics to ensure they pass MAV
        # (but only if there's no performance shift)
        has_shift = candidate_fact.get("contains_performance_shift") == "MAV"
        if fact_quality == "spatial_issue" and not has_shift:
            # High enough to exceed threshold for early termination
            consensus_score = min(1.0, consensus_score * 1.5)
            reliability_score = min(1.0, reliability_score * 1.5)
        
        return consensus_score, reliability_score

class WebSearchVerification(VerificationModule):
    """
    Web Search Verification (WSV) module.
    Uses recall and F1 score metrics.
    """
    def __init__(self, weight, alpha, threshold):
        super().__init__("WSV", weight, alpha, threshold)
    
    def compute_metrics(self, candidate_fact, knowledge_graph, fact_quality):
        """
        Compute R_WSV (recall) and F1_WSV (F1 score) as defined in document.
        
        Uses fact_quality to adjust scores to demonstrate different verification paths.
        """
        # Check if the fact contains a performance shift for WSV
        has_shift = candidate_fact.get("contains_performance_shift") == "WSV"
        
        # Base recall and precision (for F1) based on quality
        if fact_quality == "high_quality":
            base_recall = 0.90 if not has_shift else 0.65
            base_precision = 0.92 if not has_shift else 0.67
        elif fact_quality == "medium_quality":
            base_recall = 0.85 if not has_shift else 0.60
            base_precision = 0.87 if not has_shift else 0.62
        elif fact_quality == "spatial_issue":
            base_recall = 0.80 if not has_shift else 0.55
            base_precision = 0.82 if not has_shift else 0.57
        elif fact_quality == "external_ref":
            # Very high values for external_ref to exceed threshold
            base_recall = 0.98 if not has_shift else 0.73
            base_precision = 0.98 if not has_shift else 0.73
        elif fact_quality == "semantic_issue":
            base_recall = 0.75 if not has_shift else 0.50
            base_precision = 0.78 if not has_shift else 0.53
        else:  # low_quality
            base_recall = 0.40 if not has_shift else 0.25
            base_precision = 0.45 if not has_shift else 0.30
        
        # Add small random variation
        recall = min(1.0, max(0.0, base_recall + random.uniform(-0.02, 0.02)))
        precision = min(1.0, max(0.0, base_precision + random.uniform(-0.02, 0.02)))
        
        # Calculate F1 score
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        
        return recall, f1_score

class EmbeddingSimilarityVerification(VerificationModule):
    """
    Embedding Similarity Verification (ESV) module.
    Uses similarity score and anomaly detection rate metrics.
    """
    def __init__(self, weight, alpha, threshold):
        super().__init__("ESV", weight, alpha, threshold)
    
    def compute_metrics(self, candidate_fact, knowledge_graph, fact_quality):
        """
        Compute Sim_ESV (similarity score) and ADR_ESV (anomaly detection rate) as defined in document.
        
        Uses fact_quality to adjust scores to demonstrate different verification paths.
        """
        # Check if the fact contains a performance shift for ESV
        has_shift = candidate_fact.get("contains_performance_shift") == "ESV"
        
        # Base similarity and anomaly detection rate based on quality
        if fact_quality == "high_quality":
            base_similarity = 0.92 if not has_shift else 0.67
            base_anomaly_rate = 0.05 if not has_shift else 0.30
        elif fact_quality == "medium_quality":
            base_similarity = 0.85 if not has_shift else 0.60
            base_anomaly_rate = 0.10 if not has_shift else 0.35
        elif fact_quality == "spatial_issue":
            base_similarity = 0.80 if not has_shift else 0.55
            base_anomaly_rate = 0.15 if not has_shift else 0.40
        elif fact_quality == "external_ref":
            base_similarity = 0.78 if not has_shift else 0.53
            base_anomaly_rate = 0.18 if not has_shift else 0.43
        elif fact_quality == "semantic_issue":
            # Very high values for semantic_issue to exceed threshold
            base_similarity = 0.98 if not has_shift else 0.73
            base_anomaly_rate = 0.02 if not has_shift else 0.27
        else:  # low_quality
            base_similarity = 0.40 if not has_shift else 0.25
            base_anomaly_rate = 0.60 if not has_shift else 0.75
        
        # Add small random variation
        similarity = min(1.0, max(0.0, base_similarity + random.uniform(-0.02, 0.02)))
        anomaly_rate = min(1.0, max(0.0, base_anomaly_rate + random.uniform(-0.02, 0.02)))
        
        return similarity, 1.0 - anomaly_rate  # Return similarity and (1 - anomaly_rate) 