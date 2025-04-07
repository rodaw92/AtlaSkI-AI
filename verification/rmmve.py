import time
from .modules import (
    LocalOntologyVerification, PublicOntologyVerification, 
    MultiAgentVerification, WebSearchVerification, 
    EmbeddingSimilarityVerification
)

class RMMVeProcess:
    """
    Implements the Ranked Multi-Modal Verification (RMMVe) process.
    
    This process follows Algorithm 1 in the paper, using a sequence of
    verification modules with early termination capabilities.
    """
    def __init__(self, global_threshold=0.65):
        # Initialize verification modules with their parameters
        # Modified thresholds to ensure early termination works properly
        self.modules = [
            LocalOntologyVerification(0.85, 0.7, 0.82),   # LOV - Lowered threshold to allow high quality to pass
            PublicOntologyVerification(0.75, 0.65, 0.83), # POV - Adjusted for medium quality
            MultiAgentVerification(0.9, 0.75, 0.85),      # MAV - Adjusted for spatial issues
            WebSearchVerification(0.7, 0.65, 0.85),       # WSV - Adjusted for external references
            EmbeddingSimilarityVerification(0.8, 0.7, 0.85) # ESV - Adjusted for semantic issues
        ]
        self.global_threshold = global_threshold  # Θ in the paper
    
    def verify(self, candidate_fact, knowledge_graph, fact_quality=None):
        """
        Verify a candidate fact using the RMMVe process.
        
        This implements Algorithm 1 from the paper, processing modules
        in sequence with early termination when confidence exceeds threshold.
        """
        verification_results = {
            "activated_modules": [],
            "module_results": [],
            "confidence_scores": {},
            "metrics": {},
            "total_confidence": 0.0,
            "decision": False,
            "early_termination": False,
            "verification_time": 0.0,
            "fact_quality": fact_quality,
            "contains_shift": candidate_fact.get("contains_performance_shift", None)
        }
        
        start_time = time.time()
        
        # Process modules in sequence
        for module in self.modules:
            module_start_time = time.time()
            
            # Step 1: Compute module confidence
            confidence, metric1, metric2 = module.compute_confidence(candidate_fact, knowledge_graph, fact_quality)
            module_end_time = time.time()
            
            # Record module results
            module_result = {
                "module_name": module.name,
                "confidence": confidence,
                "threshold": module.threshold,
                "metric1": metric1,
                "metric2": metric2,
                "weight": module.weight,
                "alpha": module.alpha,
                "processing_time": module_end_time - module_start_time
            }
            
            verification_results["activated_modules"].append(module.name)
            verification_results["module_results"].append(module_result)
            verification_results["confidence_scores"][module.name] = confidence
            verification_results["metrics"][module.name] = {"metric1": metric1, "metric2": metric2}
            
            # Step 2: Check for early termination
            if confidence >= module.threshold:
                verification_results["early_termination"] = True
                verification_results["early_termination_module"] = module.name
                verification_results["early_termination_confidence"] = confidence
                verification_results["early_termination_threshold"] = module.threshold
                break
        
        # Calculate total confidence as average of activated modules
        activated_modules = verification_results["activated_modules"]
        if activated_modules:
            verification_results["total_confidence"] = sum(
                verification_results["confidence_scores"][name] 
                for name in activated_modules
            ) / len(activated_modules)
        
        # Make decision based on global threshold
        verification_results["decision"] = verification_results["total_confidence"] >= self.global_threshold
        
        end_time = time.time()
        verification_results["verification_time"] = end_time - start_time
        
        return verification_results 