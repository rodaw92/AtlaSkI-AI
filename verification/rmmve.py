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
    def __init__(self, global_threshold=0.70):
        # Initialize verification modules with their parameters
        # Stricter thresholds to properly reject problematic data
        self.modules = [
            LocalOntologyVerification(0.85, 0.7, 0.88),   # LOV - Strict ontology checking
            PublicOntologyVerification(0.75, 0.65, 0.88), # POV - Strict standard compliance
            MultiAgentVerification(0.9, 0.75, 0.90),      # MAV - Very strict physics checking
            WebSearchVerification(0.7, 0.65, 0.88),       # WSV - Strict external validation
            EmbeddingSimilarityVerification(0.8, 0.7, 0.88) # ESV - Strict semantic checking
        ]
        self.global_threshold = global_threshold  # Θ in the paper
    
    def verify(self, candidate_fact, knowledge_graph, fact_quality=None, llm_confidence=1.0):
        """
        Verify a candidate fact using the RMMVe process.

        This implements Algorithm 1 from the paper, processing modules
        in sequence with early termination when confidence exceeds threshold.

        Args:
            candidate_fact: The fact to verify
            knowledge_graph: The knowledge graph context
            fact_quality: Quality level of the fact
            llm_confidence: LLM confidence score (0.0-1.0) to weight module scores
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
        
        # Track activated modules and their contributions
        activated_modules = []
        total_weighted_confidence = 0.0
        total_weights = 0.0

        # Process modules in sequence
        for module in self.modules:
            module_start_time = time.time()

            # Step 1: Compute module confidence (weighted by LLM confidence)
            confidence, metric1, metric2 = module.compute_confidence(
                candidate_fact,
                knowledge_graph,
                fact_quality,
                llm_confidence
            )
            module_end_time = time.time()

            # Record module results (all modules, even if not activated)
            module_result = {
                "module_name": module.name,
                "confidence": confidence,
                "threshold": module.threshold,
                "activated": False,  # Will be set to True if activated
                "metric1": metric1,
                "metric2": metric2,
                "weight": module.weight,
                "alpha": module.alpha,
                "processing_time": module_end_time - module_start_time
            }

            verification_results["module_results"].append(module_result)
            verification_results["confidence_scores"][module.name] = confidence
            verification_results["metrics"][module.name] = {"metric1": metric1, "metric2": metric2}

            # Step 2: Check activation criterion
            # Scale the activation threshold by LLM confidence so that
            # high-confidence facts are not unfairly penalized, and
            # low-confidence facts remain hard to activate.
            effective_threshold = module.threshold * llm_confidence

            if confidence >= effective_threshold:
                # Module is activated - add to cumulative confidence
                activated_modules.append(module.name)
                total_weighted_confidence += confidence * module.weight
                total_weights += module.weight
                module_result["activated"] = True

                # Calculate current cumulative confidence
                current_cumulative = total_weighted_confidence / total_weights if total_weights > 0 else 0

                # Step 3: Check for early termination
                if current_cumulative >= self.global_threshold:
                    verification_results["early_termination"] = True
                    verification_results["early_termination_module"] = module.name
                    verification_results["early_termination_confidence"] = confidence
                    verification_results["early_termination_threshold"] = effective_threshold
                    verification_results["early_termination_cumulative"] = current_cumulative
                    break
        
        # Store activated modules and final cumulative confidence
        verification_results["activated_modules"] = activated_modules
        verification_results["total_confidence"] = total_weighted_confidence / total_weights if total_weights > 0 else 0.0
        
        # Make decision based on global threshold
        verification_results["decision"] = verification_results["total_confidence"] >= self.global_threshold
        
        end_time = time.time()
        verification_results["verification_time"] = end_time - start_time
        
        return verification_results 