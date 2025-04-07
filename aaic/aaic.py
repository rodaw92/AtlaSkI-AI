import math
import time
import pandas as pd
from datetime import datetime

class AAIC:
    """
    Implements the Autonomous Adaptive Intelligence Cycle (AAIC) for TruthFlow.
    
    This system dynamically adjusts the parameters of verification modules
    based on their performance over time, as described in the paper:
    
    1. CGR-CUSUM monitors module performance
    2. When cumulative sum exceeds threshold h, it triggers parameter updates
    3. Updates include weight (w), threshold (θ), and alpha (α) adjustments
    4. Weights are normalized to sum to 4 (for 5-module system)
    """
    def __init__(self, rmmve, h=1.5, k=0.05, gamma=0.1, eta=0.1, eta_prime=0.1):
        self.rmmve = rmmve
        self.h = h  # Threshold for detecting performance shifts (LOWERED FROM 5.0 to 1.5)
        self.k = k  # Allowance parameter (k=0.05 in paper)
        self.gamma = gamma  # Scaling factor for weight updates
        self.eta = eta  # Learning rate for threshold updates (η=0.1 in paper)
        self.eta_prime = eta_prime  # Learning rate for alpha updates
        
        # Target performance level μ_0 for each module (typically 0.8)
        self.target_performance = {module.name: 0.8 for module in self.rmmve.modules}
        
        # Initialize cumulative sums S_i(t) for each module
        self.cumulative_sums = {module.name: 0.0 for module in self.rmmve.modules}
        
        # Store historical performance for TPR/FPR calculations
        self.performance_history = {module.name: [] for module in self.rmmve.modules}
        
        # Update history - track all updates and their effects
        self.update_history = []
        
        # Parameter history - track how parameters evolve over time
        self.parameter_history = {
            module.name: {
                "weights": [],
                "thresholds": [],
                "alphas": [],
                "performances": [],
                "cum_sums": [],
                "timestamps": []
            } for module in self.rmmve.modules
        }
        
        # Track detected shifts
        self.detected_shifts = []
    
    def cgr_cusum_monitor(self, module_name, performance):
        """
        Monitor module performance using CGR-CUSUM algorithm.
        
        Implements the formula from the paper:
        S_i(t) = max(0, S_i(t-1) + [p_i(t) - μ_0 - k])
        
        Returns True if a significant performance shift is detected (S_i(t) ≥ h).
        """
        target = self.target_performance[module_name]  # μ_0 in the paper
        
        # Update cumulative sum using the formula from the paper
        previous_sum = self.cumulative_sums[module_name]  # S_i(t-1)
        current_sum = max(0, previous_sum + performance - target - self.k)  # S_i(t)
        self.cumulative_sums[module_name] = current_sum
        
        # Store performance for history
        self.performance_history[module_name].append(performance)
        
        # Detect if performance shift exceeds threshold (S_i(t) ≥ h)
        return current_sum >= self.h
    
    def update_weight(self, module, cumulative_sum):
        """
        Update module weight using exponential weights algorithm.
        
        Implements Eq.6 from the paper:
        w_i(t+1) = w_i(t)*exp[-γ*S_i(t)]
        """
        old_weight = module.weight
        # Apply exponential weights algorithm (Eq.6)
        new_weight = old_weight * math.exp(-self.gamma * cumulative_sum)
        return new_weight
    
    def update_threshold(self, module, performance):
        """
        Update confidence threshold using gradient ascent.
        
        Implements Eq.8 from the paper:
        θ_i(t+1) = θ_i(t) + η*(∂U(θ_i)/∂θ_i)
        
        Uses a simplified gradient estimation based on TPR change.
        """
        old_threshold = module.threshold
        
        # Use historical performance to estimate TPR change direction
        history = self.performance_history[module.name]
        if len(history) > 1:
            # Estimate if TPR is improving or declining
            recent_avg = sum(history[-3:]) / min(3, len(history))
            older_avg = sum(history[:-3]) / max(1, len(history) - 3)
            tpr_change = 0.05 * (recent_avg - older_avg)  # Simplified gradient
        else:
            # If limited history, use target-performance difference
            tpr_change = 0.05 * (self.target_performance[module.name] - performance)
        
        # Apply gradient ascent update (Eq.8)
        new_threshold = max(0.1, min(0.95, old_threshold + self.eta * tpr_change))
        return new_threshold
    
    def update_alpha(self, module, performance):
        """
        Update internal weighting factor alpha using gradient ascent.
        
        Implements Eq.9 from the paper:
        α_i(t+1) = α_i(t) + η'*(∂U_i(α_i)/∂α_i)
        
        Uses a simplified gradient estimation.
        """
        old_alpha = module.alpha
        
        # Use historical performance to estimate utility gradient
        history = self.performance_history[module.name]
        if len(history) > 1:
            # Adapt alpha based on recent performance trends
            recent_avg = sum(history[-3:]) / min(3, len(history))
            older_avg = sum(history[:-3]) / max(1, len(history) - 3)
            utility_change = 0.05 * (recent_avg - older_avg)  # Simplified gradient
        else:
            # If limited history, use target-performance difference
            utility_change = 0.05 * (self.target_performance[module.name] - performance)
        
        # Apply gradient ascent update (Eq.9)
        new_alpha = max(0.1, min(0.9, old_alpha + self.eta_prime * utility_change))
        return new_alpha
    
    def normalize_weights(self):
        """
        Normalize weights to sum to 4 as per Eq.7 in the paper.
        
        w_i(t+1) = (w_i(t+1) / ∑w_j(t+1)) × 4
        
        This creates a statistical baseline with mean weight of 0.8,
        allowing easy identification of above/below average contributors.
        """
        total_weight = sum(module.weight for module in self.rmmve.modules)
        if total_weight > 0:
            for module in self.rmmve.modules:
                module.weight = (module.weight / total_weight) * 4
    
    def update_module_parameters(self, module):
        """
        Update a module's parameters based on its performance.
        This implements the adaptive parameter updates from the paper.
        """
        # Get latest performance (using first metric as performance indicator)
        if not module.performance_history:
            return {"module": module.name, "update": "No performance history", "detected": False}
        
        latest_performance = module.performance_history[-1]["metric1"]
        current_time = time.time()
        
        # Add to parameter history
        self.parameter_history[module.name]["weights"].append(module.weight)
        self.parameter_history[module.name]["thresholds"].append(module.threshold)
        self.parameter_history[module.name]["alphas"].append(module.alpha)
        self.parameter_history[module.name]["performances"].append(latest_performance)
        self.parameter_history[module.name]["cum_sums"].append(self.cumulative_sums[module.name])
        self.parameter_history[module.name]["timestamps"].append(current_time)
        
        # Check if performance shift is detected using CGR-CUSUM
        shift_detected = self.cgr_cusum_monitor(module.name, latest_performance)
        
        update_info = {
            "module": module.name,
            "performance": latest_performance,
            "cumulative_sum": self.cumulative_sums[module.name],
            "detected": shift_detected,
            "old_params": {"weight": module.weight, "threshold": module.threshold, "alpha": module.alpha},
            "new_params": {"weight": module.weight, "threshold": module.threshold, "alpha": module.alpha},
            "timestamp": current_time
        }
        
        if shift_detected:
            # Store shift information before updates
            self.detected_shifts.append({
                "module": module.name,
                "timestamp": current_time,
                "cumulative_sum": self.cumulative_sums[module.name],
                "performance": latest_performance,
                "old_weight": module.weight,
                "old_threshold": module.threshold,
                "old_alpha": module.alpha
            })
            
            # Step 2 from Algorithm 1: Update weight using exponential weights algorithm (Eq.6)
            module.weight = self.update_weight(module, self.cumulative_sums[module.name])
            
            # Step 3 from Algorithm 1: Update threshold using gradient ascent (Eq.8)
            module.threshold = self.update_threshold(module, latest_performance)
            
            # Step 4 from Algorithm 1: Update alpha using gradient ascent (Eq.9)
            module.alpha = self.update_alpha(module, latest_performance)
            
            # Reset cumulative sum after adjustment as per CGR-CUSUM methodology
            self.cumulative_sums[module.name] = 0.0
            
            # Update the detected shift records with the new parameters
            self.detected_shifts[-1].update({
                "new_weight": module.weight,
                "new_threshold": module.threshold,
                "new_alpha": module.alpha,
                "weight_change": module.weight - update_info["old_params"]["weight"],
                "threshold_change": module.threshold - update_info["old_params"]["threshold"],
                "alpha_change": module.alpha - update_info["old_params"]["alpha"]
            })
            
            update_info["new_params"] = {"weight": module.weight, "threshold": module.threshold, "alpha": module.alpha}
            update_info["update"] = "Parameters updated according to AAIC Algorithm"
        else:
            # No update needed (Step 7-9 in Algorithm 1)
            update_info["update"] = "No update needed (S_i(t) < h)"
        
        return update_info
    
    def update_all_modules(self):
        """
        Update parameters for all modules based on their performance.
        This implements Algorithm 1 (AAIC within TruthFlow) from the paper.
        """
        updates = []
        
        # For each verification module M_i (Steps 1-10 in Algorithm 1)
        for module in self.rmmve.modules:
            update_info = self.update_module_parameters(module)
            updates.append(update_info)
        
        # Normalize weights (Step 11 in Algorithm 1)
        self.normalize_weights()
        
        self.update_history.append({
            "timestamp": time.time(),
            "updates": updates
        })
        
        return updates
    
    def get_parameter_history_df(self):
        """Get parameter history as a DataFrame for visualization."""
        data = []
        
        for module_name, history in self.parameter_history.items():
            for i in range(len(history["timestamps"])):
                data.append({
                    "Module": module_name,
                    "Timestamp": datetime.fromtimestamp(history["timestamps"][i]).strftime("%Y-%m-%d %H:%M:%S"),
                    "Weight": history["weights"][i],
                    "Threshold": history["thresholds"][i],
                    "Alpha": history["alphas"][i],
                    "Performance": history["performances"][i],
                    "Cumulative Sum": history["cum_sums"][i]
                })
        
        return pd.DataFrame(data)
    
    def get_detected_shifts_df(self):
        """Get detected shifts as a DataFrame for visualization."""
        if not self.detected_shifts:
            return pd.DataFrame()
        
        return pd.DataFrame(self.detected_shifts) 