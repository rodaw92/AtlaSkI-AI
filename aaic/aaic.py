import math
import time
import pandas as pd
from datetime import datetime
from models.constants import CUSUM_SIGMA, CUSUM_K, CUSUM_H


class AAIC:
    """
    Autonomous Adaptive Intelligence Cycle (AAIC).

    Monitors per-module precision via CGR-CUSUM and adapts parameters
    Φ = (w, θ, α) when distribution shift is detected (G_i(n) ≥ h).

    CGR-CUSUM (Eq. 24):
        G_i(n) = max(0, G_i(n-1) + [p_i(n) − μ_0 − k])
    where k = 0.5σ, alarm threshold h = 5σ.

    Three-level adaptation on alarm:
        Weight  (Eq. 25): w_i ← w_i · exp[−γ · G_i(t)], renormalise
        Threshold (Eq. 26): θ_i ← θ_i + η · sign(FPR_i − FNR_i)
        Alpha   (Eq. 27): α_i ← α_i + η' · ∂L_i/∂α_i, clip [0,1]
    """

    def __init__(self, rmmve, sigma=CUSUM_SIGMA, k=None, h=None,
                 gamma=0.01, eta=0.05, eta_prime=0.02):
        self.rmmve = rmmve
        self.sigma = sigma
        self.k = k if k is not None else 0.5 * sigma      # k = 0.5σ
        self.h = h if h is not None else 5.0 * sigma       # h = 5σ
        self.gamma = gamma        # Decay rate for weight (Eq. 25)
        self.eta = eta            # Learning rate for threshold (Eq. 26)
        self.eta_prime = eta_prime  # Learning rate for alpha (Eq. 27)

        self.target_performance = {m.name: 0.8 for m in self.rmmve.modules}
        self.cumulative_sums = {m.name: 0.0 for m in self.rmmve.modules}
        self.performance_history = {m.name: [] for m in self.rmmve.modules}

        # Track TP/FP/TN/FN counts for FPR/FNR estimation
        self._tp = {m.name: 0 for m in self.rmmve.modules}
        self._fp = {m.name: 0 for m in self.rmmve.modules}
        self._tn = {m.name: 0 for m in self.rmmve.modules}
        self._fn = {m.name: 0 for m in self.rmmve.modules}

        self.update_history = []
        self.parameter_history = {
            m.name: {
                "weights": [], "thresholds": [], "alphas": [],
                "performances": [], "cum_sums": [], "timestamps": [],
            }
            for m in self.rmmve.modules
        }
        self.detected_shifts = []

    # ------------------------------------------------------------------
    # CGR-CUSUM monitoring (Eq. 24)
    # ------------------------------------------------------------------
    def cgr_cusum_monitor(self, module_name, performance):
        """
        G_i(n) = max(0, G_i(n-1) + [p_i(n) − μ_0 − k])
        Returns True when G_i(n) ≥ h (alarm).
        """
        mu0 = self.target_performance[module_name]
        prev = self.cumulative_sums[module_name]
        current = max(0.0, prev + (performance - mu0 - self.k))
        self.cumulative_sums[module_name] = current
        self.performance_history[module_name].append(performance)
        return current >= self.h

    # ------------------------------------------------------------------
    # FPR / FNR tracking
    # ------------------------------------------------------------------
    def record_outcome(self, module_name, predicted_positive, actual_positive):
        """Record a single validation outcome for FPR/FNR estimation."""
        if predicted_positive and actual_positive:
            self._tp[module_name] += 1
        elif predicted_positive and not actual_positive:
            self._fp[module_name] += 1
        elif not predicted_positive and actual_positive:
            self._fn[module_name] += 1
        else:
            self._tn[module_name] += 1

    def _fpr(self, module_name):
        fp = self._fp[module_name]
        tn = self._tn[module_name]
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0

    def _fnr(self, module_name):
        fn = self._fn[module_name]
        tp = self._tp[module_name]
        return fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # ------------------------------------------------------------------
    # Parameter updates
    # ------------------------------------------------------------------
    def update_weight(self, module, cumulative_sum):
        """Eq. 25: w_i ← w_i · exp[−γ · G_i(t)]"""
        return module.weight * math.exp(-self.gamma * cumulative_sum)

    def update_threshold(self, module):
        """Eq. 26: θ_i ← θ_i + η · sign(FPR_i − FNR_i)"""
        fpr = self._fpr(module.name)
        fnr = self._fnr(module.name)
        diff = fpr - fnr
        if diff > 0:
            direction = 1.0
        elif diff < 0:
            direction = -1.0
        else:
            direction = 0.0
        new_threshold = module.threshold + self.eta * direction
        return max(0.1, min(0.95, new_threshold))

    def update_alpha(self, module, performance):
        """
        Eq. 27: α_i ← α_i + η' · ∂L_i/∂α_i, clip [0,1].

        Gradient estimated via finite differences on recent performance:
        ∂L/∂α ≈ [L(α+δ) − L(α−δ)] / (2δ), where L = −log(perf).
        """
        history = module.performance_history
        if len(history) < 2:
            return module.alpha

        recent = [h.get("metric1", 0.5) for h in history[-3:]]
        older = [h.get("metric2", 0.5) for h in history[-3:]]
        avg_m1 = sum(recent) / len(recent) if recent else 0.5
        avg_m2 = sum(older) / len(older) if older else 0.5

        loss_higher_alpha = -math.log(max(1e-10, 0.6 * avg_m1 + 0.4 * avg_m2))
        loss_lower_alpha = -math.log(max(1e-10, 0.4 * avg_m1 + 0.6 * avg_m2))
        gradient = (loss_higher_alpha - loss_lower_alpha) / 0.4

        new_alpha = module.alpha + self.eta_prime * gradient
        return max(0.0, min(1.0, new_alpha))

    def normalize_weights(self):
        """Renormalise weights so Σ w_i = 1 (Eq. 25 second step)."""
        total = sum(m.weight for m in self.rmmve.modules)
        if total > 0:
            for m in self.rmmve.modules:
                m.weight = m.weight / total

    # ------------------------------------------------------------------
    # Per-module update cycle
    # ------------------------------------------------------------------
    def update_module_parameters(self, module):
        if not module.performance_history:
            return {"module": module.name, "update": "No performance history", "detected": False}

        latest = module.performance_history[-1]["metric1"]
        now = time.time()

        self.parameter_history[module.name]["weights"].append(module.weight)
        self.parameter_history[module.name]["thresholds"].append(module.threshold)
        self.parameter_history[module.name]["alphas"].append(module.alpha)
        self.parameter_history[module.name]["performances"].append(latest)
        self.parameter_history[module.name]["cum_sums"].append(self.cumulative_sums[module.name])
        self.parameter_history[module.name]["timestamps"].append(now)

        shift_detected = self.cgr_cusum_monitor(module.name, latest)

        info = {
            "module": module.name,
            "performance": latest,
            "cumulative_sum": self.cumulative_sums[module.name],
            "detected": shift_detected,
            "old_params": {"weight": module.weight, "threshold": module.threshold, "alpha": module.alpha},
            "new_params": {"weight": module.weight, "threshold": module.threshold, "alpha": module.alpha},
            "timestamp": now,
        }

        if shift_detected:
            self.detected_shifts.append({
                "module": module.name, "timestamp": now,
                "cumulative_sum": self.cumulative_sums[module.name],
                "performance": latest,
                "old_weight": module.weight,
                "old_threshold": module.threshold,
                "old_alpha": module.alpha,
            })

            module.weight = self.update_weight(module, self.cumulative_sums[module.name])
            module.threshold = self.update_threshold(module)
            module.alpha = self.update_alpha(module, latest)

            self.cumulative_sums[module.name] = 0.0

            self.detected_shifts[-1].update({
                "new_weight": module.weight,
                "new_threshold": module.threshold,
                "new_alpha": module.alpha,
                "weight_change": module.weight - info["old_params"]["weight"],
                "threshold_change": module.threshold - info["old_params"]["threshold"],
                "alpha_change": module.alpha - info["old_params"]["alpha"],
            })

            info["new_params"] = {"weight": module.weight, "threshold": module.threshold, "alpha": module.alpha}
            info["update"] = "Parameters updated (AAIC)"
        else:
            info["update"] = "No update needed (G_i < h)"

        return info

    def update_all_modules(self):
        updates = []
        for module in self.rmmve.modules:
            updates.append(self.update_module_parameters(module))
        self.normalize_weights()
        self.update_history.append({"timestamp": time.time(), "updates": updates})
        return updates

    # ------------------------------------------------------------------
    # Convenience / visualisation
    # ------------------------------------------------------------------
    def get_parameter_history_df(self):
        data = []
        for name, hist in self.parameter_history.items():
            for i in range(len(hist["timestamps"])):
                data.append({
                    "Module": name,
                    "Timestamp": datetime.fromtimestamp(hist["timestamps"][i]).strftime("%Y-%m-%d %H:%M:%S"),
                    "Weight": hist["weights"][i],
                    "Threshold": hist["thresholds"][i],
                    "Alpha": hist["alphas"][i],
                    "Performance": hist["performances"][i],
                    "Cumulative Sum": hist["cum_sums"][i],
                })
        return pd.DataFrame(data)

    def get_detected_shifts_df(self):
        if not self.detected_shifts:
            return pd.DataFrame()
        return pd.DataFrame(self.detected_shifts)
