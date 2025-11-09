"""
Evaluation Metrics for ATLASky-AI Verification

Implements standard classification metrics as defined in Section 6.1.2:
- Precision: TP/(TP+FP)
- Recall: TP/(TP+FN)
- F1-Score: Harmonic mean
- False Positive Rate (FPR): FP/(FP+TN)

In our verification context:
- Positive class = REJECT (detecting incorrect facts)
- TP = Incorrect facts correctly rejected
- TN = Correct facts correctly accepted
- FP = Correct facts incorrectly rejected (false alarms)
- FN = Incorrect facts incorrectly accepted (dangerous!)
"""

import numpy as np
from typing import List, Dict, Tuple

class VerificationMetrics:
    """Calculate verification performance metrics"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all counters"""
        self.tp = 0  # True positives: incorrect facts rejected
        self.tn = 0  # True negatives: correct facts accepted
        self.fp = 0  # False positives: correct facts rejected (false alarms)
        self.fn = 0  # False negatives: incorrect facts accepted (DANGEROUS!)

    def update(self, ground_truth_labels: List[bool],
               predicted_decisions: List[bool]):
        """
        Update metrics with batch of predictions

        Args:
            ground_truth_labels: True if fact is CORRECT, False if INCORRECT
            predicted_decisions: True if ACCEPTED, False if REJECTED
        """
        for gt_correct, pred_accept in zip(ground_truth_labels, predicted_decisions):
            if not gt_correct and not pred_accept:
                # Incorrect fact correctly rejected
                self.tp += 1
            elif gt_correct and pred_accept:
                # Correct fact correctly accepted
                self.tn += 1
            elif gt_correct and not pred_accept:
                # Correct fact incorrectly rejected (false alarm)
                self.fp += 1
            else:  # not gt_correct and pred_accept
                # Incorrect fact incorrectly accepted (DANGEROUS!)
                self.fn += 1

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all metrics

        Returns dict with:
            - precision: TP/(TP+FP)
            - recall: TP/(TP+FN)
            - f1: Harmonic mean
            - fpr: FP/(FP+TN)
            - accuracy: (TP+TN)/(TP+TN+FP+FN)
        """
        # Precision: Of all rejections, how many were truly incorrect?
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

        # Recall: Of all incorrect facts, how many did we catch?
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

        # F1: Harmonic mean of precision and recall
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # FPR: Of all correct facts, how many did we falsely reject?
        fpr = self.fp / (self.fp + self.tn) if (self.fp + self.tn) > 0 else 0.0

        # Overall accuracy
        total = self.tp + self.tn + self.fp + self.fn
        accuracy = (self.tp + self.tn) / total if total > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fpr': fpr,
            'accuracy': accuracy,
            'tp': self.tp,
            'tn': self.tn,
            'fp': self.fp,
            'fn': self.fn,
            'total': total
        }

    def print_confusion_matrix(self):
        """Print confusion matrix for visualization"""
        print("\nConfusion Matrix:")
        print("                    Predicted")
        print("                Accept    Reject")
        print(f"Actual Correct   {self.tn:5d}    {self.fp:5d}   (FP = false alarms)")
        print(f"       Incorrect {self.fn:5d}    {self.tp:5d}   (FN = DANGEROUS!)")
        print()

    def print_summary(self):
        """Print formatted metrics summary"""
        metrics = self.compute_metrics()

        print("\n" + "="*60)
        print("VERIFICATION METRICS SUMMARY")
        print("="*60)
        self.print_confusion_matrix()
        print(f"Precision:  {metrics['precision']:.4f}  (TP/(TP+FP))")
        print(f"Recall:     {metrics['recall']:.4f}  (TP/(TP+FN))")
        print(f"F1-Score:   {metrics['f1']:.4f}  (Harmonic mean)")
        print(f"FPR:        {metrics['fpr']:.4f}  ({metrics['fpr']*100:.2f}%) (FP/(FP+TN))")
        print(f"Accuracy:   {metrics['accuracy']:.4f}")
        print("="*60)
        print(f"\nKey Insights:")
        print(f"  • False Alarms (FP): {self.fp} out of {self.fp + self.tn} correct facts")
        print(f"  • Missed Errors (FN): {self.fn} out of {self.tp + self.fn} incorrect facts")
        print(f"  • Total Facts: {metrics['total']}")
        print()


def evaluate_verification_results(
    facts: List[Dict],
    verification_results: List[Dict],
    ground_truth_labels: List[bool]
) -> Dict[str, float]:
    """
    Evaluate verification performance on a dataset

    Args:
        facts: List of fact dictionaries
        verification_results: List of verification result dictionaries from RMMVe
        ground_truth_labels: List of booleans (True = correct fact, False = incorrect)

    Returns:
        Dictionary of metrics
    """
    metrics = VerificationMetrics()

    # Extract decisions from verification results
    predicted_decisions = [result['decision'] for result in verification_results]

    # Update metrics
    metrics.update(ground_truth_labels, predicted_decisions)

    # Print summary
    metrics.print_summary()

    return metrics.compute_metrics()


def compare_configurations(
    config_results: Dict[str, Dict[str, float]],
    metric: str = 'f1'
) -> None:
    """
    Compare metrics across different configurations

    Args:
        config_results: Dict mapping config names to metric dictionaries
        metric: Which metric to use for comparison
    """
    print(f"\n{'='*70}")
    print(f"CONFIGURATION COMPARISON (sorted by {metric.upper()})")
    print(f"{'='*70}")
    print(f"{'Configuration':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FPR':>10}")
    print(f"{'-'*70}")

    # Sort by specified metric (descending)
    sorted_configs = sorted(
        config_results.items(),
        key=lambda x: x[1][metric],
        reverse=True
    )

    for config_name, metrics in sorted_configs:
        print(f"{config_name:<25} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
              f"{metrics['f1']:>10.4f} {metrics['fpr']:>10.4f}")

    print(f"{'='*70}\n")
