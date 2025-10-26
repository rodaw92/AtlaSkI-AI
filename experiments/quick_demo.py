"""
Quick Demonstration: How to Run Experiments on ATLASky-AI

This script shows how to evaluate ATLASky-AI on different fact quality types
and measure the metrics (Precision, Recall, F1, FPR) as discussed in the paper.

This demonstrates the experimental methodology without requiring full dataset implementation.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.knowledge_graph import create_sample_knowledge_graph
from verification.rmmve import RMMVeProcess
from data.generators import generate_candidate_fact_with_quality
from experiments.metrics.evaluation import VerificationMetrics

def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def run_quality_experiment(kg, rmmve, quality_type: str, num_facts: int = 50):
    """
    Run experiment on facts of a specific quality type

    Args:
        kg: Knowledge graph
        rmmve: Verification system
        quality_type: Quality level to generate
        num_facts: Number of facts to test

    Returns:
        metrics dict
    """
    print_header(f"Testing Quality Type: {quality_type.upper()}")

    # Define what "correct" means for each quality type
    # In real evaluation, this would come from human annotations
    quality_correctness = {
        'high_quality': True,      # These are correct facts
        'medium_quality': True,    # These are correct facts
        'spatial_issue': False,    # These have errors
        'external_ref': True,      # These are correct (need validation)
        'semantic_issue': False,   # These have semantic drift
        'low_quality': False       # These are clearly incorrect
    }

    is_quality_correct = quality_correctness.get(quality_type, True)

    print(f"\nGenerating {num_facts} facts with quality: {quality_type}")
    print(f"Expected correctness: {is_quality_correct}")

    # Initialize metrics
    metrics_calculator = VerificationMetrics()

    # Generate and verify facts
    accept_count = 0
    reject_count = 0

    for i in range(num_facts):
        # Generate fact
        fact, _ = generate_candidate_fact_with_quality(quality_type, introduce_shift=False)

        # Run verification
        result = rmmve.verify(fact, kg, quality_type)
        decision = result['decision']  # True = Accept, False = Reject

        if decision:
            accept_count += 1
        else:
            reject_count += 1

        # Update metrics
        # ground_truth_label: True if fact is CORRECT
        # predicted_decision: True if ACCEPTED
        metrics_calculator.update([is_quality_correct], [decision])

    # Print results
    print(f"\nVerification Decisions:")
    print(f"  Accepted: {accept_count}/{num_facts} ({accept_count/num_facts*100:.1f}%)")
    print(f"  Rejected: {reject_count}/{num_facts} ({reject_count/num_facts*100:.1f}%)")

    metrics_calculator.print_summary()

    return metrics_calculator.compute_metrics()


def main():
    print_header("ATLASky-AI: Dataset Type Experiment Demonstration")
    print("\nThis demo shows how to evaluate ATLASky-AI on different fact quality types")
    print("and measure the standard metrics: Precision, Recall, F1, FPR\n")
    print("In a full evaluation, you would:")
    print("  1. Load real datasets (AddQual, NASA ASRS, Zenodo CAD, MIMIC-IV)")
    print("  2. Have human-annotated ground truth labels")
    print("  3. Run verification on each dataset")
    print("  4. Calculate metrics to compare performance")
    print("\nThis demo simulates that process using synthetic data with quality labels.")

    # Initialize system
    kg = create_sample_knowledge_graph()
    rmmve = RMMVeProcess(global_threshold=0.65)

    # Test different quality types (simulating different dataset characteristics)
    quality_types = [
        ('high_quality', 'Manufacturing-style: High precision, tight tolerances'),
        ('spatial_issue', 'CAD Assembly-style: Spatial reasoning challenges'),
        ('semantic_issue', 'Aviation-style: Semantic drift in narratives'),
        ('low_quality', 'All types: Clear errors that should be rejected')
    ]

    all_results = {}

    for quality_type, description in quality_types:
        print(f"\n{'-'*80}")
        print(f"Dataset Analogy: {description}")
        print(f"{'-'*80}")

        metrics = run_quality_experiment(kg, rmmve, quality_type, num_facts=50)
        all_results[quality_type] = metrics

    # Print comparison
    print_header("SUMMARY: Performance Across Quality Types")
    print("\nThis simulates Table 2 from the paper (Main Results)\n")
    print(f"{'Quality Type':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FPR':>10}")
    print("-" * 70)

    for quality_type, metrics in all_results.items():
        print(f"{quality_type:<20} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
              f"{metrics['f1']:>10.4f} {metrics['fpr']:>10.4f}")

    # Calculate averages
    avg_precision = sum(m['precision'] for m in all_results.values()) / len(all_results)
    avg_recall = sum(m['recall'] for m in all_results.values()) / len(all_results)
    avg_f1 = sum(m['f1'] for m in all_results.values()) / len(all_results)
    avg_fpr = sum(m['fpr'] for m in all_results.values()) / len(all_results)

    print("-" * 70)
    print(f"{'AVERAGE':<20} {avg_precision:>10.4f} {avg_recall:>10.4f} "
          f"{avg_f1:>10.4f} {avg_fpr:>10.4f}")
    print("=" * 70)

    print("\n" + "=" * 80)
    print("KEY INSIGHTS FROM RESULTS")
    print("=" * 80)
    print("\n1. PRECISION: Fraction of rejected facts that were truly incorrect")
    print("   • High precision = few false alarms (good for production)")
    print("   • Target: >0.90 for safety-critical systems")

    print("\n2. RECALL: Fraction of incorrect facts successfully caught")
    print("   • High recall = catches most errors (critical for safety)")
    print("   • Target: >0.90 to prevent dangerous false negatives")

    print("\n3. F1-SCORE: Harmonic mean balancing precision and recall")
    print("   • Balanced performance indicator")
    print("   • Target: >0.90 for production deployment")

    print("\n4. FPR (False Positive Rate): Fraction of correct facts incorrectly rejected")
    print("   • Critical metric for user trust and review fatigue")
    print("   • Low FPR (<5%) means minimal manual review burden")
    print("   • Paper shows 39-57% FPR reduction vs baselines")

    print("\n" + "=" * 80)
    print("EXTENDING TO REAL DATASETS")
    print("=" * 80)
    print("\nTo run full experiments as in the paper:")
    print("\n1. **Manufacturing/Aerospace** (AddQual-style)")
    print("   • Load 18-month inspection logs")
    print("   • Human-annotate ground truth (correct/incorrect)")
    print("   • Test: Spatial consistency ψ_s, tolerance validation")
    print("   • Expected: P=0.94, R=0.91, F1=0.92, FPR=2.6%")

    print("\n2. **Aviation Safety** (NASA ASRS-style)")
    print("   • Load incident reports from aviation database")
    print("   • Extract structured facts from narratives")
    print("   • Test: Temporal consistency ψ_t, causal relationships")
    print("   • Expected: P=0.93, R=0.94, F1=0.93, FPR=3.2%")

    print("\n3. **CAD Assembly** (Zenodo-style)")
    print("   • Load STEP AP242 assembly models")
    print("   • Generate part-to-part relationship facts")
    print("   • Test: 3D geometric reasoning, spatial ψ_s")
    print("   • Expected: P=0.96, R=0.93, F1=0.94, FPR=2.9%")

    print("\n4. **Clinical/Healthcare** (MIMIC-IV-style)")
    print("   • Load patient transfer event logs")
    print("   • Extract facts with temporal/spatial coordinates")
    print("   • Test: Clinical workflow compliance, temporal ψ_t")
    print("   • Expected: P=0.95, R=0.95, F1=0.95, FPR=4.1%")

    print("\n" + "=" * 80)
    print("FILES IN experiments/ DIRECTORY")
    print("=" * 80)
    print("\n• README.md - Comprehensive experiment guide")
    print("• run_experiments.py - Full experiment runner (needs real data)")
    print("• quick_demo.py - This demo (works with current system)")
    print("• datasets/ - Dataset generators (templates for real data)")
    print("• metrics/ - Evaluation metrics (Precision, Recall, F1, FPR)")
    print("• results/ - Output directory for experiment results")

    print("\n✓ Demo completed successfully!")
    print("\nNext steps:")
    print("  1. Load your own dataset")
    print("  2. Add ground truth labels (correct/incorrect)")
    print("  3. Run: python3 experiments/run_experiments.py --dataset your_data")
    print("  4. Analyze results to measure performance")


if __name__ == '__main__':
    main()
