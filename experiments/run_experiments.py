"""
ATLASky-AI Experimental Evaluation Runner

Runs experiments on different dataset types and reports verification performance metrics.
Usage:
    python3 run_experiments.py --all
    python3 run_experiments.py --dataset manufacturing
    python3 run_experiments.py --dataset aviation --num-facts 200
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.knowledge_graph import create_sample_knowledge_graph
from verification.rmmve import RMMVeProcess
from experiments.metrics.evaluation import VerificationMetrics, evaluate_verification_results
from experiments.datasets import (
    manufacturing_data,
    aviation_data,
    cad_data,
    healthcare_data
)

# Dataset registry
DATASETS = {
    'manufacturing': manufacturing_data,
    'aviation': aviation_data,
    'cad': cad_data,
    'healthcare': healthcare_data
}


def print_header(title):
    """Print section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def run_single_experiment(dataset_name: str, num_facts: int = 100,
                         detailed: bool = False, save_results: bool = True):
    """
    Run experiment on a single dataset type

    Args:
        dataset_name: Name of dataset ('manufacturing', 'aviation', etc.)
        num_facts: Number of facts to generate
        detailed: Whether to print detailed per-fact results
        save_results: Whether to save results to JSON

    Returns:
        Dictionary of metrics
    """
    print_header(f"Dataset: {dataset_name.upper()}")

    # Get dataset module
    if dataset_name not in DATASETS:
        print(f"Error: Unknown dataset '{dataset_name}'")
        print(f"Available: {list(DATASETS.keys())}")
        return None

    dataset_module = DATASETS[dataset_name]

    # Print dataset info
    info = dataset_module.get_dataset_info()
    print(f"\nDataset Information:")
    print(f"  Name: {info['name']}")
    print(f"  Challenge: {info['challenge']}")
    print(f"  Primary Test: {info['primary_test']}")
    print(f"  Error Types:")
    for error_type in info['error_types']:
        print(f"    • {error_type}")
    print(f"  Example: {info['example']}")
    print(f"\nExpected Performance:")
    for metric, value in info['expected_performance'].items():
        print(f"  {metric.upper()}: {value:.4f}")

    # Generate dataset
    print(f"\nGenerating {num_facts} synthetic facts...")
    if dataset_name == 'manufacturing':
        facts, labels = dataset_module.generate_manufacturing_facts(num_facts)
    elif dataset_name == 'aviation':
        facts, labels = dataset_module.generate_aviation_facts(num_facts)
    elif dataset_name == 'cad':
        facts, labels = dataset_module.generate_cad_facts(num_facts)
    elif dataset_name == 'healthcare':
        facts, labels = dataset_module.generate_healthcare_facts(num_facts)

    # Create knowledge graph
    kg = create_sample_knowledge_graph()

    # Initialize verification system
    rmmve = RMMVeProcess(global_threshold=0.65)

    print(f"Running verification on {len(facts)} facts...")
    start_time = time.time()

    # Run verification
    verification_results = []
    for i, fact in enumerate(facts):
        if detailed and i < 5:  # Show first 5 for detailed mode
            print(f"\n  Fact {i+1}/{len(facts)}: {fact.get('subject_entity_id', 'N/A')}")

        result = rmmve.verify(fact, kg, fact_quality=None)
        verification_results.append(result)

        if detailed and i < 5:
            print(f"    Decision: {'ACCEPT' if result['decision'] else 'REJECT'}")
            print(f"    Confidence: {result['total_confidence']:.4f}")
            print(f"    Activated: {', '.join(result['activated_modules'])}")

    end_time = time.time()
    total_time = end_time - start_time

    # Evaluate results
    print_header("EVALUATION RESULTS")
    metrics = evaluate_verification_results(facts, verification_results, labels)

    # Add timing metrics
    metrics['total_time_seconds'] = total_time
    metrics['avg_time_per_fact_ms'] = (total_time / len(facts)) * 1000
    metrics['facts_per_second'] = len(facts) / total_time

    print(f"\nTiming:")
    print(f"  Total Time: {total_time:.2f} seconds")
    print(f"  Avg Per Fact: {metrics['avg_time_per_fact_ms']:.2f} ms")
    print(f"  Throughput: {metrics['facts_per_second']:.1f} facts/sec")

    # Module activation statistics
    module_activations = {}
    for result in verification_results:
        for module_name in result['activated_modules']:
            module_activations[module_name] = module_activations.get(module_name, 0) + 1

    print(f"\nModule Activation Rates:")
    for module, count in sorted(module_activations.items()):
        rate = (count / len(facts)) * 100
        print(f"  {module}: {count}/{len(facts)} ({rate:.1f}%)")

    # Early termination statistics
    early_term_count = sum(1 for r in verification_results if r['early_termination'])
    early_term_rate = (early_term_count / len(facts)) * 100
    print(f"\nEarly Termination:")
    print(f"  Count: {early_term_count}/{len(facts)} ({early_term_rate:.1f}%)")

    # Save results if requested
    if save_results:
        output_dir = Path(__file__).parent / 'results'
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f'{dataset_name}_results.json'

        results_data = {
            'dataset': dataset_name,
            'num_facts': len(facts),
            'metrics': metrics,
            'module_activations': module_activations,
            'early_termination_rate': early_term_rate,
            'dataset_info': info
        }

        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"\nResults saved to: {output_file}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Run ATLASky-AI experiments on different dataset types'
    )
    parser.add_argument(
        '--dataset',
        choices=list(DATASETS.keys()),
        help='Dataset type to run (manufacturing, aviation, cad, healthcare)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run experiments on all dataset types'
    )
    parser.add_argument(
        '--num-facts',
        type=int,
        default=100,
        help='Number of facts to generate per dataset (default: 100)'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Print detailed per-fact results'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/',
        help='Output directory for results (default: results/)'
    )

    args = parser.parse_args()

    print_header("ATLASky-AI Experimental Evaluation")
    print("Testing verification performance across different dataset types")

    # Run experiments
    all_metrics = {}

    if args.all:
        # Run on all datasets
        for dataset_name in DATASETS.keys():
            metrics = run_single_experiment(
                dataset_name,
                num_facts=args.num_facts,
                detailed=args.detailed
            )
            if metrics:
                all_metrics[dataset_name] = metrics

    elif args.dataset:
        # Run on single dataset
        metrics = run_single_experiment(
            args.dataset,
            num_facts=args.num_facts,
            detailed=args.detailed
        )
        if metrics:
            all_metrics[args.dataset] = metrics

    else:
        parser.print_help()
        return

    # Print summary if multiple datasets
    if len(all_metrics) > 1:
        print_header("SUMMARY: All Datasets")
        print(f"\n{'Dataset':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FPR':>10}")
        print("-" * 70)
        for dataset, metrics in all_metrics.items():
            print(f"{dataset:<20} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
                  f"{metrics['f1']:>10.4f} {metrics['fpr']:>10.4f}")

        # Calculate averages
        avg_precision = sum(m['precision'] for m in all_metrics.values()) / len(all_metrics)
        avg_recall = sum(m['recall'] for m in all_metrics.values()) / len(all_metrics)
        avg_f1 = sum(m['f1'] for m in all_metrics.values()) / len(all_metrics)
        avg_fpr = sum(m['fpr'] for m in all_metrics.values()) / len(all_metrics)

        print("-" * 70)
        print(f"{'AVERAGE':<20} {avg_precision:>10.4f} {avg_recall:>10.4f} "
              f"{avg_f1:>10.4f} {avg_fpr:>10.4f}")
        print("=" * 70)

    print("\n✓ Experiments completed successfully!")
    print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
