"""
ATLASky-AI Verification Demo
Demonstrates the complete verification pipeline with different fact qualities
"""

from models.knowledge_graph import create_sample_knowledge_graph
from verification.rmmve import RMMVeProcess
from aaic.aaic import AAIC
from data.generators import generate_candidate_fact_with_quality
import time

def print_header(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def print_result(result, candidate_fact):
    """Print verification result in a readable format"""
    print(f"\nCandidate Fact Quality: {result['fact_quality']}")
    print(f"Decision: {'✅ ACCEPT' if result['decision'] else '❌ REJECT'}")
    print(f"Cumulative Confidence: {result['total_confidence']:.4f} (Threshold: 0.65)")
    print(f"Early Termination: {result['early_termination']}")
    if result['early_termination']:
        print(f"Terminated at Module: {result.get('early_termination_module', 'N/A')}")
    print(f"Verification Time: {result['verification_time']*1000:.2f} ms")
    print(f"\nActivated Modules: {', '.join(result['activated_modules'])}")
    
    print("\nModule Scores:")
    for module_result in result['module_results']:
        name = module_result['module_name']
        conf = module_result['confidence']
        thresh = module_result['threshold']
        m1 = module_result['metric1']
        m2 = module_result['metric2']
        activated = "✓" if conf >= thresh else "✗"
        print(f"  {name}: {conf:.4f} (threshold: {thresh:.2f}) [{activated}]")
        print(f"       Metric1: {m1:.4f} | Metric2: {m2:.4f}")

def main():
    print_header("ATLASky-AI Verification System Demo")
    print("\nThis demo showcases the updated methodology with:")
    print("  • Physical consistency predicates (ψ_s, ψ_t, Ψ)")
    print("  • Formal STKG definition (G = V, E, O, T, Ψ)")
    print("  • Adaptive parameter tuning (AAIC with CGR-CUSUM)")
    print("  • Five-layer Defense-in-Depth verification (LOV→POV→MAV→WSV→ESV)")
    
    # Initialize system
    kg = create_sample_knowledge_graph()
    rmmve = RMMVeProcess(global_threshold=0.65)
    aaic = AAIC(rmmve, h=1.5, k=0.05, gamma=0.01, eta=0.05, eta_prime=0.02)
    
    print(f"\nKnowledge Graph: {len(kg.entities)} entities, {len(kg.relationships)} relationships")
    print(f"Verification Modules: {len(rmmve.modules)}")
    print(f"  • LOV (Lexical-Ontological Verification)")
    print(f"  • POV (Protocol-Ontology Verification)")
    print(f"  • MAV (Motion-Aware Verification) - using ψ_s, ψ_t predicates")
    print(f"  • WSV (Web-Source Verification)")
    print(f"  • ESV (Embedding Similarity Verification)")
    
    # Test different quality levels
    test_cases = [
        ("high_quality", "High quality fact - should pass early at LOV/POV"),
        ("medium_quality", "Medium quality - needs multiple modules"),
        ("spatial_issue", "Spatial/temporal issue - should be caught by MAV"),
        ("low_quality", "Low quality fact - should be rejected"),
    ]
    
    for quality, description in test_cases:
        print_header(f"Test Case: {quality.upper()}")
        print(f"Description: {description}")
        
        # Generate candidate fact (returns tuple: fact, quality_level)
        candidate_fact, _ = generate_candidate_fact_with_quality(quality)
        print(f"\nGenerated fact structure:")
        print(f"  Report ID: {candidate_fact['spatiotemporal_inspection_data']['report_id']}")
        
        # Extract key information
        inspection_data = candidate_fact['spatiotemporal_inspection_data']
        if 'inspection_data' in inspection_data:
            measurements = inspection_data['inspection_data']['inspection_measurements']
            if measurements:
                m = measurements[0]
                print(f"  Component: {m.get('component_id', 'N/A')}")
                print(f"  Feature: {m.get('feature_name', 'N/A')}")
                print(f"  Deviation: {m.get('deviation_mm', 'N/A')} mm")
        
        if 'spatial_data' in inspection_data and inspection_data['spatial_data']:
            coords = inspection_data['spatial_data'][0].get('coordinates', {})
            print(f"  Location: ({coords.get('x_coord', 0):.1f}, {coords.get('y_coord', 0):.1f}, {coords.get('z_coord', 0):.1f})")
        
        if 'temporal_data' in inspection_data and inspection_data['temporal_data']:
            print(f"  Timestamp: {inspection_data['temporal_data'][0].get('timestamp', 'N/A')}")
        
        # Run verification
        start_time = time.time()
        result = rmmve.verify(candidate_fact, kg, quality)
        end_time = time.time()
        
        # Print results
        print_result(result, candidate_fact)
        
        # Update AAIC (simulate performance monitoring)
        for module in rmmve.modules:
            if module.name in result['confidence_scores']:
                module.performance_history.append({
                    'metric1': result['metrics'][module.name]['metric1'],
                    'metric2': result['metrics'][module.name]['metric2']
                })
        
        time.sleep(0.5)  # Brief pause between tests
    
    # Demonstrate AAIC adaptation
    print_header("AAIC Adaptive Parameter Updates")
    print("\nCurrent Module Parameters:")
    for module in rmmve.modules:
        print(f"\n{module.name}:")
        print(f"  Weight (w_i): {module.weight:.4f}")
        print(f"  Threshold (θ_i): {module.threshold:.4f}")
        print(f"  Alpha (α_i): {module.alpha:.4f}")
        print(f"  CUSUM Stat (G_i): {aaic.cumulative_sums[module.name]:.4f}")
    
    # Simulate performance monitoring and updates
    print("\nSimulating performance monitoring...")
    for i in range(3):
        for module in rmmve.modules:
            # Simulate some performance variation
            perf = 0.75 + (i * 0.05)
            module.performance_history.append({'metric1': perf, 'metric2': perf})
        
        updates = aaic.update_all_modules()
        
        for update in updates:
            if update.get('detected', False):
                print(f"\n⚠ Performance shift detected in {update['module']}!")
                print(f"  CUSUM: {update['cumulative_sum']:.4f} > threshold (h={aaic.h})")
                print(f"  Parameter updates applied:")
                print(f"    Weight: {update['old_params']['weight']:.4f} → {update['new_params']['weight']:.4f}")
                print(f"    Threshold: {update['old_params']['threshold']:.4f} → {update['new_params']['threshold']:.4f}")
                print(f"    Alpha: {update['old_params']['alpha']:.4f} → {update['new_params']['alpha']:.4f}")
    
    print_header("Demo Complete")
    print("\nThe system demonstrated:")
    print("  ✓ Multi-modal fact verification with early termination")
    print("  ✓ Physical consistency checking using ψ_s and ψ_t predicates")
    print("  ✓ Different error detection across quality levels")
    print("  ✓ Adaptive parameter tuning via AAIC")
    print("  ✓ CGR-CUSUM performance monitoring")
    print("\nTo run the interactive dashboard:")
    print("  $ streamlit run app.py")
    print("\nFor detailed methodology, see the updated preliminaries section.")

if __name__ == "__main__":
    main()
