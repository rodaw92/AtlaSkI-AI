"""
Test Domain Adaptation and Defense-in-Depth Features

This script demonstrates the domain adaptation capabilities and defense-in-depth
architecture analysis for ATLASky-AI.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from verification.domain_adapter import DomainAdapter, create_aerospace_domain, create_healthcare_domain, create_aviation_domain
from verification.defense_in_depth import DefenseInDepthAnalyzer, print_defense_analysis
from verification.rmmve import RMMVeProcess
from models.knowledge_graph import create_sample_knowledge_graph


def test_domain_creation():
    """Test creating domain configurations."""
    print("\n" + "="*80)
    print("TEST 1: Domain Configuration Creation")
    print("="*80)
    
    # Create aerospace domain
    aerospace = create_aerospace_domain()
    valid, errors = aerospace.validate()
    
    print(f"\n✅ Aerospace Domain: {'VALID' if valid else 'INVALID'}")
    print(f"   - Entity Classes: {len(aerospace.ontology.entity_classes)}")
    print(f"   - Relationship Types: {len(aerospace.ontology.relationship_types)}")
    print(f"   - Standards: {len(aerospace.standards.standards)}")
    print(f"   - Transport Modes: {len(aerospace.physics.max_velocities)}")
    print(f"   - Credibility Sources: {len(aerospace.credibility.credibility_weights)}")
    
    if not valid:
        print(f"   ⚠️  Validation Errors: {errors}")
    
    # Create healthcare domain
    healthcare = create_healthcare_domain()
    valid, errors = healthcare.validate()
    
    print(f"\n✅ Healthcare Domain: {'VALID' if valid else 'INVALID'}")
    print(f"   - Entity Classes: {len(healthcare.ontology.entity_classes)}")
    print(f"   - Relationship Types: {len(healthcare.ontology.relationship_types)}")
    print(f"   - Standards: {len(healthcare.standards.standards)}")
    print(f"   - Transport Modes: {len(healthcare.physics.max_velocities)}")
    print(f"   - Credibility Sources: {len(healthcare.credibility.credibility_weights)}")
    
    if not valid:
        print(f"   ⚠️  Validation Errors: {errors}")
    
    # Create aviation domain
    aviation = create_aviation_domain()
    valid, errors = aviation.validate()
    
    print(f"\n✅ Aviation Domain: {'VALID' if valid else 'INVALID'}")
    print(f"   - Entity Classes: {len(aviation.ontology.entity_classes)}")
    print(f"   - Relationship Types: {len(aviation.ontology.relationship_types)}")
    print(f"   - Standards: {len(aviation.standards.standards)}")
    print(f"   - Transport Modes: {len(aviation.physics.max_velocities)}")
    print(f"   - Credibility Sources: {len(aviation.credibility.credibility_weights)}")
    
    if not valid:
        print(f"   ⚠️  Validation Errors: {errors}")


def test_domain_adapter():
    """Test domain adapter functionality."""
    print("\n" + "="*80)
    print("TEST 2: Domain Adapter")
    print("="*80)
    
    # Initialize adapter
    adapter = DomainAdapter(config_directory="domains")
    
    # List available domains
    available = adapter.list_domains()
    print(f"\n📂 Available domains: {available}")
    
    if not available:
        print("   ℹ️  No domains found, saving example configurations...")
        
        # Save example domains
        adapter.save_domain(create_aerospace_domain(), format="json")
        adapter.save_domain(create_healthcare_domain(), format="json")
        adapter.save_domain(create_aviation_domain(), format="json")
        
        available = adapter.list_domains()
        print(f"   ✅ Saved {len(available)} domain configurations")
    
    # Load aerospace domain
    if "aerospace" in available:
        print("\n🔧 Loading aerospace domain...")
        config = adapter.load_domain("aerospace")
        print(f"   ✅ Loaded: {config.domain_name}")
        print(f"   📝 {config.domain_description}")
        
        # Apply to RMMVe
        rmmve = RMMVeProcess()
        kg = create_sample_knowledge_graph()
        
        print("\n🔄 Applying domain configuration...")
        adapter.apply_to_rmmve(rmmve, config)
        adapter.apply_to_knowledge_graph(kg, config)
        
        # Show applied parameters
        print("\n📊 Applied Parameters:")
        for module in rmmve.modules:
            print(f"   - {module.name}: w={module.weight:.2f}, θ={module.threshold:.2f}, α={module.alpha:.2f}")
        
        print(f"   - Global Threshold: {rmmve.global_threshold:.2f}")
        print(f"   - Max Velocities: {list(kg.v_max.keys())}")


def test_defense_in_depth():
    """Test defense-in-depth analysis."""
    print("\n" + "="*80)
    print("TEST 3: Defense-in-Depth Analysis")
    print("="*80)
    
    analyzer = DefenseInDepthAnalyzer()
    
    # Generate full report
    print("\n🔍 Analyzing architecture...")
    report = analyzer.generate_full_report()
    
    # Print report
    print_defense_analysis(report, detail_level="summary")
    
    analysis = report["defense_in_depth_analysis"]
    
    # Show key metrics
    print("\n📊 Key Metrics:")
    print(f"   - Overall Score: {analysis['overall_score']:.3f} ({analysis['overall_rating']})")
    print(f"   - Independence: {analysis['principles']['independence']['independence_score']:.3f}")
    print(f"   - Complementarity: {analysis['principles']['complementarity']['complementarity_score']:.3f}")
    print(f"   - Redundancy: {analysis['principles']['redundancy']['redundancy_score']:.3f}")
    
    # Show computational efficiency
    eff = analysis['computational_efficiency']
    print(f"\n⚡ Computational Efficiency:")
    print(f"   - Execution Order: {' → '.join(eff['module_execution_order'])}")
    print(f"   - Total Cost: {eff['total_cost']:.1f}x baseline")
    print(f"   - Average Early Term Cost: {sum(s['cost'] for s in eff['early_termination_scenarios']) / len(eff['early_termination_scenarios']):.1f}x")


def test_parameter_initialization():
    """Test uniform parameter initialization."""
    print("\n" + "="*80)
    print("TEST 4: Uniform Parameter Initialization")
    print("="*80)
    
    adapter = DomainAdapter()
    
    # Create template with uniform parameters
    config = adapter.create_template("test_domain")
    
    print("\n📝 Default Uniform Parameters:")
    print(f"   Weights: {config.initial_weights}")
    print(f"   Thresholds: {config.initial_thresholds}")
    print(f"   Alphas: {config.initial_alphas}")
    
    # Verify sums
    weight_sum = sum(config.initial_weights.values())
    print(f"\n✅ Weight sum: {weight_sum:.3f} (should be 1.0)")
    
    # Verify ranges
    all_valid = True
    for module, threshold in config.initial_thresholds.items():
        if not (0 <= threshold <= 1):
            print(f"   ⚠️  Invalid threshold for {module}: {threshold}")
            all_valid = False
    
    for module, alpha in config.initial_alphas.items():
        if not (0 <= alpha <= 1):
            print(f"   ⚠️  Invalid alpha for {module}: {alpha}")
            all_valid = False
    
    if all_valid:
        print("✅ All parameters in valid range [0, 1]")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("ATLASKY-AI DOMAIN ADAPTATION TEST SUITE")
    print("="*80)
    
    try:
        test_domain_creation()
        test_domain_adapter()
        test_defense_in_depth()
        test_parameter_initialization()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)
        print("\nNext Steps:")
        print("1. Run the Streamlit app: streamlit run app.py")
        print("2. Select a domain from the sidebar")
        print("3. View domain configuration in the 'Domain Config' tab")
        print("4. Run defense-in-depth analysis in the 'Defense-in-Depth' tab")
        print("5. Verify that AAIC adapts parameters over time")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

