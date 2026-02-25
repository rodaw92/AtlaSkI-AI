"""
Defense-in-Depth Architecture Analysis for ATLASky-AI

Implements the Defense-in-Depth analysis from Section 4.3 of the paper,
demonstrating how the five-layer verification achieves robustness through:
1. Independence: Modules operate on distinct information sources
2. Complementarity: Modules target different error classes
3. Redundancy: Multiple modules provide overlapping coverage
"""

from typing import Dict, List, Set, Any, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModuleCharacteristics:
    """Characteristics of a verification module for defense-in-depth analysis."""
    name: str
    information_sources: Set[str]
    target_error_classes: Set[str]
    backup_modules: List[str]
    computational_cost: float  # Relative cost (1.0 = baseline)
    typical_false_positive_rate: float
    typical_false_negative_rate: float


class DefenseInDepthAnalyzer:
    """
    Analyzes the Defense-in-Depth properties of the verification architecture.
    
    Demonstrates:
    1. Independence through distinct information sources
    2. Complementarity through error class targeting
    3. Redundancy through overlapping coverage
    """
    
    def __init__(self):
        # Define module characteristics based on paper methodology
        self.modules = {
            "LOV": ModuleCharacteristics(
                name="LOV",
                information_sources={"domain_ontology", "entity_classes", "relation_types"},
                target_error_classes={"semantic_drift", "ontology_violation"},
                backup_modules=["ESV"],
                computational_cost=5.0,      # 5 ms (Table 1)
                typical_false_positive_rate=0.08,
                typical_false_negative_rate=0.12
            ),
            "POV": ModuleCharacteristics(
                name="POV",
                information_sources={"industry_standards", "protocol_libraries", "terminology"},
                target_error_classes={"content_hallucination", "standard_violation"},
                backup_modules=["WSV"],
                computational_cost=15.0,     # 15 ms (Table 1)
                typical_false_positive_rate=0.11,
                typical_false_negative_rate=0.09
            ),
            "MAV": ModuleCharacteristics(
                name="MAV",
                information_sources={"physics_models", "spatiotemporal_coords", "velocity_constraints"},
                target_error_classes={"st_inconsistency", "physics_violation"},
                backup_modules=[],
                computational_cost=50.0,     # 50 ms (Table 1)
                typical_false_positive_rate=0.03,
                typical_false_negative_rate=0.05
            ),
            "WSV": ModuleCharacteristics(
                name="WSV",
                information_sources={"web_sources", "external_databases", "authoritative_apis"},
                target_error_classes={"content_hallucination", "external_contradiction"},
                backup_modules=["POV"],
                computational_cost=120.0,    # 120 ms (Table 1)
                typical_false_positive_rate=0.07,
                typical_false_negative_rate=0.10
            ),
            "ESV": ModuleCharacteristics(
                name="ESV",
                information_sources={"embedding_vectors", "historical_facts", "statistical_models"},
                target_error_classes={"semantic_drift", "statistical_anomaly"},
                backup_modules=["LOV"],
                computational_cost=800.0,    # 800 ms (Table 1)
                typical_false_positive_rate=0.05,
                typical_false_negative_rate=0.08
            )
        }
    
    def analyze_independence(self) -> Dict[str, Any]:
        """
        Analyze Principle 1: Independence
        
        Modules operate on distinct information sources. Single-point failures
        in one module do not compromise others.
        
        Returns:
            Analysis report with independence metrics
        """
        logger.info("Analyzing Independence Principle...")
        
        # Extract all information sources
        all_sources = set()
        module_sources = {}
        
        for name, module in self.modules.items():
            module_sources[name] = module.information_sources
            all_sources.update(module.information_sources)
        
        # Calculate source overlap matrix
        overlap_matrix = {}
        for m1_name, m1_sources in module_sources.items():
            overlap_matrix[m1_name] = {}
            for m2_name, m2_sources in module_sources.items():
                if m1_name == m2_name:
                    overlap_matrix[m1_name][m2_name] = 1.0
                else:
                    # Jaccard similarity
                    intersection = len(m1_sources & m2_sources)
                    union = len(m1_sources | m2_sources)
                    overlap_matrix[m1_name][m2_name] = intersection / union if union > 0 else 0.0
        
        # Calculate average independence (lower overlap = higher independence)
        total_overlap = 0
        comparisons = 0
        for m1 in self.modules.keys():
            for m2 in self.modules.keys():
                if m1 != m2:
                    total_overlap += overlap_matrix[m1][m2]
                    comparisons += 1
        
        average_overlap = total_overlap / comparisons if comparisons > 0 else 0
        independence_score = 1.0 - average_overlap  # Higher = more independent
        
        analysis = {
            "principle": "Independence",
            "total_information_sources": len(all_sources),
            "module_sources": {name: list(sources) for name, sources in module_sources.items()},
            "overlap_matrix": overlap_matrix,
            "average_overlap": average_overlap,
            "independence_score": independence_score,
            "interpretation": self._interpret_independence(independence_score),
            "single_point_failure_risk": "LOW" if independence_score > 0.7 else "MEDIUM" if independence_score > 0.4 else "HIGH"
        }
        
        logger.info(f"  Independence Score: {independence_score:.3f}")
        logger.info(f"  Single-Point Failure Risk: {analysis['single_point_failure_risk']}")
        
        return analysis
    
    def analyze_complementarity(self) -> Dict[str, Any]:
        """
        Analyze Principle 2: Complementarity
        
        Modules target different error classes. Errors that evade one module
        are caught by others through orthogonal validation approaches.
        
        Returns:
            Analysis report with complementarity metrics
        """
        logger.info("Analyzing Complementarity Principle...")
        
        # Extract all error classes
        all_error_classes = set()
        module_targets = {}
        
        for name, module in self.modules.items():
            module_targets[name] = module.target_error_classes
            all_error_classes.update(module.target_error_classes)
        
        # Build error class coverage matrix
        coverage_matrix = {}
        for error_class in all_error_classes:
            coverage_matrix[error_class] = []
            for name, module in self.modules.items():
                if error_class in module.target_error_classes:
                    coverage_matrix[error_class].append(name)
        
        # Calculate complementarity metrics
        single_coverage = sum(1 for modules in coverage_matrix.values() if len(modules) == 1)
        dual_coverage = sum(1 for modules in coverage_matrix.values() if len(modules) == 2)
        multi_coverage = sum(1 for modules in coverage_matrix.values() if len(modules) > 2)
        
        # Calculate target diversity (how many different error classes each module targets)
        target_diversity = {
            name: len(module.target_error_classes)
            for name, module in self.modules.items()
        }
        
        avg_targets_per_module = sum(target_diversity.values()) / len(target_diversity)
        
        # Calculate complementarity score
        # High score = good distribution of targets across modules
        complementarity_score = (dual_coverage + 2 * multi_coverage) / len(all_error_classes)
        
        analysis = {
            "principle": "Complementarity",
            "total_error_classes": len(all_error_classes),
            "error_classes": list(all_error_classes),
            "coverage_matrix": {error: modules for error, modules in coverage_matrix.items()},
            "module_targets": {name: list(targets) for name, targets in module_targets.items()},
            "single_coverage_errors": single_coverage,
            "dual_coverage_errors": dual_coverage,
            "multi_coverage_errors": multi_coverage,
            "avg_targets_per_module": avg_targets_per_module,
            "complementarity_score": complementarity_score,
            "interpretation": self._interpret_complementarity(complementarity_score),
            "evasion_examples": self._generate_evasion_examples()
        }
        
        logger.info(f"  Complementarity Score: {complementarity_score:.3f}")
        logger.info(f"  Error classes with single coverage: {single_coverage}")
        logger.info(f"  Error classes with dual coverage: {dual_coverage}")
        
        return analysis
    
    def analyze_redundancy(self) -> Dict[str, Any]:
        """
        Analyze Principle 3: Redundancy
        
        Multiple modules provide overlapping coverage. Backup modules maintain
        detection capability even if primary modules fail.
        
        Returns:
            Analysis report with redundancy metrics
        """
        logger.info("Analyzing Redundancy Principle...")
        
        # Build redundancy chains
        redundancy_chains = {}
        for name, module in self.modules.items():
            redundancy_chains[name] = {
                "primary_targets": list(module.target_error_classes),
                "backups": module.backup_modules,
                "has_backup": len(module.backup_modules) > 0
            }
        
        # Calculate backup coverage
        modules_with_backup = sum(1 for chain in redundancy_chains.values() if chain["has_backup"])
        backup_coverage_rate = modules_with_backup / len(self.modules)
        
        # Identify critical single points (no backup)
        critical_modules = [
            name for name, chain in redundancy_chains.items()
            if not chain["has_backup"]
        ]
        
        # Calculate overlapping error detection
        error_redundancy = {}
        for name, module in self.modules.items():
            for error_class in module.target_error_classes:
                if error_class not in error_redundancy:
                    error_redundancy[error_class] = []
                error_redundancy[error_class].append(name)
        
        # Count error classes with redundant detection
        redundant_errors = sum(1 for detectors in error_redundancy.values() if len(detectors) >= 2)
        redundancy_rate = redundant_errors / len(error_redundancy) if error_redundancy else 0
        
        # Calculate redundancy score
        # Higher score = more backup coverage and redundant detection
        redundancy_score = (backup_coverage_rate + redundancy_rate) / 2
        
        analysis = {
            "principle": "Redundancy",
            "redundancy_chains": redundancy_chains,
            "modules_with_backup": modules_with_backup,
            "backup_coverage_rate": backup_coverage_rate,
            "critical_modules": critical_modules,
            "error_redundancy": {error: detectors for error, detectors in error_redundancy.items()},
            "redundant_error_classes": redundant_errors,
            "redundancy_rate": redundancy_rate,
            "redundancy_score": redundancy_score,
            "interpretation": self._interpret_redundancy(redundancy_score),
            "failure_scenarios": self._generate_failure_scenarios(critical_modules)
        }
        
        logger.info(f"  Redundancy Score: {redundancy_score:.3f}")
        logger.info(f"  Backup Coverage Rate: {backup_coverage_rate:.3f}")
        logger.info(f"  Critical Modules (no backup): {critical_modules}")
        
        return analysis
    
    def analyze_computational_efficiency(self) -> Dict[str, Any]:
        """
        Analyze computational efficiency of sequential ordering.
        
        Sequential ordering prioritizes computational efficiency by executing
        lowest-cost modules first with early termination capabilities.
        
        Returns:
            Analysis report with efficiency metrics
        """
        logger.info("Analyzing Computational Efficiency...")
        
        # Get modules in execution order
        module_order = ["LOV", "POV", "MAV", "WSV", "ESV"]
        
        # Calculate cumulative cost
        cumulative_costs = []
        total_cost = 0
        for name in module_order:
            total_cost += self.modules[name].computational_cost
            cumulative_costs.append({
                "module": name,
                "module_cost": self.modules[name].computational_cost,
                "cumulative_cost": total_cost
            })
        
        # Calculate cost distribution
        costs = [self.modules[name].computational_cost for name in module_order]
        avg_cost = sum(costs) / len(costs)
        
        # Calculate early termination benefit
        # If facts terminate at each stage, what's the average cost?
        early_term_scenarios = []
        for i, name in enumerate(module_order):
            scenario_cost = sum(self.modules[m].computational_cost for m in module_order[:i+1])
            early_term_scenarios.append({
                "terminates_at": name,
                "cost": scenario_cost,
                "savings": total_cost - scenario_cost,
                "savings_percent": (1 - scenario_cost / total_cost) * 100
            })
        
        analysis = {
            "module_execution_order": module_order,
            "module_costs": {name: self.modules[name].computational_cost for name in module_order},
            "cumulative_costs": cumulative_costs,
            "total_cost": total_cost,
            "average_cost_per_module": avg_cost,
            "early_termination_scenarios": early_term_scenarios,
            "interpretation": self._interpret_efficiency(costs)
        }
        
        logger.info(f"  Total Cost (all modules): {total_cost:.1f}x")
        logger.info(f"  Average Cost (with early termination): {sum(s['cost'] for s in early_term_scenarios) / len(early_term_scenarios):.1f}x")
        
        return analysis
    
    def generate_full_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive Defense-in-Depth analysis report.
        
        Returns:
            Complete analysis covering all three principles plus efficiency
        """
        logger.info("Generating Defense-in-Depth Analysis Report...")
        
        independence = self.analyze_independence()
        complementarity = self.analyze_complementarity()
        redundancy = self.analyze_redundancy()
        efficiency = self.analyze_computational_efficiency()
        
        # Calculate overall defense-in-depth score
        overall_score = (
            independence["independence_score"] * 0.35 +
            complementarity["complementarity_score"] * 0.35 +
            redundancy["redundancy_score"] * 0.30
        )
        
        report = {
            "defense_in_depth_analysis": {
                "overall_score": overall_score,
                "overall_rating": self._rate_overall(overall_score),
                "principles": {
                    "independence": independence,
                    "complementarity": complementarity,
                    "redundancy": redundancy
                },
                "computational_efficiency": efficiency,
                "summary": self._generate_summary(independence, complementarity, redundancy, overall_score)
            }
        }
        
        logger.info(f"Overall Defense-in-Depth Score: {overall_score:.3f}")
        logger.info(f"Rating: {report['defense_in_depth_analysis']['overall_rating']}")
        
        return report
    
    # Helper methods
    
    def _interpret_independence(self, score: float) -> str:
        """Interpret independence score."""
        if score > 0.8:
            return "EXCELLENT - Modules are highly independent with minimal source overlap"
        elif score > 0.6:
            return "GOOD - Modules have reasonable independence with limited shared sources"
        elif score > 0.4:
            return "MODERATE - Some modules share information sources"
        else:
            return "POOR - High information source overlap between modules"
    
    def _interpret_complementarity(self, score: float) -> str:
        """Interpret complementarity score."""
        if score > 0.8:
            return "EXCELLENT - Error classes have strong multi-module coverage"
        elif score > 0.6:
            return "GOOD - Most error classes targeted by multiple modules"
        elif score > 0.4:
            return "MODERATE - Some error classes have limited coverage"
        else:
            return "POOR - Many error classes only detected by single modules"
    
    def _interpret_redundancy(self, score: float) -> str:
        """Interpret redundancy score."""
        if score > 0.8:
            return "EXCELLENT - Strong backup coverage for most modules"
        elif score > 0.6:
            return "GOOD - Adequate backup coverage with few single points"
        elif score > 0.4:
            return "MODERATE - Some critical modules lack backups"
        else:
            return "POOR - Limited backup coverage, high single-point failure risk"
    
    def _interpret_efficiency(self, costs: List[float]) -> str:
        """Interpret computational efficiency ordering."""
        if costs == sorted(costs):
            return "OPTIMAL - Modules ordered by increasing computational cost"
        else:
            return "SUBOPTIMAL - Module ordering could be improved for efficiency"
    
    def _rate_overall(self, score: float) -> str:
        """Rate overall defense-in-depth architecture."""
        if score > 0.8:
            return "EXCELLENT"
        elif score > 0.6:
            return "GOOD"
        elif score > 0.4:
            return "MODERATE"
        else:
            return "NEEDS IMPROVEMENT"
    
    def _generate_evasion_examples(self) -> List[Dict[str, str]]:
        """Generate examples of how errors evade certain modules but are caught by others."""
        return [
            {
                "error_type": "ontology_compliant_fabrication",
                "evades": ["LOV", "MAV"],
                "caught_by": ["POV", "WSV"],
                "explanation": "Fact uses correct ontology terms but fabricates content, caught by external evidence checks"
            },
            {
                "error_type": "physics_violation",
                "evades": ["LOV", "POV", "WSV", "ESV"],
                "caught_by": ["MAV"],
                "explanation": "Fact passes semantic validation but violates velocity constraints"
            },
            {
                "error_type": "semantic_drift",
                "evades": ["POV", "MAV", "WSV"],
                "caught_by": ["LOV", "ESV"],
                "explanation": "Fact diverges from domain ontology, detected by both ontology and embedding modules"
            },
            {
                "error_type": "hallucination",
                "evades": ["LOV", "MAV"],
                "caught_by": ["POV", "WSV"],
                "explanation": "Fabricated content with no external corroboration"
            }
        ]
    
    def _generate_failure_scenarios(self, critical_modules: List[str]) -> List[Dict[str, str]]:
        """Generate failure scenarios for critical modules without backups."""
        scenarios = []
        for module in critical_modules:
            module_info = self.modules[module]
            scenarios.append({
                "module": module,
                "targets": list(module_info.target_error_classes),
                "risk": "HIGH" if len(module_info.target_error_classes) > 1 else "MEDIUM",
                "mitigation": f"If {module} fails, no backup module can detect {', '.join(module_info.target_error_classes)}",
                "recommendation": f"Consider adding redundant detection for {module}'s error classes"
            })
        return scenarios
    
    def _generate_summary(self, independence: Dict, complementarity: Dict, 
                         redundancy: Dict, overall_score: float) -> str:
        """Generate executive summary of defense-in-depth analysis."""
        return f"""
The ATLASky-AI verification architecture achieves {self._rate_overall(overall_score)} defense-in-depth 
(score: {overall_score:.3f}) through:

1. INDEPENDENCE ({independence['independence_score']:.3f}): {independence['interpretation']}
   - {independence['total_information_sources']} distinct information sources across modules
   - Single-point failure risk: {independence['single_point_failure_risk']}

2. COMPLEMENTARITY ({complementarity['complementarity_score']:.3f}): {complementarity['interpretation']}
   - {complementarity['total_error_classes']} error classes targeted
   - {complementarity['dual_coverage_errors']} errors with dual coverage
   - {complementarity['multi_coverage_errors']} errors with multi-module coverage

3. REDUNDANCY ({redundancy['redundancy_score']:.3f}): {redundancy['interpretation']}
   - {redundancy['modules_with_backup']}/{len(self.modules)} modules have backup coverage
   - {redundancy['redundant_error_classes']} error classes have redundant detection
   - Critical modules: {', '.join(redundancy['critical_modules']) if redundancy['critical_modules'] else 'None'}

Sequential ordering prioritizes computational efficiency with early termination,
reducing average cost while maintaining comprehensive verification coverage.
        """.strip()


def print_defense_analysis(report: Dict[str, Any], detail_level: str = "summary"):
    """
    Pretty-print defense-in-depth analysis report.
    
    Args:
        report: Report dictionary from DefenseInDepthAnalyzer
        detail_level: 'summary', 'detailed', or 'full'
    """
    analysis = report["defense_in_depth_analysis"]
    
    print("\n" + "="*80)
    print("DEFENSE-IN-DEPTH ARCHITECTURE ANALYSIS")
    print("="*80)
    
    print(f"\nOverall Score: {analysis['overall_score']:.3f} ({analysis['overall_rating']})")
    
    if detail_level in ["summary", "detailed", "full"]:
        print("\n" + "-"*80)
        print("EXECUTIVE SUMMARY")
        print("-"*80)
        print(analysis['summary'])
    
    if detail_level in ["detailed", "full"]:
        print("\n" + "-"*80)
        print("PRINCIPLE 1: INDEPENDENCE")
        print("-"*80)
        ind = analysis['principles']['independence']
        print(f"Score: {ind['independence_score']:.3f}")
        print(f"Interpretation: {ind['interpretation']}")
        print(f"Total Information Sources: {ind['total_information_sources']}")
        print(f"Single-Point Failure Risk: {ind['single_point_failure_risk']}")
        
        print("\n" + "-"*80)
        print("PRINCIPLE 2: COMPLEMENTARITY")
        print("-"*80)
        comp = analysis['principles']['complementarity']
        print(f"Score: {comp['complementarity_score']:.3f}")
        print(f"Interpretation: {comp['interpretation']}")
        print(f"Total Error Classes: {comp['total_error_classes']}")
        print(f"  - Single coverage: {comp['single_coverage_errors']}")
        print(f"  - Dual coverage: {comp['dual_coverage_errors']}")
        print(f"  - Multi coverage: {comp['multi_coverage_errors']}")
        
        print("\n" + "-"*80)
        print("PRINCIPLE 3: REDUNDANCY")
        print("-"*80)
        red = analysis['principles']['redundancy']
        print(f"Score: {red['redundancy_score']:.3f}")
        print(f"Interpretation: {red['interpretation']}")
        print(f"Backup Coverage Rate: {red['backup_coverage_rate']:.1%}")
        print(f"Critical Modules: {', '.join(red['critical_modules']) if red['critical_modules'] else 'None'}")
    
    if detail_level == "full":
        print("\n" + "-"*80)
        print("EVASION EXAMPLES")
        print("-"*80)
        for example in analysis['principles']['complementarity']['evasion_examples']:
            print(f"\n{example['error_type']}:")
            print(f"  Evades: {', '.join(example['evades'])}")
            print(f"  Caught by: {', '.join(example['caught_by'])}")
            print(f"  Explanation: {example['explanation']}")
        
        print("\n" + "-"*80)
        print("COMPUTATIONAL EFFICIENCY")
        print("-"*80)
        eff = analysis['computational_efficiency']
        print(f"Execution Order: {' → '.join(eff['module_execution_order'])}")
        print(f"Total Cost (all modules): {eff['total_cost']:.1f}x")
        print("\nEarly Termination Scenarios:")
        for scenario in eff['early_termination_scenarios']:
            print(f"  Terminate at {scenario['terminates_at']}: "
                  f"{scenario['cost']:.1f}x (saves {scenario['savings_percent']:.1f}%)")
    
    print("\n" + "="*80 + "\n")

