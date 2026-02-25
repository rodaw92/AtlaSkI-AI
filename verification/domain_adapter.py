"""
Domain Adaptation System for ATLASky-AI

Implements the Domain Adaptation Protocol from Section 4.4 of the paper,
enabling deployment in new domains through configuration of five key components:
1. Domain Ontology (O)
2. Industry Standards (M2)
3. Physical Constraints (M3)
4. Source Credibility (M4)
5. Domain Embeddings (M5)
"""

import json
import yaml
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DomainOntologyConfig:
    """
    Domain Ontology Configuration (Component 1)
    
    Defines entity classes C, relation types R_o, and attributes A
    per Definition 1 (STKG) in the paper.
    
    Typical domains require:
    - 50-200 entity classes
    - 20-50 relation types
    """
    entity_classes: List[Dict[str, Any]] = field(default_factory=list)
    relationship_types: List[Dict[str, Any]] = field(default_factory=list)
    attributes: List[Dict[str, Any]] = field(default_factory=list)
    spatiotemporal_required: bool = True
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate ontology configuration."""
        errors = []
        
        if not self.entity_classes:
            errors.append("At least one entity class required")
        
        if not self.relationship_types:
            errors.append("At least one relationship type required")
        
        if len(self.entity_classes) < 10 or len(self.entity_classes) > 500:
            logger.warning(f"Entity class count ({len(self.entity_classes)}) outside typical range (50-200)")
        
        if len(self.relationship_types) < 5 or len(self.relationship_types) > 100:
            logger.warning(f"Relationship type count ({len(self.relationship_types)}) outside typical range (20-50)")
        
        return len(errors) == 0, errors


@dataclass
class IndustryStandardsConfig:
    """
    Industry Standards Configuration (Component 2)
    
    Loads domain-specific compliance frameworks for M2 (POV module):
    - STEP AP242 (ISO 10303) for aerospace
    - HL7 FHIR for healthcare
    - ISA-95 for manufacturing
    
    Multiple standards can be integrated.
    """
    standards: List[Dict[str, Any]] = field(default_factory=list)
    terminology_sources: List[str] = field(default_factory=list)
    protocol_libraries: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate standards configuration."""
        errors = []
        
        if not self.standards and not self.terminology_sources:
            errors.append("At least one standard or terminology source required")
        
        return len(errors) == 0, errors


@dataclass
class PhysicalConstraintsConfig:
    """
    Physical Constraints Configuration (Component 3)
    
    Specifies parameters for M3 (MAV module):
    - Maximum velocities v_max for transport modes
    - Facility geometry (building layouts, restricted zones)
    - Physics predicates (ψ_s, ψ_t, Ψ)
    """
    max_velocities: Dict[str, float] = field(default_factory=lambda: {
        "manual_handling": 2.0,      # m/s
        "forklift": 5.0,             # m/s
        "automated_conveyor": 1.0,   # m/s
        "drone": 15.0,               # m/s
        "ambulance": 20.0,           # m/s
        "aircraft": 250.0            # m/s
    })
    
    facility_geometry: Dict[str, Any] = field(default_factory=dict)
    restricted_zones: List[Dict[str, Any]] = field(default_factory=list)
    temporal_resolution: float = 1.0  # seconds
    spatial_resolution: float = 0.1   # meters
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate physical constraints configuration."""
        errors = []
        
        if not self.max_velocities:
            errors.append("At least one transport mode velocity required")
        
        for mode, velocity in self.max_velocities.items():
            if velocity <= 0:
                errors.append(f"Invalid velocity for {mode}: {velocity}")
            if velocity > 1000:
                logger.warning(f"Unusually high velocity for {mode}: {velocity} m/s")
        
        return len(errors) == 0, errors


@dataclass
class SourceCredibilityConfig:
    """
    Source Credibility Configuration (Component 4)
    
    Sets credibility weights w_cred,i for M4 (WSV module).
    Uses domain-appropriate hierarchies:
    - Government > Manufacturer > Academic > News > Forums
    
    Top 20% of sources typically receive w >= 0.8
    """
    credibility_weights: Dict[str, float] = field(default_factory=lambda: {
        "government": 1.0,
        "regulatory": 0.95,
        "manufacturer": 0.85,
        "academic": 0.75,
        "industry_standards": 0.90,
        "news": 0.50,
        "forums": 0.30
    })
    
    source_mappings: Dict[str, str] = field(default_factory=dict)
    authoritative_sources: List[str] = field(default_factory=list)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate source credibility configuration."""
        errors = []
        
        if not self.credibility_weights:
            errors.append("At least one credibility weight required")
        
        for source, weight in self.credibility_weights.items():
            if weight < 0 or weight > 1:
                errors.append(f"Invalid weight for {source}: {weight} (must be 0-1)")
        
        # Check that top 20% have weight >= 0.8
        sorted_weights = sorted(self.credibility_weights.values(), reverse=True)
        top_20_percent = sorted_weights[:max(1, len(sorted_weights) // 5)]
        if top_20_percent and min(top_20_percent) < 0.8:
            logger.warning("Top 20% of sources should have weight >= 0.8")
        
        return len(errors) == 0, errors


@dataclass
class DomainEmbeddingsConfig:
    """
    Domain Embeddings Configuration (Component 5)
    
    Configuration for M5 (ESV module):
    - Training corpus requirements (minimum 10K historical facts)
    - Model architecture (sentence-transformers, domain-adapted BERT)
    - Retraining schedule (quarterly as domains evolve)
    """
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    training_corpus_size: int = 10000  # Minimum required
    training_data_path: Optional[str] = None
    retraining_frequency: str = "quarterly"
    similarity_threshold: float = 0.7
    anomaly_detection_threshold: float = 0.3
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate embeddings configuration."""
        errors = []
        
        if self.training_corpus_size < 10000:
            errors.append(f"Training corpus size ({self.training_corpus_size}) below minimum (10,000)")
        
        if self.embedding_dim <= 0:
            errors.append(f"Invalid embedding dimension: {self.embedding_dim}")
        
        if not 0 <= self.similarity_threshold <= 1:
            errors.append(f"Invalid similarity threshold: {self.similarity_threshold}")
        
        if not 0 <= self.anomaly_detection_threshold <= 1:
            errors.append(f"Invalid anomaly detection threshold: {self.anomaly_detection_threshold}")
        
        return len(errors) == 0, errors


@dataclass
class DomainConfiguration:
    """
    Complete Domain Configuration
    
    Aggregates all five components for domain adaptation.
    Initialize with uniform parameters Φ^(0) = (w^(0), θ^(0), α^(0)):
    - w_i = 0.2 (uniform weights)
    - θ_i = 0.5 (uniform thresholds)
    - α_i = 0.5 (uniform alpha)
    
    AAIC then adapts using domain-specific validation samples.
    """
    domain_name: str
    domain_description: str
    
    # Five components
    ontology: DomainOntologyConfig
    standards: IndustryStandardsConfig
    physics: PhysicalConstraintsConfig
    credibility: SourceCredibilityConfig
    embeddings: DomainEmbeddingsConfig
    
    # Initial verification parameters
    initial_weights: Dict[str, float] = field(default_factory=lambda: {
        "LOV": 0.2,
        "POV": 0.2,
        "MAV": 0.2,
        "WSV": 0.2,
        "ESV": 0.2
    })
    
    initial_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "LOV": 0.5,
        "POV": 0.5,
        "MAV": 0.5,
        "WSV": 0.5,
        "ESV": 0.5
    })
    
    initial_alphas: Dict[str, float] = field(default_factory=lambda: {
        "LOV": 0.5,
        "POV": 0.5,
        "MAV": 0.5,
        "WSV": 0.5,
        "ESV": 0.5
    })
    
    global_threshold: float = 0.65  # Θ (global acceptance threshold)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate complete configuration."""
        all_errors = []
        
        # Validate each component
        valid_ontology, ontology_errors = self.ontology.validate()
        all_errors.extend([f"Ontology: {e}" for e in ontology_errors])
        
        valid_standards, standards_errors = self.standards.validate()
        all_errors.extend([f"Standards: {e}" for e in standards_errors])
        
        valid_physics, physics_errors = self.physics.validate()
        all_errors.extend([f"Physics: {e}" for e in physics_errors])
        
        valid_credibility, credibility_errors = self.credibility.validate()
        all_errors.extend([f"Credibility: {e}" for e in credibility_errors])
        
        valid_embeddings, embeddings_errors = self.embeddings.validate()
        all_errors.extend([f"Embeddings: {e}" for e in embeddings_errors])
        
        # Validate weights sum to 1.0
        weight_sum = sum(self.initial_weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            all_errors.append(f"Initial weights sum to {weight_sum}, should sum to 1.0")
        
        # Validate threshold range
        for module, threshold in self.initial_thresholds.items():
            if not 0 <= threshold <= 1:
                all_errors.append(f"Invalid threshold for {module}: {threshold}")
        
        # Validate alpha range
        for module, alpha in self.initial_alphas.items():
            if not 0 <= alpha <= 1:
                all_errors.append(f"Invalid alpha for {module}: {alpha}")
        
        return len(all_errors) == 0, all_errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DomainConfiguration':
        """Create from dictionary."""
        return cls(
            domain_name=data["domain_name"],
            domain_description=data["domain_description"],
            ontology=DomainOntologyConfig(**data["ontology"]),
            standards=IndustryStandardsConfig(**data["standards"]),
            physics=PhysicalConstraintsConfig(**data["physics"]),
            credibility=SourceCredibilityConfig(**data["credibility"]),
            embeddings=DomainEmbeddingsConfig(**data["embeddings"]),
            initial_weights=data.get("initial_weights", {}),
            initial_thresholds=data.get("initial_thresholds", {}),
            initial_alphas=data.get("initial_alphas", {}),
            global_threshold=data.get("global_threshold", 0.65)
        )


class DomainAdapter:
    """
    Domain Adapter for ATLASky-AI
    
    Manages domain configurations and applies them to the verification system.
    Supports loading, validating, and switching between domain configurations.
    """
    
    def __init__(self, config_directory: str = "domains"):
        self.config_directory = Path(config_directory)
        self.config_directory.mkdir(exist_ok=True)
        
        self.current_domain: Optional[DomainConfiguration] = None
        self.available_domains: Dict[str, Path] = {}
        
        self._scan_domains()
    
    def _scan_domains(self):
        """Scan config directory for available domain configurations."""
        for config_file in self.config_directory.glob("*.json"):
            domain_name = config_file.stem
            self.available_domains[domain_name] = config_file
        
        for config_file in self.config_directory.glob("*.yaml"):
            domain_name = config_file.stem
            self.available_domains[domain_name] = config_file
        
        logger.info(f"Found {len(self.available_domains)} domain configurations")
    
    def load_domain(self, domain_name: str) -> DomainConfiguration:
        """
        Load a domain configuration by name.
        
        Args:
            domain_name: Name of the domain to load
            
        Returns:
            DomainConfiguration object
            
        Raises:
            ValueError: If domain not found or validation fails
        """
        if domain_name not in self.available_domains:
            raise ValueError(f"Domain '{domain_name}' not found. Available: {list(self.available_domains.keys())}")
        
        config_path = self.available_domains[domain_name]
        
        # Load based on file format
        if config_path.suffix == ".json":
            with open(config_path, 'r') as f:
                data = json.load(f)
        elif config_path.suffix in [".yaml", ".yml"]:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        # Create configuration object
        config = DomainConfiguration.from_dict(data)
        
        # Validate
        valid, errors = config.validate()
        if not valid:
            error_msg = "\n".join(errors)
            raise ValueError(f"Invalid configuration for domain '{domain_name}':\n{error_msg}")
        
        logger.info(f"Successfully loaded domain: {domain_name}")
        self.current_domain = config
        
        return config
    
    def save_domain(self, config: DomainConfiguration, format: str = "json"):
        """
        Save a domain configuration.
        
        Args:
            config: DomainConfiguration to save
            format: Output format ('json' or 'yaml')
        """
        # Validate first
        valid, errors = config.validate()
        if not valid:
            error_msg = "\n".join(errors)
            raise ValueError(f"Cannot save invalid configuration:\n{error_msg}")
        
        # Determine output path
        output_path = self.config_directory / f"{config.domain_name}.{format}"
        
        # Save based on format
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
        elif format in ["yaml", "yml"]:
            with open(output_path, 'w') as f:
                yaml.dump(config.to_dict(), f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved domain configuration: {output_path}")
        
        # Rescan to update available domains
        self._scan_domains()
    
    def apply_to_rmmve(self, rmmve_process, config: Optional[DomainConfiguration] = None):
        """
        Apply domain configuration to RMMVe process.
        
        Args:
            rmmve_process: RMMVeProcess instance to configure
            config: DomainConfiguration to apply (uses current_domain if None)
        """
        if config is None:
            config = self.current_domain
        
        if config is None:
            raise ValueError("No domain configuration specified")
        
        logger.info(f"Applying domain configuration: {config.domain_name}")
        
        # Apply weights
        for i, module in enumerate(rmmve_process.modules):
            if module.name in config.initial_weights:
                module.weight = config.initial_weights[module.name]
                logger.debug(f"  {module.name}: weight = {module.weight}")
        
        # Apply thresholds
        for i, module in enumerate(rmmve_process.modules):
            if module.name in config.initial_thresholds:
                module.threshold = config.initial_thresholds[module.name]
                logger.debug(f"  {module.name}: threshold = {module.threshold}")
        
        # Apply alphas
        for i, module in enumerate(rmmve_process.modules):
            if module.name in config.initial_alphas:
                module.alpha = config.initial_alphas[module.name]
                logger.debug(f"  {module.name}: alpha = {module.alpha}")
        
        # Apply global threshold
        rmmve_process.global_threshold = config.global_threshold
        
        logger.info("Domain configuration applied successfully")
    
    def apply_to_knowledge_graph(self, knowledge_graph, config: Optional[DomainConfiguration] = None):
        """
        Apply domain configuration to knowledge graph.
        
        Args:
            knowledge_graph: SpatiotemporalKnowledgeGraph instance
            config: DomainConfiguration to apply (uses current_domain if None)
        """
        if config is None:
            config = self.current_domain
        
        if config is None:
            raise ValueError("No domain configuration specified")
        
        logger.info(f"Applying physics constraints from: {config.domain_name}")
        
        # Apply max velocities
        knowledge_graph.v_max = config.physics.max_velocities.copy()
        
        # Ensure 'default' key exists (use the first transport mode or manual_handling)
        if 'default' not in knowledge_graph.v_max:
            if 'manual_handling' in knowledge_graph.v_max:
                knowledge_graph.v_max['default'] = knowledge_graph.v_max['manual_handling']
            elif knowledge_graph.v_max:
                # Use first available transport mode as default
                knowledge_graph.v_max['default'] = list(knowledge_graph.v_max.values())[0]
            else:
                # Fallback to 2.0 m/s
                knowledge_graph.v_max['default'] = 2.0
        
        # Apply spatial/temporal resolutions
        if hasattr(knowledge_graph, 'spatial_resolution'):
            knowledge_graph.spatial_resolution = config.physics.spatial_resolution
        
        if hasattr(knowledge_graph, 'temporal_resolution'):
            knowledge_graph.temporal_resolution = config.physics.temporal_resolution
        
        logger.info("Physics constraints applied successfully")
    
    def create_template(self, domain_name: str) -> DomainConfiguration:
        """
        Create a template domain configuration with default values.
        
        Args:
            domain_name: Name for the new domain
            
        Returns:
            DomainConfiguration with default values
        """
        return DomainConfiguration(
            domain_name=domain_name,
            domain_description=f"Configuration for {domain_name} domain",
            ontology=DomainOntologyConfig(),
            standards=IndustryStandardsConfig(),
            physics=PhysicalConstraintsConfig(),
            credibility=SourceCredibilityConfig(),
            embeddings=DomainEmbeddingsConfig()
        )
    
    def list_domains(self) -> List[str]:
        """List all available domain configurations."""
        return list(self.available_domains.keys())
    
    def get_current_domain(self) -> Optional[DomainConfiguration]:
        """Get currently loaded domain configuration."""
        return self.current_domain


# Example domain configurations
def create_aerospace_domain() -> DomainConfiguration:
    """Create aerospace manufacturing domain configuration."""
    return DomainConfiguration(
        domain_name="aerospace",
        domain_description="Aerospace manufacturing and quality inspection",
        ontology=DomainOntologyConfig(
            entity_classes=[
                {"name": "Blade", "type": "AerospaceEntity", "spatiotemporal": True},
                {"name": "EngineSet", "type": "AerospaceEntity", "spatiotemporal": True},
                {"name": "InspectionMeasurement", "type": "Measurement", "spatiotemporal": True}
            ],
            relationship_types=[
                {"name": "containsBlade", "domain": "aerospace"},
                {"name": "hasMeasurement", "domain": "aerospace"},
                {"name": "locatedAt", "spatiotemporal": True}
            ],
            attributes=["material", "dimensions", "tolerance_mm", "actual_value_mm"]
        ),
        standards=IndustryStandardsConfig(
            standards=[
                {"name": "STEP_AP242", "version": "ISO_10303", "type": "CAD_data_exchange"},
                {"name": "AS9100", "version": "Rev_D", "type": "quality_management"}
            ],
            terminology_sources=["ISO", "ASTM", "AMS"],
            protocol_libraries={"measurement": "ISO_1101"}
        ),
        physics=PhysicalConstraintsConfig(
            max_velocities={
                "default": 2.0,
                "manual_handling": 2.0,
                "forklift": 5.0,
                "automated_conveyor": 1.0,
                "crane": 3.0
            },
            temporal_resolution=1.0,
            spatial_resolution=0.001  # 1mm precision for aerospace
        ),
        credibility=SourceCredibilityConfig(
            credibility_weights={
                "FAA": 1.0,
                "NASA": 0.95,
                "manufacturer_spec": 0.90,
                "industry_standard": 0.85,
                "academic": 0.70
            },
            authoritative_sources=["FAA", "NASA", "Boeing", "Airbus"]
        ),
        embeddings=DomainEmbeddingsConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            training_corpus_size=15000,
            retraining_frequency="quarterly"
        ),
        initial_weights={
            "LOV": 0.25,  # Higher weight for ontology in aerospace
            "POV": 0.25,  # High importance of standards
            "MAV": 0.20,
            "WSV": 0.15,
            "ESV": 0.15
        },
        initial_thresholds={
            "LOV": 0.7,
            "POV": 0.75,
            "MAV": 0.65,
            "WSV": 0.60,
            "ESV": 0.65
        }
    )


def create_healthcare_domain() -> DomainConfiguration:
    """Create healthcare domain configuration."""
    return DomainConfiguration(
        domain_name="healthcare",
        domain_description="Healthcare facility patient tracking and transfers",
        ontology=DomainOntologyConfig(
            entity_classes=[
                {"name": "Patient", "type": "HealthcareEntity", "spatiotemporal": True},
                {"name": "CareUnit", "type": "HealthcareEntity", "spatiotemporal": True},
                {"name": "ClinicalTransfer", "type": "Event", "spatiotemporal": True}
            ],
            relationship_types=[
                {"name": "transferred", "domain": "healthcare"},
                {"name": "admittedTo", "domain": "healthcare"},
                {"name": "locatedAt", "spatiotemporal": True}
            ],
            attributes=["patient_id", "unit_type", "transfer_reason", "medical_record"]
        ),
        standards=IndustryStandardsConfig(
            standards=[
                {"name": "HL7_FHIR", "version": "R4", "type": "healthcare_interoperability"},
                {"name": "HIPAA", "version": "current", "type": "privacy_compliance"}
            ],
            terminology_sources=["SNOMED_CT", "ICD-10", "LOINC"],
            protocol_libraries={"transfer": "Joint_Commission_Standards"}
        ),
        physics=PhysicalConstraintsConfig(
            max_velocities={
                "default": 2.0,
                "wheelchair": 2.0,
                "stretcher": 3.0,
                "ambulance": 20.0,
                "walking": 1.5
            },
            temporal_resolution=60.0,  # 1 minute for healthcare
            spatial_resolution=1.0     # 1 meter precision
        ),
        credibility=SourceCredibilityConfig(
            credibility_weights={
                "CDC": 1.0,
                "WHO": 0.95,
                "hospital_EHR": 0.90,
                "medical_journal": 0.85,
                "clinical_trial": 0.80
            },
            authoritative_sources=["CDC", "WHO", "NIH", "FDA"]
        ),
        embeddings=DomainEmbeddingsConfig(
            model_name="emilyalsentzer/Bio_ClinicalBERT",  # Domain-adapted BERT
            training_corpus_size=20000,
            retraining_frequency="quarterly"
        ),
        initial_weights={
            "LOV": 0.20,
            "POV": 0.30,  # Higher weight for protocol compliance
            "MAV": 0.25,  # Important for transfer timing
            "WSV": 0.15,
            "ESV": 0.10
        },
        initial_thresholds={
            "LOV": 0.65,
            "POV": 0.80,  # Strict protocol compliance
            "MAV": 0.70,
            "WSV": 0.60,
            "ESV": 0.65
        }
    )


def create_aviation_domain() -> DomainConfiguration:
    """Create aviation safety domain configuration."""
    return DomainConfiguration(
        domain_name="aviation",
        domain_description="Aviation safety incident reporting and analysis",
        ontology=DomainOntologyConfig(
            entity_classes=[
                {"name": "SafetyIncident", "type": "AviationEntity", "spatiotemporal": True},
                {"name": "Aircraft", "type": "AviationEntity", "spatiotemporal": True},
                {"name": "Operator", "type": "Person", "spatiotemporal": False}
            ],
            relationship_types=[
                {"name": "hasOperator", "domain": "aviation"},
                {"name": "occurredAt", "spatiotemporal": True},
                {"name": "caused", "causal": True}
            ],
            attributes=["incident_id", "severity", "aircraft_type", "flight_phase"]
        ),
        standards=IndustryStandardsConfig(
            standards=[
                {"name": "ICAO_Annex_13", "version": "2020", "type": "accident_investigation"},
                {"name": "FAA_Part_121", "version": "current", "type": "operations"}
            ],
            terminology_sources=["ICAO", "FAA", "EASA"],
            protocol_libraries={"reporting": "ASRS_taxonomy"}
        ),
        physics=PhysicalConstraintsConfig(
            max_velocities={
                "default": 5.0,
                "ground_ops": 5.0,
                "taxi": 15.0,
                "takeoff": 100.0,
                "cruise": 250.0
            },
            temporal_resolution=1.0,
            spatial_resolution=10.0  # 10 meter precision
        ),
        credibility=SourceCredibilityConfig(
            credibility_weights={
                "NTSB": 1.0,
                "FAA": 0.95,
                "ICAO": 0.95,
                "airline_report": 0.85,
                "pilot_report": 0.75
            },
            authoritative_sources=["NTSB", "FAA", "ICAO", "EASA"]
        ),
        embeddings=DomainEmbeddingsConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            training_corpus_size=12000,
            retraining_frequency="quarterly"
        )
    )

