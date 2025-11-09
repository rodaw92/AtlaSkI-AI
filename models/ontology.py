"""
ATLASky-AI Ontology for Spatiotemporal Knowledge Graph Verification

This ontology defines the formal structure for 4D Spatiotemporal Knowledge Graphs (STKGs)
used in the ATLASky-AI verification system. It supports multi-domain verification across
aerospace, healthcare, aviation, manufacturing, and CAD domains.

Ontology Structure:
- Entity Classes: Hierarchical classification with attributes
- Relationship Types: Spatiotemporal and domain-specific relations
- Physical Constraints: Consistency predicates (ψ_s, ψ_t, Ψ)
- Domain Rules: Standards, protocols, and tolerances
- Error Types: Verification failure classifications
"""

from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import math


@dataclass
class EntityClass:
    """Definition of an entity class with attributes and constraints."""
    name: str
    description: str
    parent_class: Optional[str] = None
    required_attributes: List[str] = None
    optional_attributes: List[str] = None
    spatiotemporal_required: bool = False
    domain: str = "general"

    def __post_init__(self):
        if self.required_attributes is None:
            self.required_attributes = []
        if self.optional_attributes is None:
            self.optional_attributes = []


@dataclass
class RelationshipType:
    """Definition of a relationship type with constraints."""
    name: str
    description: str
    domain_constraints: Dict[str, Any] = None
    spatiotemporal_constraints: Dict[str, Any] = None
    symmetric: bool = False
    transitive: bool = False

    def __post_init__(self):
        if self.domain_constraints is None:
            self.domain_constraints = {}
        if self.spatiotemporal_constraints is None:
            self.spatiotemporal_constraints = {}


@dataclass
class PhysicalConstraint:
    """Physical constraint definition for spatiotemporal consistency."""
    name: str
    description: str
    predicate_function: callable
    parameters: Dict[str, Any] = None
    domain: str = "general"

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class DomainRule:
    """Domain-specific rule for standards compliance."""
    name: str
    description: str
    domain: str
    rule_type: str  # 'tolerance', 'protocol', 'standard', 'constraint'
    parameters: Dict[str, Any]
    violation_severity: str = "medium"  # 'low', 'medium', 'high', 'critical'


@dataclass
class ErrorType:
    """Verification error classification."""
    name: str
    description: str
    category: str  # 'semantic', 'spatial', 'temporal', 'content', 'structural'
    detection_module: str  # 'LOV', 'POV', 'MAV', 'WSV', 'ESV'
    typical_false_positive_rate: float
    examples: List[str]


class ATLASOntology:
    """
    Comprehensive ontology for ATLASky-AI verification system.

    Defines the formal structure for multi-domain spatiotemporal knowledge graphs
    supporting physics-based verification and domain compliance checking.
    """

    def __init__(self):
        self.entity_classes = {}
        self.relationship_types = {}
        self.physical_constraints = {}
        self.domain_rules = {}
        self.error_types = {}

        self._initialize_ontology()

    def _initialize_ontology(self):
        """Initialize the complete ontology structure."""
        self._define_entity_classes()
        self._define_relationship_types()
        self._define_physical_constraints()
        self._define_domain_rules()
        self._define_error_types()

    def _define_entity_classes(self):
        """Define hierarchical entity classes with attributes."""

        # Base classes
        self.entity_classes["Entity"] = EntityClass(
            name="Entity",
            description="Root entity class",
            required_attributes=["entity_id"],
            optional_attributes=["description", "metadata"]
        )

        self.entity_classes["PhysicalEntity"] = EntityClass(
            name="PhysicalEntity",
            description="Entities with physical presence",
            parent_class="Entity",
            spatiotemporal_required=True,
            required_attributes=["material", "dimensions"],
            optional_attributes=["weight", "temperature", "pressure"]
        )

        self.entity_classes["SpatiotemporalEntity"] = EntityClass(
            name="SpatiotemporalEntity",
            description="Entities with spatiotemporal coordinates",
            parent_class="Entity",
            spatiotemporal_required=True,
            required_attributes=["x_coord", "y_coord", "z_coord", "timestamp"]
        )

        # Aerospace domain
        self.entity_classes["AerospaceEntity"] = EntityClass(
            name="AerospaceEntity",
            description="Aerospace manufacturing and assembly entities",
            parent_class="PhysicalEntity",
            domain="aerospace",
            required_attributes=["component_id", "material", "lifecycle_hours"],
            optional_attributes=["serial_number", "certification_status"]
        )

        self.entity_classes["Blade"] = EntityClass(
            name="Blade",
            description="Turbine blade components",
            parent_class="AerospaceEntity",
            domain="aerospace",
            required_attributes=["blade_type", "alloy_composition"],
            optional_attributes=["stress_test_results", "fatigue_life"]
        )

        self.entity_classes["EngineSet"] = EntityClass(
            name="EngineSet",
            description="Aircraft engine assemblies",
            parent_class="AerospaceEntity",
            domain="aerospace",
            required_attributes=["engine_model", "thrust_rating"],
            optional_attributes=["maintenance_schedule", "flight_hours"]
        )

        self.entity_classes["InspectionMeasurement"] = EntityClass(
            name="InspectionMeasurement",
            description="Quality inspection measurements",
            parent_class="SpatiotemporalEntity",
            domain="aerospace",
            required_attributes=["measurement_id", "feature_name", "nominal_value_mm",
                              "actual_value_mm", "tolerance_mm", "inspection_tool"],
            optional_attributes=["deviation_mm", "surface_side", "operator_id"]
        )

        # Healthcare domain
        self.entity_classes["HealthcareEntity"] = EntityClass(
            name="HealthcareEntity",
            description="Healthcare and clinical entities",
            parent_class="Entity",
            domain="healthcare",
            required_attributes=["facility_id", "regulatory_compliance"],
            optional_attributes=["capacity", "specialization"]
        )

        self.entity_classes["Patient"] = EntityClass(
            name="Patient",
            description="Patient entities with medical records",
            parent_class="SpatiotemporalEntity",
            domain="healthcare",
            required_attributes=["patient_id", "medical_record_number", "current_condition"],
            optional_attributes=["admission_date", "discharge_date", "treatment_plan"]
        )

        self.entity_classes["CareUnit"] = EntityClass(
            name="CareUnit",
            description="Hospital care units and departments",
            parent_class="HealthcareEntity",
            domain="healthcare",
            required_attributes=["unit_type", "capacity", "coordinates"],
            optional_attributes=["specialization", "equipment_list", "staff_count"]
        )

        self.entity_classes["ClinicalTransfer"] = EntityClass(
            name="ClinicalTransfer",
            description="Patient transfers between care units",
            parent_class="SpatiotemporalEntity",
            domain="healthcare",
            required_attributes=["transfer_id", "from_unit", "to_unit", "transfer_reason"],
            optional_attributes=["transfer_duration", "transport_method", "accompanying_staff"]
        )

        # Aviation domain
        self.entity_classes["AviationEntity"] = EntityClass(
            name="AviationEntity",
            description="Aviation safety and incident entities",
            parent_class="Entity",
            domain="aviation",
            required_attributes=["incident_id", "severity_level"],
            optional_attributes=["aircraft_type", "flight_phase", "weather_conditions"]
        )

        self.entity_classes["SafetyIncident"] = EntityClass(
            name="SafetyIncident",
            description="Aviation safety reporting incidents",
            parent_class="SpatiotemporalEntity",
            domain="aviation",
            required_attributes=["narrative_text", "contributing_factors", "corrective_actions"],
            optional_attributes=["flight_number", "altitude", "airspace_class"]
        )

        # CAD/Engineering domain
        self.entity_classes["CADEntity"] = EntityClass(
            name="CADEntity",
            description="CAD model and engineering entities",
            parent_class="PhysicalEntity",
            domain="cad",
            required_attributes=["model_id", "assembly_version"],
            optional_attributes=["design_software", "geometric_constraints"]
        )

        self.entity_classes["CADAssembly"] = EntityClass(
            name="CADAssembly",
            description="CAD assembly models with parts",
            parent_class="CADEntity",
            domain="cad",
            required_attributes=["part_count", "geometric_constraints"],
            optional_attributes=["interference_checks", "kinematic_constraints"]
        )

        self.entity_classes["CADFeature"] = EntityClass(
            name="CADFeature",
            description="Geometric features in CAD models",
            parent_class="CADEntity",
            domain="cad",
            required_attributes=["feature_type", "geometric_parameters"],
            optional_attributes=["tolerance_class", "surface_finish"]
        )

    def _define_relationship_types(self):
        """Define relationship types with constraints."""

        # Spatial relationships
        self.relationship_types["locatedAt"] = RelationshipType(
            name="locatedAt",
            description="Entity located at spatial coordinates",
            spatiotemporal_constraints={
                "requires_coordinates": True,
                "temporal_consistency": True
            }
        )

        self.relationship_types["adjacentTo"] = RelationshipType(
            name="adjacentTo",
            description="Entities are spatially adjacent",
            symmetric=True,
            spatiotemporal_constraints={
                "max_distance": 1.0,  # meters
                "same_timestamp_window": True
            }
        )

        self.relationship_types["containedBy"] = RelationshipType(
            name="containedBy",
            description="Entity is contained within another",
            spatiotemporal_constraints={
                "geometric_containment": True
            }
        )

        # Temporal relationships
        self.relationship_types["occurredDuring"] = RelationshipType(
            name="occurredDuring",
            description="Event occurred during time period",
            spatiotemporal_constraints={
                "temporal_overlap": True
            }
        )

        self.relationship_types["precededBy"] = RelationshipType(
            name="precededBy",
            description="Event was preceded by another",
            spatiotemporal_constraints={
                "temporal_ordering": "before"
            }
        )

        self.relationship_types["caused"] = RelationshipType(
            name="caused",
            description="Entity caused another event",
            spatiotemporal_constraints={
                "causal_temporal_order": True
            }
        )

        # Domain-specific relationships
        self.relationship_types["containsBlade"] = RelationshipType(
            name="containsBlade",
            description="Engine assembly contains turbine blade",
            domain_constraints={
                "domain": "aerospace",
                "valid_subjects": ["EngineSet"],
                "valid_objects": ["Blade"]
            }
        )

        self.relationship_types["hasMeasurement"] = RelationshipType(
            name="hasMeasurement",
            description="Component has inspection measurement",
            domain_constraints={
                "domain": "aerospace",
                "valid_subjects": ["Blade", "EngineSet"],
                "valid_objects": ["InspectionMeasurement"]
            }
        )

        self.relationship_types["transferred"] = RelationshipType(
            name="transferred",
            description="Patient transferred between care units",
            domain_constraints={
                "domain": "healthcare",
                "valid_subjects": ["Patient"],
                "valid_objects": ["CareUnit"]
            },
            spatiotemporal_constraints={
                "requires_transfer_time": True,
                "protocol_compliance": True
            }
        )

        self.relationship_types["hasOperator"] = RelationshipType(
            name="hasOperator",
            description="Incident has operator involvement",
            domain_constraints={
                "domain": "aviation",
                "valid_subjects": ["SafetyIncident"]
            }
        )

        self.relationship_types["relatedTo"] = RelationshipType(
            name="relatedTo",
            description="Generic relationship between entities",
            symmetric=True
        )

    def _define_physical_constraints(self):
        """Define physical constraints for spatiotemporal consistency."""

        # Spatial consistency (ψ_s)
        self.physical_constraints["spatial_consistency"] = PhysicalConstraint(
            name="spatial_consistency",
            description="No entity exists at two separated locations simultaneously",
            predicate_function=self._check_spatial_consistency,
            parameters={
                "temporal_resolution": 1.0,  # seconds
                "spatial_resolution": 0.1   # meters
            }
        )

        # Temporal consistency (ψ_t)
        self.physical_constraints["temporal_consistency"] = PhysicalConstraint(
            name="temporal_consistency",
            description="Travel time must be physically feasible",
            predicate_function=self._check_temporal_consistency,
            parameters={
                "transport_modes": {
                    "manual": 2.0,      # m/s
                    "forklift": 5.0,    # m/s
                    "conveyor": 1.0,    # m/s
                    "ambulance": 20.0,  # m/s
                    "aircraft": 250.0   # m/s
                }
            }
        )

        # Combined physical consistency (Ψ)
        self.physical_constraints["physical_consistency"] = PhysicalConstraint(
            name="physical_consistency",
            description="Combined spatial and temporal consistency",
            predicate_function=self._check_physical_consistency,
            parameters={
                "combine_mode": "AND"  # Both ψ_s and ψ_t must hold
            }
        )

        # Domain-specific constraints
        self.physical_constraints["aerospace_tolerance"] = PhysicalConstraint(
            name="aerospace_tolerance",
            description="Manufacturing tolerance compliance",
            predicate_function=self._check_aerospace_tolerance,
            parameters={
                "default_tolerance": 0.1,  # ±0.1mm
                "critical_features": ["clearance", "fit", "alignment"]
            },
            domain="aerospace"
        )

        self.physical_constraints["healthcare_protocol"] = PhysicalConstraint(
            name="healthcare_protocol",
            description="Clinical transfer protocol compliance",
            predicate_function=self._check_healthcare_protocol,
            parameters={
                "min_transfer_times": {
                    ("MICU", "SICU"): 15,   # minutes
                    ("MICU", "OR"): 20,
                    ("SICU", "OR"): 20,
                    ("OR", "Recovery"): 10,
                    ("Recovery", "Floor"): 30
                }
            },
            domain="healthcare"
        )

    def _define_domain_rules(self):
        """Define domain-specific rules and standards."""

        # Aerospace rules
        self.domain_rules["dimensional_tolerance"] = DomainRule(
            name="dimensional_tolerance",
            description="Component dimensions must be within specified tolerances",
            domain="aerospace",
            rule_type="tolerance",
            parameters={
                "tolerance_range_mm": 0.1,
                "measurement_features": ["clearance", "fit", "alignment"],
                "inspection_tools": ["3D_Scanner", "CMM", "Laser_Tracker"]
            },
            violation_severity="high"
        )

        self.domain_rules["material_certification"] = DomainRule(
            name="material_certification",
            description="Materials must meet aerospace certification standards",
            domain="aerospace",
            rule_type="standard",
            parameters={
                "certification_standards": ["AMS", "ASTM", "MIL-SPEC"],
                "required_testing": ["chemical_composition", "mechanical_properties", "fatigue_life"]
            },
            violation_severity="critical"
        )

        # Healthcare rules
        self.domain_rules["patient_transfer_protocol"] = DomainRule(
            name="patient_transfer_protocol",
            description="Patient transfers must follow clinical protocols",
            domain="healthcare",
            rule_type="protocol",
            parameters={
                "min_transfer_times": {
                    "icu_to_icu": 15,      # minutes
                    "icu_to_or": 20,
                    "or_to_recovery": 10,
                    "recovery_to_floor": 30
                },
                "required_staff": ["RN", "MD", "transport_team"],
                "documentation_required": ["handover_note", "vital_signs", "medication_list"]
            },
            violation_severity="critical"
        )

        self.domain_rules["medication_administration"] = DomainRule(
            name="medication_administration",
            description="Medication administration must follow safety protocols",
            domain="healthcare",
            rule_type="protocol",
            parameters={
                "verification_steps": ["right_patient", "right_medication", "right_dose", "right_route", "right_time"],
                "documentation_required": ["administration_time", "patient_response", "adverse_reactions"]
            },
            violation_severity="critical"
        )

        # Aviation rules
        self.domain_rules["incident_reporting"] = DomainRule(
            name="incident_reporting",
            description="Safety incidents must be reported within time limits",
            domain="aviation",
            rule_type="standard",
            parameters={
                "reporting_deadlines": {
                    "serious_incident": 72,  # hours
                    "accident": 24
                },
                "required_elements": ["narrative", "contributing_factors", "corrective_actions"]
            },
            violation_severity="high"
        )

        # CAD/Engineering rules
        self.domain_rules["geometric_constraint"] = DomainRule(
            name="geometric_constraint",
            description="CAD models must satisfy geometric constraints",
            domain="cad",
            rule_type="constraint",
            parameters={
                "constraint_types": ["parallelism", "perpendicularity", "coaxiality", "symmetry"],
                "tolerance_classes": ["fine", "medium", "coarse"],
                "interference_check_required": True
            },
            violation_severity="high"
        )

    def _define_error_types(self):
        """Define verification error types and failure modes."""

        # Semantic drift errors
        self.error_types["semantic_drift"] = ErrorType(
            name="semantic_drift",
            description="Facts deviate from domain ontology or terminology standards",
            category="semantic",
            detection_module="LOV",
            typical_false_positive_rate=0.08,
            examples=[
                "Incorrect entity classification",
                "Invalid relationship type usage",
                "Terminology misuse"
            ]
        )

        # Content hallucination errors
        self.error_types["content_hallucination"] = ErrorType(
            name="content_hallucination",
            description="Fabricated facts not grounded in reality or evidence",
            category="content",
            detection_module="POV",
            typical_false_positive_rate=0.11,
            examples=[
                "Non-existent component measurements",
                "Fabricated patient transfers",
                "Invented incident details"
            ]
        )

        # Spatial inconsistency errors
        self.error_types["spatial_inconsistency"] = ErrorType(
            name="spatial_inconsistency",
            description="Entity exists at multiple locations simultaneously",
            category="spatial",
            detection_module="MAV",
            typical_false_positive_rate=0.03,
            examples=[
                "Component measured at two locations at same time",
                "Patient in two care units simultaneously",
                "Aircraft at conflicting positions"
            ]
        )

        # Temporal inconsistency errors
        self.error_types["temporal_inconsistency"] = ErrorType(
            name="temporal_inconsistency",
            description="Events violate temporal ordering or travel time constraints",
            category="temporal",
            detection_module="MAV",
            typical_false_positive_rate=0.04,
            examples=[
                "Effect precedes cause",
                "Impossible travel times",
                "Protocol violations in transfer timing"
            ]
        )

        # External validation errors
        self.error_types["external_validation_failure"] = ErrorType(
            name="external_validation_failure",
            description="Facts contradict authoritative external sources",
            category="content",
            detection_module="WSV",
            typical_false_positive_rate=0.07,
            examples=[
                "Contradicts regulatory standards",
                "Conflicts with manufacturer specifications",
                "Inconsistent with facility records"
            ]
        )

        # Embedding statistical errors
        self.error_types["embedding_anomaly"] = ErrorType(
            name="embedding_anomaly",
            description="Statistical outliers in embedding space",
            category="structural",
            detection_module="ESV",
            typical_false_positive_rate=0.05,
            examples=[
                "Unusual feature combinations",
                "Outlier measurement patterns",
                "Anomalous relationship patterns"
            ]
        )

    # Physical constraint checking functions
    def _check_spatial_consistency(self, fact: Dict, context: List[Dict]) -> bool:
        """Check spatial consistency predicate ψ_s."""
        if not fact.get("spatiotemporal"):
            return True

        fact_coords = fact["spatiotemporal"]
        fact_time = fact_coords.get("timestamp")
        entity_id = fact.get("subject_entity_id") or fact.get("entity_id")

        if not entity_id or not fact_time:
            return True

        # Check against other facts for same entity
        for other_fact in context:
            if (other_fact.get("subject_entity_id") or other_fact.get("entity_id")) != entity_id:
                continue

            other_coords = other_fact.get("spatiotemporal", {})
            other_time = other_coords.get("timestamp")

            if not other_time:
                continue

            # Check temporal proximity
            time_diff = abs(self._time_difference(fact_time, other_time))
            if time_diff > self.physical_constraints["spatial_consistency"].parameters["temporal_resolution"]:
                continue

            # Check spatial separation
            distance = self._euclidean_distance(fact_coords, other_coords)
            if distance > self.physical_constraints["spatial_consistency"].parameters["spatial_resolution"]:
                return False  # Violation found

        return True

    def _check_temporal_consistency(self, fact: Dict, context: List[Dict]) -> bool:
        """Check temporal consistency predicate ψ_t."""
        if not fact.get("spatiotemporal"):
            return True

        entity_id = fact.get("subject_entity_id") or fact.get("entity_id")
        if not entity_id:
            return True

        # Find previous location of same entity
        current_coords = fact["spatiotemporal"]
        current_time = current_coords.get("timestamp")

        if not current_time:
            return True

        previous_fact = None
        min_time_diff = float('inf')

        for other_fact in context:
            if (other_fact.get("subject_entity_id") or other_fact.get("entity_id")) != entity_id:
                continue

            other_coords = other_fact.get("spatiotemporal", {})
            other_time = other_coords.get("timestamp")

            if not other_time:
                continue

            time_diff = self._time_difference(current_time, other_time)
            if time_diff > 0 and time_diff < min_time_diff:  # Previous event
                min_time_diff = time_diff
                previous_fact = other_fact

        if not previous_fact:
            return True  # No previous event to check against

        # Check travel time constraint
        prev_coords = previous_fact["spatiotemporal"]
        distance = self._euclidean_distance(current_coords, prev_coords)

        transport_mode = fact.get("transport_mode", "manual")
        v_max = self.physical_constraints["temporal_consistency"].parameters["transport_modes"].get(transport_mode, 2.0)

        required_time = distance / v_max if v_max > 0 else 0

        return min_time_diff >= required_time

    def _check_physical_consistency(self, fact: Dict, context: List[Dict]) -> bool:
        """Check combined physical consistency Ψ."""
        spatial_ok = self._check_spatial_consistency(fact, context)
        temporal_ok = self._check_temporal_consistency(fact, context)
        return spatial_ok and temporal_ok

    def _check_aerospace_tolerance(self, fact: Dict, context: List[Dict]) -> bool:
        """Check aerospace tolerance compliance."""
        if fact.get("entity_class") != "InspectionMeasurement":
            return True

        tolerance = fact.get("tolerance_mm", 0.1)
        actual = fact.get("actual_value_mm", 0)
        nominal = fact.get("nominal_value_mm", 0)

        deviation = abs(actual - nominal)
        return deviation <= tolerance

    def _check_healthcare_protocol(self, fact: Dict, context: List[Dict]) -> bool:
        """Check healthcare protocol compliance."""
        if fact.get("relationship_type") != "transferred":
            return True

        from_unit = fact.get("from_unit")
        to_unit = fact.get("to_unit")
        transfer_time = fact.get("transfer_duration_minutes", 0)

        min_times = self.physical_constraints["healthcare_protocol"].parameters["min_transfer_times"]
        required_time = min_times.get((from_unit, to_unit), 15)

        return transfer_time >= required_time

    # Utility functions
    def _euclidean_distance(self, coords1: Dict, coords2: Dict) -> float:
        """Calculate Euclidean distance between coordinates."""
        x1 = coords1.get("x_coord", 0)
        y1 = coords1.get("y_coord", 0)
        z1 = coords1.get("z_coord", 0)

        x2 = coords2.get("x_coord", 0)
        y2 = coords2.get("y_coord", 0)
        z2 = coords2.get("z_coord", 0)

        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

    def _time_difference(self, time1: str, time2: str) -> float:
        """Calculate time difference in seconds."""
        try:
            dt1 = datetime.fromisoformat(time1.replace('Z', '+00:00'))
            dt2 = datetime.fromisoformat(time2.replace('Z', '+00:00'))
            return (dt1 - dt2).total_seconds()
        except:
            return float('inf')

    # Ontology query methods
    def get_entity_class(self, name: str) -> Optional[EntityClass]:
        """Get entity class definition."""
        return self.entity_classes.get(name)

    def get_relationship_type(self, name: str) -> Optional[RelationshipType]:
        """Get relationship type definition."""
        return self.relationship_types.get(name)

    def get_physical_constraint(self, name: str) -> Optional[PhysicalConstraint]:
        """Get physical constraint definition."""
        return self.physical_constraints.get(name)

    def get_domain_rules(self, domain: str) -> List[DomainRule]:
        """Get domain rules for a specific domain."""
        return [rule for rule in self.domain_rules.values() if rule.domain == domain]

    def get_error_types_by_module(self, module: str) -> List[ErrorType]:
        """Get error types detected by a specific module."""
        return [error for error in self.error_types.values() if error.detection_module == module]

    def validate_entity(self, entity: Dict) -> List[str]:
        """Validate entity against ontology constraints."""
        violations = []

        entity_class_name = entity.get("entity_class")
        entity_class = self.get_entity_class(entity_class_name)

        if not entity_class:
            violations.append(f"Unknown entity class: {entity_class_name}")
            return violations

        # Check required attributes
        for attr in entity_class.required_attributes:
            if attr not in entity and attr not in entity.get("attributes", {}):
                violations.append(f"Missing required attribute: {attr}")

        # Check spatiotemporal requirements
        if entity_class.spatiotemporal_required:
            st = entity.get("spatiotemporal", {})
            if not st or not all(k in st for k in ["x_coord", "y_coord", "z_coord", "timestamp"]):
                violations.append("Missing required spatiotemporal coordinates")

        return violations

    def validate_relationship(self, relationship: Dict) -> List[str]:
        """Validate relationship against ontology constraints."""
        violations = []

        rel_type_name = relationship.get("relationship_type")
        rel_type = self.get_relationship_type(rel_type_name)

        if not rel_type:
            violations.append(f"Unknown relationship type: {rel_type_name}")
            return violations

        # Check domain constraints
        domain_constraints = rel_type.domain_constraints
        if domain_constraints:
            valid_subjects = domain_constraints.get("valid_subjects", [])
            valid_objects = domain_constraints.get("valid_objects", [])

            subject_class = relationship.get("subject_entity_class")
            object_class = relationship.get("object_entity_class")

            if valid_subjects and subject_class not in valid_subjects:
                violations.append(f"Invalid subject class {subject_class} for relationship {rel_type_name}")

            if valid_objects and object_class not in valid_objects:
                violations.append(f"Invalid object class {object_class} for relationship {rel_type_name}")

        return violations

    def check_physical_consistency(self, fact: Dict, context: List[Dict] = None) -> Dict[str, bool]:
        """Check all applicable physical constraints for a fact."""
        if context is None:
            context = []

        results = {}

        # Check spatial consistency
        results["spatial_consistency"] = self._check_spatial_consistency(fact, context)

        # Check temporal consistency
        results["temporal_consistency"] = self._check_temporal_consistency(fact, context)

        # Check combined consistency
        results["physical_consistency"] = results["spatial_consistency"] and results["temporal_consistency"]

        # Check domain-specific constraints
        entity_class = fact.get("entity_class")
        if entity_class:
            if "InspectionMeasurement" in entity_class:
                results["aerospace_tolerance"] = self._check_aerospace_tolerance(fact, context)
            elif "ClinicalTransfer" in entity_class:
                results["healthcare_protocol"] = self._check_healthcare_protocol(fact, context)

        return results


# Global ontology instance
ONTOLOGY = ATLASOntology()
