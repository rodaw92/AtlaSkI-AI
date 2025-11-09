import random
import math
from datetime import datetime, timedelta
from .constants import (
    ENTITY_CLASSES, RELATIONSHIP_TYPES, BLADE_COMPONENTS,
    ENGINE_SETS, MEASUREMENT_IDS, BLADE_FEATURES, SURFACE_SIDES,
    TAU_RES, SIGMA_RES, V_MAX
)
from .ontology import ONTOLOGY

class KnowledgeGraph:
    """
    4D Spatiotemporal Knowledge Graph (STKG) implementation.

    Based on Definition 1 from preliminaries:
    A 4D STKG is a tuple G = (V, E, O, T, Psi) where:
    - V: set of versioned entities with immutable attributes and mutable state
    - E: set of directed edges representing relationships
    - O = (C, R_o, A): domain ontology with entity classes, relation types, and attributes
    - T: (V ∪ E) → R³ × R maps entities/relations to spatiotemporal coordinates (x,y,z,t)
    - Psi: (V ∪ E) → {0,1} physical consistency predicate ensuring facts obey physical laws
    """
    def __init__(self):
        # V: Versioned entities
        self.entities = {}  # id -> Entity data

        # E: Relationships (directed edges)
        self.relationships = []  # List of relationship objects

        # O: Domain ontology = (C, R_o, A) - now using comprehensive ontology
        self.ontology = ONTOLOGY
        self.entity_classes = set(ENTITY_CLASSES)  # Legacy support
        self.relation_types = set(RELATIONSHIP_TYPES)  # Legacy support
        self.attributes = {}  # A: attribute definitions per entity class

        # T: Spatiotemporal mapping (implicit in entity/relationship data)
        # Each entity/relationship stores its spatiotemporal coordinates

        # Physical parameters for Psi predicate - now from ontology
        self.tau_res = TAU_RES  # Temporal resolution
        self.sigma_res = SIGMA_RES  # Spatial resolution
        self.v_max = V_MAX  # Maximum velocities by transport mode
        
    def add_entity(self, entity_id, entity_class, attributes, spatiotemporal=None, validate=True):
        """Add an entity to the knowledge graph."""
        entity_data = {
            "entity_id": entity_id,
            "entity_class": entity_class,
            "attributes": attributes,
            "spatiotemporal": spatiotemporal or {}
        }

        # Validate against ontology if requested
        if validate:
            violations = self.ontology.validate_entity(entity_data)
            if violations:
                raise ValueError(f"Ontology validation failed for entity {entity_id}: {violations}")

        self.entities[entity_id] = entity_data
        self.entity_classes.add(entity_class)
    
    def add_relationship(self, rel_id, subject_id, relation_type, object_id, validate=True):
        """Add a relationship to the knowledge graph."""
        relationship_data = {
            "relationship_id": rel_id,
            "subject_entity_id": subject_id,
            "relationship_type": relation_type,
            "object_entity_id": object_id,
            "subject_entity_class": self.entities.get(subject_id, {}).get("entity_class"),
            "object_entity_class": self.entities.get(object_id, {}).get("entity_class")
        }

        # Validate against ontology if requested
        if validate:
            violations = self.ontology.validate_relationship(relationship_data)
            if violations:
                raise ValueError(f"Ontology validation failed for relationship {rel_id}: {violations}")

        self.relationships.append(relationship_data)
        self.relation_types.add(relation_type)
    
    def get_entity(self, entity_id):
        """Get an entity by its ID."""
        return self.entities.get(entity_id)
    
    def get_relationships(self, entity_id=None, relation_type=None):
        """Get relationships, filtered by entity ID or type if provided."""
        if entity_id is None and relation_type is None:
            return self.relationships
        
        result = []
        for rel in self.relationships:
            if entity_id and (rel["subject_entity_id"] == entity_id or rel["object_entity_id"] == entity_id):
                result.append(rel)
            elif relation_type and rel["relationship_type"] == relation_type:
                result.append(rel)
        return result
    
    def has_entity_class(self, entity_class):
        """Check if an entity class exists in the knowledge graph."""
        return entity_class in self.entity_classes
    
    def has_relation_type(self, relation_type):
        """Check if a relation type exists in the knowledge graph."""
        return relation_type in self.relation_types

    def get_spatiotemporal_coord(self, fact):
        """
        Extract spatiotemporal coordinates from a fact.
        Returns (x, y, z, t) tuple or None if not available.
        """
        if not fact:
            return None

        st = fact.get("spatiotemporal", {})
        if not st:
            return None

        x = st.get("x_coord", 0.0)
        y = st.get("y_coord", 0.0)
        z = st.get("z_coord", 0.0)
        t = st.get("timestamp", None)

        if t is None:
            return None

        # Convert timestamp to datetime if string
        if isinstance(t, str):
            try:
                t = datetime.fromisoformat(t.replace('Z', '+00:00'))
            except:
                return None

        return (x, y, z, t)

    def euclidean_distance(self, coord1, coord2):
        """
        Calculate Euclidean distance between two spatial coordinates.
        coord1, coord2: (x, y, z, t) tuples
        Returns: distance in meters
        """
        if not coord1 or not coord2:
            return float('inf')

        x1, y1, z1, _ = coord1
        x2, y2, z2, _ = coord2

        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

    def time_difference(self, coord1, coord2):
        """
        Calculate time difference between two temporal coordinates.
        Returns: time difference in seconds
        """
        if not coord1 or not coord2:
            return float('inf')

        _, _, _, t1 = coord1
        _, _, _, t2 = coord2

        if isinstance(t1, datetime) and isinstance(t2, datetime):
            return abs((t2 - t1).total_seconds())

        return float('inf')

    def psi_s(self, fact):
        """
        Spatial Consistency predicate (Definition 2 from preliminaries).

        A fact d satisfies spatial consistency if no entity exists at two
        separated locations within the same time window:

        psi_s(d) = 1 ⟺ ∄d': same_entity(d, d') ∧ |t_d - t_d'| < tau_res
                          ∧ dist(d, d') > sigma_res

        Returns: 1 if spatially consistent, 0 otherwise
        """
        coord = self.get_spatiotemporal_coord(fact)
        if not coord:
            return 1  # No spatiotemporal data, assume consistent

        # Get entity ID from fact
        entity_id = fact.get("subject_entity_id") or fact.get("entity_id")
        if not entity_id:
            return 1

        # Check all other facts involving the same entity
        for other_fact in self.get_all_facts():
            other_entity_id = other_fact.get("subject_entity_id") or other_fact.get("entity_id")

            if other_entity_id != entity_id:
                continue  # Different entity

            other_coord = self.get_spatiotemporal_coord(other_fact)
            if not other_coord:
                continue

            # Check if within same time window
            time_diff = self.time_difference(coord, other_coord)
            if time_diff >= self.tau_res:
                continue  # Outside temporal resolution window

            # Check if at separated locations
            distance = self.euclidean_distance(coord, other_coord)
            if distance > self.sigma_res:
                # Violation: same entity at two separated locations at same time
                return 0

        return 1

    def psi_t(self, fact, transport_mode="default"):
        """
        Temporal Consistency predicate (Definition 3 from preliminaries).

        A fact d satisfies temporal consistency if causally prior facts
        precede d temporally and travel time constraints are satisfied:

        psi_t(d) = 1 ⟺ t₂ > t₁ ∧ (t₂ - t₁) ≥ travel_time(ℓ₁, ℓ₂)

        where travel_time(ℓ₁, ℓ₂) = dist(ℓ₁, ℓ₂) / v_max

        Args:
            fact: The fact to check
            transport_mode: Transport mode for velocity limit ("manual", "forklift", etc.)

        Returns: 1 if temporally consistent, 0 otherwise
        """
        coord = self.get_spatiotemporal_coord(fact)
        if not coord:
            return 1  # No spatiotemporal data, assume consistent

        # Get entity ID from fact
        entity_id = fact.get("subject_entity_id") or fact.get("entity_id")
        if not entity_id:
            return 1

        # Get maximum velocity for transport mode
        v_max = self.v_max.get(transport_mode, self.v_max.get("default", 2.0))

        # Find previous location of the same entity
        _, _, _, t_current = coord
        previous_fact = None
        previous_coord = None

        for other_fact in self.get_all_facts():
            other_entity_id = other_fact.get("subject_entity_id") or other_fact.get("entity_id")

            if other_entity_id != entity_id:
                continue  # Different entity

            other_coord = self.get_spatiotemporal_coord(other_fact)
            if not other_coord:
                continue

            _, _, _, t_other = other_coord

            # Find the most recent previous fact
            if isinstance(t_other, datetime) and isinstance(t_current, datetime):
                if t_other < t_current:
                    if previous_coord is None or t_other > previous_coord[3]:
                        previous_fact = other_fact
                        previous_coord = other_coord

        if not previous_coord:
            return 1  # No previous location, assume consistent

        # Check temporal ordering
        _, _, _, t_prev = previous_coord
        if not (t_current > t_prev):
            return 0  # Temporal ordering violated

        # Check travel time constraint
        distance = self.euclidean_distance(previous_coord, coord)
        time_diff = self.time_difference(previous_coord, coord)

        required_travel_time = distance / v_max if v_max > 0 else 0

        if time_diff < required_travel_time:
            # Violation: insufficient time to travel the distance
            return 0

        return 1

    def Psi(self, fact, transport_mode="default"):
        """
        Physical Consistency predicate (combines spatial and temporal).

        A fact is physically consistent if:
        Psi(d) = psi_s(d) ∧ psi_t(d) = 1

        Uses ontology-based checking when available.

        Returns: 1 if physically consistent, 0 otherwise
        """
        # Try ontology-based checking first
        try:
            consistency_results = self.ontology.check_physical_consistency(fact, self.get_all_facts())
            if "physical_consistency" in consistency_results:
                return 1 if consistency_results["physical_consistency"] else 0
        except:
            # Fall back to legacy methods
            pass

        # Legacy fallback
        return 1 if (self.psi_s(fact) == 1 and self.psi_t(fact, transport_mode) == 1) else 0

    def get_all_facts(self):
        """
        Get all facts in the knowledge graph (entities + relationships).
        Returns list of fact dictionaries.
        """
        facts = []

        # Add all entities as facts
        for entity_id, entity_data in self.entities.items():
            facts.append(entity_data)

        # Add all relationships as facts
        for rel in self.relationships:
            # Enhance relationship with subject/object spatiotemporal data
            subject_entity = self.entities.get(rel["subject_entity_id"])
            if subject_entity:
                rel_with_st = rel.copy()
                rel_with_st["spatiotemporal"] = subject_entity.get("spatiotemporal", {})
                facts.append(rel_with_st)

        return facts

    def get_domain_rules(self, domain: str):
        """Get domain-specific rules from the ontology."""
        return self.ontology.get_domain_rules(domain)

    def get_error_types_by_module(self, module: str):
        """Get error types detected by a specific verification module."""
        return self.ontology.get_error_types_by_module(module)

    def validate_fact_against_domain(self, fact: dict, domain: str):
        """Validate a fact against domain-specific rules."""
        violations = []

        domain_rules = self.get_domain_rules(domain)
        for rule in domain_rules:
            # Apply domain-specific validation logic
            if rule.rule_type == "tolerance" and "tolerance" in str(fact).lower():
                # Check tolerance compliance
                if "deviation_mm" in fact:
                    deviation = abs(fact.get("deviation_mm", 0))
                    tolerance = rule.parameters.get("tolerance_range_mm", 0.1)
                    if deviation > tolerance:
                        violations.append(f"Tolerance violation: {deviation}mm > {tolerance}mm")

            elif rule.rule_type == "protocol" and fact.get("relationship_type") == "transferred":
                # Check protocol compliance for transfers
                transfer_time = fact.get("transfer_duration_minutes", 0)
                min_times = rule.parameters.get("min_transfer_times", {})
                from_unit = fact.get("from_unit")
                to_unit = fact.get("to_unit")

                if (from_unit, to_unit) in min_times:
                    required = min_times[(from_unit, to_unit)]
                    if transfer_time < required:
                        violations.append(f"Protocol violation: {transfer_time}min < {required}min required")

        return violations

    def check_fact_consistency(self, fact: dict):
        """Comprehensive consistency check using ontology."""
        results = {
            "ontology_validation": [],
            "physical_consistency": {},
            "domain_violations": []
        }

        # Ontology validation
        if "entity_id" in fact or "subject_entity_id" in fact:
            results["ontology_validation"] = self.ontology.validate_entity(fact)
        elif "relationship_id" in fact:
            results["ontology_validation"] = self.ontology.validate_relationship(fact)

        # Physical consistency
        results["physical_consistency"] = self.ontology.check_physical_consistency(fact, self.get_all_facts())

        # Domain-specific validation
        domain = self._infer_fact_domain(fact)
        if domain:
            results["domain_violations"] = self.validate_fact_against_domain(fact, domain)

        return results

    def _infer_fact_domain(self, fact: dict) -> str:
        """Infer the domain of a fact based on its characteristics."""
        entity_class = fact.get("entity_class", "")
        relationship_type = fact.get("relationship_type", "")

        if any(term in entity_class.lower() for term in ["blade", "engine", "inspection", "measurement"]):
            return "aerospace"
        elif any(term in entity_class.lower() for term in ["patient", "care", "clinical", "transfer"]):
            return "healthcare"
        elif any(term in entity_class.lower() for term in ["safety", "incident", "aviation"]):
            return "aviation"
        elif any(term in entity_class.lower() for term in ["cad", "assembly", "geometric"]):
            return "cad"

        return "general"

    def __str__(self):
        return f"KnowledgeGraph(entities={len(self.entities)}, relationships={len(self.relationships)}, ontology={len(self.ontology.entity_classes)} classes)"

def create_sample_knowledge_graph():
    """Create a sample knowledge graph with aerospace entities."""
    kg = KnowledgeGraph()

    # Add engine sets (skip validation for demo)
    for i, engine_id in enumerate(ENGINE_SETS[:4]):
        kg.add_entity(
            entity_id=engine_id,
            entity_class="EngineSet",
            attributes={
                "entityID": engine_id,
                "description": f"Aircraft Engine Set {i+1}",
                "engine_model": f"Model-{i+1}",
                "thrust_rating": f"{random.randint(50000, 100000)} lbf"
            },
            spatiotemporal={
                "x_coord": round(random.uniform(-50, 50), 1),
                "y_coord": round(random.uniform(-50, 50), 1),
                "z_coord": round(random.uniform(0, 100), 1),
                "timestamp": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat() + "Z"
            },
            validate=False  # Skip validation for demo
        )
    
    # Add blade components
    for i, blade_id in enumerate(BLADE_COMPONENTS[:6]):
        kg.add_entity(
            entity_id=blade_id,
            entity_class="Blade",
            attributes={
                "componentID": blade_id,
                "material": random.choice(["Titanium", "Inconel", "Nickel Alloy"]),
                "lifecycle_hours": round(random.uniform(1000, 5000), 0),
                "blade_type": random.choice(["Compressor", "Turbine", "Fan"]),
                "alloy_composition": "Ti-6Al-4V"
            },
            spatiotemporal={
                "x_coord": round(random.uniform(-10, 10), 1),
                "y_coord": round(random.uniform(-10, 10), 1),
                "z_coord": round(random.uniform(50, 300), 1),
                "timestamp": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat() + "Z"
            },
            validate=False  # Skip validation for demo
        )
    
    # Add measurements
    for i, measurement_id in enumerate(MEASUREMENT_IDS[:3]):
        for j, blade_id in enumerate(BLADE_COMPONENTS[:3]):
            entity_id = f"{measurement_id}_{blade_id}"
            kg.add_entity(
                entity_id=entity_id,
                entity_class="InspectionMeasurement",
                attributes={
                    "measurementID": measurement_id,
                    "componentID": blade_id,
                    "featureName": BLADE_FEATURES[i % len(BLADE_FEATURES)],
                    "nominalValue_mm": round(random.uniform(2.0, 5.0), 2),
                    "actualValue_mm": round(random.uniform(2.0, 5.0), 2),
                    "toleranceRange_mm": f"±{round(random.uniform(0.080, 0.150), 3)}",
                    "surfaceSide": SURFACE_SIDES[i % len(SURFACE_SIDES)],
                    "inspectionTool": "3D_Scanner_Unit",
                    "status": "PASS",
                    "deviation_mm": round(random.uniform(-0.05, 0.05), 3),
                    "tolerance_mm": 0.1,
                    "operator_id": f"Tech_{random.randint(1, 10)}"
                },
                spatiotemporal={
                    "x_coord": round(random.uniform(-10, 10), 1),
                    "y_coord": round(random.uniform(-10, 10), 1),
                    "z_coord": round(random.uniform(50, 300), 1),
                    "timestamp": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat() + "Z"
                },
                validate=False  # Skip validation for demo
            )
    
    # Add relationships
    # 1. Engine sets contain blades
    for i, engine_id in enumerate(ENGINE_SETS[:4]):
        blade_id = BLADE_COMPONENTS[i % len(BLADE_COMPONENTS)]
        kg.add_relationship(
            rel_id=f"R_contain_{i}",
            subject_id=engine_id,
            relation_type="containsBlade",
            object_id=blade_id,
            validate=False  # Skip validation for demo
        )

    # 2. Blades have measurements
    rel_count = 0
    for blade_id in BLADE_COMPONENTS[:3]:
        for measurement_id in MEASUREMENT_IDS[:3]:
            entity_id = f"{measurement_id}_{blade_id}"
            if entity_id in kg.entities:
                kg.add_relationship(
                    rel_id=f"R_measure_{rel_count}",
                    subject_id=blade_id,
                    relation_type="hasMeasurement",
                    object_id=entity_id,
                    validate=False  # Skip validation for demo
                )
                rel_count += 1
    
    return kg 