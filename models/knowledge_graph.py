import random
from datetime import datetime, timedelta
from .constants import (
    ENTITY_CLASSES, RELATIONSHIP_TYPES, BLADE_COMPONENTS, 
    ENGINE_SETS, MEASUREMENT_IDS, BLADE_FEATURES, SURFACE_SIDES
)

class KnowledgeGraph:
    """Simplified 4D Spatiotemporal Knowledge Graph representation."""
    def __init__(self):
        self.entities = {}  # id -> Entity data
        self.relationships = []  # List of relationship objects
        self.entity_classes = set(ENTITY_CLASSES)  # Set of all entity classes
        self.relation_types = set(RELATIONSHIP_TYPES)  # Set of all relation types
        
    def add_entity(self, entity_id, entity_class, attributes, spatiotemporal=None):
        """Add an entity to the knowledge graph."""
        self.entities[entity_id] = {
            "entity_id": entity_id,
            "entity_class": entity_class,
            "attributes": attributes,
            "spatiotemporal": spatiotemporal or {}
        }
        self.entity_classes.add(entity_class)
    
    def add_relationship(self, rel_id, subject_id, relation_type, object_id):
        """Add a relationship to the knowledge graph."""
        self.relationships.append({
            "relationship_id": rel_id,
            "subject_entity_id": subject_id,
            "relationship_type": relation_type,
            "object_entity_id": object_id
        })
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
    
    def __str__(self):
        return f"KnowledgeGraph(entities={len(self.entities)}, relationships={len(self.relationships)})"

def create_sample_knowledge_graph():
    """Create a sample knowledge graph with aerospace entities."""
    kg = KnowledgeGraph()
    
    # Add engine sets
    for i, engine_id in enumerate(ENGINE_SETS[:4]):
        kg.add_entity(
            entity_id=engine_id,
            entity_class="EngineSet",
            attributes={
                "entityID": engine_id,
                "description": f"Aircraft Engine Set {i+1}"
            },
            spatiotemporal={
                "x_coord": round(random.uniform(-50, 50), 1),
                "y_coord": round(random.uniform(-50, 50), 1),
                "z_coord": round(random.uniform(0, 100), 1),
                "timestamp": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat() + "Z"
            }
        )
    
    # Add blade components
    for i, blade_id in enumerate(BLADE_COMPONENTS[:6]):
        kg.add_entity(
            entity_id=blade_id,
            entity_class="Blade",
            attributes={
                "componentID": blade_id,
                "material": random.choice(["Titanium", "Inconel", "Nickel Alloy"]),
                "lifecycle_hours": round(random.uniform(1000, 5000), 0)
            },
            spatiotemporal={
                "x_coord": round(random.uniform(-10, 10), 1),
                "y_coord": round(random.uniform(-10, 10), 1),
                "z_coord": round(random.uniform(50, 300), 1),
                "timestamp": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat() + "Z"
            }
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
                    "status": "PASS"
                },
                spatiotemporal={
                    "x_coord": round(random.uniform(-10, 10), 1),
                    "y_coord": round(random.uniform(-10, 10), 1),
                    "z_coord": round(random.uniform(50, 300), 1),
                    "timestamp": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat() + "Z"
                }
            )
    
    # Add relationships
    # 1. Engine sets contain blades
    for i, engine_id in enumerate(ENGINE_SETS[:4]):
        blade_id = BLADE_COMPONENTS[i % len(BLADE_COMPONENTS)]
        kg.add_relationship(
            rel_id=f"R_contain_{i}",
            subject_id=engine_id,
            relation_type="containsBlade",
            object_id=blade_id
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
                    object_id=entity_id
                )
                rel_count += 1
    
    return kg 