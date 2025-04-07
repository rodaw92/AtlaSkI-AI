import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import random
import json
from datetime import datetime, timedelta
import math
from collections import defaultdict
import itertools
import base64
from io import BytesIO

# -----------------------------------------------------------------------------
# Configuration and Constants
# -----------------------------------------------------------------------------
# Configure page settings with improved layout
st.set_page_config(
    page_title="ATLASky-AI Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, professional appearance
def local_css():
    return """
    <style>
        /* Main page styling */
        .main {
            background-color: #f9fafb;
        }
        
        /* Header styling */
        .header-container {
            background-color: #1e3a8a;
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            background-image: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        }
        
        .header-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        .header-subtitle {
            font-size: 1.2rem;
            opacity: 0.8;
        }
        
        /* Card styling */
        .card {
            background-color: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            margin-bottom: 1rem;
        }
        
        .metric-card {
            background-color: white;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            text-align: center;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #1e3a8a;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #6b7280;
            margin-bottom: 0.5rem;
        }
        
        /* Status indicators */
        .status-verified {
            display: inline-block;
            background-color: #10b981;
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 500;
        }
        
        .status-rejected {
            display: inline-block;
            background-color: #ef4444;
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 500;
        }
        
        .status-waiting {
            display: inline-block;
            background-color: #f59e0b;
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 500;
        }
        
        /* Quality indicator colors */
        .quality-high {
            color: #10b981;
            font-weight: 600;
        }
        
        .quality-medium {
            color: #3b82f6;
            font-weight: 600;
        }
        
        .quality-spatial {
            color: #f59e0b;
            font-weight: 600;
        }
        
        .quality-external {
            color: #8b5cf6;
            font-weight: 600;
        }
        
        .quality-semantic {
            color: #06b6d4;
            font-weight: 600;
        }
        
        .quality-low {
            color: #ef4444;
            font-weight: 600;
        }
        
        /* Alert boxes */
        .alert-info {
            background-color: #eff6ff;
            border-left: 4px solid #3b82f6;
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 4px;
        }
        
        .alert-warning {
            background-color: #fffbeb;
            border-left: 4px solid #f59e0b;
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 4px;
        }
        
        .alert-success {
            background-color: #ecfdf5;
            border-left: 4px solid #10b981;
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 4px;
        }
        
        .alert-error {
            background-color: #fef2f2;
            border-left: 4px solid #ef4444;
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 4px;
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #1e293b;
            color: white;
        }
        
        /* Improve button styling */
        .stButton>button {
            width: 100%;
            border-radius: 6px;
            font-weight: 500;
            transition: all 0.2s;
        }
        
        /* Data tables */
        .dataframe {
            border-collapse: collapse;
            width: 100%;
            border-radius: 8px;
            overflow: hidden;
        }
        
        .dataframe th {
            background-color: #f3f4f6;
            padding: 0.75rem;
            text-align: left;
            font-weight: 600;
            color: #374151;
        }
        
        .dataframe td {
            padding: 0.75rem;
            border-top: 1px solid #e5e7eb;
        }
        
        .dataframe tr:hover {
            background-color: #f9fafb;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            font-size: 14px;
            font-weight: 500;
            background-color: #f3f4f6;
            border-radius: 6px 6px 0 0;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: white;
            border-top: 3px solid #1e3a8a;
        }
    </style>
    """

st.markdown(local_css(), unsafe_allow_html=True)

# Entity classes and relationship types (moved from inline to config)
ENTITY_CLASSES = [
    "AerospaceEntity", "Aircraft", "Blade", "CADFeature", "CADModel", 
    "EngineSet", "Feature", "InspectionMeasurement", "PhysicalEntity", 
    "SpatiotemporalEntity"
]

RELATIONSHIP_TYPES = [
    "adjacentTo", "caused", "containsBlade", "contributedTo", "hasAircraft",
    "hasMeasurement", "hasOperator", "isContainedBy", "ledTo", "locatedAt",
    "occurredDuring", "relatedTo"
]

BLADE_FEATURES = [
    "Blade Root - Simple Mount", 
    "High Pressure - Pitch Distance",
    "Leading Edge - Pressure Side",
    "Trailing Edge - Suction Side",
    "Blade Tip - Clearance Gap"
]

BLADE_COMPONENTS = [
    "TurbineBlade_Alpha", "TurbineBlade_Beta", "TurbineBlade_Gamma",
    "TurbineBlade_Delta", "TurbineBlade_Epsilon", "TurbineBlade_Zeta",
    "TurbineBlade_Eta", "TurbineBlade_Theta", "TurbineBlade_Iota",
    "TurbineBlade_Kappa"
]

ENGINE_SETS = [
    "EngineSet_1", "EngineSet_2", "EngineSet_3", "EngineSet_4", 
    "EngineSet_5", "EngineSet_6", "EngineSet_7"
]

MEASUREMENT_IDS = [
    "BladeRoot_Simple_C1", "HighPressureSidePitch_G2", 
    "LeadingEdge_Pressure_Z3", "TrailingEdge_Suction_T4",
    "BladeTip_Clearance_E5"
]

SURFACE_SIDES = ["Structural", "Pressure", "Suction", "Tip", "Root"]

# Mapping of quality levels to colors for visualization
QUALITY_COLOR_MAP = {
    "high_quality": "#10b981",  # Green
    "medium_quality": "#3b82f6",  # Blue
    "spatial_issue": "#f59e0b",  # Orange
    "external_ref": "#8b5cf6",  # Purple
    "semantic_issue": "#06b6d4",  # Teal
    "low_quality": "#ef4444"     # Red
}

# -----------------------------------------------------------------------------
# Knowledge Graph Implementation
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Data Generation Functions 
# -----------------------------------------------------------------------------
def generate_candidate_fact_with_quality(quality_level, introduce_shift=False):
    """
    Generate a random aerospace inspection candidate.
    
    Parameters:
    quality_level (str): Determines the quality of the generated fact
    introduce_shift (bool): If True, introduces performance shifts to trigger AAIC
    """
    # Select components for the fact
    engine_set = random.choice(ENGINE_SETS)
    blade_component = random.choice(BLADE_COMPONENTS)
    measurement_id = random.choice(MEASUREMENT_IDS)
    feature_name = random.choice(BLADE_FEATURES)
    
    # Generate report ID
    current_date = datetime.now().strftime("%Y-%m-%d")
    report_id = f"{engine_set}_{current_date}"
    
    # Generate random inspection values
    nominal_value = round(random.uniform(2.0, 5.0), 2)
    
    # Determine actual_value based on quality
    if quality_level == "high_quality":
        # Very small deviation - highly reliable measurement
        actual_value = round(nominal_value + random.uniform(-0.01, 0.01), 3)
    elif quality_level == "medium_quality" or quality_level == "spatial_issue":
        # Moderate deviation but within tolerance
        actual_value = round(nominal_value + random.uniform(-0.03, 0.03), 3)
    else:
        # Larger deviation
        actual_value = round(nominal_value + random.uniform(-0.07, 0.07), 3)
    
    deviation = round(actual_value - nominal_value, 3)
    tolerance = f"±{round(random.uniform(0.080, 0.150), 3)}"
    surface_side = random.choice(SURFACE_SIDES)
    
    # Generate timestamps (within the past 24 hours)
    now = datetime.now()
    random_hours = random.randint(0, 23)
    random_minutes = random.randint(0, 59)
    random_seconds = random.randint(0, 59)
    timestamp = (now - timedelta(hours=random_hours, minutes=random_minutes, seconds=random_seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Generate spatial coordinates
    x_coord = round(random.uniform(-10.0, 10.0), 2)
    y_coord = round(random.uniform(-10.0, 10.0), 2)
    z_coord = round(random.uniform(50.0, 300.0), 1)
    
    # Create relationships
    relationships = [
        {
            "relationship_id": "R1",
            "subject_entity_id": engine_set,
            "relationship_type": "containsBlade",
            "object_entity_id": blade_component
        },
        {
            "relationship_id": "R2",
            "subject_entity_id": blade_component,
            "relationship_type": "hasMeasurement",
            "object_entity_id": measurement_id
        }
    ]
    
    # Base fact structure
    fact = {
        "spatiotemporal_inspection_data": {
            "report_id": report_id,
            "inspection_data": {
                "inspection_measurements": [
                    {
                        "component_id": blade_component,
                        "measurement_id": measurement_id,
                        "feature_name": feature_name,
                        "nominal_value_mm": nominal_value,
                        "actual_value_mm": actual_value,
                        "deviation_mm": deviation,
                        "tolerance_range_mm": tolerance,
                        "surface_side": surface_side,
                        "inspection_tool": "3D_Scanner_Unit",
                        "status": "PASS" if abs(deviation) <= float(tolerance.replace("±", "")) else "FAIL"
                    }
                ]
            },
            "temporal_data": [
                {
                    "entity_id": measurement_id,
                    "timestamp": timestamp,
                    "event_type_or_feature": feature_name
                }
            ],
            "spatial_data": [
                {
                    "entity_id": measurement_id,
                    "coordinates": {
                        "x_coord": x_coord,
                        "y_coord": y_coord,
                        "z_coord": z_coord
                    }
                }
            ],
            "relationships": relationships
        }
    }
    
    # Modify fact based on quality level
    if quality_level == "high_quality":
        # No modifications - this is a high-quality fact that should pass LOV easily
        pass
    
    elif quality_level == "medium_quality":
        # Slightly increase deviation but still valid
        fact["spatiotemporal_inspection_data"]["inspection_data"]["inspection_measurements"][0]["deviation_mm"] = round(deviation * 1.5, 3)
    
    elif quality_level == "spatial_issue":
        # Introduce a potential spatial issue - mismatched coordinates or entity IDs
        if random.random() < 0.5:
            # Slightly change the entity_id in spatial data
            fact["spatiotemporal_inspection_data"]["spatial_data"][0]["entity_id"] = measurement_id + "_variant"
        else:
            # Use unusual coordinates that need spatial verification
            fact["spatiotemporal_inspection_data"]["spatial_data"][0]["coordinates"]["z_coord"] = round(z_coord * 1.5, 1)
    
    elif quality_level == "external_ref":
        # Add references or terminology that might need web search verification
        extra_terms = ["Proprietary_Method_XR7", "Advanced_Inspection_Protocol", "ISO9001_Certification"]
        fact["spatiotemporal_inspection_data"]["inspection_data"]["inspection_measurements"][0]["reference_standard"] = random.choice(extra_terms)
    
    elif quality_level == "semantic_issue":
        # Add slightly unusual feature descriptions or values that need semantic checking
        unusual_features = [
            "Modified Blade Edge Profile",
            "Experimental Coating Layer",
            "Non-standard Surface Treatment"
        ]
        fact["spatiotemporal_inspection_data"]["inspection_data"]["inspection_measurements"][0]["feature_name"] = random.choice(unusual_features)
    
    elif quality_level == "low_quality":
        # Introduce multiple issues that would cause rejection
        # 1. Invalid entity
        fact["spatiotemporal_inspection_data"]["inspection_data"]["inspection_measurements"][0]["component_id"] = f"Unknown_{random.randint(1000, 9999)}"
        # 2. Bad timestamp
        fact["spatiotemporal_inspection_data"]["temporal_data"][0]["timestamp"] = f"202X-{random.randint(1, 12)}-{random.randint(1, 30)}"
        # 3. Missing coordinate
        coord_to_remove = random.choice(["x_coord", "y_coord", "z_coord"])
        del fact["spatiotemporal_inspection_data"]["spatial_data"][0]["coordinates"][coord_to_remove]
    
    # Introduce additional shift to trigger AAIC
    if introduce_shift:
        # Choose random module to affect
        module_to_shift = random.choice(["LOV", "POV", "MAV", "WSV", "ESV"])
        
        # Make more dramatic quality changes to increase performance variation
        if module_to_shift == "LOV":
            # Make entity class more ambiguous to reduce LOV performance
            fact["spatiotemporal_inspection_data"]["inspection_data"]["inspection_measurements"][0]["component_id"] += "_significantly_modified"
            # Also add invalid component to dramatically reduce precision
            if random.random() < 0.5:
                fact["spatiotemporal_inspection_data"]["inspection_data"]["inspection_measurements"].append({
                    "component_id": "Invalid_Component_" + str(random.randint(1000, 9999)),
                    "measurement_id": "Invalid_" + measurement_id,
                    "feature_name": "Invalid Feature",
                    "nominal_value_mm": 999.99,
                    "actual_value_mm": 0.01,
                    "deviation_mm": -999.98,
                    "tolerance_range_mm": "±0.1",
                    "surface_side": "Invalid",
                    "inspection_tool": "3D_Scanner_Unit",
                    "status": "FAIL"
                })
        
        elif module_to_shift == "POV":
            # Add unusual attribute that might affect POV
            fact["spatiotemporal_inspection_data"]["inspection_data"]["inspection_measurements"][0]["non_standard_attribute"] = True
            fact["spatiotemporal_inspection_data"]["inspection_data"]["inspection_measurements"][0]["severely_out_of_domain"] = True
            fact["spatiotemporal_inspection_data"]["inspection_data"]["inspection_measurements"][0]["unknown_context"] = "completely unexpected value"
        
        elif module_to_shift == "MAV":
            # Introduce severe temporal inconsistency
            current_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            shifted_time = current_time - timedelta(days=random.randint(30, 100))  # Much bigger shift
            fact["spatiotemporal_inspection_data"]["temporal_data"][0]["timestamp"] = shifted_time.isoformat().replace('+00:00', 'Z')
            # Also add spatial inconsistency
            fact["spatiotemporal_inspection_data"]["spatial_data"][0]["coordinates"]["z_coord"] = round(z_coord * 10, 1)  # 10x shift
        
        elif module_to_shift == "WSV":
            # Add unusual domain-specific terms that might affect WSV recall
            fact["spatiotemporal_inspection_data"]["inspection_data"]["inspection_measurements"][0]["proprietary_code"] = f"XYZ{random.randint(1000, 9999)}"
            fact["spatiotemporal_inspection_data"]["inspection_data"]["inspection_measurements"][0]["external_reference"] = "ISO9001:NonStandard"
            fact["spatiotemporal_inspection_data"]["inspection_data"]["inspection_measurements"][0]["certification_level"] = "OutOfDomain_Level_" + str(random.randint(10, 50))
        
        elif module_to_shift == "ESV":
            # Add semantically divergent content
            fact["spatiotemporal_inspection_data"]["inspection_data"]["inspection_measurements"][0]["semantic_context"] = "experimental_non_standard_procedure"
            fact["spatiotemporal_inspection_data"]["inspection_data"]["inspection_measurements"][0]["semantic_category"] = "completely_unknown_terminology"
            fact["spatiotemporal_inspection_data"]["inspection_data"]["inspection_measurements"][0]["interpretation"] = "outside_of_embedding_space"
        
        # Tag the fact as containing a performance shift
        fact["contains_performance_shift"] = module_to_shift
    
    return fact, quality_level

def generate_candidate_fact(introduce_shift=False):
    """Generate a random candidate fact with varying quality levels."""
    # Define the distribution of quality levels
    quality_levels = [
        "high_quality",   # 25% - Will pass LOV
        "medium_quality", # 20% - Will pass LOV+POV
        "spatial_issue",  # 15% - Will need MAV
        "external_ref",   # 15% - Will need WSV
        "semantic_issue", # 15% - Will need ESV
        "low_quality"     # 10% - Likely to be rejected
    ]
    
    weights = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]
    
    # Randomly select a quality level based on the weights
    selected_quality = random.choices(quality_levels, weights=weights, k=1)[0]
    
    # Generate the fact with the selected quality level
    fact, quality = generate_candidate_fact_with_quality(selected_quality, introduce_shift)
    return fact, quality

# -----------------------------------------------------------------------------
# Verification Modules
# -----------------------------------------------------------------------------
class VerificationModule:
    """Base class for verification modules in the RMMVe process."""
    def __init__(self, name, weight, alpha, threshold):
        self.name = name
        self.weight = weight  # wi in Eq.1
        self.alpha = alpha    # αi in Eq.1
        self.threshold = threshold  # θi for early termination
        self.performance_history = []
    
    def compute_metrics(self, candidate_fact, knowledge_graph, fact_quality):
        """Compute the two metrics used in confidence calculation."""
        raise NotImplementedError("Subclasses must implement compute_metrics")
    
    def compute_confidence(self, candidate_fact, knowledge_graph, fact_quality=None):
        """
        Compute confidence score using Eq.1 from RMMVe:
        C_module_i(dk) = wi(αi·Metric1^(i)(dk) + (1-αi)·Metric2^(i)(dk))
        """
        metric1, metric2 = self.compute_metrics(candidate_fact, knowledge_graph, fact_quality)
        confidence = self.weight * (self.alpha * metric1 + (1 - self.alpha) * metric2)
        
        # Record metrics for performance history
        self.performance_history.append({
            "metric1": metric1,
            "metric2": metric2,
            "confidence": confidence,
            "timestamp": time.time()
        })
        
        return confidence, metric1, metric2

class LocalOntologyVerification(VerificationModule):
    """
    Local Ontology Verification (LOV) module.
    Uses precision and recall against the local ontology.
    """
    def __init__(self, weight, alpha, threshold):
        super().__init__("LOV", weight, alpha, threshold)
    
    def compute_metrics(self, candidate_fact, knowledge_graph, fact_quality):
        """
        Compute P_LOV (precision) and R_LOV (recall) as defined in document.
        
        Uses fact_quality to adjust scores to demonstrate different verification paths.
        """
        # Check if the fact contains a performance shift for LOV
        has_shift = candidate_fact.get("contains_performance_shift") == "LOV"
        
        # Base precision and recall based on quality
        if fact_quality == "high_quality":
            # Set very high precision and recall for high quality facts to exceed threshold
            base_precision = 0.98 if not has_shift else 0.60
            base_recall = 0.95 if not has_shift else 0.55
        elif fact_quality == "medium_quality":
            # High but not enough to exceed threshold
            base_precision = 0.85 if not has_shift else 0.50
            base_recall = 0.80 if not has_shift else 0.45
        elif fact_quality == "spatial_issue":
            base_precision = 0.80 if not has_shift else 0.45
            base_recall = 0.75 if not has_shift else 0.40
        elif fact_quality == "external_ref":
            base_precision = 0.75 if not has_shift else 0.40
            base_recall = 0.70 if not has_shift else 0.35
        elif fact_quality == "semantic_issue":
            base_precision = 0.70 if not has_shift else 0.35
            base_recall = 0.65 if not has_shift else 0.30
        else:  # low_quality
            base_precision = 0.50 if not has_shift else 0.20
            base_recall = 0.45 if not has_shift else 0.15
        
        # Add small random variation
        precision = min(1.0, max(0.0, base_precision + random.uniform(-0.05, 0.05)))
        recall = min(1.0, max(0.0, base_recall + random.uniform(-0.05, 0.05)))
        
        return precision, recall

class PublicOntologyVerification(VerificationModule):
    """
    Public Ontology Verification (POV) module.
    Uses accuracy and coverage metrics.
    """
    def __init__(self, weight, alpha, threshold):
        super().__init__("POV", weight, alpha, threshold)
    
    def compute_metrics(self, candidate_fact, knowledge_graph, fact_quality):
        """
        Compute A_POV (accuracy) and CA_POV (coverage) as defined in document.
        
        Uses fact_quality to adjust scores to demonstrate different verification paths.
        """
        # Check if the fact contains a performance shift for POV
        has_shift = candidate_fact.get("contains_performance_shift") == "POV"
        
        # Base accuracy and coverage based on quality
        if fact_quality == "high_quality":
            # High quality should already terminate at LOV, but still set high values
            base_accuracy = 0.93 if not has_shift else 0.70
            base_coverage = 0.88 if not has_shift else 0.65
        elif fact_quality == "medium_quality":
            # Very high values to allow early termination at POV
            base_accuracy = 0.95 if not has_shift else 0.72
            base_coverage = 0.92 if not has_shift else 0.68
        elif fact_quality == "spatial_issue":
            # Good but not enough to exceed threshold
            base_accuracy = 0.80 if not has_shift else 0.60
            base_coverage = 0.75 if not has_shift else 0.55
        elif fact_quality == "external_ref":
            base_accuracy = 0.75 if not has_shift else 0.55
            base_coverage = 0.70 if not has_shift else 0.50
        elif fact_quality == "semantic_issue":
            base_accuracy = 0.70 if not has_shift else 0.50
            base_coverage = 0.65 if not has_shift else 0.45
        else:  # low_quality
            base_accuracy = 0.50 if not has_shift else 0.35
            base_coverage = 0.45 if not has_shift else 0.30
        
        # Add small random variation
        accuracy = min(1.0, max(0.0, base_accuracy + random.uniform(-0.02, 0.02)))
        coverage = min(1.0, max(0.0, base_coverage + random.uniform(-0.02, 0.02)))
        
        return accuracy, coverage

class Agent:
    """Agent for Multi-Agent Verification (MAV)."""
    def __init__(self, name, reliability):
        self.name = name  # Agent name (Temporal, Spatial, or 4D Consistency)
        self.reliability = reliability  # Reliability factor rᵢ ∈ [0,1]
    
    def validate(self, candidate_fact, fact_quality):
        """
        Validate a candidate fact.
        Returns 1 if valid, 0 otherwise.
        Uses fact_quality to determine validation outcome.
        """
        # Check if this fact contains a performance shift for MAV
        has_shift = candidate_fact.get("contains_performance_shift") == "MAV"
        
        if has_shift:
            # Lower validation probability with shift
            return random.choices([0, 1], weights=[0.7, 0.3], k=1)[0]
        
        if self.name == "Temporal":
            # Temporal agent validation
            if fact_quality in ["high_quality", "medium_quality"]:
                return 1
            elif fact_quality == "spatial_issue":
                return 1  # Temporal aspects are fine even with spatial issues
            elif fact_quality == "external_ref":
                return random.choices([0, 1], weights=[0.2, 0.8], k=1)[0]
            elif fact_quality == "semantic_issue":
                return random.choices([0, 1], weights=[0.3, 0.7], k=1)[0]
            else:  # low_quality
                return 0
        
        elif self.name == "Spatial":
            # Spatial agent validation
            if fact_quality in ["high_quality", "medium_quality"]:
                return 1
            elif fact_quality == "spatial_issue":
                return 0  # Specifically fails on spatial issues
            elif fact_quality == "external_ref":
                return random.choices([0, 1], weights=[0.3, 0.7], k=1)[0]
            elif fact_quality == "semantic_issue":
                return random.choices([0, 1], weights=[0.4, 0.6], k=1)[0]
            else:  # low_quality
                return 0
        
        elif self.name == "4D Consistency":
            # 4D Consistency agent validation
            if fact_quality == "high_quality":
                return 1
            elif fact_quality == "medium_quality":
                return random.choices([0, 1], weights=[0.1, 0.9], k=1)[0]
            elif fact_quality == "spatial_issue":
                return 0  # Fails on 4D consistency with spatial issues
            elif fact_quality == "external_ref":
                return random.choices([0, 1], weights=[0.4, 0.6], k=1)[0]
            elif fact_quality == "semantic_issue":
                return random.choices([0, 1], weights=[0.5, 0.5], k=1)[0]
            else:  # low_quality
                return 0
        
        return 0  # Default case

class MultiAgentVerification(VerificationModule):
    """
    Multi-Agent Verification (MAV) module using Shapley value integration.
    Uses consensus score and reliability metrics.
    """
    def __init__(self, weight, alpha, threshold):
        super().__init__("MAV", weight, alpha, threshold)
        # Create the three agents mentioned in the paper
        self.agents = [
            Agent("Temporal", 0.9),
            Agent("Spatial", 0.85),
            Agent("4D Consistency", 0.95)
        ]
    
    def compute_shapley_values(self, validation_results):
        """
        Compute Shapley values for each agent based on their validation results.
        This follows the formula in the paper:
        
        ϕ_Ai = ∑_{S⊆N∖{Ai}} (|S|!(|N|-|S|-1)!/|N|!) [v(S∪{Ai})-v(S)]
        
        where v(S) = (1/m) ∑_{Aj∈S} V_Aj(dk)
        """
        agents = list(validation_results.keys())
        n = len(agents)
        shapley_values = {}
        
        # Calculate Shapley value for each agent
        for agent in agents:
            shapley_value = 0
            
            # Generate all subsets of agents excluding the current agent
            other_agents = [a for a in agents if a != agent]
            
            for subset_size in range(len(other_agents) + 1):
                for subset in itertools.combinations(other_agents, subset_size):
                    s = len(subset)
                    
                    # Calculate |S|!(|N|-|S|-1)!/|N|!
                    subset_weight = (math.factorial(s) * math.factorial(n - s - 1)) / math.factorial(n)
                    
                    # Calculate v(S)
                    if subset:
                        v_s = sum(validation_results[a] for a in subset) / len(subset)
                    else:
                        v_s = 0
                    
                    # Calculate v(S∪{Ai})
                    subset_with_agent = list(subset) + [agent]
                    v_s_with_agent = sum(validation_results[a] for a in subset_with_agent) / len(subset_with_agent)
                    
                    # Add to Shapley value
                    shapley_value += subset_weight * (v_s_with_agent - v_s)
            
            shapley_values[agent] = shapley_value
        
        return shapley_values
    
    def compute_metrics(self, candidate_fact, knowledge_graph, fact_quality):
        """
        Compute CS_MAV (consensus score) and R_MAV (reliability) as defined in document.
        
        CS_MAV = ∑(i=1 to m) ϕ_Ai * V_Ai(dk)
        R_MAV = ∑(i=1 to m) ϕ_Ai * r_Ai
        """
        # Get validation results from each agent
        validation_results = {}
        for agent in self.agents:
            validation_results[agent.name] = agent.validate(candidate_fact, fact_quality)
        
        # Compute Shapley values
        shapley_values = self.compute_shapley_values(validation_results)
        
        # Calculate consensus score (CS_MAV)
        consensus_score = sum(shapley_values[agent.name] * validation_results[agent.name] 
                              for agent in self.agents)
        
        # Calculate reliability score (R_MAV)
        reliability_score = sum(shapley_values[agent.name] * agent.reliability 
                               for agent in self.agents)
        
        # For spatial_issue facts, increase metrics to ensure they pass MAV
        # (but only if there's no performance shift)
        has_shift = candidate_fact.get("contains_performance_shift") == "MAV"
        if fact_quality == "spatial_issue" and not has_shift:
            # High enough to exceed threshold for early termination
            consensus_score = min(1.0, consensus_score * 1.5)
            reliability_score = min(1.0, reliability_score * 1.5)
        
        return consensus_score, reliability_score

class WebSearchVerification(VerificationModule):
    """
    Web Search Verification (WSV) module.
    Uses recall and F1 score metrics.
    """
    def __init__(self, weight, alpha, threshold):
        super().__init__("WSV", weight, alpha, threshold)
    
    def compute_metrics(self, candidate_fact, knowledge_graph, fact_quality):
        """
        Compute R_WSV (recall) and F1_WSV (F1 score) as defined in document.
        
        Uses fact_quality to adjust scores to demonstrate different verification paths.
        """
        # Check if the fact contains a performance shift for WSV
        has_shift = candidate_fact.get("contains_performance_shift") == "WSV"
        
        # Base recall and precision (for F1) based on quality
        if fact_quality == "high_quality":
            base_recall = 0.90 if not has_shift else 0.65
            base_precision = 0.92 if not has_shift else 0.67
        elif fact_quality == "medium_quality":
            base_recall = 0.85 if not has_shift else 0.60
            base_precision = 0.87 if not has_shift else 0.62
        elif fact_quality == "spatial_issue":
            base_recall = 0.80 if not has_shift else 0.55
            base_precision = 0.82 if not has_shift else 0.57
        elif fact_quality == "external_ref":
            # Very high values for external_ref to exceed threshold
            base_recall = 0.98 if not has_shift else 0.73
            base_precision = 0.98 if not has_shift else 0.73
        elif fact_quality == "semantic_issue":
            base_recall = 0.75 if not has_shift else 0.50
            base_precision = 0.78 if not has_shift else 0.53
        else:  # low_quality
            base_recall = 0.40 if not has_shift else 0.25
            base_precision = 0.45 if not has_shift else 0.30
        
        # Add small random variation
        recall = min(1.0, max(0.0, base_recall + random.uniform(-0.02, 0.02)))
        precision = min(1.0, max(0.0, base_precision + random.uniform(-0.02, 0.02)))
        
        # Calculate F1 score
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        
        return recall, f1_score

class EmbeddingSimilarityVerification(VerificationModule):
    """
    Embedding Similarity Verification (ESV) module.
    Uses similarity score and anomaly detection rate metrics.
    """
    def __init__(self, weight, alpha, threshold):
        super().__init__("ESV", weight, alpha, threshold)
    
    def compute_metrics(self, candidate_fact, knowledge_graph, fact_quality):
        """
        Compute Sim_ESV (similarity score) and ADR_ESV (anomaly detection rate) as defined in document.
        
        Uses fact_quality to adjust scores to demonstrate different verification paths.
        """
        # Check if the fact contains a performance shift for ESV
        has_shift = candidate_fact.get("contains_performance_shift") == "ESV"
        
        # Base similarity and anomaly detection rate based on quality
        if fact_quality == "high_quality":
            base_similarity = 0.92 if not has_shift else 0.67
            base_anomaly_rate = 0.05 if not has_shift else 0.30
        elif fact_quality == "medium_quality":
            base_similarity = 0.85 if not has_shift else 0.60
            base_anomaly_rate = 0.10 if not has_shift else 0.35
        elif fact_quality == "spatial_issue":
            base_similarity = 0.80 if not has_shift else 0.55
            base_anomaly_rate = 0.15 if not has_shift else 0.40
        elif fact_quality == "external_ref":
            base_similarity = 0.78 if not has_shift else 0.53
            base_anomaly_rate = 0.18 if not has_shift else 0.43
        elif fact_quality == "semantic_issue":
            # Very high values for semantic_issue to exceed threshold
            base_similarity = 0.98 if not has_shift else 0.73
            base_anomaly_rate = 0.02 if not has_shift else 0.27
        else:  # low_quality
            base_similarity = 0.40 if not has_shift else 0.25
            base_anomaly_rate = 0.60 if not has_shift else 0.75
        
        # Add small random variation
        similarity = min(1.0, max(0.0, base_similarity + random.uniform(-0.02, 0.02)))
        anomaly_rate = min(1.0, max(0.0, base_anomaly_rate + random.uniform(-0.02, 0.02)))
        
        return similarity, 1.0 - anomaly_rate  # Return similarity and (1 - anomaly_rate)

# -----------------------------------------------------------------------------
# RMMVe Process Implementation
# -----------------------------------------------------------------------------
class RMMVeProcess:
    """
    Implements the Ranked Multi-Modal Verification (RMMVe) process.
    
    This process follows Algorithm 1 in the paper, using a sequence of
    verification modules with early termination capabilities.
    """
    def __init__(self, global_threshold=0.65):
        # Initialize verification modules with their parameters
        # Modified thresholds to ensure early termination works properly
        self.modules = [
            LocalOntologyVerification(0.85, 0.7, 0.82),   # LOV - Lowered threshold to allow high quality to pass
            PublicOntologyVerification(0.75, 0.65, 0.83), # POV - Adjusted for medium quality
            MultiAgentVerification(0.9, 0.75, 0.85),      # MAV - Adjusted for spatial issues
            WebSearchVerification(0.7, 0.65, 0.85),       # WSV - Adjusted for external references
            EmbeddingSimilarityVerification(0.8, 0.7, 0.85) # ESV - Adjusted for semantic issues
        ]
        self.global_threshold = global_threshold  # Θ in the paper
    
    def verify(self, candidate_fact, knowledge_graph, fact_quality=None):
        """
        Verify a candidate fact using the RMMVe process.
        
        This implements Algorithm 1 from the paper, processing modules
        in sequence with early termination when confidence exceeds threshold.
        """
        verification_results = {
            "activated_modules": [],
            "module_results": [],
            "confidence_scores": {},
            "metrics": {},
            "total_confidence": 0.0,
            "decision": False,
            "early_termination": False,
            "verification_time": 0.0,
            "fact_quality": fact_quality,
            "contains_shift": candidate_fact.get("contains_performance_shift", None)
        }
        
        start_time = time.time()
        
        # Process modules in sequence
        for module in self.modules:
            module_start_time = time.time()
            
            # Step 1: Compute module confidence
            confidence, metric1, metric2 = module.compute_confidence(candidate_fact, knowledge_graph, fact_quality)
            module_end_time = time.time()
            
            # Record module results
            module_result = {
                "module_name": module.name,
                "confidence": confidence,
                "threshold": module.threshold,
                "metric1": metric1,
                "metric2": metric2,
                "weight": module.weight,
                "alpha": module.alpha,
                "processing_time": module_end_time - module_start_time
            }
            
            verification_results["activated_modules"].append(module.name)
            verification_results["module_results"].append(module_result)
            verification_results["confidence_scores"][module.name] = confidence
            verification_results["metrics"][module.name] = {"metric1": metric1, "metric2": metric2}
            
            # Step 2: Check for early termination
            if confidence >= module.threshold:
                verification_results["early_termination"] = True
                verification_results["early_termination_module"] = module.name
                verification_results["early_termination_confidence"] = confidence
                verification_results["early_termination_threshold"] = module.threshold
                break
        
        # Calculate total confidence as average of activated modules
        activated_modules = verification_results["activated_modules"]
        if activated_modules:
            verification_results["total_confidence"] = sum(
                verification_results["confidence_scores"][name] 
                for name in activated_modules
            ) / len(activated_modules)
        
        # Make decision based on global threshold
        verification_results["decision"] = verification_results["total_confidence"] >= self.global_threshold
        
        end_time = time.time()
        verification_results["verification_time"] = end_time - start_time
        
        return verification_results

# -----------------------------------------------------------------------------
# AAIC (Autonomous Adaptive Intelligence Cycle) Implementation
# -----------------------------------------------------------------------------
class AAIC:
    """
    Implements the Autonomous Adaptive Intelligence Cycle (AAIC) for TruthFlow.
    
    This system dynamically adjusts the parameters of verification modules
    based on their performance over time, as described in the paper:
    
    1. CGR-CUSUM monitors module performance
    2. When cumulative sum exceeds threshold h, it triggers parameter updates
    3. Updates include weight (w), threshold (θ), and alpha (α) adjustments
    4. Weights are normalized to sum to 4 (for 5-module system)
    """
    def __init__(self, rmmve, h=1.5, k=0.05, gamma=0.1, eta=0.1, eta_prime=0.1):
        self.rmmve = rmmve
        self.h = h  # Threshold for detecting performance shifts (LOWERED FROM 5.0 to 1.5)
        self.k = k  # Allowance parameter (k=0.05 in paper)
        self.gamma = gamma  # Scaling factor for weight updates
        self.eta = eta  # Learning rate for threshold updates (η=0.1 in paper)
        self.eta_prime = eta_prime  # Learning rate for alpha updates
        
        # Target performance level μ_0 for each module (typically 0.8)
        self.target_performance = {module.name: 0.8 for module in self.rmmve.modules}
        
        # Initialize cumulative sums S_i(t) for each module
        self.cumulative_sums = {module.name: 0.0 for module in self.rmmve.modules}
        
        # Store historical performance for TPR/FPR calculations
        self.performance_history = {module.name: [] for module in self.rmmve.modules}
        
        # Update history - track all updates and their effects
        self.update_history = []
        
        # Parameter history - track how parameters evolve over time
        self.parameter_history = {
            module.name: {
                "weights": [],
                "thresholds": [],
                "alphas": [],
                "performances": [],
                "cum_sums": [],
                "timestamps": []
            } for module in self.rmmve.modules
        }
        
        # Track detected shifts
        self.detected_shifts = []
    
    def cgr_cusum_monitor(self, module_name, performance):
        """
        Monitor module performance using CGR-CUSUM algorithm.
        
        Implements the formula from the paper:
        S_i(t) = max(0, S_i(t-1) + [p_i(t) - μ_0 - k])
        
        Returns True if a significant performance shift is detected (S_i(t) ≥ h).
        """
        target = self.target_performance[module_name]  # μ_0 in the paper
        
        # Update cumulative sum using the formula from the paper
        previous_sum = self.cumulative_sums[module_name]  # S_i(t-1)
        current_sum = max(0, previous_sum + performance - target - self.k)  # S_i(t)
        self.cumulative_sums[module_name] = current_sum
        
        # Store performance for history
        self.performance_history[module_name].append(performance)
        
        # Detect if performance shift exceeds threshold (S_i(t) ≥ h)
        return current_sum >= self.h
    
    def update_weight(self, module, cumulative_sum):
        """
        Update module weight using exponential weights algorithm.
        
        Implements Eq.6 from the paper:
        w_i(t+1) = w_i(t)*exp[-γ*S_i(t)]
        """
        old_weight = module.weight
        # Apply exponential weights algorithm (Eq.6)
        new_weight = old_weight * math.exp(-self.gamma * cumulative_sum)
        return new_weight
    
    def update_threshold(self, module, performance):
        """
        Update confidence threshold using gradient ascent.
        
        Implements Eq.8 from the paper:
        θ_i(t+1) = θ_i(t) + η*(∂U(θ_i)/∂θ_i)
        
        Uses a simplified gradient estimation based on TPR change.
        """
        old_threshold = module.threshold
        
        # Use historical performance to estimate TPR change direction
        history = self.performance_history[module.name]
        if len(history) > 1:
            # Estimate if TPR is improving or declining
            recent_avg = sum(history[-3:]) / min(3, len(history))
            older_avg = sum(history[:-3]) / max(1, len(history) - 3)
            tpr_change = 0.05 * (recent_avg - older_avg)  # Simplified gradient
        else:
            # If limited history, use target-performance difference
            tpr_change = 0.05 * (self.target_performance[module.name] - performance)
        
        # Apply gradient ascent update (Eq.8)
        new_threshold = max(0.1, min(0.95, old_threshold + self.eta * tpr_change))
        return new_threshold
    
    def update_alpha(self, module, performance):
        """
        Update internal weighting factor alpha using gradient ascent.
        
        Implements Eq.9 from the paper:
        α_i(t+1) = α_i(t) + η'*(∂U_i(α_i)/∂α_i)
        
        Uses a simplified gradient estimation.
        """
        old_alpha = module.alpha
        
        # Use historical performance to estimate utility gradient
        history = self.performance_history[module.name]
        if len(history) > 1:
            # Adapt alpha based on recent performance trends
            recent_avg = sum(history[-3:]) / min(3, len(history))
            older_avg = sum(history[:-3]) / max(1, len(history) - 3)
            utility_change = 0.05 * (recent_avg - older_avg)  # Simplified gradient
        else:
            # If limited history, use target-performance difference
            utility_change = 0.05 * (self.target_performance[module.name] - performance)
        
        # Apply gradient ascent update (Eq.9)
        new_alpha = max(0.1, min(0.9, old_alpha + self.eta_prime * utility_change))
        return new_alpha
    
    def normalize_weights(self):
        """
        Normalize weights to sum to 4 as per Eq.7 in the paper.
        
        w_i(t+1) = (w_i(t+1) / ∑w_j(t+1)) × 4
        
        This creates a statistical baseline with mean weight of 0.8,
        allowing easy identification of above/below average contributors.
        """
        total_weight = sum(module.weight for module in self.rmmve.modules)
        if total_weight > 0:
            for module in self.rmmve.modules:
                module.weight = (module.weight / total_weight) * 4
    
    def update_module_parameters(self, module):
        """
        Update a module's parameters based on its performance.
        This implements the adaptive parameter updates from the paper.
        """
        # Get latest performance (using first metric as performance indicator)
        if not module.performance_history:
            return {"module": module.name, "update": "No performance history", "detected": False}
        
        latest_performance = module.performance_history[-1]["metric1"]
        current_time = time.time()
        
        # Add to parameter history
        self.parameter_history[module.name]["weights"].append(module.weight)
        self.parameter_history[module.name]["thresholds"].append(module.threshold)
        self.parameter_history[module.name]["alphas"].append(module.alpha)
        self.parameter_history[module.name]["performances"].append(latest_performance)
        self.parameter_history[module.name]["cum_sums"].append(self.cumulative_sums[module.name])
        self.parameter_history[module.name]["timestamps"].append(current_time)
        
        # Check if performance shift is detected using CGR-CUSUM
        shift_detected = self.cgr_cusum_monitor(module.name, latest_performance)
        
        update_info = {
            "module": module.name,
            "performance": latest_performance,
            "cumulative_sum": self.cumulative_sums[module.name],
            "detected": shift_detected,
            "old_params": {"weight": module.weight, "threshold": module.threshold, "alpha": module.alpha},
            "new_params": {"weight": module.weight, "threshold": module.threshold, "alpha": module.alpha},
            "timestamp": current_time
        }
        
        if shift_detected:
            # Store shift information before updates
            self.detected_shifts.append({
                "module": module.name,
                "timestamp": current_time,
                "cumulative_sum": self.cumulative_sums[module.name],
                "performance": latest_performance,
                "old_weight": module.weight,
                "old_threshold": module.threshold,
                "old_alpha": module.alpha
            })
            
            # Step 2 from Algorithm 1: Update weight using exponential weights algorithm (Eq.6)
            module.weight = self.update_weight(module, self.cumulative_sums[module.name])
            
            # Step 3 from Algorithm 1: Update threshold using gradient ascent (Eq.8)
            module.threshold = self.update_threshold(module, latest_performance)
            
            # Step 4 from Algorithm 1: Update alpha using gradient ascent (Eq.9)
            module.alpha = self.update_alpha(module, latest_performance)
            
            # Reset cumulative sum after adjustment as per CGR-CUSUM methodology
            self.cumulative_sums[module.name] = 0.0
            
            # Update the detected shift records with the new parameters
            self.detected_shifts[-1].update({
                "new_weight": module.weight,
                "new_threshold": module.threshold,
                "new_alpha": module.alpha,
                "weight_change": module.weight - update_info["old_params"]["weight"],
                "threshold_change": module.threshold - update_info["old_params"]["threshold"],
                "alpha_change": module.alpha - update_info["old_params"]["alpha"]
            })
            
            update_info["new_params"] = {"weight": module.weight, "threshold": module.threshold, "alpha": module.alpha}
            update_info["update"] = "Parameters updated according to AAIC Algorithm"
        else:
            # No update needed (Step 7-9 in Algorithm 1)
            update_info["update"] = "No update needed (S_i(t) < h)"
        
        return update_info
    
    def update_all_modules(self):
        """
        Update parameters for all modules based on their performance.
        This implements Algorithm 1 (AAIC within TruthFlow) from the paper.
        """
        updates = []
        
        # For each verification module M_i (Steps 1-10 in Algorithm 1)
        for module in self.rmmve.modules:
            update_info = self.update_module_parameters(module)
            updates.append(update_info)
        
        # Normalize weights (Step 11 in Algorithm 1)
        self.normalize_weights()
        
        self.update_history.append({
            "timestamp": time.time(),
            "updates": updates
        })
        
        return updates
    
    def get_parameter_history_df(self):
        """Get parameter history as a DataFrame for visualization."""
        data = []
        
        for module_name, history in self.parameter_history.items():
            for i in range(len(history["timestamps"])):
                data.append({
                    "Module": module_name,
                    "Timestamp": datetime.fromtimestamp(history["timestamps"][i]).strftime("%Y-%m-%d %H:%M:%S"),
                    "Weight": history["weights"][i],
                    "Threshold": history["thresholds"][i],
                    "Alpha": history["alphas"][i],
                    "Performance": history["performances"][i],
                    "Cumulative Sum": history["cum_sums"][i]
                })
        
        return pd.DataFrame(data)
    
    def get_detected_shifts_df(self):
        """Get detected shifts as a DataFrame for visualization."""
        if not self.detected_shifts:
            return pd.DataFrame()
        
        return pd.DataFrame(self.detected_shifts)
def create_parameter_change_table(shift):
    """Compatibility function that redirects to the display_parameter_change function."""
    if shift is not None:
        return display_parameter_change(shift)
    return None
# -----------------------------------------------------------------------------
# Visualization Functions
# -----------------------------------------------------------------------------
def create_fact_preview_card(fact):
    """Create a formatted preview of a candidate fact as an HTML card."""
    fact_data = fact.get("spatiotemporal_inspection_data", {})
    report_id = fact_data.get("report_id", "N/A")
    
    preview = f"""
    <div class="card">
        <h3>Report ID: {report_id}</h3>
    """
    
    # Add inspection measurements
    if "inspection_data" in fact_data and "inspection_measurements" in fact_data["inspection_data"]:
        preview += "<h4>Measurements</h4>"
        
        for i, item in enumerate(fact_data["inspection_data"]["inspection_measurements"]):
            component_id = item.get("component_id", "N/A")
            measurement_id = item.get("measurement_id", "N/A")
            feature_name = item.get("feature_name", "N/A")
            nominal = item.get("nominal_value_mm", "N/A")
            actual = item.get("actual_value_mm", "N/A")
            deviation = item.get("deviation_mm", "N/A")
            status = item.get("status", "N/A")
            
            status_class = "status-verified" if status == "PASS" else "status-rejected"
            
            preview += f"""
            <div style="margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid #e5e7eb;">
                <div><strong>Component:</strong> {component_id}</div>
                <div><strong>ID:</strong> {measurement_id}</div>
                <div><strong>Feature:</strong> {feature_name}</div>
                <div><strong>Nominal:</strong> {nominal} mm</div>
                <div><strong>Actual:</strong> {actual} mm</div>
                <div><strong>Deviation:</strong> {deviation} mm</div>
                <div><strong>Status:</strong> <span class="{status_class}">{status}</span></div>
            </div>
            """
    
    # Add temporal data
    if "temporal_data" in fact_data:
        preview += "<h4>Temporal Data</h4>"
        preview += "<div style='margin-bottom: 12px;'>"
        
        for i, item in enumerate(fact_data["temporal_data"]):
            entity_id = item.get("entity_id", "N/A")
            timestamp = item.get("timestamp", "N/A")
            event_type = item.get("event_type_or_feature", "N/A")
            
            preview += f"<div>Entity: {entity_id}, Time: {timestamp}, Event: {event_type}</div>"
        
        preview += "</div>"
    
    # Add spatial data
    if "spatial_data" in fact_data:
        preview += "<h4>Spatial Data</h4>"
        preview += "<div style='margin-bottom: 12px;'>"
        
        for i, item in enumerate(fact_data["spatial_data"]):
            entity_id = item.get("entity_id", "N/A")
            coords = item.get("coordinates", {})
            x = coords.get("x_coord", "N/A")
            y = coords.get("y_coord", "N/A")
            z = coords.get("z_coord", "N/A")
            
            preview += f"<div>Entity: {entity_id}, Coordinates: ({x}, {y}, {z})</div>"
        
        preview += "</div>"
    
    # Add relationships
    if "relationships" in fact_data:
        preview += "<h4>Relationships</h4>"
        preview += "<div>"
        
        for i, rel in enumerate(fact_data["relationships"]):
            rel_id = rel.get("relationship_id", "N/A")
            subject = rel.get("subject_entity_id", "N/A")
            rel_type = rel.get("relationship_type", "N/A")
            object_id = rel.get("object_entity_id", "N/A")
            
            preview += f"<div>{rel_id}: {subject} --{rel_type}--> {object_id}</div>"
        
        preview += "</div>"
    
    preview += "</div>"
    return preview

def plot_verification_results_plotly(verification_results):
    """Plot the verification results as an interactive bar chart using Plotly."""
    # Extract data for plotting
    module_names = []
    confidences = []
    thresholds = []
    
    for result in verification_results["module_results"]:
        module_names.append(result["module_name"])
        confidences.append(result["confidence"])
        thresholds.append(result["threshold"])
    
    # Create figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add confidence bars
    fig.add_trace(
        go.Bar(
            x=module_names,
            y=confidences,
            name="Confidence",
            marker_color='#3b82f6',
            text=[f'{conf:.2f}' for conf in confidences],
            textposition='auto',
        )
    )
    
    # Add threshold bars
    fig.add_trace(
        go.Bar(
            x=module_names,
            y=thresholds,
            name="Threshold",
            marker_color='#f43f5e',
            text=[f'{thr:.2f}' for thr in thresholds],
            textposition='auto',
        )
    )
    
    # Add global threshold line
    if "total_confidence" in verification_results:
        fig.add_trace(
            go.Scatter(
                x=module_names,
                y=[verification_results["total_confidence"]] * len(module_names),
                mode='lines',
                name='Total Confidence',
                line=dict(color='#1e3a8a', width=2, dash='dash'),
            )
        )
    
    # Customize layout
    fig.update_layout(
        title='Verification Results by Module',
        xaxis_title='Module',
        yaxis_title='Score',
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        height=400,
    )
    
    # Highlight early termination if applicable
    if verification_results.get("early_termination") and verification_results.get("early_termination_module"):
        early_term_module = verification_results["early_termination_module"]
        early_term_index = module_names.index(early_term_module)
        confidence = confidences[early_term_index]
        
        fig.add_annotation(
            x=module_names[early_term_index],
            y=confidence + 0.1,
            text="Early Termination",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#ef4444",
            font=dict(size=12, color="#ef4444", family="Arial, sans-serif"),
            align="center",
        )
    
    return fig

def plot_metrics_plotly(verification_results):
    """Plot the metrics used in each module's confidence calculation using Plotly."""
    # Extract data for plotting
    module_names = []
    metric1_values = []
    metric2_values = []
    metric_names = []
    
    metric_mappings = {
        "LOV": ("Precision", "Recall"),
        "POV": ("Accuracy", "Coverage"),
        "MAV": ("Consensus", "Reliability"),
        "WSV": ("Recall", "F1 Score"),
        "ESV": ("Similarity", "1-Anomaly Rate")
    }
    
    for result in verification_results["module_results"]:
        module_name = result["module_name"]
        module_names.append(module_name)
        metric1_values.append(result["metric1"])
        metric2_values.append(result["metric2"])
        metric_names.append(metric_mappings.get(module_name, ("Metric 1", "Metric 2")))
    
    # Create custom x labels with module name and metric name
    x_labels = []
    for module, (metric1, metric2) in zip(module_names, metric_names):
        x_labels.extend([f"{module}: {metric1}", f"{module}: {metric2}"])
    
    # Flatten metrics for grouped bar chart
    all_metrics = []
    for m1, m2 in zip(metric1_values, metric2_values):
        all_metrics.extend([m1, m2])
    
    # Create color mapping for better visualization
    colors = []
    for module in module_names:
        colors.extend(['#3b82f6', '#10b981'])  # Blue for Metric 1, Green for Metric 2
    
    # Create figure
    fig = go.Figure()
    
    # Add bars for all metrics
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=all_metrics,
            marker_color=colors,
            text=[f'{m:.2f}' for m in all_metrics],
            textposition='auto',
        )
    )
    
    # Customize layout
    fig.update_layout(
        title='Verification Metrics by Module',
        xaxis_title='Module and Metric',
        yaxis_title='Value',
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=10)
        ),
        margin=dict(l=20, r=20, t=60, b=100),
        height=450,
    )
    
    return fig

def plot_aaic_cumulative_sums_plotly(parameter_history_df):
    """Plot the CGR-CUSUM cumulative sums for each module using Plotly."""
    if parameter_history_df.empty:
        return None
    
    modules = parameter_history_df["Module"].unique()
    
    # Create dataframe for plotting by module iteration instead of timestamp
    plot_data = []
    for module in modules:
        module_data = parameter_history_df[parameter_history_df["Module"] == module]
        for i, row in enumerate(module_data.itertuples()):
            plot_data.append({
                "Module": row.Module,
                "Iteration": i + 1,
                "Cumulative Sum": row._7,  # Index of "Cumulative Sum" in the tuple
                "Timestamp": row.Timestamp,
                "Performance": row.Performance
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create figure
    fig = go.Figure()
    
    # Set color map for modules
    color_map = {
        "LOV": "#3b82f6",  # Blue
        "POV": "#10b981",  # Green
        "MAV": "#f59e0b",  # Orange
        "WSV": "#8b5cf6",  # Purple
        "ESV": "#06b6d4"   # Teal
    }
    
    # Add lines for each module
    for module in modules:
        module_data = plot_df[plot_df["Module"] == module]
        # Convert to lists to ensure Plotly compatibility
        x_values = module_data["Iteration"].tolist()
        y_values = module_data["Cumulative Sum"].tolist()
        custom_data = module_data["Performance"].tolist()
        
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines+markers',
                name=module,
                line=dict(color=color_map.get(module, "#000000"), width=2),
                hovertemplate=
                "<b>%{x}</b><br>" +
                "Module: " + module + "<br>" +
                "Cum. Sum: %{y:.3f}<br>" +
                "Performance: %{customdata:.3f}<extra></extra>",
                customdata=custom_data
            )
        )
    
    # Add threshold line
    h_value = 5.0  # Default value
    fig.add_trace(
        go.Scatter(
            x=[1, plot_df["Iteration"].max()],
            y=[h_value, h_value],
            mode='lines',
            name=f'Threshold (h={h_value})',
            line=dict(color='#ef4444', width=2, dash='dash'),
        )
    )
    
    # Customize layout
    fig.update_layout(
        title='CGR-CUSUM Monitoring',
        xaxis_title='Verification Iteration',
        yaxis_title='Cumulative Sum',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        height=400,
        hovermode="closest"
    )
    
    return fig

def plot_aaic_parameter_evolution_plotly(parameter_history_df, param_type):
    """Plot the evolution of module parameters over time using Plotly."""
    if parameter_history_df.empty:
        return None
    
    modules = parameter_history_df["Module"].unique()
    
    # Create dataframe for plotting by module iteration instead of timestamp
    plot_data = []
    for module in modules:
        module_data = parameter_history_df[parameter_history_df["Module"] == module]
        for i, row in enumerate(module_data.itertuples()):
            param_value = getattr(row, param_type)
            plot_data.append({
                "Module": row.Module,
                "Iteration": i + 1,
                "Value": param_value,
                "Timestamp": row.Timestamp,
                "Cumulative Sum": row._7  # Index of Cumulative Sum in tuple
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create figure
    fig = go.Figure()
    
    # Set color map for modules
    color_map = {
        "LOV": "#3b82f6",  # Blue
        "POV": "#10b981",  # Green
        "MAV": "#f59e0b",  # Orange
        "WSV": "#8b5cf6",  # Purple
        "ESV": "#06b6d4"   # Teal
    }
    
    # Add lines for each module
    for module in modules:
        module_data = plot_df[plot_df["Module"] == module]
        # Convert to list to ensure Plotly compatibility
        x_values = module_data["Iteration"].tolist()
        y_values = module_data["Value"].tolist()
        cum_sums = module_data["Cumulative Sum"].tolist()
        
        # Find points where parameter changes occurred (where cum_sum likely exceeded threshold)
        change_points = []
        change_values = []
        
        for i in range(1, len(y_values)):
            if abs(y_values[i] - y_values[i-1]) > 0.001:  # Detect significant parameter change
                change_points.append(x_values[i])
                change_values.append(y_values[i])
        
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines+markers',
                name=module,
                line=dict(color=color_map.get(module, "#000000"), width=2),
                hovertemplate=
                "<b>Iteration %{x}</b><br>" +
                "Module: " + module + "<br>" +
                f"{param_type}: %{{y:.3f}}<br>" +
                "Cum. Sum: %{customdata:.3f}<extra></extra>",
                customdata=cum_sums
            )
        )
        
        # Add markers for parameter change points
        if change_points:
            fig.add_trace(
                go.Scatter(
                    x=change_points,
                    y=change_values,
                    mode='markers',
                    marker=dict(
                        symbol='star',
                        size=12,
                        color='red',
                        line=dict(width=2, color='DarkSlateGrey')
                    ),
                    name=f'{module} Parameter Change',
                    hovertemplate=
                    "<b>Parameter Updated</b><br>" +
                    "Module: " + module + "<br>" +
                    f"New {param_type}: %{{y:.3f}}<extra></extra>"
                )
            )
    
    # Param-specific styling
    if param_type == "Weight":
        title = "Module Weights Evolution"
        y_title = "Weight (w)"
    elif param_type == "Threshold":
        title = "Module Thresholds Evolution"
        y_title = "Threshold (θ)"
    elif param_type == "Alpha":
        title = "Module Alpha Parameters Evolution"
        y_title = "Alpha (α)"
    elif param_type == "Performance":
        title = "Module Performance Evolution"
        y_title = "Performance"
    else:
        title = f"{param_type} Evolution"
        y_title = param_type
    
    # Customize layout
    fig.update_layout(
        title=title,
        xaxis_title='Verification Iteration',
        yaxis_title=y_title,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        height=400,
        hovermode="closest"
    )
    
    return fig

def plot_verification_history_plotly(history_df):
    """Plot verification history metrics using Plotly."""
    if history_df.empty:
        return None
    
    # Create figure for confidence over time
    fig = go.Figure()
    
    # Convert range to list for Plotly - range() objects aren't directly accepted
    x_values = list(range(1, len(history_df) + 1))
    
    fig.add_trace(
        go.Scatter(
            x=x_values,  # Using list instead of range object
            y=history_df["Total Confidence"],
            mode='lines+markers',
            name='Total Confidence',
            line=dict(color='#3b82f6', width=2),
            marker=dict(
                size=8,
                color=history_df["Decision"].apply(lambda x: '#10b981' if "Verified" in x else '#ef4444'),
                symbol=history_df["Early Termination"].apply(lambda x: 'circle' if x == "✓ Yes" else 'x')
            ),
            hovertemplate=
            "<b>Verification %{x}</b><br>" +
            "Confidence: %{y:.3f}<br>" +
            "Decision: %{customdata[0]}<br>" +
            "Early Term: %{customdata[1]}<br>" +
            "Fact Quality: %{customdata[2]}<extra></extra>",
            customdata=list(zip(
                history_df["Decision"],
                history_df["Early Termination"],
                history_df["Fact Quality"]
            ))
        )
    )
    
    # Add threshold line (using list for x values)
    fig.add_trace(
        go.Scatter(
            x=[1, len(history_df)],
            y=[0.65, 0.65],  # Default global threshold
            mode='lines',
            name='Global Threshold',
            line=dict(color='#6b7280', width=2, dash='dash'),
        )
    )
    
    # Customize layout
    fig.update_layout(
        title='Verification Confidence History',
        xaxis_title='Verification Number',
        yaxis_title='Total Confidence',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        height=400,
        hovermode="closest"
    )
    
    return fig

def plot_quality_distribution_plotly(history_df):
    """Plot fact quality distribution as a pie chart using Plotly."""
    if history_df.empty:
        return None
    
    # Count quality occurrences
    quality_counts = history_df["Fact Quality"].value_counts().reset_index()
    quality_counts.columns = ["Quality", "Count"]
    
    # Calculate percentages
    total = quality_counts["Count"].sum()
    quality_counts["Percentage"] = quality_counts["Count"] / total * 100
    
    # Create color map that matches the app's styling
    color_map = {
        "high_quality": "#10b981",  # Green
        "medium_quality": "#3b82f6",  # Blue
        "spatial_issue": "#f59e0b",  # Orange
        "external_ref": "#8b5cf6",  # Purple
        "semantic_issue": "#06b6d4",  # Teal
        "low_quality": "#ef4444"     # Red
    }
    
    colors = [color_map.get(q, "#6b7280") for q in quality_counts["Quality"]]
    
    # Create figure
    fig = go.Figure(
        data=[
            go.Pie(
                labels=quality_counts["Quality"],
                values=quality_counts["Percentage"],
                hole=0.4,
                marker=dict(colors=colors),
                textinfo='label+percent',
                insidetextorientation='radial',
                hovertemplate="<b>%{label}</b><br>Count: %{value:.1f}%<extra></extra>"
            )
        ]
    )
    
    # Customize layout
    fig.update_layout(
        title='Distribution of Fact Qualities',
        margin=dict(l=20, r=20, t=60, b=20),
        height=400,
    )
    
    return fig

def plot_early_term_by_module_plotly(history_df):
    """Plot early termination by module as a bar chart using Plotly."""
    if history_df.empty:
        return None
    
    # Get early termination counts by module
    term_counts = history_df["Term. Module"].value_counts().reset_index()
    term_counts.columns = ["Module", "Count"]
    term_counts = term_counts[term_counts["Module"] != "None"]
    
    if term_counts.empty:
        return None
    
    # Calculate percentages
    total = term_counts["Count"].sum()
    term_counts["Percentage"] = term_counts["Count"] / total * 100
    
    # Set color map for modules
    color_map = {
        "LOV": "#3b82f6",  # Blue
        "POV": "#10b981",  # Green
        "MAV": "#f59e0b",  # Orange
        "WSV": "#8b5cf6",  # Purple
        "ESV": "#06b6d4"   # Teal
    }
    
    colors = [color_map.get(m, "#6b7280") for m in term_counts["Module"]]
    
    # Create figure
    fig = go.Figure(
        data=[
            go.Bar(
                x=term_counts["Module"],
                y=term_counts["Percentage"],
                marker_color=colors,
                text=[f'{p:.1f}%' for p in term_counts["Percentage"]],
                textposition='auto',
                hovertemplate="<b>%{x}</b><br>Percentage: %{y:.1f}%<br>Count: %{customdata}<extra></extra>",
                customdata=term_counts["Count"]
            )
        ]
    )
    
    # Customize layout
    fig.update_layout(
        title='Distribution of Early Terminations by Module',
        xaxis_title='Module',
        yaxis_title='Percentage of Early Terminations',
        margin=dict(l=20, r=20, t=60, b=20),
        height=400,
    )
    
    return fig

def create_module_performance_gauge(module_name, confidence, threshold):
    """Create a gauge chart for module performance using Plotly."""
    # Set color based on confidence vs threshold
    if confidence >= threshold:
        color = "#10b981"  # Green for above threshold
    elif confidence >= 0.7 * threshold:
        color = "#f59e0b"  # Orange for close to threshold
    else:
        color = "#ef4444"  # Red for below threshold
    
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence,
            domain=dict(x=[0, 1], y=[0, 1]),
            title=dict(text=module_name, font=dict(size=16)),
            gauge=dict(
                axis=dict(range=[0, 1], tickwidth=1),
                bar=dict(color=color),
                bgcolor="white",
                borderwidth=2,
                bordercolor="gray",
                steps=[
                    dict(range=[0, 0.5], color="#fee2e2"),
                    dict(range=[0.5, 0.8], color="#fef3c7"),
                    dict(range=[0.8, 1.0], color="#d1fae5")
                ],
                threshold=dict(
                    line=dict(color="red", width=4),
                    thickness=0.75,
                    value=threshold
                )
            ),
            number=dict(
                font=dict(color=color),
                suffix=""
            )
        )
    )
    
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    
    return fig

def get_quality_badge(quality):
    """Return HTML for a quality badge."""
    quality_classes = {
        "high_quality": "quality-high",
        "medium_quality": "quality-medium",
        "spatial_issue": "quality-spatial",
        "external_ref": "quality-external",
        "semantic_issue": "quality-semantic",
        "low_quality": "quality-low"
    }
    
    quality_names = {
        "high_quality": "High Quality",
        "medium_quality": "Medium Quality",
        "spatial_issue": "Spatial Issue",
        "external_ref": "External Reference",
        "semantic_issue": "Semantic Issue",
        "low_quality": "Low Quality"
    }
    
    cls = quality_classes.get(quality, "")
    name = quality_names.get(quality, quality)
    
    return f'<span class="{cls}">{name}</span>'

def get_decision_badge(decision):
    """Return HTML for a decision badge."""
    if decision:
        return '<span class="status-verified">VERIFIED</span>'
    else:
        return '<span class="status-rejected">REJECTED</span>'

def create_gauge_grid(verification_results):
    """Create a grid of gauge charts for all modules."""
    if not verification_results["module_results"]:
        return None
    
    gauges = []
    for result in verification_results["module_results"]:
        module_name = result["module_name"]
        confidence = result["confidence"]
        threshold = result["threshold"]
        
        gauge = create_module_performance_gauge(module_name, confidence, threshold)
        gauges.append(gauge)
    
    return gauges

def format_shift_alert(shift_module):
    """Format an alert for performance shift."""
    if not shift_module:
        return ""
    
    return f"""
    <div class="alert-warning">
        <strong>⚠️ Performance Shift Detected:</strong> This fact contains a shift in the {shift_module} module
        that may trigger adaptation by the AAIC system.
    </div>
    """

def display_parameter_change(shift):
    """Display parameter changes using Streamlit native components instead of HTML."""
    if shift is None or (hasattr(shift, 'empty') and shift.empty):
        return
    
    # Convert to dictionary if it's a pandas Series
    if hasattr(shift, 'to_dict'):
        shift = shift.to_dict()
    
    # Create an expander with module name
    with st.expander(f"Shift: Module {shift['module']}", expanded=True):
        # Display metrics at the top
        cols = st.columns(3)
        with cols[0]:
            st.metric("Cumulative Sum", f"{shift['cumulative_sum']:.3f}")
        with cols[1]:
            st.metric("Performance", f"{shift['performance']*100:.1f}%")
        with cols[2]:
            timestamp = datetime.fromtimestamp(shift['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
            st.metric("Timestamp", timestamp)
        
        st.markdown("### Parameter Adjustments")
        
        # Create a dataframe for parameter changes
        param_data = {
            "Parameter": ["Weight (w)", "Threshold (θ)", "Alpha (α)"],
            "Old Value": [
                f"{shift['old_weight']:.4f}",
                f"{shift['old_threshold']:.4f}",
                f"{shift['old_alpha']:.4f}"
            ],
            "New Value": [
                f"{shift['new_weight']:.4f}",
                f"{shift['new_threshold']:.4f}",
                f"{shift['new_alpha']:.4f}"
            ]
        }
        
        # Calculate changes and format with arrows
        weight_pct = round((shift["weight_change"] / shift["old_weight"]) * 100, 1)
        threshold_pct = round((shift["threshold_change"] / shift["old_threshold"]) * 100, 1)
        alpha_pct = round((shift["alpha_change"] / shift["old_alpha"]) * 100, 1)
        
        param_data["Change"] = [
            f"{'↑' if weight_pct >= 0 else '↓'} {abs(weight_pct)}%",
            f"{'↑' if threshold_pct >= 0 else '↓'} {abs(threshold_pct)}%",
            f"{'↑' if alpha_pct >= 0 else '↓'} {abs(alpha_pct)}%"
        ]
        
        # Add descriptions
        param_data["Description"] = [
            "Module influence",
            "Early term. threshold",
            "Metric balance"
        ]
        
        # Display as table
        st.table(pd.DataFrame(param_data))
        
        st.info("Parameters adjusted according to AAIC Algorithm (Algorithm 1) from the paper.")

# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    # ----- Initialize Application State -----
    
    # Initialize session state
    if "kg" not in st.session_state:
        st.session_state.kg = create_sample_knowledge_graph()
    
    if "rmmve" not in st.session_state:
        st.session_state.rmmve = RMMVeProcess(global_threshold=0.65)
    
    if "aaic" not in st.session_state:
        st.session_state.aaic = AAIC(st.session_state.rmmve)
    
    if "candidate_fact" not in st.session_state:
        st.session_state.candidate_fact, st.session_state.fact_quality = generate_candidate_fact()
    
    if "verification_history" not in st.session_state:
        st.session_state.verification_history = []
    
    if "verification_count" not in st.session_state:
        st.session_state.verification_count = 0
    
    # Add trigger for introducing shifts
    if "introduce_shift" not in st.session_state:
        st.session_state.introduce_shift = False
    
    if "shift_module" not in st.session_state:
        st.session_state.shift_module = "LOV"
    
    # ----- Header Section -----
    st.markdown(
    """
    <div class="header-container">
        <div class="header-title">ATLASky-AI</div>
        <div class="header-subtitle">Aerospace Spatiotemporal Knowledge Graph Verification System</div>
    </div>
    """, 
    unsafe_allow_html=True
)
    
    # ----- Sidebar -----
    with st.sidebar:
        st.markdown("<h3>Control Panel</h3>", unsafe_allow_html=True)
        
        st.markdown("### Data Generation")
        if st.button("Generate New Candidate Fact", key="gen_new_fact"):
            st.session_state.candidate_fact, st.session_state.fact_quality = generate_candidate_fact(st.session_state.introduce_shift)
            st.rerun()
        
        # Allow manual selection of quality level for testing
        quality_options = [
            "high_quality", "medium_quality", "spatial_issue", 
            "external_ref", "semantic_issue", "low_quality"
        ]
        
        quality_level = st.selectbox(
            "Select Fact Quality Level",
            options=quality_options,
            index=quality_options.index(st.session_state.fact_quality) if hasattr(st.session_state, 'fact_quality') else 0,
            help="Choose quality level for testing different verification paths"
        )
        
        if quality_level != st.session_state.fact_quality:
            st.session_state.candidate_fact, st.session_state.fact_quality = generate_candidate_fact_with_quality(quality_level, st.session_state.introduce_shift)
            st.rerun()
        
        st.markdown("### Knowledge Graph")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Entities</div>
            <div class="metric-value">{len(st.session_state.kg.entities)}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Relationships</div>
            <div class="metric-value">{len(st.session_state.kg.relationships)}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # AAIC Controls
        st.markdown("### AAIC Parameters")
        
        with st.expander("Adaptive Parameters", expanded=False):
            h_param = st.slider("CGR-CUSUM Threshold (h)", 0.0, 10.0, st.session_state.aaic.h, 0.5,
                                help="Threshold for detecting performance shifts")
            if h_param != st.session_state.aaic.h:
                st.session_state.aaic.h = h_param
            
            k_param = st.slider("Allowance Parameter (k)", 0.01, 0.2, st.session_state.aaic.k, 0.01,
                                help="Allowance in CGR-CUSUM algorithm")
            if k_param != st.session_state.aaic.k:
                st.session_state.aaic.k = k_param
            
            eta_param = st.slider("Learning Rate (η)", 0.01, 0.5, st.session_state.aaic.eta, 0.01,
                                help="Learning rate for threshold updates")
            if eta_param != st.session_state.aaic.eta:
                st.session_state.aaic.eta = eta_param
            
            gamma_param = st.slider("Weight Scaling (γ)", 0.01, 0.5, st.session_state.aaic.gamma, 0.01,
                                help="Scaling factor for weight updates")
            if gamma_param != st.session_state.aaic.gamma:
                st.session_state.aaic.gamma = gamma_param
        
        # Performance Shift Controls
        st.markdown("### Performance Testing")
        
        st.markdown("""
        <div class="alert-info">
            Performance shift testing allows you to simulate degraded performance 
            in specific modules to trigger AAIC adaptation.
        </div>
        """, unsafe_allow_html=True)
        
        st.session_state.introduce_shift = st.checkbox("Introduce Performance Shifts", 
                                                      value=st.session_state.introduce_shift,
                                                      help="When enabled, generated facts will contain anomalies that reduce performance")
        
        if st.session_state.introduce_shift:
            st.session_state.shift_module = st.selectbox(
                "Module to affect",
                options=["LOV", "POV", "MAV", "WSV", "ESV"],
                index=["LOV", "POV", "MAV", "WSV", "ESV"].index(st.session_state.shift_module),
                help="Select which verification module will experience reduced performance"
            )
            
            st.info(f"Next fact will have a performance shift in the {st.session_state.shift_module} module")
        
        # Batch Processing Button
        st.markdown("### Batch Processing")
        
        st.markdown("""
        <div class="alert-info">
            Run batch processing to generate more verification data and trigger AAIC parameter adjustments.
            More runs increase the likelihood of detecting performance shifts.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            batch_size = st.number_input("Number of facts to process", min_value=1, max_value=100, value=10)
        
        with col2:
            force_shifts = st.checkbox("Force Performance Shifts", value=False, 
                                      help="When enabled, guarantees at least some performance shifts will be generated")
        
        if st.button(f"Process {batch_size} Facts", key="batch_verify"):
            # Add a progress bar
            progress_bar = st.progress(0)
            
            # Keep track of shifts detected
            shifts_detected = 0
            
            for i in range(batch_size):
                # Update progress
                progress_bar.progress((i + 1) / batch_size)
                
                # For forced shifts, periodically generate a fact with a shift
                # to ensure AAIC parameter adjustments are triggered
                introduce_shift = st.session_state.introduce_shift
                if force_shifts and i % 3 == 0:  # Every 3rd iteration
                    introduce_shift = True
                
                # Generate new fact
                fact, quality = generate_candidate_fact(introduce_shift)
                
                # Run verification
                verification_results = st.session_state.rmmve.verify(fact, st.session_state.kg, quality)
                
                # Record verification
                verification_record = {
                    "timestamp": time.time(),
                    "candidate_fact": fact,
                    "verification_results": verification_results,
                    "fact_quality": quality
                }
                
                # Run AAIC
                aaic_updates = st.session_state.aaic.update_all_modules()
                verification_record["aaic_updates"] = aaic_updates
                
                # Count shifts detected
                for update in aaic_updates:
                    if update["detected"]:
                        shifts_detected += 1
                
                # Add to history
                st.session_state.verification_history.append(verification_record)
                st.session_state.verification_count += 1
            
            # Generate new candidate fact
            st.session_state.candidate_fact, st.session_state.fact_quality = generate_candidate_fact(st.session_state.introduce_shift)
            
            # Show message about shifts
            if shifts_detected > 0:
                st.success(f"✅ Batch completed with {shifts_detected} performance shifts detected and parameters adjusted!")
            else:
                st.info("Batch completed without any parameter adjustments. Try running again or check CGR-CUSUM parameters.")
            
            # Refresh
            st.rerun()
    
    # ----- Main Content Area -----
    
    # Create tabs for different views
    tab_verification, tab_aaic, tab_parameters, tab_history = st.tabs([
        "💠 Verification Process", 
        "🔄 AAIC Monitoring", 
        "📊 Parameter Evolution",
        "📜 Verification History"
    ])
    
    with tab_verification:  # Verification Process Tab
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("## Candidate Fact")
            
            # Display quality level with modern styling
            quality_color = QUALITY_COLOR_MAP.get(st.session_state.fact_quality, "#6b7280")
            st.markdown(
                f"""
                <div style='background-color: {quality_color}; padding: 12px; border-radius: 8px; 
                     color: white; margin-bottom: 15px; display: flex; align-items: center;'>
                    <span style='font-size: 24px; margin-right: 10px;'>📋</span>
                    <div>
                        <strong style='font-size: 16px;'>Fact Quality:</strong><br>
                        <span style='font-size: 20px; font-weight: 500;'>{st.session_state.fact_quality.replace('_', ' ').title()}</span>
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Show if the fact contains a performance shift
            if "contains_performance_shift" in st.session_state.candidate_fact:
                shift_module = st.session_state.candidate_fact["contains_performance_shift"]
                st.markdown(format_shift_alert(shift_module), unsafe_allow_html=True)
            
            with st.expander("Fact Details", expanded=True):
                st.markdown(create_fact_preview_card(st.session_state.candidate_fact), unsafe_allow_html=True)
            
            # Add JSON expander to show raw data
            with st.expander("Raw JSON Data", expanded=False):
                # Use json.dumps with indentation for pretty printing
                st.code(json.dumps(st.session_state.candidate_fact, indent=2), language="json")
            
            # Verification control
            st.markdown("## Verification Process")
            
            st.markdown("""
            <div class="alert-info">
                Click "Run Verification" to apply the RMMVe process to the candidate fact.
                The verification system will attempt to sequentially apply verification modules
                with the possibility of early termination when sufficient confidence is reached.
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Run Verification", key="run_verify"):
                with st.spinner("Running verification..."):
                    # Run verification
                    verification_results = st.session_state.rmmve.verify(
                        st.session_state.candidate_fact, 
                        st.session_state.kg,
                        st.session_state.fact_quality
                    )
                    
                    # Record verification
                    verification_record = {
                        "timestamp": time.time(),
                        "candidate_fact": st.session_state.candidate_fact,
                        "verification_results": verification_results,
                        "fact_quality": st.session_state.fact_quality
                    }
                    
                    # Run AAIC
                    aaic_updates = st.session_state.aaic.update_all_modules()
                    verification_record["aaic_updates"] = aaic_updates
                    
                    # Add to history
                    st.session_state.verification_history.append(verification_record)
                    st.session_state.verification_count += 1
                    
                    # Generate new candidate fact
                    st.session_state.candidate_fact, st.session_state.fact_quality = generate_candidate_fact(st.session_state.introduce_shift)
                    
                    # Refresh
                    st.rerun()
        
        with col2:
            st.markdown("## Verification Results")
            
            if st.session_state.verification_history:
                # Get latest verification
                latest = st.session_state.verification_history[-1]
                results = latest["verification_results"]
                
                # Display fact quality
                quality = results.get("fact_quality", "unknown")
                quality_color = QUALITY_COLOR_MAP.get(quality, "#6b7280")
                
                # Create a results header with fancy styling
                st.markdown(
                    f"""
                    <div style='display: flex; margin-bottom: 15px;'>
                        <div style='background-color: {quality_color}; padding: 12px; border-radius: 8px; 
                             color: white; flex: 1; margin-right: 10px;'>
                            <span style='font-size: 16px;'>Fact Quality</span><br>
                            <span style='font-size: 20px; font-weight: 500;'>{quality.replace('_', ' ').title()}</span>
                        </div>
                        <div style='background-color: {"#10b981" if results["decision"] else "#ef4444"}; 
                                    padding: 12px; border-radius: 8px; color: white; flex: 1;'>
                            <span style='font-size: 16px;'>Decision</span><br>
                            <span style='font-size: 20px; font-weight: 500;'>{results["decision"] and "VERIFIED" or "REJECTED"}</span>
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Show if this fact contained a performance shift
                if results.get("contains_shift"):
                    st.markdown(format_shift_alert(results['contains_shift']), unsafe_allow_html=True)
                
                # Display verification summary in a card
                st.markdown(
                    f"""
                    <div class="card">
                        <h3>Verification Summary</h3>
                        <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 10px;">
                            <div style="flex: 1; min-width: 150px;">
                                <div class="metric-label">Total Confidence</div>
                                <div class="metric-value" style="color: {'#10b981' if results['total_confidence'] >= st.session_state.rmmve.global_threshold else '#ef4444'};">
                                    {results['total_confidence']:.3f}
                                </div>
                            </div>
                            <div style="flex: 1; min-width: 150px;">
                                <div class="metric-label">Global Threshold</div>
                                <div class="metric-value">{st.session_state.rmmve.global_threshold:.3f}</div>
                            </div>
                            <div style="flex: 1; min-width: 150px;">
                                <div class="metric-label">Processing Time</div>
                                <div class="metric-value">{results['verification_time']:.3f}s</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Show early termination information prominently
                if results.get("early_termination"):
                    module_name = results.get("early_termination_module")
                    confidence = results.get("early_termination_confidence", 0)
                    threshold = results.get("early_termination_threshold", 0)
                    
                    st.markdown(
                        f"""
                        <div style='background-color: #10b981; color: white; padding: 15px; border-radius: 8px; margin: 15px 0;'>
                            <div style='display: flex; align-items: center;'>
                                <span style='font-size: 24px; margin-right: 15px;'>🚀</span>
                                <div>
                                    <h3 style='margin:0; font-size: 18px;'>Early Termination Achieved!</h3>
                                    <p style='margin-bottom: 5px;'>Verification stopped at <b>{module_name}</b> module</p>
                                    <p style='margin:0;'>Module confidence: <b>{confidence:.3f}</b> ≥ threshold: <b>{threshold:.3f}</b></p>
                                </div>
                            </div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        """
                        <div class="alert-warning">
                            <strong>⚠️ No early termination.</strong> All verification modules were used in the process.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Create a grid of gauge charts for module performance
                st.markdown("### Module Performance")
                gauges = create_gauge_grid(results)
                if gauges:
                    gauge_cols = st.columns(len(gauges))
                    for i, (col, gauge) in enumerate(zip(gauge_cols, gauges)):
                        with col:
                            st.plotly_chart(gauge, use_container_width=True)
                
                # Plot verification results using Plotly
                st.markdown("### Module Confidence vs. Thresholds")
                fig1 = plot_verification_results_plotly(results)
                st.plotly_chart(fig1, use_container_width=True)
                
                # Plot metrics
                st.markdown("### Verification Metrics")
                fig2 = plot_metrics_plotly(results)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.markdown(
                    """
                    <div class="alert-info">
                        <span style="font-size: 24px; margin-right: 10px;">ℹ️</span>
                        <span>No verification results yet. Run verification to see results.</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    with tab_aaic:  # AAIC Monitoring Tab
        st.markdown("## AAIC Monitoring")
        
        # Explanation of AAIC
        st.markdown(
            """
            <div class="card">
                <h3>Autonomous Adaptive Intelligence Cycle</h3>
                <p>
                    The AAIC continuously monitors the performance of each verification module
                    and adaptively adjusts their parameters when significant shifts in performance are detected.
                </p>
                <p>
                    This monitoring uses the <strong>CGR-CUSUM algorithm</strong> to detect when a module's performance 
                    deviates significantly from its target performance level. When the cumulative sum exceeds the 
                    threshold (h), the AAIC updates the module's parameters:
                </p>
                <ol>
                    <li><strong>Weight (w)</strong> - Using exponential weights algorithm</li>
                    <li><strong>Threshold (θ)</strong> - Using gradient ascent</li>
                    <li><strong>Alpha (α)</strong> - Using gradient ascent</li>
                </ol>
                <p>
                    These updates help the system adapt to changing data characteristics and maintain optimal verification performance.
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Display CGR-CUSUM Monitoring
        st.markdown("### CGR-CUSUM Monitoring")
        
        param_history_df = st.session_state.aaic.get_parameter_history_df()
        if not param_history_df.empty:
            # Display plot
            cusum_fig = plot_aaic_cumulative_sums_plotly(param_history_df)
            if cusum_fig:
                st.plotly_chart(cusum_fig, use_container_width=True)
            
            # Show current cumulative sums with nicer styling
            st.markdown("### Current Cumulative Sums")
            
            cum_sum_data = []
            for module_name, cum_sum in st.session_state.aaic.cumulative_sums.items():
                cum_sum_data.append({
                    "Module": module_name, 
                    "Cumulative Sum": cum_sum, 
                    "Threshold": st.session_state.aaic.h
                })
            
            # Create a grid of modules showing cumulative sums
            cols = st.columns(len(cum_sum_data))
            for i, (col, data) in enumerate(zip(cols, cum_sum_data)):
                with col:
                    # Determine color based on proximity to threshold
                    if data["Cumulative Sum"] > 0.8 * data["Threshold"]:
                        color = "#f59e0b"  # Orange - warning
                    elif data["Cumulative Sum"] > 0.5 * data["Threshold"]:
                        color = "#3b82f6"  # Blue - moderate
                    else:
                        color = "#10b981"  # Green - good
                    
                    # Calculate percentage of threshold
                    pct = (data["Cumulative Sum"] / data["Threshold"]) * 100
                    
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">{data["Module"]}</div>
                            <div class="metric-value" style="color: {color};">{data["Cumulative Sum"]:.3f}</div>
                            <div style="font-size: 0.8rem; color: #6b7280;">{pct:.1f}% of threshold</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            
            # Display detected shifts
            shifts_df = st.session_state.aaic.get_detected_shifts_df()
            if not shifts_df.empty:
                st.markdown("### Detected Parameter Shifts")
                
                for i, shift in shifts_df.iterrows():
                    module = shift["module"]
                    timestamp = datetime.fromtimestamp(shift["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                    cum_sum = shift["cumulative_sum"]
                    performance = shift["performance"]
                    
                    st.markdown(
                        f"""
                        <div class="card">
                            <h3>Shift {i+1}: Module {module}</h3>
                            <div style="display: flex; margin-bottom: 10px;">
                                <div style="flex: 1;">
                                    <div class="metric-label">Timestamp</div>
                                    <div style="font-weight: 500;">{timestamp}</div>
                                </div>
                                <div style="flex: 1;">
                                    <div class="metric-label">Cumulative Sum</div>
                                    <div style="font-weight: 500;">{cum_sum:.3f}</div>
                                </div>
                                <div style="flex: 1;">
                                    <div class="metric-label">Performance</div>
                                    <div style="font-weight: 500;">{performance:.3f}</div>
                                </div>
                            </div>
                            
                            <h4>Parameter Adjustments</h4>
                            {create_parameter_change_table(shift)}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.markdown(
                    """
                    <div class="alert-info">
                        <span style="font-size: 24px; margin-right: 10px;">ℹ️</span>
                        <span>No parameter shifts detected yet. Run more verifications or lower the h threshold to see shifts.</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                """
                <div class="alert-info">
                    <span style="font-size: 24px; margin-right: 10px;">ℹ️</span>
                    <span>No performance data available yet. Run verification to generate performance history.</span>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    with tab_parameters:  # Parameter Evolution Tab
        st.markdown("## Parameter Evolution")
        
        # Explanation of module parameters
        st.markdown(
            """
            <div class="card">
                <h3>Adaptive Parameters</h3>
                <p>
                    Each verification module has three key parameters that the AAIC system adapts based on performance:
                </p>
                <ul>
                    <li>
                        <strong>Weight (w)</strong>: The relative importance of the module in the verification process. 
                        Higher weights mean the module has more influence on the final confidence score.
                    </li>
                    <li>
                        <strong>Threshold (θ)</strong>: The minimum confidence level required for early termination. 
                        If a module's confidence exceeds this threshold, verification can stop early at this module.
                    </li>
                    <li>
                        <strong>Alpha (α)</strong>: The balance between the two metrics used in each module. 
                        Controls how much each metric contributes to the module's overall confidence.
                    </li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        param_history_df = st.session_state.aaic.get_parameter_history_df()
        if not param_history_df.empty:
            # Create parameter evolution plots
            st.markdown("### Parameter Evolution Charts")
            
            # Create tabs for different parameter types
            param_tabs = st.tabs(["Weights", "Thresholds", "Alpha Values", "Performance"])
            
            with param_tabs[0]:
                weight_fig = plot_aaic_parameter_evolution_plotly(param_history_df, "Weight")
                if weight_fig:
                    st.plotly_chart(weight_fig, use_container_width=True)
            
            with param_tabs[1]:
                threshold_fig = plot_aaic_parameter_evolution_plotly(param_history_df, "Threshold")
                if threshold_fig:
                    st.plotly_chart(threshold_fig, use_container_width=True)
            
            with param_tabs[2]:
                alpha_fig = plot_aaic_parameter_evolution_plotly(param_history_df, "Alpha")
                if alpha_fig:
                    st.plotly_chart(alpha_fig, use_container_width=True)
            
            with param_tabs[3]:
                performance_fig = plot_aaic_parameter_evolution_plotly(param_history_df, "Performance")
                if performance_fig:
                    st.plotly_chart(performance_fig, use_container_width=True)
            
            # Display current module parameters
            st.markdown("### Current Module Parameters")
            
            # Create parameter cards for each module
            module_cols = st.columns(len(st.session_state.rmmve.modules))
            
            for i, (col, module) in enumerate(zip(module_cols, st.session_state.rmmve.modules)):
                with col:
                    st.markdown(
                        f"""
                        <div class="card">
                            <h3>{module.name}</h3>
                            <div style="margin-bottom: 8px;">
                                <div class="metric-label">Weight (w)</div>
                                <div style="font-size: 1.5rem; font-weight: 600; color: #3b82f6;">{module.weight:.4f}</div>
                            </div>
                            <div style="margin-bottom: 8px;">
                                <div class="metric-label">Threshold (θ)</div>
                                <div style="font-size: 1.5rem; font-weight: 600; color: #f59e0b;">{module.threshold:.4f}</div>
                            </div>
                            <div>
                                <div class="metric-label">Alpha (α)</div>
                                <div style="font-size: 1.5rem; font-weight: 600; color: #8b5cf6;">{module.alpha:.4f}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.markdown(
                """
                <div class="alert-info">
                    <span style="font-size: 24px; margin-right: 10px;">ℹ️</span>
                    <span>No parameter evolution data available yet. Run verification to generate parameter history.</span>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    with tab_history:  # Verification History Tab
        st.markdown("## Verification History")
        
        # Display key metrics at the top
        if st.session_state.verification_count > 0:
            metrics_cols = st.columns(4)
            
            with metrics_cols[0]:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Total Verifications</div>
                        <div class="metric-value">{st.session_state.verification_count}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            if st.session_state.verification_history:
                # Calculate statistics
                history_df = pd.DataFrame([
                    {
                        "Time": datetime.fromtimestamp(record["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
                        "Fact Quality": record["verification_results"].get("fact_quality", "unknown"),
                        "Contains Shift": record["verification_results"].get("contains_shift", "None"),
                        "Total Confidence": round(record["verification_results"]["total_confidence"], 4),
                        "Decision": "✅ Verified" if record['verification_results']['decision'] else "❌ Rejected",
                        "Modules Used": len(record['verification_results']['activated_modules']),
                        "Early Termination": "✓ Yes" if record['verification_results'].get('early_termination') else "✗ No",
                        "Term. Module": record['verification_results'].get('early_termination_module', "None")
                    }
                    for record in st.session_state.verification_history
                ])
                
                verify_rate = len(history_df[history_df["Decision"].str.contains("Verified")]) / len(history_df) * 100
                early_term_rate = len(history_df[history_df["Early Termination"] == "✓ Yes"]) / len(history_df) * 100
                avg_modules = history_df["Modules Used"].mean()
                
                with metrics_cols[1]:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Verification Rate</div>
                            <div class="metric-value">{verify_rate:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                with metrics_cols[2]:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Early Termination Rate</div>
                            <div class="metric-value">{early_term_rate:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                with metrics_cols[3]:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Avg. Modules Used</div>
                            <div class="metric-value">{avg_modules:.2f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Display performance trends
                st.markdown("### Performance Trends")
                history_tabs = st.tabs(["Confidence History", "Quality Distribution", "Early Termination Distribution"])
                
                with history_tabs[0]:
                    # Plot confidence history
                    confidence_fig = plot_verification_history_plotly(history_df)
                    if confidence_fig:
                        st.plotly_chart(confidence_fig, use_container_width=True)
                
                with history_tabs[1]:
                    # Plot quality distribution
                    quality_fig = plot_quality_distribution_plotly(history_df)
                    if quality_fig:
                        st.plotly_chart(quality_fig, use_container_width=True)
                
                with history_tabs[2]:
                    # Plot early termination by module
                    term_fig = plot_early_term_by_module_plotly(history_df)
                    if term_fig:
                        st.plotly_chart(term_fig, use_container_width=True)
                    else:
                        st.info("No early terminations recorded yet.")
                
                # Detailed verification history table
                st.markdown("### Detailed History")
                
                # Add quality color highlighting
                def format_quality(row):
                    quality = row["Fact Quality"]
                    quality_color = QUALITY_COLOR_MAP.get(quality, "#6b7280")
                    quality_name = quality.replace("_", " ").title()
                    return f'<span style="color: {quality_color}; font-weight: bold;">{quality_name}</span>'
                
                def format_decision(row):
                    if "Verified" in row["Decision"]:
                        return '<span class="status-verified">VERIFIED</span>'
                    else:
                        return '<span class="status-rejected">REJECTED</span>'
                
                # Apply formatting
                history_df["Formatted Quality"] = history_df.apply(format_quality, axis=1)
                history_df["Formatted Decision"] = history_df.apply(format_decision, axis=1)
                
                # Display table with formatted columns
                display_cols = ["Time", "Formatted Quality", "Total Confidence", "Formatted Decision", 
                               "Modules Used", "Early Termination", "Term. Module"]
                
                st.markdown(
                    """
                    <style>
                    .dataframe th {
                        text-align: center !important;
                    }
                    </style>
                    """, 
                    unsafe_allow_html=True
                )
                
                st.write(
                    history_df[display_cols].to_html(
                        escape=False, 
                        index=False,
                        classes='dataframe'
                    ), 
                    unsafe_allow_html=True
                )
            else:
                with metrics_cols[1]:
                    st.metric("Verification Rate", "0%")
                with metrics_cols[2]:
                    st.metric("Early Termination Rate", "0%")
                with metrics_cols[3]:
                    st.metric("Avg. Modules Used", "0")
                
                st.info("No verification history details available yet.")
        else:
            st.markdown(
                """
                <div class="alert-info">
                    <span style="font-size: 24px; margin-right: 10px;">ℹ️</span>
                    <span>No verification history yet. Run verification to generate history.</span>
                </div>
                """,
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()