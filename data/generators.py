import random
from datetime import datetime, timedelta
from models.constants import (
    ENGINE_SETS, BLADE_COMPONENTS, MEASUREMENT_IDS, 
    BLADE_FEATURES, SURFACE_SIDES
)

def generate_candidate_fact_with_quality(quality_level, introduce_shift=False):
    """
    Generate a random aerospace inspection candidate with actual quality issues.
    
    Parameters:
    quality_level (str): Determines the quality and type of issues in the generated fact
        - high_quality: Clean, well-formed facts with correct ontology and physics
        - medium_quality: Minor issues but generally acceptable
        - semantic_issue: Ontology violations (wrong entity classes, invalid relationships)
        - spatial_issue: Impossible spatial/temporal constraints (bilocation, impossible velocity)
        - external_ref: Fabricated references that can't be externally validated
        - low_quality: Multiple issues (semantic + spatial + invalid data)
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
        tolerance = f"±{round(random.uniform(0.080, 0.150), 3)}"
    elif quality_level == "medium_quality":
        # Moderate deviation but within tolerance
        actual_value = round(nominal_value + random.uniform(-0.03, 0.03), 3)
        tolerance = f"±{round(random.uniform(0.080, 0.150), 3)}"
    elif quality_level == "semantic_issue":
        # Use wrong terminology or invalid entity classes
        actual_value = round(nominal_value + random.uniform(-0.02, 0.02), 3)
        tolerance = f"±{round(random.uniform(0.080, 0.150), 3)}"
        # Will inject semantic errors below
    elif quality_level == "spatial_issue":
        # Normal values but will create impossible spatial/temporal constraints
        actual_value = round(nominal_value + random.uniform(-0.02, 0.02), 3)
        tolerance = f"±{round(random.uniform(0.080, 0.150), 3)}"
        # Will inject spatial errors below
    elif quality_level == "external_ref":
        # Create fabricated references
        actual_value = round(nominal_value + random.uniform(-0.02, 0.02), 3)
        tolerance = f"±{round(random.uniform(0.080, 0.150), 3)}"
        # Will use fake standards/tools
    else:  # low_quality
        # Larger deviation and multiple issues
        actual_value = round(nominal_value + random.uniform(-0.15, 0.15), 3)
        tolerance = f"±{round(random.uniform(0.020, 0.050), 3)}"  # Very tight tolerance
    
    deviation = round(actual_value - nominal_value, 3)
    surface_side = random.choice(SURFACE_SIDES)
    
    # Generate timestamps (within the past 24 hours)
    now = datetime.now()
    random_hours = random.randint(0, 23)
    random_minutes = random.randint(0, 59)
    random_seconds = random.randint(0, 59)
    timestamp = (now - timedelta(hours=random_hours, minutes=random_minutes, seconds=random_seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Generate spatial coordinates based on quality
    if quality_level == "spatial_issue":
        # Create impossible locations (e.g., extremely far away requiring impossible velocity)
        x_coord = round(random.uniform(100.0, 200.0), 2)  # Very far from origin
        y_coord = round(random.uniform(100.0, 200.0), 2)
        z_coord = round(random.uniform(50.0, 300.0), 1)
        # Make timestamp very recent so travel would be impossible
        timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        # Normal spatial coordinates
        x_coord = round(random.uniform(-10.0, 10.0), 2)
        y_coord = round(random.uniform(-10.0, 10.0), 2)
        z_coord = round(random.uniform(50.0, 300.0), 1)
    
    # Inject quality-specific issues into relationships and inspection tool
    if quality_level == "semantic_issue":
        # INJECT SEMANTIC ERRORS: Wrong relationship types or invalid entity classes
        relationships = [
            {
                "relationship_id": "R1",
                "subject_entity_id": engine_set,
                "relationship_type": "manufacturedBy",  # WRONG: Should be "containsBlade"
                "object_entity_id": blade_component
            },
            {
                "relationship_id": "R2",
                "subject_entity_id": blade_component,
                "relationship_type": "isPartOf",  # WRONG: Should be "hasMeasurement"
                "object_entity_id": measurement_id
            }
        ]
        inspection_tool = "UltraSonicInspector"  # Non-standard tool name
    elif quality_level == "external_ref":
        # INJECT EXTERNAL REFERENCE ERRORS: Fabricated tools and standards
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
        inspection_tool = "FabricatedScanner_XZ9000"  # FABRICATED: Can't be externally validated
    elif quality_level == "low_quality":
        # INJECT MULTIPLE ERRORS: Wrong relationships + fabricated tools
        relationships = [
            {
                "relationship_id": "R1",
                "subject_entity_id": engine_set,
                "relationship_type": "linkedTo",  # WRONG: Invalid relationship
                "object_entity_id": blade_component
            },
            {
                "relationship_id": "R2",
                "subject_entity_id": blade_component,
                "relationship_type": "contains",  # WRONG: Invalid relationship
                "object_entity_id": measurement_id
            }
        ]
        inspection_tool = "UnknownTool_123"  # FABRICATED
    else:
        # Normal, valid relationships
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
        inspection_tool = "3D_Scanner_Unit"  # Standard tool
    
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
                        "inspection_tool": inspection_tool,
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
    
    # Additional modifications based on quality level
    if quality_level == "spatial_issue":
        # INJECT SPATIAL ERRORS: Add conflicting spatial data (entity in two places at once)
        # Add a second spatial entry with different coordinates at same/similar time
        fact["spatiotemporal_inspection_data"]["spatial_data"].append({
            "entity_id": measurement_id,
            "coordinates": {
                "x_coord": x_coord + 50.0,  # 50m away
                "y_coord": y_coord + 50.0,
                "z_coord": z_coord
            }
        })
        # Add another temporal entry very close in time (impossible movement)
        timestamp2 = (now - timedelta(seconds=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
        fact["spatiotemporal_inspection_data"]["temporal_data"].append({
            "entity_id": measurement_id,
            "timestamp": timestamp2,
            "event_type_or_feature": feature_name
        })
    
    elif quality_level == "external_ref":
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