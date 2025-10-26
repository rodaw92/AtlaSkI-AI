# Entity classes and relationship types
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

# Physical consistency parameters (from Definition 2 & 3 in preliminaries)
# Temporal resolution (seconds) - minimum time window for distinguishing events
TAU_RES = 1.0  # 1 second temporal resolution

# Spatial resolution (meters) - minimum distinguishable distance
SIGMA_RES = 0.1  # 0.1 meters spatial resolution

# Maximum velocities for different transport modes (m/s)
V_MAX = {
    "manual": 2.0,        # Manual handling: 2 m/s
    "forklift": 5.0,      # Forklift: 5 m/s
    "conveyor": 1.0,      # Automated conveyor: 1 m/s
    "drone": 15.0,        # Drone delivery: 15 m/s
    "default": 2.0        # Default to manual handling
}

# Semantic drift threshold (from Definition 6)
THETA_DRIFT = 0.15  # KL divergence threshold for detecting semantic drift

# Review margin for three-way decision
EPSILON_REVIEW = 0.1  # Uncertainty margin for borderline cases

# Mapping of quality levels to colors for visualization
QUALITY_COLOR_MAP = {
    "high_quality": "#10b981",  # Green
    "medium_quality": "#3b82f6",  # Blue
    "spatial_issue": "#f59e0b",  # Orange
    "external_ref": "#8b5cf6",  # Purple
    "semantic_issue": "#06b6d4",  # Teal
    "low_quality": "#ef4444"     # Red
} 