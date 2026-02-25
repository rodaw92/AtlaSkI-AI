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

# Review margin for three-way decision (Equation 23)
EPSILON_REVIEW = 0.1  # Uncertainty margin for borderline cases

# Veto thresholds τ_veto per domain (severe physical violations)
VETO_THRESHOLDS = {
    "healthcare": 0.5,
    "aerospace": 0.30,
    "default": 0.35,
}

# Critical modules whose veto check applies to ST facts
CRITICAL_MODULES = {"MAV"}

# CGR-CUSUM parameters derived from σ (baseline precision std dev)
CUSUM_SIGMA = 0.1        # Baseline standard deviation of module precision
CUSUM_K = 0.5 * CUSUM_SIGMA   # Allowable slack k = 0.5σ
CUSUM_H = 5.0 * CUSUM_SIGMA   # Alarm threshold h = 5σ

# Module computational costs in milliseconds (Table 1)
MODULE_COSTS_MS = {
    "LOV": 5,
    "POV": 15,
    "MAV": 50,
    "WSV": 120,
    "ESV": 800,
}

# Minimum process durations (seconds) for MAV Process(d_k)
MIN_PROCESS_DURATIONS = {
    "aerospace": {
        "engine_installation": 45 * 60,
        "blade_inspection": 15 * 60,
        "measurement": 5 * 60,
        "default": 5 * 60,
    },
    "healthcare": {
        "icu_discharge": 17 * 60,
        "patient_transfer": 10 * 60,
        "surgery_prep": 30 * 60,
        "default": 10 * 60,
    },
    "default": {
        "default": 5 * 60,
    },
}

# Standard terminology sets for POV (Metric 1)
STANDARD_TERMINOLOGIES = {
    "aerospace": {
        "entities": {
            "Blade", "EngineSet", "InspectionMeasurement", "AerospaceEntity",
            "TurbineBlade", "CompressorBlade", "FanBlade", "Assembly",
        },
        "relations": {
            "containsBlade", "hasMeasurement", "locatedAt", "installedAt",
            "isContainedBy",
        },
        "tools": {
            "3D_Scanner_Unit", "CMM", "Laser_Tracker", "Coordinate_Measuring_Machine",
            "Profilometer", "Surface_Roughness_Tester",
        },
        "features": {
            "Blade Root", "Leading Edge", "Trailing Edge", "Blade Tip",
            "Pressure Side", "Suction Side", "Clearance Gap", "Pitch Distance",
        },
        "standards": {"STEP AP242", "AS9100", "AMS", "ISO 10303", "ASTM"},
    },
    "healthcare": {
        "entities": {
            "Patient", "CareUnit", "ClinicalTransfer", "Medication",
            "MICU", "SICU", "CCU", "OR", "Recovery", "Floor",
        },
        "relations": {
            "transferred", "administeredTo", "performedAt", "locatedIn",
            "admittedTo", "dischargedFrom",
        },
        "tools": set(),
        "features": set(),
        "standards": {"HL7 FHIR", "SNOMED CT", "ICD-10", "HIPAA", "LOINC"},
    },
}

# Mapping of quality levels to colors for visualization
QUALITY_COLOR_MAP = {
    "high_quality": "#10b981",  # Green
    "medium_quality": "#3b82f6",  # Blue
    "spatial_issue": "#f59e0b",  # Orange
    "external_ref": "#8b5cf6",  # Purple
    "semantic_issue": "#06b6d4",  # Teal
    "low_quality": "#ef4444"     # Red
} 