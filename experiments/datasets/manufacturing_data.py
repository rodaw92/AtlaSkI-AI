"""
Manufacturing/Aerospace Dataset Generator (AddQual-style)

Tests: Micro-tolerance precision (±0.1mm), spatial consistency ψ_s
Error Types: Hallucinated tolerances, impossible measurements, spatial violations
"""

import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

def generate_manufacturing_facts(num_facts: int = 100) -> Tuple[List[Dict], List[bool]]:
    """
    Generate synthetic manufacturing facts similar to AddQual dataset

    Returns:
        facts: List of fact dictionaries
        labels: List of booleans (True = correct, False = incorrect)
    """
    facts = []
    labels = []

    components = [f"TurbineBlade_{chr(65+i)}" for i in range(10)]
    features = [
        "Blade Root - Simple Mount",
        "High Pressure - Pitch Distance",
        "Leading Edge - Pressure Side",
        "Trailing Edge - Suction Side",
        "Blade Tip - Clearance Gap"
    ]

    # Generate mix of correct and incorrect facts
    for i in range(num_facts):
        component = random.choice(components)
        feature = random.choice(features)

        # Base measurement
        nominal_value = round(random.uniform(2.0, 5.0), 2)
        tolerance = 0.1  # ±0.1mm standard tolerance

        # Decide if this fact should be correct or have an error
        error_type = random.choice(['correct', 'correct', 'correct',  # 60% correct
                                    'tolerance_violation', 'spatial_violation',
                                    'hallucinated_value', 'temporal_violation'])

        is_correct = (error_type == 'correct')

        if error_type == 'correct':
            # Generate correct measurement within tolerance
            actual_value = round(nominal_value + random.uniform(-tolerance, tolerance), 3)
            x = round(random.uniform(5, 15), 1)
            y = round(random.uniform(15, 25), 1)
            z = round(random.uniform(100, 200), 1)

        elif error_type == 'tolerance_violation':
            # Measurement exceeds tolerance (hallucinated)
            actual_value = round(nominal_value + random.uniform(0.2, 0.5), 3)
            x = round(random.uniform(5, 15), 1)
            y = round(random.uniform(15, 25), 1)
            z = round(random.uniform(100, 200), 1)

        elif error_type == 'spatial_violation':
            # Measurement OK but spatial coordinates inconsistent
            actual_value = round(nominal_value + random.uniform(-tolerance, tolerance), 3)
            # Same component at two far locations at same time (violates ψ_s)
            x = round(random.uniform(50, 100), 1)  # Far from typical range
            y = round(random.uniform(50, 100), 1)
            z = round(random.uniform(500, 600), 1)

        elif error_type == 'hallucinated_value':
            # Completely fabricated measurement
            actual_value = round(random.uniform(10, 20), 3)  # Way off nominal
            x = round(random.uniform(5, 15), 1)
            y = round(random.uniform(15, 25), 1)
            z = round(random.uniform(100, 200), 1)

        else:  # temporal_violation
            # Measurement at impossible time (future or past)
            actual_value = round(nominal_value + random.uniform(-tolerance, tolerance), 3)
            x = round(random.uniform(5, 15), 1)
            y = round(random.uniform(15, 25), 1)
            z = round(random.uniform(100, 200), 1)

        # Generate timestamp
        now = datetime.now()
        if error_type == 'temporal_violation':
            # Invalid timestamp
            timestamp = (now + timedelta(days=random.randint(1, 30))).isoformat() + "Z"
        else:
            # Valid timestamp (recent past)
            timestamp = (now - timedelta(hours=random.randint(1, 240))).isoformat() + "Z"

        # Create fact
        fact = {
            "subject_entity_id": component,
            "relationship_type": "hasMeasurement",
            "object_entity_id": f"Measurement_{i}",
            "feature_name": feature,
            "nominal_value_mm": nominal_value,
            "actual_value_mm": actual_value,
            "deviation_mm": round(actual_value - nominal_value, 3),
            "tolerance_mm": tolerance,
            "status": "PASS" if abs(actual_value - nominal_value) <= tolerance else "FAIL",
            "spatiotemporal": {
                "x_coord": x,
                "y_coord": y,
                "z_coord": z,
                "timestamp": timestamp
            },
            "error_type": error_type
        }

        facts.append(fact)
        labels.append(is_correct)

    return facts, labels


def get_dataset_info() -> Dict:
    """Return metadata about this dataset type"""
    return {
        'name': 'Manufacturing/Aerospace',
        'challenge': 'Micro-tolerance precision (±0.1mm)',
        'primary_test': 'Spatial consistency ψ_s, measurement validation',
        'error_types': [
            'Hallucinated tolerance values',
            'Impossible measurements',
            'Spatial violations (ψ_s)'
        ],
        'example': 'Turbine blade deviation: 0.023mm at (10.5, 20.3, 150.2)',
        'expected_performance': {
            'precision': 0.94,
            'recall': 0.91,
            'f1': 0.92,
            'fpr': 0.026
        }
    }
