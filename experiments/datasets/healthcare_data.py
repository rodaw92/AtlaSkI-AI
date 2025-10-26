"""
Clinical/Healthcare Dataset Generator (MIMIC-IV-style)

Tests: Temporal consistency ψ_t, clinical workflow compliance
Error Types: Protocol violations, impossible transfer times
"""

import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

def generate_healthcare_facts(num_facts: int = 100) -> Tuple[List[Dict], List[bool]]:
    """Generate synthetic patient transfer facts similar to MIMIC-IV dataset"""
    facts = []
    labels = []

    # Care units with typical facility coordinates
    care_units = {
        "MICU": (10, 20, 3),    # Medical ICU
        "SICU": (45, 20, 3),     # Surgical ICU
        "CCU": (80, 20, 3),      # Cardiac Care
        "OR": (25, 50, 2),       # Operating Room
        "Recovery": (60, 50, 2), # Recovery Ward
        "Floor": (40, 80, 4)     # Regular floor
    }

    # Minimum transfer times (minutes) based on protocols
    min_transfer_times = {
        ("MICU", "SICU"): 15,    # ICU-to-ICU handoff
        ("MICU", "OR"): 20,      # ICU-to-OR prep
        ("SICU", "OR"): 20,
        ("OR", "Recovery"): 10,   # Post-op transfer
        ("Recovery", "Floor"): 30 # Full assessment
    }

    patient_ids = [f"P{str(i).zfill(6)}" for i in range(1000, 1200)]

    for i in range(num_facts):
        patient_id = random.choice(patient_ids)
        from_unit = random.choice(list(care_units.keys()))
        to_unit = random.choice([u for u in care_units.keys() if u != from_unit])

        # Get minimum transfer time
        transfer_key = (from_unit, to_unit)
        min_time = min_transfer_times.get(transfer_key, 15)  # Default 15 min

        error_type = random.choice(['correct', 'correct', 'correct',  # 60% correct
                                    'protocol_violation', 'temporal_impossibility',
                                    'spatial_violation'])

        is_correct = (error_type == 'correct')

        base_time = datetime.now() - timedelta(days=random.randint(1, 30))

        if error_type == 'correct':
            # Valid transfer with appropriate time
            transfer_duration = random.randint(min_time, min_time + 30)
            actual_distance = sum((a-b)**2 for a, b in zip(care_units[from_unit], care_units[to_unit]))**0.5

            fact = {
                "subject_entity_id": patient_id,
                "relationship_type": "transferred",
                "object_entity_id": from_unit,
                "target_entity_id": to_unit,
                "transfer_duration_minutes": transfer_duration,
                "min_protocol_time_minutes": min_time,
                "spatiotemporal": {
                    "from_location": from_unit,
                    "to_location": to_unit,
                    "from_coords": care_units[from_unit],
                    "to_coords": care_units[to_unit],
                    "distance_meters": actual_distance,
                    "start_timestamp": base_time.isoformat() + "Z",
                    "end_timestamp": (base_time + timedelta(minutes=transfer_duration)).isoformat() + "Z"
                },
                "error_type": error_type
            }

        elif error_type == 'protocol_violation':
            # Transfer time violates clinical protocol (too fast)
            transfer_duration = random.randint(1, min_time - 1)  # Below minimum!
            actual_distance = sum((a-b)**2 for a, b in zip(care_units[from_unit], care_units[to_unit]))**0.5

            fact = {
                "subject_entity_id": patient_id,
                "relationship_type": "transferred",
                "object_entity_id": from_unit,
                "target_entity_id": to_unit,
                "transfer_duration_minutes": transfer_duration,
                "min_protocol_time_minutes": min_time,
                "spatiotemporal": {
                    "from_location": from_unit,
                    "to_location": to_unit,
                    "from_coords": care_units[from_unit],
                    "to_coords": care_units[to_unit],
                    "distance_meters": actual_distance,
                    "start_timestamp": base_time.isoformat() + "Z",
                    "end_timestamp": (base_time + timedelta(minutes=transfer_duration)).isoformat() + "Z"
                },
                "error_type": error_type
            }

        elif error_type == 'temporal_impossibility':
            # Physically impossible transfer time (violates ψ_t)
            transfer_duration = 1  # 1 minute for cross-building transfer!
            actual_distance = sum((a-b)**2 for a, b in zip(care_units[from_unit], care_units[to_unit]))**0.5

            fact = {
                "subject_entity_id": patient_id,
                "relationship_type": "transferred",
                "object_entity_id": from_unit,
                "target_entity_id": to_unit,
                "transfer_duration_minutes": transfer_duration,
                "min_protocol_time_minutes": min_time,
                "spatiotemporal": {
                    "from_location": from_unit,
                    "to_location": to_unit,
                    "from_coords": care_units[from_unit],
                    "to_coords": care_units[to_unit],
                    "distance_meters": actual_distance,
                    "start_timestamp": base_time.isoformat() + "Z",
                    "end_timestamp": (base_time + timedelta(minutes=transfer_duration)).isoformat() + "Z"
                },
                "error_type": error_type
            }

        else:  # spatial_violation
            # Patient at two locations simultaneously (violates ψ_s)
            transfer_duration = 0  # Simultaneous!
            actual_distance = sum((a-b)**2 for a, b in zip(care_units[from_unit], care_units[to_unit]))**0.5

            fact = {
                "subject_entity_id": patient_id,
                "relationship_type": "at_two_locations",
                "object_entity_id": from_unit,
                "target_entity_id": to_unit,
                "transfer_duration_minutes": transfer_duration,
                "min_protocol_time_minutes": min_time,
                "spatiotemporal": {
                    "from_location": from_unit,
                    "to_location": to_unit,
                    "from_coords": care_units[from_unit],
                    "to_coords": care_units[to_unit],
                    "distance_meters": actual_distance,
                    "start_timestamp": base_time.isoformat() + "Z",
                    "end_timestamp": base_time.isoformat() + "Z"  # Same time!
                },
                "error_type": error_type
            }

        facts.append(fact)
        labels.append(is_correct)

    return facts, labels


def get_dataset_info() -> Dict:
    """Return metadata about this dataset type"""
    return {
        'name': 'Clinical/Healthcare (MIMIC-IV-style)',
        'challenge': 'Clinical workflow compliance',
        'primary_test': 'Temporal consistency ψ_t, protocol validation',
        'error_types': [
            'Protocol violations (too fast transfers)',
            'Temporal impossibilities (instant transfers)',
            'Spatial violations (patient at two locations)'
        ],
        'example': 'Patient transfer ICU→OR in 5min (violates 20min protocol)',
        'expected_performance': {
            'precision': 0.95,
            'recall': 0.95,
            'f1': 0.95,
            'fpr': 0.041
        }
    }
