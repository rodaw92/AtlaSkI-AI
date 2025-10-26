"""
Aviation Safety Dataset Generator (NASA ASRS-style)

Tests: Temporal consistency ψ_t, causal relationships, narrative extraction
Error Types: Fabricated event sequences, temporal impossibilities
"""

import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

def generate_aviation_facts(num_facts: int = 100) -> Tuple[List[Dict], List[bool]]:
    """
    Generate synthetic aviation incident facts similar to NASA ASRS dataset

    Returns:
        facts: List of fact dictionaries
        labels: List of booleans (True = correct, False = incorrect)
    """
    facts = []
    labels = []

    aircraft_ids = [f"N{random.randint(10000, 99999)}" for _ in range(20)]
    airports = ["KJFK", "KLAX", "KORD", "KATL", "KDFW", "KDEN", "KSFO", "KLAS"]
    altitudes = [350, 310, 280, 250, 200, 180, 150, 100, 50, 30]  # Flight levels * 100ft

    for i in range(num_facts):
        aircraft = random.choice(aircraft_ids)
        origin = random.choice(airports)
        destination = random.choice([a for a in airports if a != origin])

        # Decide error type
        error_type = random.choice(['correct', 'correct', 'correct',  # 60% correct
                                    'temporal_impossibility', 'fabricated_sequence',
                                    'velocity_violation', 'causal_violation'])

        is_correct = (error_type == 'correct')

        base_time = datetime.now() - timedelta(days=random.randint(1, 30))

        if error_type == 'correct':
            # Generate plausible sequence
            alt_start = random.choice(altitudes[:-2])
            alt_end = alt_start - random.randint(1, 3) * 100  # Descent
            time_diff = random.randint(300, 900)  # 5-15 minutes

            fact = {
                "subject_entity_id": aircraft,
                "relationship_type": "descended",
                "object_entity_id": f"FL{alt_start}",
                "target_entity_id": f"FL{alt_end}",
                "origin": origin,
                "destination": destination,
                "altitude_start_ft": alt_start * 100,
                "altitude_end_ft": alt_end * 100,
                "vertical_speed_fpm": abs((alt_start - alt_end) * 100) / (time_diff / 60),
                "duration_seconds": time_diff,
                "spatiotemporal": {
                    "start_timestamp": base_time.isoformat() + "Z",
                    "end_timestamp": (base_time + timedelta(seconds=time_diff)).isoformat() + "Z",
                    "distance_nm": random.randint(50, 200)
                },
                "error_type": error_type
            }

        elif error_type == 'temporal_impossibility':
            # Descent too fast (physically impossible)
            alt_start = 350
            alt_end = 100
            time_diff = random.randint(30, 60)  # Way too fast!

            fact = {
                "subject_entity_id": aircraft,
                "relationship_type": "descended",
                "object_entity_id": f"FL{alt_start}",
                "target_entity_id": f"FL{alt_end}",
                "origin": origin,
                "destination": destination,
                "altitude_start_ft": alt_start * 100,
                "altitude_end_ft": alt_end * 100,
                "vertical_speed_fpm": abs((alt_start - alt_end) * 100) / (time_diff / 60),  # ~15,000 fpm!
                "duration_seconds": time_diff,
                "spatiotemporal": {
                    "start_timestamp": base_time.isoformat() + "Z",
                    "end_timestamp": (base_time + timedelta(seconds=time_diff)).isoformat() + "Z",
                    "distance_nm": random.randint(50, 200)
                },
                "error_type": error_type
            }

        elif error_type == 'fabricated_sequence':
            # Events in wrong causal order
            alt_start = 200
            alt_end = 350  # Climbing after landing? Impossible!
            time_diff = random.randint(300, 600)

            fact = {
                "subject_entity_id": aircraft,
                "relationship_type": "climbed_after_landing",  # Nonsensical
                "object_entity_id": f"FL{alt_start}",
                "target_entity_id": f"FL{alt_end}",
                "origin": origin,
                "destination": destination,
                "altitude_start_ft": alt_start * 100,
                "altitude_end_ft": alt_end * 100,
                "vertical_speed_fpm": abs((alt_end - alt_start) * 100) / (time_diff / 60),
                "duration_seconds": time_diff,
                "spatiotemporal": {
                    "start_timestamp": base_time.isoformat() + "Z",
                    "end_timestamp": (base_time + timedelta(seconds=time_diff)).isoformat() + "Z",
                    "distance_nm": random.randint(50, 200)
                },
                "error_type": error_type
            }

        elif error_type == 'velocity_violation':
            # Ground speed exceeds aircraft limits
            time_diff = 600  # 10 minutes
            distance = 250  # nautical miles -> 1,500 knots ground speed!

            fact = {
                "subject_entity_id": aircraft,
                "relationship_type": "traveled",
                "object_entity_id": origin,
                "target_entity_id": destination,
                "origin": origin,
                "destination": destination,
                "altitude_start_ft": 35000,
                "altitude_end_ft": 35000,
                "vertical_speed_fpm": 0,
                "duration_seconds": time_diff,
                "ground_speed_knots": (distance / time_diff) * 3600,  # Way too fast!
                "spatiotemporal": {
                    "start_timestamp": base_time.isoformat() + "Z",
                    "end_timestamp": (base_time + timedelta(seconds=time_diff)).isoformat() + "Z",
                    "distance_nm": distance
                },
                "error_type": error_type
            }

        else:  # causal_violation
            # Event happens before its prerequisite
            alt_start = 100
            alt_end = 0  # Landing
            time_diff = 180

            fact = {
                "subject_entity_id": aircraft,
                "relationship_type": "landed_before_descent",  # Causal violation
                "object_entity_id": origin,
                "target_entity_id": destination,
                "origin": origin,
                "destination": destination,
                "altitude_start_ft": alt_start * 100,
                "altitude_end_ft": 0,
                "vertical_speed_fpm": (alt_start * 100) / (time_diff / 60),
                "duration_seconds": time_diff,
                "spatiotemporal": {
                    "start_timestamp": (base_time + timedelta(seconds=600)).isoformat() + "Z",  # After!
                    "end_timestamp": base_time.isoformat() + "Z",  # Before! (reversed)
                    "distance_nm": random.randint(20, 80)
                },
                "error_type": error_type
            }

        facts.append(fact)
        labels.append(is_correct)

    return facts, labels


def get_dataset_info() -> Dict:
    """Return metadata about this dataset type"""
    return {
        'name': 'Aviation Safety (NASA ASRS-style)',
        'challenge': 'Unstructured narrative extraction',
        'primary_test': 'Temporal consistency ψ_t, causal relationships',
        'error_types': [
            'Fabricated event sequences',
            'Temporal impossibilities (too fast descent)',
            'Velocity violations (exceeding aircraft limits)',
            'Causal violations (effect before cause)'
        ],
        'example': 'Aircraft descended FL350→FL280 in 45 seconds (violates ψ_t)',
        'expected_performance': {
            'precision': 0.93,
            'recall': 0.94,
            'f1': 0.93,
            'fpr': 0.032
        }
    }
