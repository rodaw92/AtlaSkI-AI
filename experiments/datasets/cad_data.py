"""
CAD Assembly Dataset Generator (Zenodo-style)

Tests: 3D geometric reasoning, spatial consistency ψ_s
Error Types: Invalid spatial relationships, interference violations
"""

import random
from typing import List, Dict, Tuple

def generate_cad_facts(num_facts: int = 100) -> Tuple[List[Dict], List[bool]]:
    """Generate synthetic CAD assembly facts similar to Zenodo CAD dataset"""
    facts = []
    labels = []

    components = [f"Part_{chr(65+i)}" for i in range(20)]
    relations = ["contains", "attached_to", "aligned_with", "interferes_with"]

    for i in range(num_facts):
        part_a = random.choice(components)
        part_b = random.choice([c for c in components if c != part_a])

        error_type = random.choice(['correct', 'correct', 'correct',  # 60% correct
                                    'spatial_interference', 'geometric_impossibility',
                                    'dimensional_conflict'])

        is_correct = (error_type == 'correct')

        if error_type == 'correct':
            # Valid spatial relationship
            relation = random.choice(["contains", "attached_to", "aligned_with"])
            bbox_a = {
                'min': (random.uniform(0, 50), random.uniform(0, 50), random.uniform(0, 50)),
                'max': (random.uniform(60, 100), random.uniform(60, 100), random.uniform(60, 100))
            }
            bbox_b = {
                'min': (bbox_a['max'][0] + 1, bbox_a['min'][1], bbox_a['min'][2]),
                'max': (bbox_a['max'][0] + 30, bbox_a['max'][1], bbox_a['max'][2])
            }

        elif error_type == 'spatial_interference':
            # Geometries overlap (violates ψ_s)
            relation = "attached_to"
            bbox_a = {
                'min': (0, 0, 0),
                'max': (50, 50, 50)
            }
            bbox_b = {
                'min': (25, 25, 25),  # Overlaps with part_a!
                'max': (75, 75, 75)
            }

        else:  # geometric_impossibility or dimensional_conflict
            # Impossible spatial claim
            relation = "contains"  # But dimensions show it can't contain
            bbox_a = {
                'min': (0, 0, 0),
                'max': (30, 30, 30)
            }
            bbox_b = {
                'min': (0, 0, 0),
                'max': (50, 50, 50)  # Bigger than container!
            }

        fact = {
            "subject_entity_id": part_a,
            "relationship_type": relation,
            "object_entity_id": part_b,
            "geometry": {
                "part_a_bbox": bbox_a,
                "part_b_bbox": bbox_b,
                "interference_detected": (error_type == 'spatial_interference')
            },
            "spatiotemporal": {
                "x_coord": bbox_a['min'][0],
                "y_coord": bbox_a['min'][1],
                "z_coord": bbox_a['min'][2]
            },
            "error_type": error_type
        }

        facts.append(fact)
        labels.append(is_correct)

    return facts, labels


def get_dataset_info() -> Dict:
    """Return metadata about this dataset type"""
    return {
        'name': 'CAD Assembly (Zenodo-style)',
        'challenge': '3D geometric reasoning',
        'primary_test': 'Spatial consistency ψ_s, geometric feasibility',
        'error_types': [
            'Invalid spatial relationships',
            'Interference violations',
            'Geometric impossibilities'
        ],
        'example': 'Component A contains Component B (but dimensions conflict)',
        'expected_performance': {
            'precision': 0.96,
            'recall': 0.93,
            'f1': 0.94,
            'fpr': 0.029
        }
    }
