"""
Quality-based data generation for honest testing.

Generates raw text data with actual quality characteristics that match
the selected quality level. The verification system then honestly evaluates
these facts without artificial boosting.
"""

import random
from datetime import datetime, timedelta
from typing import Dict, Tuple


def generate_raw_text_with_quality(quality_level: str, domain: str = "aerospace") -> Tuple[str, Dict]:
    """
    Generate raw text input with quality characteristics matching the selected level.
    
    Returns: (raw_text, expected_attributes)
    """
    
    if domain == "aerospace":
        return _generate_aerospace_text(quality_level)
    elif domain == "healthcare":
        return _generate_healthcare_text(quality_level)
    elif domain == "aviation":
        return _generate_aviation_text(quality_level)
    else:
        return _generate_cad_text(quality_level)


def _generate_aerospace_text(quality_level: str) -> Tuple[str, Dict]:
    """Generate aerospace inspection text with quality-appropriate errors."""
    
    blade_ids = ["Blade Alpha", "Blade Beta", "Blade Gamma", "Blade Delta"]
    blade_id = random.choice(blade_ids)
    nominal = round(random.uniform(2.0, 5.0), 2)
    
    # Generate actual quality-based data
    if quality_level == "high_quality":
        # Perfect measurement, no errors
        actual = round(nominal + random.uniform(-0.01, 0.01), 3)
        deviation = abs(actual - nominal)
        location = "Bay 7"
        time_str = "2024-10-29T10:30:00Z"
        
        text = f"Installation completed in {location}. {blade_id} measurement: {actual} mm on leading edge. Nominal: {nominal} mm. Deviation: {deviation} mm. Tolerance: ±0.1 mm. Status: PASS."
        
    elif quality_level == "medium_quality":
        # Good measurement with minor spelling error
        actual = round(nominal + random.uniform(-0.03, 0.03), 3)
        deviation = abs(actual - nominal)
        location = "bay 7"  # Lowercase (will be corrected)
        time_str = "2024-10-29T10:30:00Z"
        
        text = f"Instalation completed in {location}. {blade_id} measurment: {actual} mm. Deviation: {deviation} mm. Tolerance check passed."
        
    elif quality_level == "spatial_issue":
        # Valid measurement but unclear location
        actual = round(nominal + random.uniform(-0.02, 0.02), 3)
        deviation = abs(actual - nominal)
        location = "somewhere in assembly area"  # Vague location
        time_str = "2024-10-29T10:30:00Z"
        
        text = f"{blade_id} inspection at {location}. Measured: {actual} mm. Nominal: {nominal} mm."
        
    elif quality_level == "semantic_issue":
        # Incorrect terminology
        actual = round(nominal + random.uniform(-0.04, 0.04), 3)
        location = "bay 7"
        time_str = "2024-10-29T10:30:00Z"
        
        text = f"{blade_id} checkup in {location}. Size measurement was {actual} millimeters. Expected around {nominal} mm."
        
    else:  # low_quality
        # Multiple issues: vague, imprecise, missing data
        actual = round(nominal + random.uniform(-0.08, 0.08), 3)
        location = "facility"
        time_str = "sometime today"
        
        text = f"Blade part inspected. Measured approximately {actual}. Seems okay."
    
    # Expected attributes after extraction
    expected = {
        "blade_id": blade_id,
        "nominal_mm": nominal,
        "actual_mm": actual,
        "deviation_mm": abs(actual - nominal),
        "tolerance_mm": 0.1,
        "location": location,
        "timestamp": time_str,
        "quality_level": quality_level
    }
    
    return text, expected


def _generate_healthcare_text(quality_level: str) -> Tuple[str, Dict]:
    """Generate healthcare transfer text with quality-appropriate errors."""
    
    patient_id = f"P{random.randint(100000, 999999)}"
    
    if quality_level == "high_quality":
        text = f"Patient {patient_id} transferred from MICU to Operating Room at 14:35 UTC. Transfer duration: 22 minutes. Accompanied by RN and MD. Vital signs stable."
        expected = {
            "patient_id": patient_id,
            "from_unit": "MICU",
            "to_unit": "Operating Room",
            "timestamp": "2024-10-29T14:35:00Z",
            "duration_min": 22
        }
        
    elif quality_level == "medium_quality":
        text = f"Patient {patient_id} tranfser from micu to OR at 14:35. Duration: 22 minutes."
        expected = {
            "patient_id": patient_id,
            "from_unit": "micu",
            "to_unit": "OR",
            "timestamp": "2024-10-29T14:35:00Z",
            "duration_min": 22
        }
        
    elif quality_level == "spatial_issue":
        text = f"Patient {patient_id} moved to surgery area. Transfer took about 20 minutes."
        expected = {
            "patient_id": patient_id,
            "from_unit": "unknown",
            "to_unit": "surgery area",
            "duration_min": 20
        }
        
    else:  # low_quality
        text = f"Patient transferred. Went to another unit."
        expected = {
            "patient_id": "unknown",
            "from_unit": "unknown",
            "to_unit": "unknown"
        }
    
    expected["quality_level"] = quality_level
    return text, expected


def _generate_aviation_text(quality_level: str) -> Tuple[str, Dict]:
    """Generate aviation incident text."""
    
    aircraft = f"N{random.randint(10000, 99999)}"
    
    if quality_level == "high_quality":
        text = f"Aircraft {aircraft} experienced moderate turbulence at FL350 during climb phase at 10:45 UTC. Altitude maintained. No injuries reported."
        expected = {"aircraft": aircraft, "event": "turbulence", "altitude": "FL350", "time": "10:45 UTC"}
    else:
        text = f"Aircraft had some issues during flight."
        expected = {"aircraft": "unknown", "event": "issues"}
    
    expected["quality_level"] = quality_level
    return text, expected


def _generate_cad_text(quality_level: str) -> Tuple[str, Dict]:
    """Generate CAD assembly text."""
    
    if quality_level == "high_quality":
        text = "Assembly clearance check: Part A-023 and Housing B-104 have 0.15mm gap. Interference check passed. Geometric constraints satisfied."
        expected = {"part1": "A-023", "part2": "B-104", "clearance": 0.15, "status": "pass"}
    else:
        text = "Assembly has some clearance issues."
        expected = {"status": "unclear"}
    
    expected["quality_level"] = quality_level
    return text, expected

