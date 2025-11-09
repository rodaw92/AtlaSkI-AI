# ATLASky-AI Experimental Evaluation Guide

## Overview

This directory contains the experimental framework for evaluating ATLASky-AI's verification performance across different dataset types, as described in Section 6 (Experimental Evaluation) of the paper.

## What's Implemented

### ✅ Core Framework Components

1. **Metrics Module** (`metrics/evaluation.py`)
   - Implements standard classification metrics:
     - **Precision**: TP/(TP+FP) - Fraction of rejected facts that were truly incorrect
     - **Recall**: TP/(TP+FN) - Fraction of incorrect facts successfully caught
     - **F1-Score**: Harmonic mean balancing both metrics
     - **FPR (False Positive Rate)**: FP/(FP+TN) - Critical for production deployment
   - Confusion matrix visualization
   - Performance comparison across configurations

2. **Dataset Templates** (`datasets/`)
   - Manufacturing/Aerospace (AddQual-style): Micro-tolerance precision testing
   - Aviation Safety (NASA ASRS-style): Temporal consistency and narrative extraction
   - CAD Assembly (Zenodo-style): 3D geometric reasoning
   - Clinical/Healthcare (MIMIC-IV-style): Workflow compliance validation

3. **Experiment Runner** (`run_experiments.py`)
   - Automated test execution across dataset types
   - Performance metric calculation
   - Module activation tracking
   - Results export to JSON

4. **Quick Demo** (`quick_demo.py`)
   - **Works with current system** - no external data needed
   - Demonstrates experimental methodology
   - Shows how to calculate Precision, Recall, F1, FPR
   - Simulates paper Table 2 (Main Results)

## Running Experiments

### Quick Start (No External Data Required)

```bash
# Run demo using built-in data generator
python3 experiments/quick_demo.py
```

This demonstrates:
- How to evaluate different fact quality types
- How metrics are calculated
- What the output format looks like
- Performance patterns across quality levels

### Full Experiments (Requires Real Datasets)

```bash
# Run on all dataset types
python3 experiments/run_experiments.py --all --num-facts 100

# Run on specific dataset
python3 experiments/run_experiments.py --dataset manufacturing --num-facts 50

# Run with detailed output
python3 experiments/run_experiments.py --dataset aviation --detailed
```

## Dataset Integration Requirements

To run full experiments as in the paper, you need:

### 1. Manufacturing/Aerospace Dataset (AddQual-style)

**Data Required:**
- 600 facts from 18-month inspection logs
- Dimensional measurements with ±0.1mm tolerance
- Spatial coordinates for each measurement
- Timestamps

**Ground Truth:**
- Human-annotated labels: correct/incorrect
- Validation by domain experts
- Focus on tolerance violations and spatial inconsistencies

**Expected Performance:**
- Precision: 0.94
- Recall: 0.91
- F1: 0.92
- FPR: 2.6%

**Primary Tests:**
- Spatial consistency ψ_s (Definition 2)
- Measurement validation
- Tolerance checking

### 2. Aviation Safety Dataset (NASA ASRS-style)

**Data Required:**
- 10,500 facts from 1,500 incident reports
- Flight events with altitudes, times, locations
- Event sequences and causal relationships

**Ground Truth:**
- Expert aviation safety analysts
- Validation against flight logs
- Physics-based feasibility checks

**Expected Performance:**
- Precision: 0.93
- Recall: 0.94
- F1: 0.93
- FPR: 3.2%

**Primary Tests:**
- Temporal consistency ψ_t (Definition 3)
- Causal relationships
- Velocity constraints

### 3. CAD Assembly Dataset (Zenodo-style)

**Data Required:**
- 520 facts from 26 STEP AP242 assembly models
- Part-to-part spatial relationships
- 3D bounding boxes and geometric constraints

**Ground Truth:**
- CAD engineers validate geometric feasibility
- Interference detection results
- Assembly sequence validation

**Expected Performance:**
- Precision: 0.96
- Recall: 0.93
- F1: 0.94
- FPR: 2.9%

**Primary Tests:**
- 3D spatial reasoning
- Geometric feasibility
- Interference detection

### 4. Clinical/Healthcare Dataset (MIMIC-IV-style)

**Data Required:**
- 1,000 facts from ICU patient transfers
- Transfer events with timestamps and locations
- Care unit coordinates and protocols

**Ground Truth:**
- Clinical workflow experts
- Protocol compliance validation
- Transfer time feasibility

**Expected Performance:**
- Precision: 0.95
- Recall: 0.95
- F1: 0.95
- FPR: 4.1%

**Primary Tests:**
- Temporal consistency ψ_t
- Protocol compliance (minimum transfer times)
- Clinical workflow validation

## Understanding the Metrics

### Why These Metrics Matter

**Precision** (TP / (TP+FP)):
- Measures trustworthiness: Of all rejected facts, how many were truly incorrect?
- High precision = Few false alarms
- Target: >0.90 for safety-critical systems
- Low precision causes review fatigue

**Recall** (TP / (TP+FN)):
- Measures safety: Of all incorrect facts, how many did we catch?
- High recall = Few dangerous errors slip through
- Target: >0.90 for production deployment
- Low recall is DANGEROUS in safety-critical systems

**F1-Score** (2 × Precision × Recall / (Precision + Recall)):
- Balances precision and recall
- Single number for overall performance
- Target: >0.90 for deployment readiness
- Prevents gaming one metric at expense of other

**FPR** (FP / (FP+TN)):
- Critical for user acceptance
- Measures: Of all correct facts, how many were falsely rejected?
- Low FPR (<5%) = Minimal manual review burden
- Paper shows 39-57% FPR reduction vs baselines
- At NASA ASRS scale: 3.2% FPR = 336 false alarms vs 788 for best baseline

### Interpreting Results

**Good Performance Indicators:**
- Precision >0.90: System is trustworthy
- Recall >0.90: System catches errors
- F1 >0.90: Balanced performance
- FPR <5%: Low review burden

**Warning Signs:**
- Precision <0.85: Too many false alarms
- Recall <0.85: Missing errors (DANGEROUS!)
- F1 <0.85: Unbalanced or poor overall performance
- FPR >10%: Excessive manual review required

## Module-Specific Analysis

The experiments also reveal which modules contribute most to each dataset type:

| Dataset Type    | Primary Modules | Why                                    |
|----------------|-----------------|----------------------------------------|
| Manufacturing  | MAV (M3)        | Tolerance checking via ψ_s             |
| Aviation       | MAV + WSV       | Temporal ψ_t + external corroboration  |
| CAD Assembly   | MAV             | 3D spatial reasoning via ψ_s           |
| Healthcare     | POV + MAV       | Protocol compliance + temporal ψ_t     |

**Key Finding from Ablation Studies (Table 3):**
- Removing MAV drops F1 from 0.92 to 0.81 (-11.9%)
- MAV catches 35% of all errors (spatiotemporal inconsistencies)
- These errors are INVISIBLE to semantic validation
- Validates core architectural innovation

## Extending to Your Domain

To run experiments on your own data:

### Step 1: Prepare Your Dataset

```python
# Create: experiments/datasets/your_domain_data.py

def generate_your_domain_facts(num_facts: int):
    """
    Generate or load facts from your domain

    Returns:
        facts: List of fact dictionaries with:
            - subject_entity_id
            - relationship_type
            - object_entity_id
            - spatiotemporal coordinates
            - domain-specific attributes
        labels: List of booleans (True = correct, False = incorrect)
    """
    facts = []
    labels = []

    # Load your data here
    # Add ground truth labels

    return facts, labels

def get_dataset_info():
    """Return metadata about your dataset"""
    return {
        'name': 'Your Domain Name',
        'challenge': 'What makes this domain difficult',
        'primary_test': 'Which physics predicates (ψ_s, ψ_t, Ψ)',
        'error_types': ['List', 'of', 'error', 'types'],
        'example': 'Example fact from your domain'
    }
```

### Step 2: Configure Domain-Specific Rules

Update `models/constants.py`:

```python
# Physical constraints for your domain
V_MAX['your_transport_mode'] = 10.0  # meters/second
TAU_RES = 2.0  # temporal resolution (seconds)
SIGMA_RES = 0.2  # spatial resolution (meters)

# Add your entity classes
ENTITY_CLASSES.extend(['YourEntity1', 'YourEntity2'])

# Add your relationship types
RELATIONSHIP_TYPES.extend(['your_relation_1', 'your_relation_2'])
```

### Step 3: Run Experiments

```bash
python3 experiments/run_experiments.py --dataset your_domain --num-facts 1000
```

### Step 4: Analyze Results

Results are saved to `experiments/results/your_domain_results.json`:

```json
{
  "dataset": "your_domain",
  "num_facts": 1000,
  "metrics": {
    "precision": 0.92,
    "recall": 0.90,
    "f1": 0.91,
    "fpr": 0.038,
    "tp": 450,
    "tn": 515,
    "fp": 20,
    "fn": 15
  },
  "module_activations": {
    "LOV": 850,
    "POV": 680,
    "MAV": 520,
    "WSV": 380,
    "ESV": 250
  }
}
```

## Troubleshooting

### Issue: All facts are accepted/rejected

**Cause**: Threshold configuration may not match your data distribution

**Solution**: Adjust global threshold in `run_experiments.py`:
```python
rmmve = RMMVeProcess(global_threshold=0.70)  # Increase for stricter
```

### Issue: Module activations seem wrong

**Cause**: Module thresholds may need tuning for your domain

**Solution**: See AAIC adaptation in `aaic/aaic.py` or adjust manually:
```python
rmmve.modules[0].threshold = 0.75  # LOV
rmmve.modules[1].threshold = 0.80  # POV
# etc.
```

### Issue: Metrics don't match paper

**Cause**: Dataset characteristics differ from paper's datasets

**Expected**: Each domain has unique error distributions
- Manufacturing: More spatial violations (ψ_s)
- Aviation: More temporal violations (ψ_t)
- Your domain may have different patterns

## Files Reference

```
experiments/
├── README.md                    # High-level overview
├── EXPERIMENTS_GUIDE.md        # This file (detailed guide)
├── quick_demo.py               # ✅ Working demo (no external data needed)
├── run_experiments.py          # Full experiment runner
├── datasets/
│   ├── manufacturing_data.py   # Manufacturing dataset template
│   ├── aviation_data.py        # Aviation dataset template
│   ├── cad_data.py            # CAD assembly dataset template
│   └── healthcare_data.py     # Healthcare dataset template
├── metrics/
│   └── evaluation.py          # Metrics calculation (P, R, F1, FPR)
└── results/                    # Output directory
    └── *.json                  # Experiment results
```

## Citation

If you use this experimental framework, please cite the ATLASky-AI paper:

```bibtex
@article{atlasky2025,
  title={ATLASky-AI: Defense-in-Depth Verification for LLM-Generated Spatiotemporal Knowledge Graphs},
  author={...},
  journal={...},
  year={2025}
}
```

## Support

For questions about experiments:
1. Check this guide and README.md
2. Run `python3 experiments/quick_demo.py` to see working example
3. Review paper Section 6 for methodology details
4. See `metrics/evaluation.py` for metric definitions

---

**Remember**: The experimental framework measures THREE error classes:
1. **Content Hallucination** (Definition 4) - Detected by POV (M2) + WSV (M4)
2. **Spatiotemporal Inconsistency** (Definition 5) - Detected by MAV (M3)
3. **Semantic Drift** (Definition 6) - Detected by LOV (M1) + ESV (M5)

Your experiments should verify the system catches all three types across your domain!
