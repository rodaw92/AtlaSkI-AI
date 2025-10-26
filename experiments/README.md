# ATLASky-AI Experimental Evaluation Framework

This directory contains scripts and datasets for evaluating ATLASky-AI's verification performance across different domain types.

## Dataset Types

Our evaluation uses 4 distinct dataset types that test different aspects of spatiotemporal knowledge verification:

### 1. **Manufacturing/Aerospace** (Micro-tolerance precision)
- **Challenge**: Tight dimensional tolerances (±0.1mm precision)
- **Tests**: Spatial consistency ψ_s, measurement validation
- **Error Types**: Hallucinated tolerance values, impossible measurements
- **Example**: "Turbine blade deviation: 0.023mm at coordinates (10.5, 20.3, 150.2)"

### 2. **Aviation Safety** (Unstructured narratives)
- **Challenge**: Extract structured facts from incident reports
- **Tests**: Temporal consistency ψ_t, causal relationships
- **Error Types**: Fabricated event sequences, temporal impossibilities
- **Example**: "Aircraft descended from FL350 to FL280 in 45 seconds" (too fast)

### 3. **CAD Assembly** (3D geometric reasoning)
- **Challenge**: Validate spatial relationships in 3D models
- **Tests**: Spatial consistency ψ_s, geometric feasibility
- **Error Types**: Invalid spatial relationships, interference violations
- **Example**: "Component A inside Component B" when geometries conflict

### 4. **Clinical/Healthcare** (Workflow compliance)
- **Challenge**: Patient transfers with protocol constraints
- **Tests**: Temporal consistency ψ_t, workflow rules
- **Error Types**: Protocol violations, impossible transfer times
- **Example**: "Patient moved ICU→OR in 5 minutes" (violates 20min protocol)

## Running Experiments

### Quick Start
```bash
# Run all dataset experiments
python3 run_experiments.py --all

# Run specific dataset type
python3 run_experiments.py --dataset manufacturing
python3 run_experiments.py --dataset aviation
python3 run_experiments.py --dataset cad
python3 run_experiments.py --dataset healthcare

# Run with detailed analysis
python3 run_experiments.py --all --detailed --output results/
```

### Metrics Reported

For each dataset, we measure:
- **Precision**: TP/(TP+FP) - Fraction of rejected facts that were truly incorrect
- **Recall**: TP/(TP+FN) - Fraction of incorrect facts successfully caught
- **F1-Score**: Harmonic mean of Precision and Recall
- **FPR (False Positive Rate)**: FP/(FP+TN) - Critical for production deployment

Where:
- TP: Incorrect facts correctly rejected ✓
- TN: Correct facts correctly accepted ✓
- FP: Correct facts incorrectly rejected ✗ (false alarms)
- FN: Incorrect facts incorrectly accepted ✗ (dangerous!)

### Expected Performance

Based on paper Section 6.3:

| Dataset Type    | Precision | Recall | F1   | FPR  | Facts |
|----------------|-----------|--------|------|------|-------|
| Manufacturing  | 0.94      | 0.91   | 0.92 | 2.6% | 600   |
| Aviation       | 0.93      | 0.94   | 0.93 | 3.2% | 10,500|
| CAD Assembly   | 0.96      | 0.93   | 0.94 | 2.9% | 520   |
| Healthcare     | 0.95      | 0.95   | 0.95 | 4.1% | 1,000 |

## Directory Structure

```
experiments/
├── README.md                    # This file
├── run_experiments.py          # Main experiment runner
├── datasets/                    # Dataset files
│   ├── manufacturing_data.py   # AddQual-style manufacturing facts
│   ├── aviation_data.py        # NASA ASRS-style incident reports
│   ├── cad_data.py             # Zenodo CAD-style assemblies
│   └── healthcare_data.py      # MIMIC-IV-style patient transfers
├── metrics/
│   └── evaluation.py           # Precision, Recall, F1, FPR computation
└── results/                     # Output directory for results
    ├── manufacturing_results.json
    ├── aviation_results.json
    ├── cad_results.json
    └── healthcare_results.json
```

## Interpreting Results

### Good Performance Indicators
- **High Precision (>0.90)**: Few false alarms, system trustworthy
- **High Recall (>0.90)**: Catches most errors, safe for production
- **Low FPR (<5%)**: Minimal manual review burden
- **Balanced F1 (>0.90)**: No precision-recall trade-off

### Module-Specific Insights
The experiments also show which modules activate most per dataset:
- **Manufacturing**: MAV dominates (tolerance checking via ψ_s)
- **Aviation**: MAV + WSV (temporal validation + external corroboration)
- **CAD**: MAV heavily used (3D spatial reasoning via ψ_s)
- **Healthcare**: POV + MAV (protocol compliance + temporal ψ_t)

## Extending to Your Domain

To test on your own data:
1. Create a new dataset file in `datasets/your_domain_data.py`
2. Follow the structure of existing datasets
3. Define your domain-specific:
   - Entity classes (ontology C)
   - Relation types (ontology R_o)
   - Physical constraints (v_max, spatial resolution)
   - Industry standards for POV module
4. Run: `python3 run_experiments.py --dataset your_domain`
