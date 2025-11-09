# Domain Adaptation Guide for ATLASky-AI

This guide explains how to deploy ATLASky-AI in new domains using the **Domain Adaptation Protocol** from Section 4.4 of the paper.

## Table of Contents

1. [Overview](#overview)
2. [Defense-in-Depth Architecture](#defense-in-depth-architecture)
3. [Five-Component Configuration](#five-component-configuration)
4. [Domain Adaptation Workflow](#domain-adaptation-workflow)
5. [Example Domains](#example-domains)
6. [Creating New Domains](#creating-new-domains)
7. [Best Practices](#best-practices)

---

## Overview

ATLASky-AI's verification system is **domain-agnostic** by design. The same verification algorithms (Equations 1-5, Algorithm 1) work across all domains—only the **configuration parameters** change.

### Key Principle

> "Verification algorithms remain unchanged across domains. Initialize with uniform parameters Φ^(0) = (w^(0), θ^(0), α^(0)) where w_i = 0.2, θ_i = 0.5, α_i = 0.5, then allow AAIC to adapt using domain-specific validation samples."

---

## Defense-in-Depth Architecture

The five-layer verification implements Defense-in-Depth through three principles:

### 1. **Independence**

Modules operate on **distinct information sources**:
- **LOV (M₁)**: Domain ontology O
- **POV (M₂)**: Industry standards and protocols
- **MAV (M₃)**: Physics models Ψ
- **WSV (M₄)**: External web sources
- **ESV (M₅)**: Learned embeddings

**Benefit**: Single-point failures in one module do not compromise others.

### 2. **Complementarity**

Modules target **different error classes**:
- **Semantic Drift**: Detected by LOV and ESV
- **Content Hallucination**: Detected by POV and WSV
- **ST-Inconsistency**: Detected by MAV (unique)

**Benefit**: Errors that evade one module are caught by others.

### 3. **Redundancy**

Multiple modules provide **overlapping coverage**:
- **Hallucinations**: POV (terminology) + WSV (external evidence)
- **Drift**: LOV (ontology) + ESV (embeddings)

**Benefit**: Backup modules maintain detection even if primary modules fail.

### Sequential Ordering

Modules execute in order of **increasing computational cost**:

```
LOV (1.0x) → POV (1.5x) → MAV (3.0x) → WSV (5.0x) → ESV (2.0x)
```

Early termination saves computation when high confidence is achieved.

---

## Five-Component Configuration

Deploying ATLASky-AI in a new domain requires configuring five components:

### Component 1: Domain Ontology (O)

**Purpose**: Define entity classes C, relation types R_o, and attributes A

**Requirements**:
- 50-200 entity classes (typical)
- 20-50 relation types (typical)
- Spatiotemporal coordinates for trackable entities

**Example (Aerospace)**:
```json
{
  "entity_classes": [
    {"name": "Blade", "type": "AerospaceEntity", "spatiotemporal": true},
    {"name": "EngineSet", "type": "AerospaceEntity", "spatiotemporal": true}
  ],
  "relationship_types": [
    {"name": "containsBlade", "domain": "aerospace"},
    {"name": "hasMeasurement", "domain": "aerospace"}
  ]
}
```

### Component 2: Industry Standards (M₂)

**Purpose**: Load domain-specific compliance frameworks

**Examples by Domain**:
- **Aerospace**: STEP AP242 (ISO 10303), AS9100
- **Healthcare**: HL7 FHIR, HIPAA, Joint Commission
- **Aviation**: ICAO Annex 13, FAA Part 121
- **Manufacturing**: ISA-95, IEC 62264

**Configuration**:
```json
{
  "standards": [
    {"name": "STEP_AP242", "version": "ISO_10303", "type": "CAD_data_exchange"},
    {"name": "AS9100", "version": "Rev_D", "type": "quality_management"}
  ],
  "terminology_sources": ["ISO", "ASTM", "AMS"]
}
```

### Component 3: Physical Constraints (M₃)

**Purpose**: Specify velocity limits and facility geometry

**Maximum Velocities** (v_max):
- Manual handling: 2 m/s
- Forklift: 5 m/s
- Automated conveyor: 1 m/s
- Ambulance: 20 m/s
- Drone: 15 m/s
- Aircraft: 250 m/s

**Configuration**:
```json
{
  "max_velocities": {
    "manual_handling": 2.0,
    "forklift": 5.0,
    "ambulance": 20.0
  },
  "temporal_resolution": 1.0,
  "spatial_resolution": 0.1
}
```

### Component 4: Source Credibility (M₄)

**Purpose**: Set credibility weights w_cred,i for authoritative sources

**Hierarchy** (domain-appropriate):
```
Government (1.0) > Regulatory (0.95) > Manufacturer (0.85) > 
Academic (0.75) > News (0.50) > Forums (0.30)
```

**Top 20% Rule**: Top 20% of sources should have w ≥ 0.8

**Configuration**:
```json
{
  "credibility_weights": {
    "FAA": 1.0,
    "NASA": 0.95,
    "manufacturer_spec": 0.90,
    "academic": 0.70
  }
}
```

### Component 5: Domain Embeddings (M₅)

**Purpose**: Train embeddings on historical facts

**Requirements**:
- **Minimum**: 10,000 historical facts
- **Model**: sentence-transformers or domain-adapted BERT
- **Retraining**: Quarterly as domains evolve

**Configuration**:
```json
{
  "model_name": "sentence-transformers/all-MiniLM-L6-v2",
  "embedding_dim": 384,
  "training_corpus_size": 15000,
  "retraining_frequency": "quarterly"
}
```

---

## Domain Adaptation Workflow

### Step 1: Create Configuration

```python
from verification.domain_adapter import DomainAdapter, DomainConfiguration
from verification.domain_adapter import (
    DomainOntologyConfig, IndustryStandardsConfig,
    PhysicalConstraintsConfig, SourceCredibilityConfig,
    DomainEmbeddingsConfig
)

# Create configuration
config = DomainConfiguration(
    domain_name="aerospace",
    domain_description="Aerospace manufacturing and quality inspection",
    ontology=DomainOntologyConfig(...),
    standards=IndustryStandardsConfig(...),
    physics=PhysicalConstraintsConfig(...),
    credibility=SourceCredibilityConfig(...),
    embeddings=DomainEmbeddingsConfig(...)
)

# Validate
valid, errors = config.validate()
if not valid:
    print("Errors:", errors)
```

### Step 2: Save Configuration

```python
adapter = DomainAdapter(config_directory="domains")
adapter.save_domain(config, format="json")
```

### Step 3: Load and Apply

```python
# Load domain
config = adapter.load_domain("aerospace")

# Apply to RMMVe process
from verification.rmmve import RMMVeProcess
rmmve = RMMVeProcess()
adapter.apply_to_rmmve(rmmve, config)

# Apply to knowledge graph
from models.knowledge_graph import SpatiotemporalKnowledgeGraph
kg = SpatiotemporalKnowledgeGraph()
adapter.apply_to_knowledge_graph(kg, config)
```

### Step 4: Initialize AAIC

```python
from aaic.aaic import AAIC

# Initialize with uniform parameters (as per paper)
aaic = AAIC(rmmve)

# AAIC will adapt parameters using validation samples
# No manual tuning required!
```

### Step 5: Verify Facts

```python
# Verify candidate facts
result = rmmve.verify(candidate_fact, kg)

# AAIC monitors and adapts
if result["decision"]:
    aaic.record_true_positive(module_name, performance)
else:
    aaic.record_false_positive(module_name, performance)
```

---

## Example Domains

### Aerospace Manufacturing

**Focus**: High-precision quality inspection, tolerance compliance

**Key Standards**: STEP AP242, AS9100, AMS

**Velocity Constraints**:
- Manual: 2 m/s
- Forklift: 5 m/s
- Crane: 3 m/s

**Configuration**: See `domains/aerospace.json`

### Healthcare Facility

**Focus**: Patient tracking, clinical transfers, protocol compliance

**Key Standards**: HL7 FHIR, HIPAA, Joint Commission

**Velocity Constraints**:
- Wheelchair: 2 m/s
- Stretcher: 3 m/s
- Ambulance: 20 m/s

**Configuration**: See `domains/healthcare.json`

### Aviation Safety

**Focus**: Incident reporting, safety analysis, regulatory compliance

**Key Standards**: ICAO Annex 13, FAA Part 121

**Velocity Constraints**:
- Taxi: 15 m/s
- Takeoff: 100 m/s
- Cruise: 250 m/s

**Configuration**: See `domains/aviation.json`

---

## Creating New Domains

### Quick Start Template

```python
# Create template
adapter = DomainAdapter()
config = adapter.create_template("my_domain")

# Customize components
config.ontology.entity_classes = [
    {"name": "MyEntity", "type": "PhysicalEntity", "spatiotemporal": True}
]

config.standards.standards = [
    {"name": "ISO_XXXX", "version": "2024", "type": "quality"}
]

config.physics.max_velocities = {
    "transport_mode_1": 5.0,
    "transport_mode_2": 10.0
}

config.credibility.credibility_weights = {
    "regulatory_body": 1.0,
    "industry_standard": 0.90
}

# Save
adapter.save_domain(config)
```

### Validation Checklist

✅ **Ontology**:
- [ ] 50-200 entity classes defined
- [ ] 20-50 relationship types defined
- [ ] Spatiotemporal requirements specified

✅ **Standards**:
- [ ] At least one industry standard loaded
- [ ] Terminology sources identified
- [ ] Protocol libraries configured

✅ **Physics**:
- [ ] Velocity limits for all transport modes
- [ ] Temporal resolution set (typically 1.0s)
- [ ] Spatial resolution set (domain-appropriate)

✅ **Credibility**:
- [ ] Weights sum to reasonable distribution
- [ ] Top 20% have w ≥ 0.8
- [ ] Authoritative sources listed

✅ **Embeddings**:
- [ ] Training corpus ≥ 10,000 facts
- [ ] Model architecture selected
- [ ] Retraining schedule defined

✅ **Parameters**:
- [ ] Initial weights sum to 1.0
- [ ] Thresholds in range [0, 1]
- [ ] Alphas in range [0, 1]

---

## Best Practices

### 1. Start with Uniform Parameters

Always initialize with **uniform parameters** as specified in the paper:

```python
initial_weights = {"LOV": 0.2, "POV": 0.2, "MAV": 0.2, "WSV": 0.2, "ESV": 0.2}
initial_thresholds = {"LOV": 0.5, "POV": 0.5, "MAV": 0.5, "WSV": 0.5, "ESV": 0.5}
initial_alphas = {"LOV": 0.5, "POV": 0.5, "MAV": 0.5, "WSV": 0.5, "ESV": 0.5}
```

Let **AAIC adapt** based on domain-specific validation samples.

### 2. Use Domain-Adapted Embeddings

For specialized domains, use domain-adapted BERT variants:
- **Healthcare**: `emilyalsentzer/Bio_ClinicalBERT`
- **Scientific**: `allenai/scibert_scivocab_uncased`
- **Legal**: `nlpaueb/legal-bert-base-uncased`
- **General**: `sentence-transformers/all-MiniLM-L6-v2`

### 3. Set Appropriate Resolutions

**Temporal Resolution**:
- Manufacturing: 1 second
- Healthcare: 60 seconds (1 minute)
- Aviation: 1 second

**Spatial Resolution**:
- Aerospace (precision): 0.001 m (1 mm)
- Healthcare: 1.0 m
- Aviation: 10.0 m

### 4. Maintain Credibility Hierarchy

Ensure credibility weights follow domain-appropriate hierarchy:

```
Regulatory > Manufacturer > Academic > Industry > News > Social
```

### 5. Retrain Embeddings Quarterly

Domains evolve—retrain embeddings every 3 months to capture new patterns.

### 6. Monitor AAIC Adaptations

Track how AAIC adjusts parameters over time:

```python
aaic.get_parameter_history()
aaic.get_detected_shifts()
```

### 7. Analyze Defense-in-Depth

Periodically analyze architecture robustness:

```python
from verification.defense_in_depth import DefenseInDepthAnalyzer

analyzer = DefenseInDepthAnalyzer()
report = analyzer.generate_full_report()
```

---

## Performance Expectations

### Typical Results (Section 5, Table 3)

| Domain | Precision | Recall | F1 | FPR | Processing Time |
|--------|-----------|--------|----|----|-----------------|
| Aerospace | 0.92 | 0.89 | 0.90 | 0.08 | 1.8s |
| Healthcare | 0.89 | 0.91 | 0.90 | 0.11 | 2.1s |
| Aviation | 0.94 | 0.88 | 0.91 | 0.06 | 1.6s |

### Early Termination Rates

Expect **60-75%** of facts to terminate before final module, saving computation.

---

## Troubleshooting

### Configuration Errors

**Problem**: "Training corpus size below minimum (10,000)"
**Solution**: Ensure at least 10K historical facts for embedding training

**Problem**: "Initial weights sum to X, should sum to 1.0"
**Solution**: Normalize weights to sum to 1.0

### Low Performance

**Problem**: Low precision/recall
**Solution**: 
1. Verify ontology completeness
2. Check standard terminology sources
3. Ensure credibility weights are accurate
4. Retrain embeddings with more data

### High False Positive Rate

**Problem**: Too many false positives
**Solution**:
1. Increase module thresholds (θ_i)
2. Increase global threshold (Θ)
3. Let AAIC adapt over time

---

## API Reference

### DomainAdapter

```python
adapter = DomainAdapter(config_directory="domains")

# Load domain
config = adapter.load_domain("aerospace")

# Save domain
adapter.save_domain(config, format="json")

# List available domains
domains = adapter.list_domains()

# Apply to RMMVe
adapter.apply_to_rmmve(rmmve_process, config)

# Apply to knowledge graph
adapter.apply_to_knowledge_graph(kg, config)
```

### DefenseInDepthAnalyzer

```python
analyzer = DefenseInDepthAnalyzer()

# Analyze principles
independence = analyzer.analyze_independence()
complementarity = analyzer.analyze_complementarity()
redundancy = analyzer.analyze_redundancy()
efficiency = analyzer.analyze_computational_efficiency()

# Full report
report = analyzer.generate_full_report()
```

---

## References

- **Paper Section 4.3**: Defense-in-Depth Architecture Analysis
- **Paper Section 4.4**: Domain Adaptation Protocol
- **Paper Section 5**: Experimental Validation (Three Domains)
- **Paper Table 3**: Performance Metrics Across Domains

---

## Support

For questions or issues:
1. Check existing domain configurations in `domains/`
2. Review example usage in `test_domain_adaptation.py`
3. Analyze defense-in-depth properties with `DefenseInDepthAnalyzer`
4. Monitor AAIC adaptations over time

**Remember**: The verification algorithms are domain-agnostic. Only configuration changes across domains!

