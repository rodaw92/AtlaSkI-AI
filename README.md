# ATLASky-AI

## Multi-Domain 4D Spatiotemporal Knowledge Graph Verification System

ATLASky-AI is a domain-adaptable verification system for 4D Spatiotemporal Knowledge Graphs (STKGs) that combines physics-based constraints with multi-modal verification to detect and prevent:

- **Content Hallucination**: Fabricated facts not grounded in reality
- **ST-Inconsistency**: Violations of physical laws (spatial/temporal)
- **Semantic Drift**: Facts that deviate from domain ontology

---

## Demo — See It In Action

### Video Walkthrough

https://github.com/rodaw92/AtlaSkI-AI/raw/main/demo/atlaskyai_demo.mp4

> **[▶️ Download/watch the full demo video](https://github.com/rodaw92/AtlaSkI-AI/raw/main/demo/atlaskyai_demo.mp4)** — Methodology overview → Fact verification → AAIC monitoring

---

### Dashboard Overview

<p align="center">
  <img src="demo/dashboard_overview.png" alt="ATLASky-AI Dashboard" width="100%">
</p>

The interactive dashboard shows the STKG formalization **G = (V, E, O, T, Ψ)** with live knowledge graph metrics and physics-based consistency predicates:

<p align="center">
  <img src="demo/physics_predicates.png" alt="Physics Predicates ψ_s, ψ_t, Ψ" width="100%">
</p>

### Five-Module Verification Pipeline

Each candidate fact passes through 5 modules sequentially (M₁→M₅), each computing dual metrics. Early termination occurs when cumulative confidence exceeds the global threshold.

<p align="center">
  <img src="demo/methodology_modules.png" alt="Five Verification Modules" width="100%">
</p>

### How Verification Works — Step by Step

The system processes each fact through a **three-stage pipeline** (Preprocessing → LLM Extraction → TruthFlow Verification), then classifies it as **ST** (spatiotemporal — has valid coordinates and timestamp) or **SEM** (semantic-only — missing or invalid coordinates). The final decision is **Accept**, **Review**, or **Reject** based on cumulative module confidence vs. the global threshold Θ.

Below are two example cases showing exactly what the system checks and why a fact passes or fails.

---

#### ✅ Case 1: High-Quality Fact → ACCEPT

**Input text:**
> *"Installation completed in Bay 7. Blade Gamma measurement: 3.02 mm on leading edge. Tolerance check passed."*

**Why this is high quality — what the system sees:**

| Property | Value | Why it matters |
|---|---|---|
| Entity | `TurbineBlade_Gamma` | Valid entity — exists in ontology class `Blade` ✓ |
| Relationship | `hasMeasurement` | Valid relationship — defined in ontology R_o ✓ |
| Location | Bay 7 → (40.0, 20.0, 0.0) | Known facility location, mapped to real coordinates ✓ |
| Timestamp | `2026-02-25T12:30:33Z` | Valid ISO 8601 UTC timestamp ✓ |
| LLM Confidence | **High (1.0)** | Complete info: numbers + location + timestamp + detailed text |

**How each module scores it:**

| Module | Score | Threshold | Activated? | What it checked |
|---|---|---|---|---|
| M₁ (LOV) | 0.700 | 0.70 | ✅ Yes | Subject `Blade` ∈ ontology, relation `hasMeasurement` ∈ ontology, object `InspectionMeasurement` ∈ ontology → Metric₁ = 1.0 |
| M₂ (POV) | 0.610 | 0.70 | ❌ No | Some terms match standards but not enough to exceed threshold |
| M₃ (MAV) | 1.000 | 0.65 | ✅ Yes | ψ_s = 1 (no bilocation), ψ_t = 1 (travel time feasible), Kinematic OK, Process OK |

**Result:** C = (0.700 × 0.25 + 1.000 × 0.20) / (0.25 + 0.20) = **0.800** ≥ Θ = 0.650 → **ACCEPT** ✅ (early termination at M₃, M₄ and M₅ skipped)

<p align="center">
  <img src="demo/result_high_quality.png" alt="High Quality Accepted" width="90%">
</p>

---

#### ❌ Case 2: Low-Quality Fact → REJECT

**Input text:**
> *"Blade part inspected. Measured approximately 3.5. Seems okay."*

**Why this is low quality — what the system sees:**

| Property | Value | Why it fails |
|---|---|---|
| Entity | `Unknown_6959` | Not in ontology — unknown entity class ✗ |
| Relationship | `linkedTo` / `contains` | Invalid — not defined in ontology R_o ✗ |
| Inspection tool | `UnknownTool_123` | Fabricated — not in standard tool list ✗ |
| Timestamp | `202X-12-5` | Unparseable — not valid ISO 8601 ✗ |
| Location | Missing coordinate axis | Incomplete spatial data ✗ |
| Fact Type | **SEM** (semantic-only) | Can't verify physics because coordinates are invalid |
| LLM Confidence | **Low (0.6)** | Vague text, no precise numbers, no clear location |

**How each module scores it:**

| Module | Score | Threshold | Activated? | What it detected |
|---|---|---|---|---|
| M₁ (LOV) | 0.533 | 0.70 | ❌ No | `Unknown_6959` not in entity classes → Metric₁ = 0.33 (only 1 of 3 structural checks pass) |
| M₂ (POV) | 0.130 | 0.70 | ❌ No | `linkedTo` not in standard terminology, `UnknownTool_123` not a recognized tool → Metric₁ = 0.20 |
| M₃ (MAV) | 1.000 | 0.65 | ⬜ Neutral | SEM fact → physics N/A, neutral score does not count toward confidence |
| M₄ (WSV) | 0.350 | 0.60 | ❌ No | No corroborating evidence found in knowledge graph → Metric₁ = 0.00 |
| M₅ (ESV) | 0.518 | 0.65 | ❌ No | Low similarity to known facts → statistical outlier detected |

**Result:** No modules activated → C = **0.000** < Θ − ε = 0.550 → **REJECT** ❌ (all 5 modules executed, none reached activation threshold)

<p align="center">
  <img src="demo/result_low_quality.png" alt="Low Quality Rejected" width="90%">
</p>

---

#### Summary: What Makes a Fact Pass or Fail?

| Check | High Quality (Accept) | Low Quality (Reject) |
|---|---|---|
| **Entity class** | Known (`Blade`, `EngineSet`) | Unknown (`Unknown_XXXX`) |
| **Relationship type** | Valid (`hasMeasurement`, `containsBlade`) | Invalid (`linkedTo`, `contains`) |
| **Timestamp** | Valid ISO 8601 (`2026-02-25T12:30:33Z`) | Unparseable (`202X-12-5`) |
| **Coordinates** | Complete (x, y, z from facility map) | Missing or incomplete |
| **Inspection tool** | Standard (`3D_Scanner_Unit`) | Fabricated (`UnknownTool_123`) |
| **LLM confidence** | High (1.0) — precise text with all details | Low (0.6) — vague, missing info |
| **Fact type** | ST (spatiotemporal) | SEM (semantic-only) |
| **Physics check** | ψ_s = 1, ψ_t = 1 (consistent) | N/A (can't check without valid coordinates) |
| **Decision** | **C = 0.80 ≥ 0.65 → Accept** | **C = 0.00 < 0.55 → Reject** |

---

### AAIC Adaptive Monitoring

The Autonomous Adaptive Intelligence Cycle (AAIC) monitors module performance via CGR-CUSUM and adapts weights, thresholds, and alpha parameters when distribution shifts are detected.

<p align="center">
  <img src="demo/aaic_monitoring.png" alt="AAIC CGR-CUSUM Monitoring" width="100%">
</p>

### CLI Demo

Run `python3 test_verification_demo.py` to see all quality cases (high, medium, spatial, low) processed through the verification pipeline from the command line.

---

## System Architecture

### 4D STKG Formalization

ATLASky-AI operates on a formal 4D STKG defined as **G = (V, E, O, T, Ψ)** where:

- **V**: Versioned entities with immutable attributes and mutable state
- **E**: Directed edges representing relationships
- **O = (C, R_o, A)**: Domain ontology with entity classes, relation types, and attributes
- **T: (V ∪ E) → ℝ³ × ℝ**: Maps entities/relations to spatiotemporal coordinates (x,y,z,t)
- **Ψ: (V ∪ E) → {0,1}**: Physical consistency predicate combining spatial (ψ_s) and temporal (ψ_t) consistency

### Physics-Based Predicates

- **ψ_s (Spatial Consistency)**: Prevents co-location violations — same entity cannot exist at two separated locations within the same time window
- **ψ_t (Temporal Consistency)**: Enforces velocity and travel-time constraints — travel time must be physically feasible given distance and maximum velocity
- **Ψ = ψ_s ∧ ψ_t**: Combined predicate ensuring full physical consistency

### Three-Stage Pipeline

1. **Stage 1 — Data Preprocessing**: Normalizes heterogeneous raw data RD into structured format RD' (OCR, spell correction, terminology standardization via ontology O, temporal alignment to UTC, spatial validation via facility maps)
2. **Stage 2 — LLM-Based Extraction**: Generates candidate facts D = L(RD'; P) using domain-specialized prompts with confidence-weighted output d_k = ⟨s, r, o, T(d_k), conf_k⟩ where conf_k ∈ {high=1.0, medium=0.8, low=0.6}
3. **Stage 3 — TruthFlow Verification**: Validates candidates through Ranked Multi-Modal Verification (RMMVe) with Autonomous Adaptive Intelligence Cycle (AAIC)

### Five-Module Verification Pipeline (RMMVe)

Each module M_i computes confidence through two complementary metrics:

**S_i(d_k) = conf_k × [α_i · Metric₁ + (1−α_i) · Metric₂]**

| Module | Full Name | Primary Target | Dual Metrics | Cost |
|--------|-----------|----------------|--------------|------|
| M₁ (LOV) | Lexical-Ontological Verification | Semantic Drift | Structural Compliance (Eq. 8) + Attribute Compliance (Eq. 9) | 5 ms |
| M₂ (POV) | Protocol-Ontology Verification | Content Hallucination | Standard Terminology Match (Eq. 10) + Cross-Standard Consistency (Eq. 11) | 15 ms |
| M₃ (MAV) | Motion-Aware Verification | ST-Inconsistency | Temporal-Spatial Validity ψ_s, ψ_t (Eq. 12) + Physical Feasibility min(Kinematic, Process) (Eq. 13-16) | 50 ms |
| M₄ (WSV) | Web-Source Verification | Content Hallucination | Source Credibility (Eq. 17) + Cross-Source Agreement (Eq. 18) | 120 ms |
| M₅ (ESV) | Embedding Similarity Verification | Semantic Drift + Hallucination | K-NN Cosine Similarity (Eq. 19) + Cluster Membership / GMM (Eq. 20) | 800 ms |

**Key mechanisms:**

- **Fact Type Classification**: Facts are classified as **ST** (spatiotemporal — has valid coordinates) or **SEM** (semantic-only). For SEM facts, MAV assigns neutral score S₃=1.0 (physics not applicable).
- **Critical Module Veto**: For ST facts, if MAV score < τ_veto (healthcare: 0.5, aerospace: 0.30) → immediate Reject regardless of other modules.
- **Early Termination**: When cumulative confidence C ≥ Θ (global threshold), remaining modules are skipped. For ST facts, early termination is suspended until M₃ has executed.
- **Three-Way Decision** (Eq. 23): Accept if C ≥ Θ; Review if Θ−ε ≤ C < Θ (ε=0.1); Reject if C < Θ−ε.

### Autonomous Adaptive Intelligence Cycle (AAIC)

AAIC monitors per-module precision via **CGR-CUSUM** (Eq. 24):

**G_i(n) = max(0, G_i(n-1) + [p_i(n) − μ_0 − k])**

where k = 0.5σ (allowable slack), h = 5σ (alarm threshold). When G_i(n) ≥ h, three-level adaptation triggers:

- **Weight** (Eq. 25): `w_i ← w_i × exp[−γ · G_i(t)]`, renormalise Σw_i = 1 (γ = 0.01)
- **Threshold** (Eq. 26): `θ_i ← θ_i + η · sign(FPR_i − FNR_i)` (η = 0.05)
- **Alpha** (Eq. 27): `α_i ← α_i + η' · ∂L_i/∂α_i`, clip [0,1] (η' = 0.02)

### Defense-in-Depth Architecture

The five-layer verification achieves robustness through three principles:

1. **Independence**: Modules operate on distinct information sources (ontology, standards, physics, web, embeddings)
2. **Complementarity**: Modules target different error classes — ontology-compliant fabrications evade M₁/M₃ but are caught by M₂/M₄
3. **Redundancy**: Hallucination covered by both M₂ (terminology) and M₄ (external evidence); Drift covered by both M₁ (ontology) and M₅ (embeddings)

### Domain Adaptation Protocol

Deploying in new domains requires configuring five components:

1. **Domain Ontology (O)**: Entity classes C, relation types R_o, attributes A (50–200 classes typical)
2. **Industry Standards (M₂)**: STEP AP242 for aerospace, HL7 FHIR for healthcare, ISA-95 for manufacturing
3. **Physical Constraints (M₃)**: Max velocities v_max, minimum process durations, facility geometry, veto threshold τ_veto
4. **Source Credibility (M₄)**: Credibility weights w_cred per source type (government > manufacturer > academic > news)
5. **Domain Embeddings (M₅)**: Sentence-transformers on ≥10K historical facts, quarterly retraining

## Key Features

- **Multi-Domain Support** — Aerospace, Healthcare, Aviation, CAD/Engineering
- **Physics-Based Verification** — Enforces ψ_s (bilocation) and ψ_t (velocity/travel-time) consistency
- **Three-Stage Pipeline** — Data Preprocessing → LLM Extraction → TruthFlow Verification
- **Adaptive Intelligence** — AAIC auto-adjusts w, θ, α via CGR-CUSUM monitoring
- **Honest Verification** — Real ontology checking, real standard terminology matching, real embedding similarity
- **ST/SEM Fact Classification** — Physics checks applied only to spatiotemporal facts
- **Critical Module Veto** — MAV can immediately reject physically impossible ST facts
- **Automatic STKG Integration** — Accepted facts added to knowledge graph
- **Interactive Visualization** — 7 dashboard tabs:
  - 📚 Methodology — STKG formalization, physics predicates, error taxonomy, five-module pipeline
  - 🌐 Domain Configuration — Load/edit domain configs (ontology, standards, physics, credibility, embeddings)
  - 🗂️ STKG Structure — Knowledge graph visualization, domain examples, ontology browser
  - 💠 Verification Process — Three-stage pipeline with upload/processing capabilities
  - 🔄 AAIC Monitoring — CGR-CUSUM tracking and parameter shift detection
  - 📊 Parameter Evolution — Weight, threshold, and alpha adaptation over time
  - 📜 Verification History — Complete audit trail of all verifications

## Getting Started

### Prerequisites

- Python 3.8+
- Dependencies listed in `requirements.txt`:
  - `streamlit>=1.24.0`
  - `pandas>=1.5.0`
  - `numpy>=1.23.0`
  - `matplotlib>=3.6.0`
  - `plotly>=5.14.0`

### Installation

```bash
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

## Code Structure

```
├── app.py                          # Main Streamlit application (7 tabs)
├── requirements.txt                # Python dependencies
├── models/
│   ├── knowledge_graph.py          # 4D STKG with physics predicates (ψ_s, ψ_t, Ψ)
│   ├── ontology.py                 # Multi-domain ontology (16 entity classes, 11 relationships)
│   └── constants.py                # Physical params, veto thresholds, CUSUM params, standard terminologies
├── verification/
│   ├── rmmve.py                    # RMMVe: ST/SEM classification, veto, 3-way decision, early termination
│   ├── modules.py                  # 5 modules (LOV, POV, MAV, WSV, ESV) with real dual-metric implementations
│   ├── base.py                     # Base module: S_i = conf_k × [α·M1 + (1−α)·M2]
│   ├── domain_adapter.py           # Domain adaptation (5-component configuration)
│   └── defense_in_depth.py         # Defense-in-Depth analysis (independence, complementarity, redundancy)
├── aaic/
│   └── aaic.py                     # CGR-CUSUM monitoring, FPR/FNR threshold, loss-gradient alpha
├── data/
│   ├── preprocessing.py            # Stage 1: RD → RD' (OCR, spell correction, temporal alignment, spatial mapping)
│   ├── llm_extraction.py           # Stage 2: D = L(RD'; P) with confidence {high:1.0, medium:0.8, low:0.6}
│   ├── generators.py               # Test data with quality-specific issues (semantic, spatial, low)
│   └── quality_based_generator.py  # Raw text generation for honest testing
├── experiments/
│   ├── run_experiments.py          # Full experiment runner across 4 domains
│   ├── quick_demo.py               # Quick CLI demo
│   ├── datasets/                   # Domain-specific dataset generators
│   └── metrics/                    # Evaluation: Precision, Recall, F1, FPR
├── visualization/                  # Plotly charts and Streamlit UI components
├── utils/                          # CSS styles
└── domains/                        # Domain configuration JSON files
```

## Usage

### Interactive Dashboard

```bash
streamlit run app.py
```

**Basic Workflow:**

1. **Generate Test Fact** (Sidebar): Select domain and quality level → Click "🎲 Generate Test Fact"
2. **Run Verification** (Verification Process Tab): Stage 1 → Stage 2 → Stage 3
3. **View Results**: Decision (Accept/Review/Reject), fact type (ST/SEM), module scores, cumulative confidence

**Or Upload Your Own Data:**
- Stage 1: Upload TXT/JSON file → Configure domain → Run preprocessing
- Stage 2: Extract facts with domain-specialized LLM prompts
- Stage 3: Verify via RMMVe + AAIC and integrate accepted facts into STKG

### Command-Line Testing

```bash
# Verification pipeline demo (high, medium, spatial, low quality)
python3 test_verification_demo.py

# Quick experiment demo with metrics
python3 experiments/quick_demo.py

# Full experiments on specific or all datasets
python3 experiments/run_experiments.py --dataset manufacturing --num-facts 100
python3 experiments/run_experiments.py --all --num-facts 100

# Domain adaptation test
python3 test_domain_adaptation.py
```

Results are saved to `experiments/results/` as JSON with complete metrics and confusion matrices.

## License

This project is for demonstration purposes only.
