# ATLASky-AI

## Multi-Domain 4D Spatiotemporal Knowledge Graph Verification System

ATLASky-AI is a domain-adaptable verification system for 4D Spatiotemporal Knowledge Graphs (STKGs) that combines physics-based constraints with multi-modal verification to detect and prevent:

- **Content Hallucination**: Fabricated facts not grounded in reality
- **ST-Inconsistency**: Violations of physical laws (spatial/temporal)
- **Semantic Drift**: Facts that deviate from domain ontology

## System Architecture

### 4D STKG Formalization

ATLASky-AI operates on a formal 4D STKG defined as **G = (V, E, O, T, Ψ)** where:

- **V**: Versioned entities with immutable attributes and mutable state
- **E**: Directed edges representing relationships
- **O = (C, R_o, A)**: Domain ontology with entity classes, relation types, and attributes
- **T: (V ∪ E) → ℝ³ × ℝ**: Maps entities/relations to spatiotemporal coordinates (x,y,z,t)
- **Ψ: (V ∪ E) → {0,1}**: Physical consistency predicate combining spatial (ψ_s) and temporal (ψ_t) consistency

### Physics-Based Predicates

- **ψ_s (Spatial Consistency)**: Prevents co-location violations (same entity at two locations simultaneously)
- **ψ_t (Temporal Consistency)**: Enforces velocity constraints (travel time must be physically feasible)
- **Ψ = ψ_s ∧ ψ_t**: Combined predicate ensuring full physical consistency

### Five-Module Verification Pipeline

The system implements the Ranked Multi-Modal Verification (RMMVe) process:

1. **Local Ontology Verification (LOV)** - Targets Semantic Drift using structural and attribute compliance
2. **Provenance-Aware Verification (POV)** - Targets Content Hallucination via lineage tracing
3. **Motion-Aware Verification (MAV)** - Targets ST-Inconsistency using physics predicates (ψ_s, ψ_t, Ψ)
4. **Workflow State Verification (WSV)** - Validates state transitions and workflow compliance
5. **External Source Verification (ESV)** - Cross-references against authoritative sources

### Autonomous Adaptive Intelligence Cycle (AAIC)

AAIC continuously monitors module performance using the CGR-CUSUM algorithm and adaptively adjusts:

- **Weights (w)**: Using exponential decay based on cumulative error
- **Thresholds (θ)**: Using gradient ascent based on FPR-FNR balance
- **Alpha (α)**: Using gradient ascent to optimize metric combination

## Key Features

- **Multi-Domain Support** - Aerospace, Healthcare, Aviation, CAD/Engineering
- **Physics-Based Verification** - Enforces spatial (ψ_s) and temporal (ψ_t) consistency using physical laws
- **Three-Stage Pipeline** - Data Preprocessing → LLM Extraction → TruthFlow Verification
- **Adaptive Intelligence** - AAIC automatically adjusts parameters based on performance monitoring
- **Honest Verification** - No artificial score boosting, real quality assessment
- **Automatic STKG Integration** - Accepted facts automatically added to knowledge graph
- **Interactive Visualization** - Comprehensive dashboards with 6 tabs:
  - 📚 Methodology - STKG formalization, physics predicates, error taxonomy
  - 🗂️ STKG Structure - Knowledge graph visualization, domain examples, ontology browser
  - 💠 Verification Process - Three-stage pipeline with upload/processing capabilities
  - 🔄 AAIC Monitoring - CGR-CUSUM tracking and parameter shift detection
  - 📊 Parameter Evolution - Weight, threshold, and alpha adaptation over time
  - 📜 Verification History - Complete audit trail of all verifications
- **Performance Metrics** - Precision, Recall, F1-Score, False Positive Rate (FPR)

## Getting Started

### Prerequisites

- Python 3.8+
- Streamlit
- Pandas
- Plotly
- NumPy
- Matplotlib

### Running the Application

```bash
cd AtlaSkI-AI
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

## Code Structure

- `app.py` - Main Streamlit application with 6 interactive tabs
- `models/` - Knowledge graph and ontology implementation
  - `knowledge_graph.py` - 4D STKG with physics predicates
  - `ontology.py` - Multi-domain ontology system (16 entity classes, 11 relationships)
  - `constants.py` - Physical parameters and domain constants
- `verification/` - Five verification modules (LOV, POV, MAV, WSV, ESV)
  - `rmmve.py` - Ranked Multi-Modal Verification with early termination
  - `modules.py` - All 5 modules with dual-metric implementation
  - `base.py` - Base module with LLM confidence weighting
- `aaic/` - Autonomous Adaptive Intelligence Cycle
  - `aaic.py` - CGR-CUSUM monitoring and parameter adaptation
- `data/` - Data processing and generation
  - `preprocessing.py` - Stage 1: RD → RD' normalization
  - `llm_extraction.py` - Stage 2: LLM extraction with prompts
  - `quality_based_generator.py` - Honest test data generation
- `experiments/` - Experimental evaluation
  - `datasets/` - Generators (aerospace, aviation, CAD, healthcare)
  - `metrics/` - Evaluation metrics (Precision, Recall, F1, FPR)
- `visualization/` - UI components and Plotly charts
- `utils/` - Utility functions and CSS styles

## Usage

### Interactive Dashboard

Launch the Streamlit dashboard to explore ATLASky-AI:

```bash
streamlit run app.py
```

The dashboard provides 6 interactive tabs:

#### 📚 Methodology Tab
- Complete STKG formalization (G = (V, E, O, T, Ψ))
- Physics predicates visualization (ψ_s, ψ_t, Ψ)
- Error taxonomy (Content Hallucination, ST-Inconsistency, Semantic Drift)
- Five-module pipeline explanation
- AAIC adaptation mechanisms

#### 🗂️ STKG Structure Tab
- Knowledge graph visualization (V, E, O, T, Ψ)
- Domain-specific STKG examples (aerospace, healthcare, aviation, CAD)
- Live metrics (entities, relationships, accepted facts)
- Recent STKG updates (last 5 accepted facts)
- Ontology browser (entity classes, relationships, constraints, rules)

#### 💠 Verification Process Tab
- Three-stage pipeline: Preprocessing → LLM Extraction → TruthFlow Verification
- Upload files (TXT, JSON, PDF) OR generate test facts
- Stage 1: Data preprocessing with before/after comparison
- Stage 2: LLM extraction with confidence scoring
- Stage 3: RMMVe + AAIC verification with Accept/Review/Reject decisions
- Real-time module performance and cumulative confidence calculation
- Automatic STKG integration for accepted facts

#### 🔄 AAIC Monitoring Tab
- Monitor CGR-CUSUM cumulative sums
- Track performance shifts in real-time
- View parameter adjustment history

#### 📊 Parameter Evolution Tab
- Visualize weight, threshold, and alpha changes over time
- Compare current vs. initial parameter values
- Understand adaptation patterns

#### 📜 Verification History Tab
- Complete audit trail of all verifications
- Performance trends and quality distribution
- Early termination statistics

### Quick Start Guide

**For detailed usage instructions, see [`HOW_TO_USE.md`](HOW_TO_USE.md)**

**Basic Workflow:**

1. **Generate Test Fact** (Sidebar):
   - Select domain (aerospace/healthcare/aviation/CAD)
   - Select quality level (high/medium/low)
   - Click "🎲 Generate Test Fact"

2. **Run Verification** (Verification Process Tab):
   - Stage 1: Click "▶️ Run Stage 1 Preprocessing"
   - Stage 2: Click "▶️ Run Stage 2 LLM Extraction"
   - Stage 3: Click "▶️ Run TruthFlow Verification"

3. **View Results**:
   - See decision (Accept/Review/Reject) in right column
   - Check STKG Structure tab to see accepted facts added to knowledge graph

**Or Upload Your Own Data:**
- Stage 1: Upload TXT/JSON file → Configure domain → Run preprocessing
- Stage 2: Extract facts from your data
- Stage 3: Verify and integrate into STKG

### Command-Line Experiments

Run comprehensive experiments on different dataset types:

```bash
# Test on specific dataset
python3 experiments/run_experiments.py --dataset manufacturing --num-facts 100

# Test on all 4 dataset types
python3 experiments/run_experiments.py --all --num-facts 100

# Quick demo (works immediately)
python3 experiments/quick_demo.py

# Detailed output
python3 experiments/run_experiments.py --dataset aviation --detailed
```

Results are saved to `experiments/results/` as JSON files with complete metrics and confusion matrices.

## License

This project is for demonstration purposes only. 