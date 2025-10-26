# ATLASky-AI

## 4D Spatiotemporal Knowledge Graph Verification System

ATLASky-AI is a novel verification system for 4D Spatiotemporal Knowledge Graphs (STKGs) that combines physics-based constraints with multi-agent verification to detect and prevent:

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

## Features

- **Physics-Based Verification** - Enforces spatial (ψ_s) and temporal (ψ_t) consistency using physical laws
- **Early Termination** - Stops verification when sufficient confidence is reached, improving efficiency
- **Adaptive Parameters** - AAIC automatically adjusts weights, thresholds, and alpha values
- **4D Spatiotemporal Knowledge Graph** - Full support for entities with (x,y,z,t) coordinates
- **Experimental Evaluation** - Test on 4 dataset types: Manufacturing, Aviation, CAD, Healthcare
- **Interactive Visualization** - Comprehensive dashboards with 6 tabs:
  - 📚 Methodology - STKG formalization, physics predicates, error taxonomy
  - 💠 Verification Process - Real-time fact verification with module performance
  - 🧪 Experimental Evaluation - Live demo on different dataset types with P/R/F1/FPR metrics
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
cd atlasky-ai
streamlit run app.py
```

## Code Structure

- `app.py` - Main Streamlit application with 6 interactive tabs
- `models/` - Knowledge graph implementation, constants, and STKG formalization
- `verification/` - Five verification modules (LOV, POV, MAV, WSV, ESV)
- `aaic/` - Autonomous Adaptive Intelligence Cycle with CGR-CUSUM monitoring
- `experiments/` - Experimental evaluation framework
  - `datasets/` - Dataset generators (manufacturing, aviation, CAD, healthcare)
  - `metrics/` - Evaluation metrics (Precision, Recall, F1, FPR)
  - `run_experiments.py` - Full experiment runner
  - `quick_demo.py` - Quick demonstration script
- `data/` - Data generation utilities
- `visualization/` - Visualization components and Plotly charts
- `utils/` - Utility functions and custom CSS styles

## Usage

### Interactive Dashboard

Launch the Streamlit dashboard to explore ATLASky-AI:

```bash
streamlit run app.py
```

The dashboard provides 6 interactive tabs:

#### 📚 Methodology Tab
- Complete STKG formalization (Definition 1: G = (V, E, O, T, Ψ))
- Physics predicates visualization (ψ_s, ψ_t, Ψ)
- Error taxonomy (Content Hallucination, ST-Inconsistency, Semantic Drift)
- Five-module pipeline explanation
- AAIC adaptation mechanisms

#### 💠 Verification Process Tab
- Generate candidate facts with varying quality levels
- Run real-time verification with RMMVe pipeline
- View module confidence scores and early termination
- Analyze verification metrics and processing time

#### 🧪 Experimental Evaluation Tab
- Select dataset type (Manufacturing, Aviation, CAD, Healthcare)
- View expected performance benchmarks
- Run live demo with configurable number of facts
- See real-time P/R/F1/FPR metrics
- Compare performance across different error patterns

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