# ATLASky-AI

## Aerospace Spatiotemporal Knowledge Graph Verification System

ATLASky-AI is a sophisticated verification system for aerospace data that combines multiple verification approaches to ensure the integrity and validity of spatiotemporal knowledge graphs in the aerospace domain.

## System Architecture

The system implements the Ranked Multi-Modal Verification (RMMVe) process, which utilizes five verification modules in sequence:

1. **Local Ontology Verification (LOV)** - Verifies facts against the local knowledge graph
2. **Public Ontology Verification (POV)** - Verifies facts against public domain knowledge
3. **Multi-Agent Verification (MAV)** - Uses multiple specialized agents for verification
4. **Web Search Verification (WSV)** - Leverages web search for external verification
5. **Embedding Similarity Verification (ESV)** - Uses semantic embeddings to verify facts

The system includes an Autonomous Adaptive Intelligence Cycle (AAIC) that continuously monitors module performance and adaptively adjusts parameters to optimize verification.

## Features

- **Early Termination** - Stops the verification process when sufficient confidence is reached
- **Adaptive Parameters** - Automatically adjusts module parameters based on performance
- **Spatiotemporal Knowledge Graph** - Handles 4D aerospace data (space + time)
- **Interactive Visualization** - Comprehensive dashboards for monitoring system performance
- **Performance Monitoring** - Tracks verification metrics and parameter evolution

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
cd atlasky
streamlit run app.py
```

## Code Structure

- `app.py` - Main application entry point
- `models/` - Knowledge graph implementation and constants
- `verification/` - Verification modules implementation
- `aaic/` - Autonomous Adaptive Intelligence Cycle implementation
- `data/` - Data generation utilities
- `visualization/` - Visualization components and plots
- `utils/` - Utility functions and styles

## Usage

The dashboard allows you to:

1. Generate sample candidate facts with varying quality levels
2. Run verification on these facts using the RMMVe process
3. Monitor verification results and confidence scores
4. Track module performance and parameter adaptations
5. Analyze verification history and quality distribution

## License

This project is for demonstration purposes only. 