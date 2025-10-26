# ATLASky-AI Quick Start Guide

## What You Just Saw

The command-line demo showed:
- ✅ **High Quality Facts**: Pass early at LOV/POV/MAV
- ✅ **Medium Quality Facts**: Need multiple modules but accepted
- ✅ **Spatial Issues**: Caught by MAV's physics-based validation
- ❌ **Low Quality Facts**: Correctly rejected

## Running the Interactive Dashboard

### 1. Start the Application

```bash
cd /home/user/AtlaSkI-AI
streamlit run app.py
```

The dashboard will start on **http://localhost:8501**

### 2. Dashboard Navigation

The dashboard has 4 main tabs:

#### Tab 1: Main Dashboard 📊
- Overview of knowledge graph statistics
- Real-time verification metrics
- Module performance gauges

#### Tab 2: Fact Verification 🔍
- **Generate Facts**: Select quality level and generate candidate facts
- **Run Verification**: Execute RMMVe process
- **View Results**: See detailed module scores, metrics, and decisions
- **Early Termination**: Watch which module triggers acceptance

#### Tab 3: AAIC Monitoring 🤖
- **CGR-CUSUM Tracking**: Monitor G_i(n) statistics
- **Parameter Evolution**: Watch w_i, θ_i, α_i adapt over time
- **Shift Detection**: See when h threshold is exceeded
- **Update History**: View all parameter adjustments

#### Tab 4: Verification History 📈
- **All Verifications**: Complete history with decisions
- **Quality Distribution**: Breakdown by fact type
- **Early Termination Stats**: Which modules terminate most
- **Performance Trends**: Accuracy and throughput over time

## Quick Test Scenarios

### Scenario 1: High-Quality Verification
1. Go to **Fact Verification** tab
2. Select **Quality Level**: `high_quality`
3. Click **Generate Candidate Fact**
4. Click **Run Verification**
5. **Expected**: Early termination at LOV or POV, confidence > 0.80

### Scenario 2: Spatial Inconsistency Detection
1. Select **Quality Level**: `spatial_issue`
2. Generate and verify
3. **Expected**: MAV module catches the issue using ψ_s/ψ_t predicates

### Scenario 3: Performance Adaptation
1. Select **Include Performance Shift**: ✓
2. Verify multiple facts (10-20)
3. Go to **AAIC Monitoring** tab
4. **Expected**: See CUSUM statistics increase, parameter updates trigger

### Scenario 4: Low-Quality Rejection
1. Select **Quality Level**: `low_quality`
2. Generate and verify
3. **Expected**: Rejection decision, confidence < 0.65

## Understanding the Output

### Module Scores Display
```
LOV: 0.8193 (threshold: 0.82) [✓]
     Metric1: 0.9878 | Metric2: 0.9080
```
- **Score**: S_i = α_i × Metric1 + (1-α_i) × Metric2
- **Threshold**: θ_i activation threshold
- **[✓]**: Module activated (score ≥ threshold)
- **Metrics**: Two complementary validation scores

### Verification Decision
```
Decision: ACCEPT
Cumulative Confidence: 0.8011 (Threshold: 0.65)
Early Termination: True at MAV
```
- **C(d_k)**: Weighted average of activated modules
- **Θ = 0.65**: Global threshold for acceptance
- **Early Termination**: Stops when C(d_k) ≥ Θ

### AAIC Updates
```
G_i(n) = 1.8 > h = 1.5  →  Update triggered
w_i: 0.8500 → 0.7853 (↓ 7.6%)
θ_i: 0.8200 → 0.8350 (↑ 1.8%)
α_i: 0.7000 → 0.7140 (↑ 2.0%)
```

## System Architecture

### Data Flow
```
Raw Data → Preprocessing → LLM Extraction → Candidate Facts
                                                    ↓
                                            TruthFlow Verification
                                                    ↓
                                    RMMVe (LOV→POV→MAV→WSV→ESV)
                                                    ↓
                                        Accept / Reject / Review
                                                    ↓
                                            AAIC Monitoring
                                                    ↓
                                        Parameter Adaptation
```

### Module Error Targeting

| Module | Targets | Detects |
|--------|---------|---------|
| LOV | Semantic Drift | Ontology violations, invalid classes |
| POV | Content Hallucination | Non-standard terminology |
| MAV | ST-Inconsistency | Physics violations (ψ_s, ψ_t) |
| WSV | Content Hallucination | Lack of external evidence |
| ESV | Semantic Drift | Statistical anomalies |

## Key Features Demonstrated

1. **Physical Consistency**: MAV uses ψ_s and ψ_t predicates (Definitions 2 & 3)
2. **Early Termination**: 40% computation reduction via confidence thresholds
3. **Defense-in-Depth**: Independent, complementary, redundant modules
4. **Adaptive Intelligence**: CGR-CUSUM monitoring + automatic parameter tuning
5. **Multi-Threat Detection**: Handles all three error classes simultaneously

## Performance Monitoring

### Metrics to Track
- **Throughput**: Facts verified per second
- **Latency**: Average verification time (target: <1 second)
- **Accuracy**: Precision/recall on labeled samples
- **Early Termination Rate**: % facts that skip expensive modules
- **Parameter Stability**: Frequency of AAIC updates

### Expected Performance
- High-quality facts: **0.02-0.05 ms** (LOV/POV termination)
- Spatial issues: **0.04-0.08 ms** (MAV termination)
- Full pipeline: **0.50-1.00 seconds** (all 5 modules)

## Advanced Configuration

### Adjust Global Threshold
```python
# In app.py or verification script
rmmve = RMMVeProcess(global_threshold=0.70)  # More strict
```

### Modify AAIC Parameters
```python
aaic = AAIC(
    rmmve,
    h=2.0,           # Higher alarm threshold (less sensitive)
    gamma=0.02,      # Faster weight decay
    eta=0.08,        # Larger threshold adjustments
    eta_prime=0.03   # Larger alpha adjustments
)
```

### Custom Module Weights
```python
# Emphasize MAV for spatiotemporal domains
rmmve.modules[2].weight = 1.0  # MAV
rmmve.modules[0].weight = 0.6  # LOV
aaic.normalize_weights()
```

## Troubleshooting

### Dashboard Not Loading
```bash
# Check if port is available
lsof -i:8501

# Use different port
streamlit run app.py --server.port 8502
```

### Module Import Errors
```bash
# Verify all dependencies
pip3 install -r requirements.txt

# Check Python version (need 3.8+)
python3 --version
```

### Slow Verification
- Check if all modules are activating (should have early termination)
- Verify threshold settings (too low = all modules run)
- Monitor module scores in dashboard

## Next Steps

1. **Explore Dashboard**: Try all quality levels and observe patterns
2. **Review Methodology**: See how implementation matches formal definitions
3. **Customize Configuration**: Adjust for your specific domain
4. **Deploy at Scale**: Use the system on real aerospace/healthcare data
5. **Monitor Performance**: Track AAIC adaptations over time

## Support

For questions or issues:
- Check the methodology section for formal definitions
- Review module documentation in `verification/modules.py`
- See STKG implementation in `models/knowledge_graph.py`
- Examine AAIC logic in `aaic/aaic.py`

---

**Demo Commands Summary**:
```bash
# Run command-line demo
python3 test_verification_demo.py

# Run interactive dashboard
streamlit run app.py

# Run with custom config
streamlit run app.py --server.port 8501 --server.headless true
```
