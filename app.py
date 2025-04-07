import streamlit as st
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime, timedelta

# Import from our modules
from models.constants import (
    ENTITY_CLASSES, RELATIONSHIP_TYPES, BLADE_FEATURES, BLADE_COMPONENTS,
    ENGINE_SETS, MEASUREMENT_IDS, SURFACE_SIDES, QUALITY_COLOR_MAP
)
from models.knowledge_graph import KnowledgeGraph, create_sample_knowledge_graph
from verification.rmmve import RMMVeProcess
from aaic.aaic import AAIC
from data.generators import generate_candidate_fact, generate_candidate_fact_with_quality
from visualization.ui_components import (
    create_fact_preview_card, get_quality_badge, get_decision_badge,
    format_shift_alert, display_parameter_change, create_gauge_grid
)
from visualization.plots import (
    plot_verification_results_plotly, plot_metrics_plotly,
    plot_aaic_cumulative_sums_plotly, plot_aaic_parameter_evolution_plotly,
    plot_verification_history_plotly, plot_quality_distribution_plotly,
    plot_early_term_by_module_plotly, create_module_performance_gauge
)
from utils.styles import local_css

# -----------------------------------------------------------------------------
# Configure page settings with improved layout
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ATLASky-AI Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(local_css(), unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    # ----- Initialize Application State -----
    
    # Initialize session state
    if "kg" not in st.session_state:
        st.session_state.kg = create_sample_knowledge_graph()
    
    if "rmmve" not in st.session_state:
        st.session_state.rmmve = RMMVeProcess(global_threshold=0.65)
    
    if "aaic" not in st.session_state:
        st.session_state.aaic = AAIC(st.session_state.rmmve)
    
    if "candidate_fact" not in st.session_state:
        st.session_state.candidate_fact, st.session_state.fact_quality = generate_candidate_fact()
    
    if "verification_history" not in st.session_state:
        st.session_state.verification_history = []
    
    if "verification_count" not in st.session_state:
        st.session_state.verification_count = 0
    
    # Add trigger for introducing shifts
    if "introduce_shift" not in st.session_state:
        st.session_state.introduce_shift = False
    
    if "shift_module" not in st.session_state:
        st.session_state.shift_module = "LOV"
    
    # ----- Header Section -----
    st.markdown(
    """
    <div class="header-container">
        <div class="header-title">ATLASky-AI</div>
        <div class="header-subtitle">Aerospace Spatiotemporal Knowledge Graph Verification System</div>
    </div>
    """, 
    unsafe_allow_html=True
)
    
    # ----- Sidebar -----
    with st.sidebar:
        st.markdown("<h3>Control Panel</h3>", unsafe_allow_html=True)
        
        st.markdown("### Data Generation")
        if st.button("Generate New Candidate Fact", key="gen_new_fact"):
            st.session_state.candidate_fact, st.session_state.fact_quality = generate_candidate_fact(st.session_state.introduce_shift)
            st.rerun()
        
        # Allow manual selection of quality level for testing
        quality_options = [
            "high_quality", "medium_quality", "spatial_issue", 
            "external_ref", "semantic_issue", "low_quality"
        ]
        
        quality_level = st.selectbox(
            "Select Fact Quality Level",
            options=quality_options,
            index=quality_options.index(st.session_state.fact_quality) if hasattr(st.session_state, 'fact_quality') else 0,
            help="Choose quality level for testing different verification paths"
        )
        
        if quality_level != st.session_state.fact_quality:
            st.session_state.candidate_fact, st.session_state.fact_quality = generate_candidate_fact_with_quality(quality_level, st.session_state.introduce_shift)
            st.rerun()
        
        st.markdown("### Knowledge Graph")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Entities</div>
            <div class="metric-value">{len(st.session_state.kg.entities)}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Relationships</div>
            <div class="metric-value">{len(st.session_state.kg.relationships)}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # AAIC Controls
        st.markdown("### AAIC Parameters")
        
        with st.expander("Adaptive Parameters", expanded=False):
            h_param = st.slider("CGR-CUSUM Threshold (h)", 0.0, 10.0, st.session_state.aaic.h, 0.5,
                                help="Threshold for detecting performance shifts")
            if h_param != st.session_state.aaic.h:
                st.session_state.aaic.h = h_param
            
            k_param = st.slider("Allowance Parameter (k)", 0.01, 0.2, st.session_state.aaic.k, 0.01,
                                help="Allowance in CGR-CUSUM algorithm")
            if k_param != st.session_state.aaic.k:
                st.session_state.aaic.k = k_param
            
            eta_param = st.slider("Learning Rate (η)", 0.01, 0.5, st.session_state.aaic.eta, 0.01,
                                help="Learning rate for threshold updates")
            if eta_param != st.session_state.aaic.eta:
                st.session_state.aaic.eta = eta_param
            
            gamma_param = st.slider("Weight Scaling (γ)", 0.01, 0.5, st.session_state.aaic.gamma, 0.01,
                                help="Scaling factor for weight updates")
            if gamma_param != st.session_state.aaic.gamma:
                st.session_state.aaic.gamma = gamma_param
        
        # Performance Shift Controls
        st.markdown("### Performance Testing")
        
        st.markdown("""
        <div class="alert-info">
            Performance shift testing allows you to simulate degraded performance 
            in specific modules to trigger AAIC adaptation.
        </div>
        """, unsafe_allow_html=True)
        
        st.session_state.introduce_shift = st.checkbox("Introduce Performance Shifts", 
                                                      value=st.session_state.introduce_shift,
                                                      help="When enabled, generated facts will contain anomalies that reduce performance")
        
        if st.session_state.introduce_shift:
            st.session_state.shift_module = st.selectbox(
                "Module to affect",
                options=["LOV", "POV", "MAV", "WSV", "ESV"],
                index=["LOV", "POV", "MAV", "WSV", "ESV"].index(st.session_state.shift_module),
                help="Select which verification module will experience reduced performance"
            )
            
            st.info(f"Next fact will have a performance shift in the {st.session_state.shift_module} module")
        
        # Batch Processing Button
        st.markdown("### Batch Processing")
        
        st.markdown("""
        <div class="alert-info">
            Run batch processing to generate more verification data and trigger AAIC parameter adjustments.
            More runs increase the likelihood of detecting performance shifts.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            batch_size = st.number_input("Number of facts to process", min_value=1, max_value=100, value=10)
        
        with col2:
            force_shifts = st.checkbox("Force Performance Shifts", value=False, 
                                      help="When enabled, guarantees at least some performance shifts will be generated")
        
        if st.button(f"Process {batch_size} Facts", key="batch_verify"):
            # Add a progress bar
            progress_bar = st.progress(0)
            
            # Keep track of shifts detected
            shifts_detected = 0
            
            for i in range(batch_size):
                # Update progress
                progress_bar.progress((i + 1) / batch_size)
                
                # For forced shifts, periodically generate a fact with a shift
                # to ensure AAIC parameter adjustments are triggered
                introduce_shift = st.session_state.introduce_shift
                if force_shifts and i % 3 == 0:  # Every 3rd iteration
                    introduce_shift = True
                
                # Generate new fact
                fact, quality = generate_candidate_fact(introduce_shift)
                
                # Run verification
                verification_results = st.session_state.rmmve.verify(fact, st.session_state.kg, quality)
                
                # Record verification
                verification_record = {
                    "timestamp": time.time(),
                    "candidate_fact": fact,
                    "verification_results": verification_results,
                    "fact_quality": quality
                }
                
                # Run AAIC
                aaic_updates = st.session_state.aaic.update_all_modules()
                verification_record["aaic_updates"] = aaic_updates
                
                # Count shifts detected
                for update in aaic_updates:
                    if update["detected"]:
                        shifts_detected += 1
                
                # Add to history
                st.session_state.verification_history.append(verification_record)
                st.session_state.verification_count += 1
            
            # Generate new candidate fact
            st.session_state.candidate_fact, st.session_state.fact_quality = generate_candidate_fact(st.session_state.introduce_shift)
            
            # Show message about shifts
            if shifts_detected > 0:
                st.success(f"✅ Batch completed with {shifts_detected} performance shifts detected and parameters adjusted!")
            else:
                st.info("Batch completed without any parameter adjustments. Try running again or check CGR-CUSUM parameters.")
            
            # Refresh
            st.rerun()
    
    # ----- Main Content Area -----
    
    # Create tabs for different views
    tab_verification, tab_aaic, tab_parameters, tab_history = st.tabs([
        "💠 Verification Process", 
        "🔄 AAIC Monitoring", 
        "📊 Parameter Evolution",
        "📜 Verification History"
    ])
    
    with tab_verification:  # Verification Process Tab
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("## Candidate Fact")
            
            # Display quality level with modern styling
            quality_color = QUALITY_COLOR_MAP.get(st.session_state.fact_quality, "#6b7280")
            st.markdown(
                f"""
                <div style='background-color: {quality_color}; padding: 12px; border-radius: 8px; 
                     color: white; margin-bottom: 15px; display: flex; align-items: center;'>
                    <span style='font-size: 24px; margin-right: 10px;'>📋</span>
                    <div>
                        <strong style='font-size: 16px;'>Fact Quality:</strong><br>
                        <span style='font-size: 20px; font-weight: 500;'>{st.session_state.fact_quality.replace('_', ' ').title()}</span>
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Show if the fact contains a performance shift
            if "contains_performance_shift" in st.session_state.candidate_fact:
                shift_module = st.session_state.candidate_fact["contains_performance_shift"]
                st.markdown(format_shift_alert(shift_module), unsafe_allow_html=True)
            
            with st.expander("Fact Details", expanded=True):
                st.markdown(create_fact_preview_card(st.session_state.candidate_fact), unsafe_allow_html=True)
            
            # Add JSON expander to show raw data
            with st.expander("Raw JSON Data", expanded=False):
                # Use json.dumps with indentation for pretty printing
                import json
                st.code(json.dumps(st.session_state.candidate_fact, indent=2), language="json")
            
            # Verification control
            st.markdown("## Verification Process")
            
            st.markdown("""
            <div class="alert-info">
                Click "Run Verification" to apply the RMMVe process to the candidate fact.
                The verification system will attempt to sequentially apply verification modules
                with the possibility of early termination when sufficient confidence is reached.
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Run Verification", key="run_verify"):
                with st.spinner("Running verification..."):
                    # Run verification
                    verification_results = st.session_state.rmmve.verify(
                        st.session_state.candidate_fact, 
                        st.session_state.kg,
                        st.session_state.fact_quality
                    )
                    
                    # Record verification
                    verification_record = {
                        "timestamp": time.time(),
                        "candidate_fact": st.session_state.candidate_fact,
                        "verification_results": verification_results,
                        "fact_quality": st.session_state.fact_quality
                    }
                    
                    # Run AAIC
                    aaic_updates = st.session_state.aaic.update_all_modules()
                    verification_record["aaic_updates"] = aaic_updates
                    
                    # Add to history
                    st.session_state.verification_history.append(verification_record)
                    st.session_state.verification_count += 1
                    
                    # Generate new candidate fact
                    st.session_state.candidate_fact, st.session_state.fact_quality = generate_candidate_fact(st.session_state.introduce_shift)
                    
                    # Refresh
                    st.rerun()
        
        with col2:
            st.markdown("## Verification Results")
            
            if st.session_state.verification_history:
                # Get latest verification
                latest = st.session_state.verification_history[-1]
                results = latest["verification_results"]
                
                # Display fact quality
                quality = results.get("fact_quality", "unknown")
                quality_color = QUALITY_COLOR_MAP.get(quality, "#6b7280")
                
                # Create a results header with fancy styling
                st.markdown(
                    f"""
                    <div style='display: flex; margin-bottom: 15px;'>
                        <div style='background-color: {quality_color}; padding: 12px; border-radius: 8px; 
                             color: white; flex: 1; margin-right: 10px;'>
                            <span style='font-size: 16px;'>Fact Quality</span><br>
                            <span style='font-size: 20px; font-weight: 500;'>{quality.replace('_', ' ').title()}</span>
                        </div>
                        <div style='background-color: {"#10b981" if results["decision"] else "#ef4444"}; 
                                    padding: 12px; border-radius: 8px; color: white; flex: 1;'>
                            <span style='font-size: 16px;'>Decision</span><br>
                            <span style='font-size: 20px; font-weight: 500;'>{results["decision"] and "VERIFIED" or "REJECTED"}</span>
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Show if this fact contained a performance shift
                if results.get("contains_shift"):
                    st.markdown(format_shift_alert(results['contains_shift']), unsafe_allow_html=True)
                
                # Display verification summary in a card
                st.markdown(
                    f"""
                    <div class="card">
                        <h3>Verification Summary</h3>
                        <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 10px;">
                            <div style="flex: 1; min-width: 150px;">
                                <div class="metric-label">Total Confidence</div>
                                <div class="metric-value" style="color: {'#10b981' if results['total_confidence'] >= st.session_state.rmmve.global_threshold else '#ef4444'};">
                                    {results['total_confidence']:.3f}
                                </div>
                            </div>
                            <div style="flex: 1; min-width: 150px;">
                                <div class="metric-label">Global Threshold</div>
                                <div class="metric-value">{st.session_state.rmmve.global_threshold:.3f}</div>
                            </div>
                            <div style="flex: 1; min-width: 150px;">
                                <div class="metric-label">Processing Time</div>
                                <div class="metric-value">{results['verification_time']:.3f}s</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Show early termination information prominently
                if results.get("early_termination"):
                    module_name = results.get("early_termination_module")
                    confidence = results.get("early_termination_confidence", 0)
                    threshold = results.get("early_termination_threshold", 0)
                    
                    st.markdown(
                        f"""
                        <div style='background-color: #10b981; color: white; padding: 15px; border-radius: 8px; margin: 15px 0;'>
                            <div style='display: flex; align-items: center;'>
                                <span style='font-size: 24px; margin-right: 15px;'>🚀</span>
                                <div>
                                    <h3 style='margin:0; font-size: 18px;'>Early Termination Achieved!</h3>
                                    <p style='margin-bottom: 5px;'>Verification stopped at <b>{module_name}</b> module</p>
                                    <p style='margin:0;'>Module confidence: <b>{confidence:.3f}</b> ≥ threshold: <b>{threshold:.3f}</b></p>
                                </div>
                            </div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        """
                        <div class="alert-warning">
                            <strong>⚠️ No early termination.</strong> All verification modules were used in the process.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Create a grid of gauge charts for module performance
                st.markdown("### Module Performance")
                gauges = create_gauge_grid(results)
                if gauges:
                    gauge_cols = st.columns(len(gauges))
                    for i, (col, gauge) in enumerate(zip(gauge_cols, gauges)):
                        with col:
                            st.plotly_chart(gauge, use_container_width=True)
                
                # Plot verification results using Plotly
                st.markdown("### Module Confidence vs. Thresholds")
                fig1 = plot_verification_results_plotly(results)
                st.plotly_chart(fig1, use_container_width=True)
                
                # Plot metrics
                st.markdown("### Verification Metrics")
                fig2 = plot_metrics_plotly(results)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.markdown(
                    """
                    <div class="alert-info">
                        <span style="font-size: 24px; margin-right: 10px;">ℹ️</span>
                        <span>No verification results yet. Run verification to see results.</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    with tab_aaic:  # AAIC Monitoring Tab
        st.markdown("## AAIC Monitoring")
        
        # Explanation of AAIC
        st.markdown(
            """
            <div class="card">
                <h3>Autonomous Adaptive Intelligence Cycle</h3>
                <p>
                    The AAIC continuously monitors the performance of each verification module
                    and adaptively adjusts their parameters when significant shifts in performance are detected.
                </p>
                <p>
                    This monitoring uses the <strong>CGR-CUSUM algorithm</strong> to detect when a module's performance 
                    deviates significantly from its target performance level. When the cumulative sum exceeds the 
                    threshold (h), the AAIC updates the module's parameters:
                </p>
                <ol>
                    <li><strong>Weight (w)</strong> - Using exponential weights algorithm</li>
                    <li><strong>Threshold (θ)</strong> - Using gradient ascent</li>
                    <li><strong>Alpha (α)</strong> - Using gradient ascent</li>
                </ol>
                <p>
                    These updates help the system adapt to changing data characteristics and maintain optimal verification performance.
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Display CGR-CUSUM Monitoring
        st.markdown("### CGR-CUSUM Monitoring")
        
        param_history_df = st.session_state.aaic.get_parameter_history_df()
        if not param_history_df.empty:
            # Display plot
            cusum_fig = plot_aaic_cumulative_sums_plotly(param_history_df)
            if cusum_fig:
                st.plotly_chart(cusum_fig, use_container_width=True)
            
            # Show current cumulative sums with nicer styling
            st.markdown("### Current Cumulative Sums")
            
            cum_sum_data = []
            for module_name, cum_sum in st.session_state.aaic.cumulative_sums.items():
                cum_sum_data.append({
                    "Module": module_name, 
                    "Cumulative Sum": cum_sum, 
                    "Threshold": st.session_state.aaic.h
                })
            
            # Create a grid of modules showing cumulative sums
            cols = st.columns(len(cum_sum_data))
            for i, (col, data) in enumerate(zip(cols, cum_sum_data)):
                with col:
                    # Determine color based on proximity to threshold
                    if data["Cumulative Sum"] > 0.8 * data["Threshold"]:
                        color = "#f59e0b"  # Orange - warning
                    elif data["Cumulative Sum"] > 0.5 * data["Threshold"]:
                        color = "#3b82f6"  # Blue - moderate
                    else:
                        color = "#10b981"  # Green - good
                    
                    # Calculate percentage of threshold
                    pct = (data["Cumulative Sum"] / data["Threshold"]) * 100
                    
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">{data["Module"]}</div>
                            <div class="metric-value" style="color: {color};">{data["Cumulative Sum"]:.3f}</div>
                            <div style="font-size: 0.8rem; color: #6b7280;">{pct:.1f}% of threshold</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            
            # Display detected shifts
            shifts_df = st.session_state.aaic.get_detected_shifts_df()
            if not shifts_df.empty:
                st.markdown("### Detected Parameter Shifts")
                
                for i, shift in shifts_df.iterrows():
                    display_parameter_change(shift)
            else:
                st.markdown(
                    """
                    <div class="alert-info">
                        <span style="font-size: 24px; margin-right: 10px;">ℹ️</span>
                        <span>No parameter shifts detected yet. Run more verifications or lower the h threshold to see shifts.</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                """
                <div class="alert-info">
                    <span style="font-size: 24px; margin-right: 10px;">ℹ️</span>
                    <span>No performance data available yet. Run verification to generate performance history.</span>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    with tab_parameters:  # Parameter Evolution Tab
        st.markdown("## Parameter Evolution")
        
        # Explanation of module parameters
        st.markdown(
            """
            <div class="card">
                <h3>Adaptive Parameters</h3>
                <p>
                    Each verification module has three key parameters that the AAIC system adapts based on performance:
                </p>
                <ul>
                    <li>
                        <strong>Weight (w)</strong>: The relative importance of the module in the verification process. 
                        Higher weights mean the module has more influence on the final confidence score.
                    </li>
                    <li>
                        <strong>Threshold (θ)</strong>: The minimum confidence level required for early termination. 
                        If a module's confidence exceeds this threshold, verification can stop early at this module.
                    </li>
                    <li>
                        <strong>Alpha (α)</strong>: The balance between the two metrics used in each module. 
                        Controls how much each metric contributes to the module's overall confidence.
                    </li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        param_history_df = st.session_state.aaic.get_parameter_history_df()
        if not param_history_df.empty:
            # Create parameter evolution plots
            st.markdown("### Parameter Evolution Charts")
            
            # Create tabs for different parameter types
            param_tabs = st.tabs(["Weights", "Thresholds", "Alpha Values", "Performance"])
            
            with param_tabs[0]:
                weight_fig = plot_aaic_parameter_evolution_plotly(param_history_df, "Weight")
                if weight_fig:
                    st.plotly_chart(weight_fig, use_container_width=True)
            
            with param_tabs[1]:
                threshold_fig = plot_aaic_parameter_evolution_plotly(param_history_df, "Threshold")
                if threshold_fig:
                    st.plotly_chart(threshold_fig, use_container_width=True)
            
            with param_tabs[2]:
                alpha_fig = plot_aaic_parameter_evolution_plotly(param_history_df, "Alpha")
                if alpha_fig:
                    st.plotly_chart(alpha_fig, use_container_width=True)
            
            with param_tabs[3]:
                performance_fig = plot_aaic_parameter_evolution_plotly(param_history_df, "Performance")
                if performance_fig:
                    st.plotly_chart(performance_fig, use_container_width=True)
            
            # Display current module parameters
            st.markdown("### Current Module Parameters")
            
            # Create parameter cards for each module
            module_cols = st.columns(len(st.session_state.rmmve.modules))
            
            for i, (col, module) in enumerate(zip(module_cols, st.session_state.rmmve.modules)):
                with col:
                    st.markdown(
                        f"""
                        <div class="card">
                            <h3>{module.name}</h3>
                            <div style="margin-bottom: 8px;">
                                <div class="metric-label">Weight (w)</div>
                                <div style="font-size: 1.5rem; font-weight: 600; color: #3b82f6;">{module.weight:.4f}</div>
                            </div>
                            <div style="margin-bottom: 8px;">
                                <div class="metric-label">Threshold (θ)</div>
                                <div style="font-size: 1.5rem; font-weight: 600; color: #f59e0b;">{module.threshold:.4f}</div>
                            </div>
                            <div>
                                <div class="metric-label">Alpha (α)</div>
                                <div style="font-size: 1.5rem; font-weight: 600; color: #8b5cf6;">{module.alpha:.4f}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.markdown(
                """
                <div class="alert-info">
                    <span style="font-size: 24px; margin-right: 10px;">ℹ️</span>
                    <span>No parameter evolution data available yet. Run verification to generate parameter history.</span>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    with tab_history:  # Verification History Tab
        st.markdown("## Verification History")
        
        # Display key metrics at the top
        if st.session_state.verification_count > 0:
            metrics_cols = st.columns(4)
            
            with metrics_cols[0]:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Total Verifications</div>
                        <div class="metric-value">{st.session_state.verification_count}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            if st.session_state.verification_history:
                # Calculate statistics
                history_df = pd.DataFrame([
                    {
                        "Time": datetime.fromtimestamp(record["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
                        "Fact Quality": record["verification_results"].get("fact_quality", "unknown"),
                        "Contains Shift": record["verification_results"].get("contains_shift", "None"),
                        "Total Confidence": round(record["verification_results"]["total_confidence"], 4),
                        "Decision": "✅ Verified" if record['verification_results']['decision'] else "❌ Rejected",
                        "Modules Used": len(record['verification_results']['activated_modules']),
                        "Early Termination": "✓ Yes" if record['verification_results'].get('early_termination') else "✗ No",
                        "Term. Module": record['verification_results'].get('early_termination_module', "None")
                    }
                    for record in st.session_state.verification_history
                ])
                
                verify_rate = len(history_df[history_df["Decision"].str.contains("Verified")]) / len(history_df) * 100
                early_term_rate = len(history_df[history_df["Early Termination"] == "✓ Yes"]) / len(history_df) * 100
                avg_modules = history_df["Modules Used"].mean()
                
                with metrics_cols[1]:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Verification Rate</div>
                            <div class="metric-value">{verify_rate:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                with metrics_cols[2]:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Early Termination Rate</div>
                            <div class="metric-value">{early_term_rate:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                with metrics_cols[3]:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Avg. Modules Used</div>
                            <div class="metric-value">{avg_modules:.2f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Display performance trends
                st.markdown("### Performance Trends")
                history_tabs = st.tabs(["Confidence History", "Quality Distribution", "Early Termination Distribution"])
                
                with history_tabs[0]:
                    # Plot confidence history
                    confidence_fig = plot_verification_history_plotly(history_df)
                    if confidence_fig:
                        st.plotly_chart(confidence_fig, use_container_width=True)
                
                with history_tabs[1]:
                    # Plot quality distribution
                    quality_fig = plot_quality_distribution_plotly(history_df)
                    if quality_fig:
                        st.plotly_chart(quality_fig, use_container_width=True)
                
                with history_tabs[2]:
                    # Plot early termination by module
                    term_fig = plot_early_term_by_module_plotly(history_df)
                    if term_fig:
                        st.plotly_chart(term_fig, use_container_width=True)
                    else:
                        st.info("No early terminations recorded yet.")
                
                # Detailed verification history table
                st.markdown("### Detailed History")
                
                # Add quality color highlighting
                def format_quality(row):
                    quality = row["Fact Quality"]
                    quality_color = QUALITY_COLOR_MAP.get(quality, "#6b7280")
                    quality_name = quality.replace("_", " ").title()
                    return f'<span style="color: {quality_color}; font-weight: bold;">{quality_name}</span>'
                
                def format_decision(row):
                    if "Verified" in row["Decision"]:
                        return '<span class="status-verified">VERIFIED</span>'
                    else:
                        return '<span class="status-rejected">REJECTED</span>'
                
                # Apply formatting
                history_df["Formatted Quality"] = history_df.apply(format_quality, axis=1)
                history_df["Formatted Decision"] = history_df.apply(format_decision, axis=1)
                
                # Display table with formatted columns
                display_cols = ["Time", "Formatted Quality", "Total Confidence", "Formatted Decision", 
                               "Modules Used", "Early Termination", "Term. Module"]
                
                st.markdown(
                    """
                    <style>
                    .dataframe th {
                        text-align: center !important;
                    }
                    </style>
                    """, 
                    unsafe_allow_html=True
                )
                
                st.write(
                    history_df[display_cols].to_html(
                        escape=False, 
                        index=False,
                        classes='dataframe'
                    ), 
                    unsafe_allow_html=True
                )
            else:
                with metrics_cols[1]:
                    st.metric("Verification Rate", "0%")
                with metrics_cols[2]:
                    st.metric("Early Termination Rate", "0%")
                with metrics_cols[3]:
                    st.metric("Avg. Modules Used", "0")
                
                st.info("No verification history details available yet.")
        else:
            st.markdown(
                """
                <div class="alert-info">
                    <span style="font-size: 24px; margin-right: 10px;">ℹ️</span>
                    <span>No verification history yet. Run verification to generate history.</span>
                </div>
                """,
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main() 