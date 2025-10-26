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
    tab_methodology, tab_verification, tab_experiments, tab_aaic, tab_parameters, tab_history = st.tabs([
        "📚 Methodology",
        "💠 Verification Process",
        "🧪 Experimental Evaluation",
        "🔄 AAIC Monitoring",
        "📊 Parameter Evolution",
        "📜 Verification History"
    ])

    with tab_methodology:  # Methodology Tab
        st.markdown("## ATLASky-AI Methodology")

        # Introduction
        st.markdown(
            """
            <div class="card">
                <h3>4D Spatiotemporal Knowledge Graph Verification</h3>
                <p>
                    ATLASky-AI is a novel verification system for 4D Spatiotemporal Knowledge Graphs (STKGs)
                    that combines physics-based constraints with multi-agent verification to detect and prevent:
                </p>
                <ul>
                    <li><strong>Content Hallucination</strong>: Fabricated facts not grounded in reality</li>
                    <li><strong>ST-Inconsistency</strong>: Violations of physical laws (spatial/temporal)</li>
                    <li><strong>Semantic Drift</strong>: Facts that deviate from domain ontology</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        # STKG Formalization
        st.markdown("### STKG Formalization")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
                <div class="card">
                    <h4>Definition 1: 4D STKG</h4>
                    <p>A 4D Spatiotemporal Knowledge Graph is a tuple:</p>
                    <p style="text-align: center; font-size: 1.2rem; color: #3b82f6; font-weight: 600;">
                        G = (V, E, O, T, Ψ)
                    </p>
                    <ul>
                        <li><strong>V</strong>: Versioned entities with immutable attributes and mutable state</li>
                        <li><strong>E</strong>: Directed edges representing relationships</li>
                        <li><strong>O = (C, R<sub>o</sub>, A)</strong>: Domain ontology
                            <ul style="margin-top: 5px;">
                                <li>C: Entity classes</li>
                                <li>R<sub>o</sub>: Relation types</li>
                                <li>A: Attributes</li>
                            </ul>
                        </li>
                        <li><strong>T: (V ∪ E) → ℝ³ × ℝ</strong>: Maps to (x,y,z,t) coordinates</li>
                        <li><strong>Ψ: (V ∪ E) → {0,1}</strong>: Physical consistency predicate</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                """
                <div class="card">
                    <h4>Current Knowledge Graph</h4>
                    <p style="margin-bottom: 10px;">Your active STKG contains:</p>
                    <div style="display: flex; gap: 10px; margin-bottom: 10px;">
                        <div style="flex: 1; background-color: #eff6ff; padding: 10px; border-radius: 5px;">
                            <div style="font-size: 2rem; font-weight: 600; color: #3b82f6; text-align: center;">
                                {entities}
                            </div>
                            <div style="text-align: center; color: #6b7280;">Entities (V)</div>
                        </div>
                        <div style="flex: 1; background-color: #f0fdf4; padding: 10px; border-radius: 5px;">
                            <div style="font-size: 2rem; font-weight: 600; color: #10b981; text-align: center;">
                                {relationships}
                            </div>
                            <div style="text-align: center; color: #6b7280;">Relationships (E)</div>
                        </div>
                    </div>
                    <div style="background-color: #fef3c7; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <div style="font-size: 1.5rem; font-weight: 600; color: #f59e0b; text-align: center;">
                            {classes}
                        </div>
                        <div style="text-align: center; color: #6b7280;">Entity Classes (C)</div>
                    </div>
                    <p style="font-size: 0.9rem; color: #6b7280; margin: 0;">
                        All facts are mapped to spatiotemporal coordinates and verified using physics predicates.
                    </p>
                </div>
                """.format(
                    entities=len(st.session_state.kg.entities),
                    relationships=len(st.session_state.kg.relationships),
                    classes=len(ENTITY_CLASSES)
                ),
                unsafe_allow_html=True
            )

        # Physics Predicates
        st.markdown("### Physics-Based Consistency Predicates")

        pred_col1, pred_col2, pred_col3 = st.columns(3)

        with pred_col1:
            st.markdown(
                """
                <div class="card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                    <h4 style="color: white;">ψ<sub>s</sub>: Spatial Consistency</h4>
                    <p style="margin-bottom: 10px; font-size: 0.95rem;">
                        <strong>Definition 2:</strong> Prevents co-location violations
                    </p>
                    <div style="background-color: rgba(255,255,255,0.2); padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <code style="color: white; font-size: 0.85rem;">
                            ∀d₁,d₂ ∈ D: (e₁=e₂) ∧ (|t₂-t₁| < τ_res)<br>
                            → dist(ℓ₁,ℓ₂) ≤ σ_res
                        </code>
                    </div>
                    <p style="font-size: 0.9rem; margin: 0;">
                        <strong>Checks:</strong> Same entity cannot be at two locations simultaneously
                    </p>
                    <p style="font-size: 0.85rem; margin-top: 5px; opacity: 0.9;">
                        σ_res = {sigma} meters
                    </p>
                </div>
                """.format(sigma=st.session_state.kg.sigma_res),
                unsafe_allow_html=True
            )

        with pred_col2:
            st.markdown(
                """
                <div class="card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white;">
                    <h4 style="color: white;">ψ<sub>t</sub>: Temporal Consistency</h4>
                    <p style="margin-bottom: 10px; font-size: 0.95rem;">
                        <strong>Definition 3:</strong> Enforces velocity constraints
                    </p>
                    <div style="background-color: rgba(255,255,255,0.2); padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <code style="color: white; font-size: 0.85rem;">
                            ∀d₁,d₂ ∈ D: (e₁=e₂) ∧ (t₂>t₁)<br>
                            → |t₂-t₁| ≥ dist(ℓ₁,ℓ₂)/v_max
                        </code>
                    </div>
                    <p style="font-size: 0.9rem; margin: 0;">
                        <strong>Checks:</strong> Travel time must be physically feasible
                    </p>
                    <p style="font-size: 0.85rem; margin-top: 5px; opacity: 0.9;">
                        v_max = 2-15 m/s (mode-dependent)
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with pred_col3:
            st.markdown(
                """
                <div class="card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white;">
                    <h4 style="color: white;">Ψ: Combined Predicate</h4>
                    <p style="margin-bottom: 10px; font-size: 0.95rem;">
                        <strong>Master Consistency:</strong> Both spatial and temporal
                    </p>
                    <div style="background-color: rgba(255,255,255,0.2); padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <code style="color: white; font-size: 0.85rem;">
                            Ψ(d) = ψ<sub>s</sub>(d) ∧ ψ<sub>t</sub>(d)
                        </code>
                    </div>
                    <p style="font-size: 0.9rem; margin: 0;">
                        <strong>Result:</strong> 1 if physically consistent, 0 otherwise
                    </p>
                    <p style="font-size: 0.85rem; margin-top: 5px; opacity: 0.9;">
                        Used by MAV module
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Error Taxonomy
        st.markdown("### Error Taxonomy")

        error_col1, error_col2, error_col3 = st.columns(3)

        with error_col1:
            st.markdown(
                """
                <div class="card">
                    <h4 style="color: #ef4444;">Content Hallucination</h4>
                    <p><strong>Definition 4:</strong> Fabricated facts not in source data</p>
                    <p style="font-size: 0.9rem; color: #6b7280; margin-bottom: 10px;">
                        Facts that cannot be traced back to authoritative sources.
                    </p>
                    <div style="background-color: #fee2e2; padding: 8px; border-radius: 5px;">
                        <p style="margin: 0; font-size: 0.85rem;"><strong>Example:</strong></p>
                        <p style="margin: 5px 0 0 0; font-size: 0.85rem;">
                            "Blade B500 measured at 0.05mm" when no such measurement exists
                        </p>
                    </div>
                    <p style="margin-top: 10px; font-size: 0.9rem;">
                        <strong>Detected by:</strong> External Source Verification (ESV)
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with error_col2:
            st.markdown(
                """
                <div class="card">
                    <h4 style="color: #f59e0b;">ST-Inconsistency</h4>
                    <p><strong>Definition 5:</strong> Violations of physical laws</p>
                    <p style="font-size: 0.9rem; color: #6b7280; margin-bottom: 10px;">
                        Facts that violate spatial (ψ<sub>s</sub>) or temporal (ψ<sub>t</sub>) consistency.
                    </p>
                    <div style="background-color: #fef3c7; padding: 8px; border-radius: 5px;">
                        <p style="margin: 0; font-size: 0.85rem;"><strong>Example:</strong></p>
                        <p style="margin: 5px 0 0 0; font-size: 0.85rem;">
                            Blade at two locations within 1 second (violates ψ<sub>s</sub>)
                        </p>
                    </div>
                    <p style="margin-top: 10px; font-size: 0.9rem;">
                        <strong>Detected by:</strong> Motion-Aware Verification (MAV)
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with error_col3:
            st.markdown(
                """
                <div class="card">
                    <h4 style="color: #8b5cf6;">Semantic Drift</h4>
                    <p><strong>Definition 6:</strong> Deviation from domain ontology</p>
                    <p style="font-size: 0.9rem; color: #6b7280; margin-bottom: 10px;">
                        Facts that don't conform to expected entity classes, relationships, or attributes.
                    </p>
                    <div style="background-color: #f3e8ff; padding: 8px; border-radius: 5px;">
                        <p style="margin: 0; font-size: 0.85rem;"><strong>Example:</strong></p>
                        <p style="margin: 5px 0 0 0; font-size: 0.85rem;">
                            Invalid relationship type or out-of-range attribute values
                        </p>
                    </div>
                    <p style="margin-top: 10px; font-size: 0.9rem;">
                        <strong>Detected by:</strong> Local Ontology Verification (LOV)
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Verification Modules
        st.markdown("### Five-Module Verification Pipeline")

        st.markdown(
            """
            <div class="card">
                <p style="margin-bottom: 15px;">
                    ATLASky-AI uses five specialized verification modules that can terminate early
                    when sufficient confidence is achieved, improving efficiency while maintaining accuracy.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        module_cols = st.columns(5)

        modules_info = [
            {
                "name": "LOV",
                "full_name": "Local Ontology Verification",
                "color": "#3b82f6",
                "targets": "Semantic Drift",
                "metrics": ["Structural Compliance", "Attribute Compliance"],
                "icon": "📚"
            },
            {
                "name": "POV",
                "full_name": "Provenance-Aware Verification",
                "color": "#10b981",
                "targets": "Content Hallucination",
                "metrics": ["Lineage Tracing", "Dependency Validation"],
                "icon": "🔍"
            },
            {
                "name": "MAV",
                "full_name": "Motion-Aware Verification",
                "color": "#f59e0b",
                "targets": "ST-Inconsistency",
                "metrics": ["Temporal-Spatial Validity (ψ<sub>s</sub>, ψ<sub>t</sub>)", "Physical Feasibility"],
                "icon": "⚡"
            },
            {
                "name": "WSV",
                "full_name": "Workflow State Verification",
                "color": "#8b5cf6",
                "targets": "Content Hallucination",
                "metrics": ["State Transition Validity", "Workflow Compliance"],
                "icon": "🔄"
            },
            {
                "name": "ESV",
                "full_name": "External Source Verification",
                "color": "#ef4444",
                "targets": "Content Hallucination",
                "metrics": ["Source Authority", "Cross-Reference Validity"],
                "icon": "🌐"
            }
        ]

        for col, module_info in zip(module_cols, modules_info):
            with col:
                st.markdown(
                    f"""
                    <div class="card" style="border-left: 4px solid {module_info['color']}; min-height: 280px;">
                        <div style="text-align: center; font-size: 2rem; margin-bottom: 10px;">
                            {module_info['icon']}
                        </div>
                        <h4 style="color: {module_info['color']}; text-align: center; margin-bottom: 5px;">
                            {module_info['name']}
                        </h4>
                        <p style="font-size: 0.75rem; text-align: center; color: #6b7280; margin-bottom: 10px;">
                            {module_info['full_name']}
                        </p>
                        <div style="background-color: #f9fafb; padding: 8px; border-radius: 5px; margin-bottom: 10px;">
                            <p style="margin: 0; font-size: 0.8rem;"><strong>Targets:</strong></p>
                            <p style="margin: 3px 0 0 0; font-size: 0.8rem; color: {module_info['color']};">
                                {module_info['targets']}
                            </p>
                        </div>
                        <p style="margin: 0; font-size: 0.75rem;"><strong>Metrics:</strong></p>
                        <ul style="margin: 5px 0 0 0; padding-left: 20px; font-size: 0.75rem;">
                            {''.join([f'<li>{metric}</li>' for metric in module_info['metrics']])}
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # AAIC Adaptation
        st.markdown("### Autonomous Adaptive Intelligence Cycle (AAIC)")

        aaic_col1, aaic_col2 = st.columns([1, 1])

        with aaic_col1:
            st.markdown(
                """
                <div class="card">
                    <h4>CGR-CUSUM Performance Monitoring</h4>
                    <p style="font-size: 0.9rem; margin-bottom: 10px;">
                        AAIC uses the CGR-CUSUM algorithm to detect performance shifts in real-time:
                    </p>
                    <div style="background-color: #eff6ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <code style="font-size: 0.85rem;">
                            G<sub>i</sub>(n) = max(0, G<sub>i</sub>(n-1) + L<sub>i</sub>(n) - k)
                        </code>
                    </div>
                    <p style="font-size: 0.85rem; color: #6b7280; margin-bottom: 10px;">
                        When G<sub>i</sub>(n) > h, a performance shift is detected and parameters are updated.
                    </p>
                    <ul style="font-size: 0.85rem;">
                        <li><strong>h</strong>: Detection threshold (default: {h})</li>
                        <li><strong>k</strong>: Allowance parameter (default: {k})</li>
                        <li><strong>L<sub>i</sub>(n)</strong>: Log-likelihood ratio of performance</li>
                    </ul>
                </div>
                """.format(h=st.session_state.aaic.h, k=st.session_state.aaic.k),
                unsafe_allow_html=True
            )

        with aaic_col2:
            st.markdown(
                """
                <div class="card">
                    <h4>Adaptive Parameter Updates</h4>
                    <p style="font-size: 0.9rem; margin-bottom: 10px;">
                        When shifts are detected, AAIC updates three key parameters:
                    </p>
                    <div style="margin-bottom: 8px;">
                        <div style="background-color: #eff6ff; padding: 8px; border-radius: 5px;">
                            <p style="margin: 0; font-size: 0.8rem;"><strong>Weight (w)</strong> - Equation 12:</p>
                            <code style="font-size: 0.75rem;">w<sub>i</sub> ← w<sub>i</sub> × exp[-γ·G<sub>i</sub>(t)]</code>
                            <p style="margin: 3px 0 0 0; font-size: 0.75rem; color: #6b7280;">γ = {gamma}</p>
                        </div>
                    </div>
                    <div style="margin-bottom: 8px;">
                        <div style="background-color: #fef3c7; padding: 8px; border-radius: 5px;">
                            <p style="margin: 0; font-size: 0.8rem;"><strong>Threshold (θ)</strong> - Equation 13:</p>
                            <code style="font-size: 0.75rem;">θ<sub>i</sub> ← θ<sub>i</sub> + η·sign(FPR - FNR)</code>
                            <p style="margin: 3px 0 0 0; font-size: 0.75rem; color: #6b7280;">η = {eta}</p>
                        </div>
                    </div>
                    <div>
                        <div style="background-color: #f3e8ff; padding: 8px; border-radius: 5px;">
                            <p style="margin: 0; font-size: 0.8rem;"><strong>Alpha (α)</strong> - Equation 14:</p>
                            <code style="font-size: 0.75rem;">α<sub>i</sub> ← α<sub>i</sub> + η'·∂L<sub>i</sub>/∂α<sub>i</sub></code>
                            <p style="margin: 3px 0 0 0; font-size: 0.75rem; color: #6b7280;">η' = {eta_prime}</p>
                        </div>
                    </div>
                </div>
                """.format(gamma=st.session_state.aaic.gamma, eta=st.session_state.aaic.eta,
                          eta_prime=st.session_state.aaic.eta_prime),
                unsafe_allow_html=True
            )

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

    with tab_experiments:  # Experimental Evaluation Tab
        st.markdown("## Experimental Evaluation")

        # Introduction
        st.markdown(
            """
            <div class="card">
                <h3>Testing ATLASky-AI on Different Dataset Types</h3>
                <p>
                    This section demonstrates how ATLASky-AI performs on facts from different domains,
                    each with unique characteristics and error patterns.
                </p>
                <p style="margin-bottom: 0;">
                    The experimental framework measures standard metrics: <strong>Precision</strong>,
                    <strong>Recall</strong>, <strong>F1-Score</strong>, and <strong>FPR</strong> (False Positive Rate).
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Dataset Type Selection
        st.markdown("### Select Dataset Type")

        dataset_types = {
            "manufacturing": {
                "name": "Manufacturing/Aerospace",
                "icon": "🏭",
                "color": "#3b82f6",
                "challenge": "Micro-tolerance precision (±0.1mm)",
                "test": "Spatial consistency ψs, measurement validation",
                "errors": ["Hallucinated tolerance values", "Impossible measurements", "Spatial violations (ψs)"],
                "example": "Turbine blade deviation: 0.023mm at (10.5, 20.3, 150.2)",
                "expected": {"P": 0.94, "R": 0.91, "F1": 0.92, "FPR": 0.026}
            },
            "aviation": {
                "name": "Aviation Safety",
                "icon": "✈️",
                "color": "#10b981",
                "challenge": "Temporal consistency and causal relationships",
                "test": "Temporal consistency ψt, causal reasoning",
                "errors": ["Temporal impossibility", "Causal violations", "Velocity violations (ψt)"],
                "example": "Aircraft descended 10,000 ft in 2 minutes at location (34.05, -118.25)",
                "expected": {"P": 0.93, "R": 0.94, "F1": 0.93, "FPR": 0.032}
            },
            "cad": {
                "name": "CAD Assembly",
                "icon": "📐",
                "color": "#f59e0b",
                "challenge": "3D geometric reasoning",
                "test": "Spatial consistency ψs, interference detection",
                "errors": ["Geometric interference", "Invalid transformations", "Spatial violations (ψs)"],
                "example": "Part P-123 at position (5.2, 3.1, 7.8) interferes with P-456",
                "expected": {"P": 0.96, "R": 0.93, "F1": 0.94, "FPR": 0.029}
            },
            "healthcare": {
                "name": "Clinical/Healthcare",
                "icon": "🏥",
                "color": "#8b5cf6",
                "challenge": "Clinical workflow compliance",
                "test": "Temporal consistency ψt, protocol validation",
                "errors": ["Protocol violations", "Temporal impossibility", "Spatial violations (ψs)"],
                "example": "Patient transferred MICU→OR in 10 minutes",
                "expected": {"P": 0.95, "R": 0.95, "F1": 0.95, "FPR": 0.041}
            }
        }

        # Display dataset cards
        dataset_cols = st.columns(4)

        selected_dataset = None
        for col, (dataset_key, dataset_info) in zip(dataset_cols, dataset_types.items()):
            with col:
                if st.button(
                    f"{dataset_info['icon']} {dataset_info['name']}",
                    key=f"dataset_{dataset_key}",
                    use_container_width=True
                ):
                    st.session_state['selected_dataset'] = dataset_key

        # Initialize selected dataset
        if 'selected_dataset' not in st.session_state:
            st.session_state['selected_dataset'] = 'manufacturing'

        selected_key = st.session_state['selected_dataset']
        dataset_info = dataset_types[selected_key]

        # Display selected dataset information
        st.markdown("---")
        st.markdown(f"## {dataset_info['icon']} {dataset_info['name']}")

        info_col1, info_col2 = st.columns([1, 1])

        with info_col1:
            st.markdown(
                f"""
                <div class="card" style="border-left: 4px solid {dataset_info['color']};">
                    <h4>Dataset Characteristics</h4>
                    <div style="margin-bottom: 10px;">
                        <p style="margin: 0; font-size: 0.85rem; color: #6b7280;">Challenge:</p>
                        <p style="margin: 3px 0 0 0; font-size: 1rem; font-weight: 500;">{dataset_info['challenge']}</p>
                    </div>
                    <div style="margin-bottom: 10px;">
                        <p style="margin: 0; font-size: 0.85rem; color: #6b7280;">Primary Test:</p>
                        <p style="margin: 3px 0 0 0; font-size: 1rem; font-weight: 500;">{dataset_info['test']}</p>
                    </div>
                    <div>
                        <p style="margin: 0; font-size: 0.85rem; color: #6b7280;">Error Types:</p>
                        <ul style="margin: 5px 0 0 0; padding-left: 20px;">
                            {''.join([f'<li style="font-size: 0.9rem;">{error}</li>' for error in dataset_info['errors']])}
                        </ul>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="card">
                    <h4>Example Fact</h4>
                    <div style="background-color: #f9fafb; padding: 10px; border-radius: 5px; border-left: 3px solid {dataset_info['color']};">
                        <code style="font-size: 0.9rem;">{dataset_info['example']}</code>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with info_col2:
            st.markdown(
                f"""
                <div class="card" style="background: linear-gradient(135deg, {dataset_info['color']} 0%, {dataset_info['color']}dd 100%); color: white;">
                    <h4 style="color: white;">Expected Performance</h4>
                    <p style="font-size: 0.9rem; opacity: 0.95; margin-bottom: 15px;">
                        Based on experimental evaluation with ground truth labels
                    </p>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                        <div style="background-color: rgba(255,255,255,0.2); padding: 10px; border-radius: 5px;">
                            <div style="font-size: 0.8rem; opacity: 0.9;">Precision</div>
                            <div style="font-size: 1.8rem; font-weight: 600;">{dataset_info['expected']['P']:.2f}</div>
                        </div>
                        <div style="background-color: rgba(255,255,255,0.2); padding: 10px; border-radius: 5px;">
                            <div style="font-size: 0.8rem; opacity: 0.9;">Recall</div>
                            <div style="font-size: 1.8rem; font-weight: 600;">{dataset_info['expected']['R']:.2f}</div>
                        </div>
                        <div style="background-color: rgba(255,255,255,0.2); padding: 10px; border-radius: 5px;">
                            <div style="font-size: 0.8rem; opacity: 0.9;">F1-Score</div>
                            <div style="font-size: 1.8rem; font-weight: 600;">{dataset_info['expected']['F1']:.2f}</div>
                        </div>
                        <div style="background-color: rgba(255,255,255,0.2); padding: 10px; border-radius: 5px;">
                            <div style="font-size: 0.8rem; opacity: 0.9;">FPR</div>
                            <div style="font-size: 1.8rem; font-weight: 600;">{dataset_info['expected']['FPR']:.3f}</div>
                        </div>
                    </div>
                    <p style="font-size: 0.8rem; margin-top: 10px; opacity: 0.9;">
                        FPR = {dataset_info['expected']['FPR']*100:.1f}% false alarms
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Live Demo Section
        st.markdown("---")
        st.markdown("### Live Demonstration")

        st.markdown(
            """
            <div class="alert-info">
                Generate a fact from the selected dataset type and run verification to see
                how ATLASky-AI detects different error patterns.
            </div>
            """,
            unsafe_allow_html=True
        )

        demo_col1, demo_col2 = st.columns(2)

        with demo_col1:
            # Number of facts to test
            num_test_facts = st.slider("Number of facts to test", 10, 100, 50, 10)

            if st.button(f"Run Demo on {dataset_info['name']}", key="run_demo", use_container_width=True):
                # Import the dataset generator
                try:
                    if selected_key == "manufacturing":
                        from experiments.datasets.manufacturing_data import generate_manufacturing_facts as generate_facts
                    elif selected_key == "aviation":
                        from experiments.datasets.aviation_data import generate_aviation_facts as generate_facts
                    elif selected_key == "cad":
                        from experiments.datasets.cad_data import generate_cad_facts as generate_facts
                    elif selected_key == "healthcare":
                        from experiments.datasets.healthcare_data import generate_healthcare_facts as generate_facts

                    # Generate facts with ground truth labels
                    with st.spinner(f"Generating {num_test_facts} {dataset_info['name']} facts..."):
                        facts, labels = generate_facts(num_test_facts)

                    # Run verification on all facts
                    from experiments.metrics.evaluation import VerificationMetrics
                    metrics_calc = VerificationMetrics()

                    verification_decisions = []

                    progress_bar = st.progress(0)
                    for i, (fact, is_correct) in enumerate(zip(facts, labels)):
                        # Run verification
                        result = st.session_state.rmmve.verify(fact, st.session_state.kg, None)
                        verification_decisions.append(result['decision'])

                        # Update metrics
                        metrics_calc.update([is_correct], [result['decision']])

                        # Update progress
                        progress_bar.progress((i + 1) / num_test_facts)

                    # Get final metrics
                    metrics = metrics_calc.compute_metrics()

                    # Store in session state
                    st.session_state['demo_results'] = {
                        'dataset': selected_key,
                        'metrics': metrics,
                        'num_facts': num_test_facts
                    }

                    st.success(f"✅ Completed verification of {num_test_facts} facts!")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error running demo: {str(e)}")

        with demo_col2:
            # Display results if available
            if 'demo_results' in st.session_state and st.session_state['demo_results']['dataset'] == selected_key:
                results = st.session_state['demo_results']
                metrics = results['metrics']

                st.markdown(
                    f"""
                    <div class="card">
                        <h4>Demo Results ({results['num_facts']} facts)</h4>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px;">
                            <div style="background-color: #eff6ff; padding: 10px; border-radius: 5px;">
                                <div style="font-size: 0.8rem; color: #6b7280;">Precision</div>
                                <div style="font-size: 1.5rem; font-weight: 600; color: #3b82f6;">{metrics['precision']:.3f}</div>
                            </div>
                            <div style="background-color: #f0fdf4; padding: 10px; border-radius: 5px;">
                                <div style="font-size: 0.8rem; color: #6b7280;">Recall</div>
                                <div style="font-size: 1.5rem; font-weight: 600; color: #10b981;">{metrics['recall']:.3f}</div>
                            </div>
                            <div style="background-color: #fef3c7; padding: 10px; border-radius: 5px;">
                                <div style="font-size: 0.8rem; color: #6b7280;">F1-Score</div>
                                <div style="font-size: 1.5rem; font-weight: 600; color: #f59e0b;">{metrics['f1']:.3f}</div>
                            </div>
                            <div style="background-color: #fee2e2; padding: 10px; border-radius: 5px;">
                                <div style="font-size: 0.8rem; color: #6b7280;">FPR</div>
                                <div style="font-size: 1.5rem; font-weight: 600; color: #ef4444;">{metrics['fpr']:.3f}</div>
                            </div>
                        </div>
                        <div style="margin-top: 15px; padding: 10px; background-color: #f9fafb; border-radius: 5px;">
                            <p style="margin: 0; font-size: 0.85rem;"><strong>Confusion Matrix:</strong></p>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 5px; margin-top: 5px;">
                                <div style="font-size: 0.8rem;">TP: {metrics['tp']}</div>
                                <div style="font-size: 0.8rem;">TN: {metrics['tn']}</div>
                                <div style="font-size: 0.8rem;">FP: {metrics['fp']}</div>
                                <div style="font-size: 0.8rem;">FN: {metrics['fn']}</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div class="alert-info">
                        <span style="font-size: 18px; margin-right: 5px;">ℹ️</span>
                        Click "Run Demo" to see verification performance on this dataset type
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Metrics Explanation
        st.markdown("---")
        st.markdown("### Performance Metrics Explained")

        metrics_cols = st.columns(4)

        with metrics_cols[0]:
            st.markdown(
                """
                <div class="card">
                    <h4 style="color: #3b82f6;">Precision</h4>
                    <p style="font-size: 0.85rem; color: #6b7280;">TP/(TP+FP)</p>
                    <p style="font-size: 0.8rem;">
                        Fraction of rejected facts that were truly incorrect.
                        High precision = few false alarms.
                    </p>
                    <p style="font-size: 0.75rem; margin: 0; color: #6b7280;">
                        <strong>Target:</strong> >0.90 for production
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with metrics_cols[1]:
            st.markdown(
                """
                <div class="card">
                    <h4 style="color: #10b981;">Recall</h4>
                    <p style="font-size: 0.85rem; color: #6b7280;">TP/(TP+FN)</p>
                    <p style="font-size: 0.8rem;">
                        Fraction of incorrect facts successfully caught.
                        High recall = catches most errors.
                    </p>
                    <p style="font-size: 0.75rem; margin: 0; color: #6b7280;">
                        <strong>Target:</strong> >0.90 for safety-critical
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with metrics_cols[2]:
            st.markdown(
                """
                <div class="card">
                    <h4 style="color: #f59e0b;">F1-Score</h4>
                    <p style="font-size: 0.85rem; color: #6b7280;">2·P·R/(P+R)</p>
                    <p style="font-size: 0.8rem;">
                        Harmonic mean balancing precision and recall.
                        Overall performance indicator.
                    </p>
                    <p style="font-size: 0.75rem; margin: 0; color: #6b7280;">
                        <strong>Target:</strong> >0.90 for deployment
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with metrics_cols[3]:
            st.markdown(
                """
                <div class="card">
                    <h4 style="color: #ef4444;">FPR</h4>
                    <p style="font-size: 0.85rem; color: #6b7280;">FP/(FP+TN)</p>
                    <p style="font-size: 0.8rem;">
                        Fraction of correct facts incorrectly rejected.
                        Low FPR = minimal review burden.
                    </p>
                    <p style="font-size: 0.75rem; margin: 0; color: #6b7280;">
                        <strong>Target:</strong> <5% for user trust
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Full Experiment Runner Info
        st.markdown("---")
        st.markdown("### Running Full Experiments")

        st.markdown(
            """
            <div class="card">
                <h4>Command-Line Experiment Runner</h4>
                <p style="margin-bottom: 10px;">
                    For comprehensive testing, use the experiment runner from the command line:
                </p>
                <div style="background-color: #1f2937; padding: 12px; border-radius: 5px; margin-bottom: 10px;">
                    <code style="color: #10b981; font-size: 0.9rem;">
                        # Test on specific dataset<br>
                        python3 experiments/run_experiments.py --dataset manufacturing --num-facts 100<br>
                        <br>
                        # Test on all dataset types<br>
                        python3 experiments/run_experiments.py --all --num-facts 100<br>
                        <br>
                        # Quick demo (works immediately)<br>
                        python3 experiments/quick_demo.py
                    </code>
                </div>
                <p style="font-size: 0.9rem; margin: 0;">
                    Results are saved to <code>experiments/results/</code> as JSON files for analysis.
                </p>
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