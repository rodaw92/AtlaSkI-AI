import streamlit as st
import pandas as pd
import numpy as np
import time
import random
import os
from datetime import datetime, timedelta

# Import from our modules
from models.constants import (
    ENTITY_CLASSES, RELATIONSHIP_TYPES, BLADE_FEATURES, BLADE_COMPONENTS,
    ENGINE_SETS, MEASUREMENT_IDS, SURFACE_SIDES, QUALITY_COLOR_MAP
)
from models.knowledge_graph import KnowledgeGraph, create_sample_knowledge_graph
from verification.rmmve import RMMVeProcess
from verification.domain_adapter import DomainAdapter
from aaic.aaic import AAIC
from data.generators import generate_candidate_fact, generate_candidate_fact_with_quality
from data.preprocessing import PREPROCESSOR, DataPreprocessor
from data.llm_extraction import LLM_EXTRACTOR, LLMExtractor
from data.quality_based_generator import generate_raw_text_with_quality
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
    page_icon="🗂️",
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
    
    # Initialize domain adapter
    if "domain_adapter" not in st.session_state:
        st.session_state.domain_adapter = DomainAdapter(config_directory="domains")
        st.session_state.current_domain_name = None
        st.session_state.selected_domain_for_generation = "aerospace"  # Default domain for generation
    
    
    # Force recreation of rmmve object to pick up signature changes (version 2)
    if "rmmve" not in st.session_state or getattr(st.session_state, 'rmmve_version', 0) < 2:
        st.session_state.rmmve = RMMVeProcess(global_threshold=0.65)
        st.session_state.rmmve_version = 2
    
    if "aaic" not in st.session_state or getattr(st.session_state, 'aaic_version', 0) < 2:
        st.session_state.aaic = AAIC(st.session_state.rmmve)
        st.session_state.aaic_version = 2
    
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
        <div class="header-subtitle">4D Spatiotemporal Knowledge Graph Verification System</div>
    </div>
    """, 
    unsafe_allow_html=True
)
    
    # ----- Sidebar -----
    with st.sidebar:
        st.markdown("<h3>Control Panel</h3>", unsafe_allow_html=True)

        # Show current global domain selection for testing - MOVED TO TOP
        st.markdown("### 🧪 Testing Domain")
        if st.session_state.current_domain_name:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px; border-radius: 8px; margin-bottom: 15px;">
                <div style="font-size: 14px; margin-bottom: 5px;">🌐 <strong>Testing Domain: {st.session_state.current_domain_name.upper()}</strong></div>
                <div style="font-size: 12px; opacity: 0.9;">
                    Generated facts will use {st.session_state.current_domain_name} domain-specific content and issues
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #f3f4f6; padding: 12px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #64748b;">
                <div style="font-size: 14px; margin-bottom: 5px;">🧪 <strong>Testing Domain: Default (Aerospace)</strong></div>
                <div style="font-size: 12px; color: #64748b;">
                    Select a domain above to use domain-specific testing content
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Testing Controls")

        # LLM Mode Selection
        use_real_llm = st.checkbox(
            "🤖 Use Real OpenAI API",
            value=False,
            help="Enable to use actual GPT-4o for extraction (requires API key). Leave unchecked to use simulation."
        )
        
        if use_real_llm:
            api_key_input = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Enter your OpenAI API key or set OPENAI_API_KEY environment variable"
            )
            if api_key_input:
                st.session_state.openai_api_key = api_key_input
                st.success("✅ API key configured!")
            elif os.environ.get("OPENAI_API_KEY"):
                st.info("✅ Using API key from environment variable")
            else:
                st.warning("⚠️ No API key provided. Will fallback to simulation if API call fails.")
        
        # Allow manual selection of quality level for testing
        quality_options = [
            "high_quality", "medium_quality", "spatial_issue", 
            "external_ref", "semantic_issue", "low_quality"
        ]
        
        quality_level = st.selectbox(
            "Fact Quality Level",
            options=quality_options,
            index=quality_options.index(st.session_state.fact_quality) if hasattr(st.session_state, 'fact_quality') else 0,
            help="Choose quality level for testing different verification paths"
        )
        
        if st.button("🎲 Generate Test Fact", key="gen_new_fact"):
            # Store selected quality and use global domain selection
            st.session_state.fact_quality = quality_level
            st.session_state.test_domain_selected = st.session_state.selected_domain_for_generation

            # Generate honest raw text with quality-based characteristics using current global domain
            raw_text, expected_attrs = generate_raw_text_with_quality(
                quality_level,
                st.session_state.current_domain_name or st.session_state.selected_domain_for_generation
            )
            
            # Store raw text for Stage 1
            st.session_state.generated_raw_text = raw_text
            st.session_state.expected_attributes = expected_attrs
            
            # Automatically run all 3 stages
            with st.spinner("Running 3-stage pipeline..."):
                # Stage 1: Data Preprocessing
                st.session_state.preprocessed_data = PREPROCESSOR.preprocess_multimodal_data(
                    raw_docs=[{"text": raw_text, "domain": test_domain}]
                )
                
                # Stage 2: LLM Extraction
                # Use API-enabled extractor if user enabled it
                if use_real_llm and "openai_api_key" in st.session_state:
                    extractor = LLMExtractor(api_key=st.session_state.openai_api_key, use_api=True)
                else:
                    extractor = LLM_EXTRACTOR  # Use default simulation
                
                st.session_state.extracted_facts = extractor.extract_knowledge(
                    preprocessed_data=st.session_state.preprocessed_data
                )
                
                # Set candidate fact to first extracted fact
                if st.session_state.extracted_facts:
                    st.session_state.candidate_fact = st.session_state.extracted_facts[0]
                else:
                    st.session_state.candidate_fact = None
            
            st.rerun()
        
        # Global Domain Selection - Affects ALL tabs and operations
        st.markdown("### 🌐 Global Domain Selection")
        st.markdown("**This domain selection affects ALL tabs and operations throughout the application**")

        available_domains = st.session_state.domain_adapter.list_domains()

        if available_domains:
            domain_options = ["None (Default)"] + available_domains
            current_index = 0
            if st.session_state.current_domain_name:
                try:
                    current_index = available_domains.index(st.session_state.current_domain_name) + 1
                except ValueError:
                    current_index = 0

            global_domain_selection = st.selectbox(
                "Active Domain (Global)",
                options=domain_options,
                index=current_index,
                key="global_domain_selector",
                help="Select a domain configuration - affects data generation, verification, and all tabs"
            )

            # Apply domain change immediately
            if global_domain_selection != "None (Default)":
                if global_domain_selection != st.session_state.current_domain_name:
                    try:
                        # Load and apply domain configuration to all systems
                        config = st.session_state.domain_adapter.load_domain(global_domain_selection)
                        st.session_state.domain_adapter.apply_to_rmmve(st.session_state.rmmve, config)
                        st.session_state.domain_adapter.apply_to_knowledge_graph(st.session_state.kg, config)
                        st.session_state.current_domain_name = global_domain_selection

                        # Set domain for data generation
                        st.session_state.selected_domain_for_generation = global_domain_selection

                        st.success(f"✅ Applied {global_domain_selection} domain globally")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error loading domain: {str(e)}")
                else:
                    st.info(f"📊 Active Domain: **{global_domain_selection}**")
                    st.session_state.selected_domain_for_generation = global_domain_selection
            else:
                if st.session_state.current_domain_name is not None:
                    st.session_state.current_domain_name = None
                    st.session_state.selected_domain_for_generation = "aerospace"  # default fallback
                    st.info("ℹ️ Using default parameters (aerospace domain for generation)")
                else:
                    st.session_state.selected_domain_for_generation = "aerospace"

            # Show current domain status
            if st.session_state.current_domain_name:
                st.markdown(f"""
                <div style="background: #f0f8ff; padding: 10px; border-radius: 6px; margin-top: 10px; border-left: 4px solid #3b82f6;">
                    <strong>🌐 Current Global Domain: {st.session_state.current_domain_name.upper()}</strong><br>
                    <small>• Verification parameters active<br>
                    • Data generation uses {st.session_state.current_domain_name} domain<br>
                    • All tabs show {st.session_state.current_domain_name} configuration</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No domain configurations found in /domains directory")
            st.session_state.selected_domain_for_generation = "aerospace"
        
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
            
            # Keep track of shifts detected and decisions
            shifts_detected = 0
            accept_count = 0
            review_count = 0
            reject_count = 0
            
            for i in range(batch_size):
                try:
                    # Update progress
                    progress_bar.progress((i + 1) / batch_size)
                    
                    # For forced shifts, periodically generate a fact with a shift
                    # to ensure AAIC parameter adjustments are triggered
                    introduce_shift = st.session_state.introduce_shift
                    if force_shifts and i % 3 == 0:  # Every 3rd iteration
                        introduce_shift = True
                    
                    # Generate quality level for this iteration (use user-selected domain)
                    import random
                    quality_levels = ["high_quality", "medium_quality", "low_quality"]
                    quality = random.choice(quality_levels)
                    # Use user-selected domain from sidebar
                    domain = st.session_state.test_domain
                    
                    # Run through the 3-stage pipeline
                    # Stage 1: Generate raw text
                    raw_text, expected_attrs = generate_raw_text_with_quality(quality, domain)
                    
                    # Stage 2: Preprocess
                    preprocessed_data = PREPROCESSOR.preprocess_multimodal_data(
                        raw_docs=[{"text": raw_text, "domain": domain}]
                    )
                    
                    # Stage 3: LLM Extract
                    # Use API-enabled extractor if user enabled it
                    if use_real_llm and "openai_api_key" in st.session_state:
                        extractor = LLMExtractor(api_key=st.session_state.openai_api_key, use_api=True)
                    else:
                        extractor = LLM_EXTRACTOR  # Use default simulation
                    
                    extracted_facts = extractor.extract_knowledge(
                        preprocessed_data=preprocessed_data
                    )
                    
                    # Get first extracted fact
                    if not extracted_facts:
                        st.warning(f"Iteration {i+1}: No facts extracted from text, skipping...")
                        continue  # Skip if no facts extracted
                    
                    fact = extracted_facts[0]
                
                except Exception as e:
                    st.error(f"Error processing fact {i+1}: {str(e)}")
                    continue
                
                # Get LLM confidence from fact, but align with selected quality level
                quality_confidence_map = {
                    "high_quality": 0.9,
                    "medium_quality": 0.8,
                    "low_quality": 0.3,
                    # Optional extensions
                    "spatial_issue": 0.6,
                    "external_ref": 0.75,
                    "semantic_issue": 0.7,
                }
                llm_confidence = quality_confidence_map.get(quality, fact.get("llm_confidence", 0.8))

                # Run verification with LLM confidence
                verification_results = st.session_state.rmmve.verify(
                    fact, 
                    st.session_state.kg, 
                    quality,
                    llm_confidence
                )
                
                # Add LLM confidence to results
                verification_results["llm_confidence"] = llm_confidence
                
                # Implement three-way decision rule
                total_confidence = verification_results["total_confidence"]
                global_threshold = st.session_state.rmmve.global_threshold
                epsilon = 0.1  # Review margin
                
                if total_confidence >= global_threshold:
                    decision = "Accept"
                    decision_color = "#10b981"
                    accept_count += 1
                elif total_confidence >= global_threshold - epsilon:
                    decision = "Review"
                    decision_color = "#f59e0b"
                    review_count += 1
                else:
                    decision = "Reject"
                    decision_color = "#ef4444"
                    reject_count += 1
                
                verification_results["decision"] = decision
                verification_results["decision_color"] = decision_color
                
                # Add accepted facts to the knowledge graph
                if decision == "Accept":
                    try:
                        # Extract entity information from fact
                        subject_id = fact.get("subject_entity_id")
                        object_id = fact.get("object_entity_id")
                        relationship_type = fact.get("relationship_type")
                        
                        # Add subject entity if not exists
                        if subject_id and subject_id not in st.session_state.kg.entities:
                            entity_class = fact.get("entity_class", "PhysicalEntity")
                            st.session_state.kg.add_entity(
                                entity_id=subject_id,
                                entity_class=entity_class,
                                attributes=fact.get("attributes", {}),
                                spatiotemporal=fact.get("spatiotemporal", {}),
                                validate=False
                            )
                        
                        # Add object entity if not exists
                        if object_id and object_id not in st.session_state.kg.entities:
                            object_class = fact.get("object_entity_class", "Entity")
                            st.session_state.kg.add_entity(
                                entity_id=object_id,
                                entity_class=object_class,
                                attributes={},
                                spatiotemporal={},
                                validate=False
                            )
                        
                        # Add relationship
                        if subject_id and object_id and relationship_type:
                            rel_id = f"R_{len(st.session_state.kg.relationships)}_{subject_id}_{relationship_type}"
                            st.session_state.kg.add_relationship(
                                rel_id=rel_id,
                                subject_id=subject_id,
                                relation_type=relationship_type,
                                object_id=object_id,
                                validate=False
                            )
                        
                        verification_results["added_to_kg"] = True
                    except Exception as e:
                        verification_results["added_to_kg"] = False
                        verification_results["kg_error"] = str(e)
                else:
                    verification_results["added_to_kg"] = False
                
                # Record verification
                verification_record = {
                    "timestamp": time.time(),
                    "candidate_fact": fact,
                    "llm_confidence": llm_confidence,
                    "verification_results": verification_results,
                    "fact_quality": quality,
                    "fact_source": "Batch Generator"
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
            
            # Clear progress bar
            progress_bar.empty()
            
            # Show comprehensive results
            st.success(f"✅ Batch completed: {accept_count} Accepted, {review_count} Review, {reject_count} Rejected")
            if shifts_detected > 0:
                st.info(f"🔄 {shifts_detected} performance shifts detected and parameters adjusted by AAIC")
            if accept_count > 0:
                st.info(f"📊 {accept_count} facts integrated into STKG")
            
            # Show detailed summary in a nice box
            st.markdown(f"""
            <div style='background-color: #dbeafe; padding: 15px; border-radius: 8px; border-left: 4px solid #3b82f6;'>
                <h4 style='margin-top: 0; color: #1e40af;'>📊 Batch Processing Complete</h4>
                <p><strong>Total Verifications:</strong> {batch_size}</p>
                <p><strong>History Count:</strong> {len(st.session_state.verification_history)} total verifications recorded</p>
                <p style='margin-bottom: 0;'><strong>✨ Check the tabs above to view:</strong></p>
                <ul style='margin-top: 5px; margin-bottom: 0;'>
                    <li><strong>📜 Verification History</strong> - See all {batch_size} verifications with decisions</li>
                    <li><strong>🔄 AAIC Monitoring</strong> - View CGR-CUSUM and performance shifts</li>
                    <li><strong>📊 Parameter Evolution</strong> - Track weight/threshold changes</li>
                    <li><strong>🗂️ STKG Structure</strong> - See {accept_count} accepted facts in knowledge graph</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # ----- Main Content Area -----
    
    # Create tabs for different views
    tab_methodology, tab_domain, tab_stkg, tab_verification, tab_aaic, tab_parameters, tab_history = st.tabs([
        "📚 Methodology",
        "🌐 Domain Config",
        "🗂️ STKG Structure",
        "💠 Verification Process",
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
    
    with tab_domain:  # Domain Configuration Tab
        st.markdown("## 🌐 Domain Configuration")

        # Show current global domain status
        if st.session_state.current_domain_name:
            st.markdown(f"""
            <div class="card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; margin-bottom: 20px;">
                <h4 style="margin: 0 0 10px 0; color: white;">🌐 Active Global Domain: {st.session_state.current_domain_name.upper()}</h4>
                <p style="margin: 0; font-size: 14px; opacity: 0.9;">
                    This tab shows the complete configuration for the currently active domain.
                    Change the global domain in the sidebar to view different configurations.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ No domain configuration active. Select a domain from the sidebar to view its configuration.")

        st.markdown("""
        <div class="card">
            <h3>Domain Adaptation Protocol</h3>
            <p>
                ATLASky-AI is domain-agnostic by design. The verification algorithms remain unchanged—only
                the <strong>configuration parameters</strong> adapt to each domain. This tab shows the five
                components required for domain adaptation:
            </p>
            <ol>
                <li><strong>Domain Ontology (O)</strong>: Entity classes, relation types, and attributes</li>
                <li><strong>Industry Standards (M₂)</strong>: Compliance frameworks and protocols</li>
                <li><strong>Physical Constraints (M₃)</strong>: Velocity limits and facility geometry</li>
                <li><strong>Source Credibility (M₄)</strong>: Credibility weights for authoritative sources</li>
                <li><strong>Domain Embeddings (M₅)</strong>: Trained embeddings on historical facts</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        # Show current domain configuration
        if st.session_state.current_domain_name:
            config = st.session_state.domain_adapter.current_domain
            
            st.markdown(f"### Active Domain: **{config.domain_name.upper()}**")
            st.info(f"📝 {config.domain_description}")
            
            # Component tabs
            comp_tabs = st.tabs(["📋 Ontology", "📜 Standards", "⚙️ Physics", "🌐 Credibility", "🔮 Embeddings", "🎛️ Parameters"])
            
            with comp_tabs[0]:  # Ontology
                st.markdown("#### Domain Ontology")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Entity Classes", len(config.ontology.entity_classes))
                    if config.ontology.entity_classes:
                        st.markdown("**Entity Classes:**")
                        entity_df = pd.DataFrame(config.ontology.entity_classes)
                        st.dataframe(entity_df, use_container_width=True)
                
                with col2:
                    st.metric("Relationship Types", len(config.ontology.relationship_types))
                    if config.ontology.relationship_types:
                        st.markdown("**Relationship Types:**")
                        rel_df = pd.DataFrame(config.ontology.relationship_types)
                        st.dataframe(rel_df, use_container_width=True)
            
            with comp_tabs[1]:  # Standards
                st.markdown("#### Industry Standards")
                
                if config.standards.standards:
                    st.markdown("**Compliance Frameworks:**")
                    standards_df = pd.DataFrame(config.standards.standards)
                    st.dataframe(standards_df, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    if config.standards.terminology_sources:
                        st.markdown("**Terminology Sources:**")
                        for source in config.standards.terminology_sources:
                            st.markdown(f"- {source}")
                
                with col2:
                    if config.standards.protocol_libraries:
                        st.markdown("**Protocol Libraries:**")
                        for key, value in config.standards.protocol_libraries.items():
                            st.markdown(f"- **{key}**: {value}")
            
            with comp_tabs[2]:  # Physics
                st.markdown("#### Physical Constraints")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Maximum Velocities (m/s):**")
                    vel_df = pd.DataFrame([
                        {"Transport Mode": mode, "v_max (m/s)": vel}
                        for mode, vel in config.physics.max_velocities.items()
                    ])
                    st.dataframe(vel_df, use_container_width=True)
                
                with col2:
                    st.metric("Temporal Resolution", f"{config.physics.temporal_resolution} seconds")
                    st.metric("Spatial Resolution", f"{config.physics.spatial_resolution} meters")
                    
                    if config.physics.facility_geometry:
                        st.markdown("**Facility Geometry:**")
                        st.json(config.physics.facility_geometry)
            
            with comp_tabs[3]:  # Credibility
                st.markdown("#### Source Credibility Weights")
                
                cred_df = pd.DataFrame([
                    {"Source": source, "Weight": weight}
                    for source, weight in config.credibility.credibility_weights.items()
                ]).sort_values("Weight", ascending=False)
                
                st.dataframe(cred_df, use_container_width=True)
                
                # Visualize top sources
                import plotly.graph_objects as go
                fig = go.Figure(data=[
                    go.Bar(
                        x=cred_df["Weight"],
                        y=cred_df["Source"],
                        orientation='h',
                        marker=dict(
                            color=cred_df["Weight"],
                            colorscale='Viridis',
                            showscale=True
                        )
                    )
                ])
                fig.update_layout(
                    title="Source Credibility Hierarchy",
                    xaxis_title="Credibility Weight",
                    yaxis_title="Source",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with comp_tabs[4]:  # Embeddings
                st.markdown("#### Domain Embeddings Configuration")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Model", config.embeddings.model_name)
                    st.metric("Embedding Dimension", config.embeddings.embedding_dim)
                    st.metric("Training Corpus Size", f"{config.embeddings.training_corpus_size:,}")
                
                with col2:
                    st.metric("Retraining Frequency", config.embeddings.retraining_frequency)
                    st.metric("Similarity Threshold", config.embeddings.similarity_threshold)
                    st.metric("Anomaly Detection Threshold", config.embeddings.anomaly_detection_threshold)
                
                if config.embeddings.training_corpus_size < 10000:
                    st.warning("⚠️ Training corpus below recommended minimum of 10,000 facts")
            
            with comp_tabs[5]:  # Parameters
                st.markdown("#### Verification Parameters")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Module Weights**")
                    weight_df = pd.DataFrame([
                        {"Module": k, "Weight": v}
                        for k, v in config.initial_weights.items()
                    ])
                    st.dataframe(weight_df, use_container_width=True)
                
                with col2:
                    st.markdown("**Module Thresholds**")
                    threshold_df = pd.DataFrame([
                        {"Module": k, "Threshold": v}
                        for k, v in config.initial_thresholds.items()
                    ])
                    st.dataframe(threshold_df, use_container_width=True)
                
                with col3:
                    st.markdown("**Module Alphas**")
                    alpha_df = pd.DataFrame([
                        {"Module": k, "Alpha": v}
                        for k, v in config.initial_alphas.items()
                    ])
                    st.dataframe(alpha_df, use_container_width=True)
                
                st.metric("Global Threshold (Θ)", config.global_threshold)
    

    with tab_stkg:  # STKG Structure Tab
        st.markdown("## 4D Spatiotemporal Knowledge Graph (STKG) Structure")
        st.markdown("---")
        
        # Introduction
        st.markdown("""
        <div class="card">
            <h3>STKG Formalization: G = (V, E, O, T, Ψ)</h3>
            <p>
                The 4D STKG is the foundation of ATLASky-AI verification. It integrates entities, relationships,
                ontology, spatiotemporal coordinates, and physical consistency predicates.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show STKG components
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### STKG Components")
            st.markdown("""
            <div class="card">
                <h4>V: Versioned Entities</h4>
                <p>Entities with immutable attributes and mutable state tracked through time.</p>
                <p><strong>Example:</strong> TurbineBlade_Alpha with material=Titanium, lifecycle_hours=2000</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
                <h4>E: Directed Relationships</h4>
                <p>Edges representing relationships between entities.</p>
                <p><strong>Example:</strong> EngineSet_1 → containsBlade → TurbineBlade_Alpha</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
                <h4>O: Domain Ontology (C, R_o, A)</h4>
                <p>Entity classes (C), relation types (R_o), and attributes (A) from ontology.</p>
                <p><strong>Classes:</strong> {len(st.session_state.kg.ontology.entity_classes)}</p>
                <p><strong>Relations:</strong> {len(st.session_state.kg.ontology.relationship_types)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Physical Predicates")
            st.markdown("""
            <div class="card">
                <h4>T: Spatiotemporal Mapping</h4>
                <p>Maps entities/relations to (x, y, z, t) coordinates in 4D spacetime.</p>
                <p><strong>Example:</strong> Blade at (10.5, 20.3, 150.2) at 2024-01-15T10:30:00Z</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
                <h4>Ψ: Physical Consistency Predicate</h4>
                <p>Combines spatial (ψ_s) and temporal (ψ_t) consistency:</p>
                <p><strong>ψ_s:</strong> No entity at two locations simultaneously</p>
                <p><strong>ψ_t:</strong> Travel time physically feasible</p>
                <p><strong>Ψ = ψ_s ∧ ψ_t</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Domain-specific STKG examples
        st.markdown("---")
        st.markdown("### Domain-Specific STKG Examples")
        
        domain_tabs = st.tabs(["🏭 Aerospace", "🏥 Healthcare", "✈️ Aviation", "🔧 CAD"])
        
        with domain_tabs[0]:  # Aerospace
            st.markdown("""
            <div class="card">
                <h4>Aerospace Manufacturing STKG</h4>
                <p><strong>Entity Classes:</strong> EngineSet, Blade, InspectionMeasurement</p>
                <p><strong>Relationships:</strong> containsBlade, hasMeasurement, locatedAt</p>
                <p><strong>Spatiotemporal:</strong> Facility coordinates (Bay 1-12), timestamps for each measurement</p>
                <p><strong>Physical Constraints:</strong> Tolerance ±0.1mm, velocity limits for manual/forklift transport</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Example Verified Fact Integration")
            st.code("""
Verified Fact:
  Subject: TurbineBlade_Gamma
  Relation: hasMeasurement
  Object: Measurement_001
  Spatiotemporal: (40.0, 20.0, 0.0) @ 2024-10-29T10:30:00Z
  Attributes: {actual_mm: 3.02, tolerance_mm: 0.1, deviation: 0.02}
  
Integration into STKG:
  1. Entity TurbineBlade_Gamma added to V with spatiotemporal coords
  2. Measurement_001 entity created
  3. Edge (TurbineBlade_Gamma → hasMeasurement → Measurement_001) added to E
  4. Physical consistency checked: ψ_s=1, ψ_t=1, Ψ=1 ✓
  5. Ontology validated: Blade ∈ C, hasMeasurement ∈ R_o ✓
            """, language="text")
        
        with domain_tabs[1]:  # Healthcare
            st.markdown("""
            <div class="card">
                <h4>Healthcare Clinical STKG</h4>
                <p><strong>Entity Classes:</strong> Patient, CareUnit, ClinicalTransfer</p>
                <p><strong>Relationships:</strong> transferred, locatedIn, administeredTo</p>
                <p><strong>Spatiotemporal:</strong> Care unit coordinates (MICU, SICU, OR), transfer timestamps</p>
                <p><strong>Physical Constraints:</strong> Minimum transfer times (ICU→OR ≥20min), patient transport protocols</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Example Verified Fact Integration")
            st.code("""
Verified Fact:
  Subject: Patient_P001234
  Relation: transferred
  Object: MICU
  Target: OR
  Spatiotemporal: (10,20,3) → (25,50,2) @ 2024-10-29T14:35:00Z
  Attributes: {transfer_duration: 22min, protocol_time: 20min}
  
Integration into STKG:
  1. Patient entity updated with new location in V
  2. Transfer event added with from/to coordinates
  3. Edge (Patient_P001234 → transferred → OR) added to E
  4. Temporal consistency: 22min ≥ 20min required ✓
  5. Protocol compliance: Transfer time meets clinical standards ✓
            """, language="text")
        
        with domain_tabs[2]:  # Aviation
            st.markdown("""
            <div class="card">
                <h4>Aviation Safety STKG</h4>
                <p><strong>Entity Classes:</strong> Aircraft, SafetyIncident, Operator, Event</p>
                <p><strong>Relationships:</strong> caused, contributedTo, occurredDuring</p>
                <p><strong>Spatiotemporal:</strong> Airspace coordinates, altitude, incident timestamps</p>
                <p><strong>Physical Constraints:</strong> Descent rate limits, causal temporal ordering</p>
            </div>
            """, unsafe_allow_html=True)
        
        with domain_tabs[3]:  # CAD
            st.markdown("""
            <div class="card">
                <h4>CAD Engineering STKG</h4>
                <p><strong>Entity Classes:</strong> CADAssembly, CADFeature, Component, Constraint</p>
                <p><strong>Relationships:</strong> containedBy, adjacentTo, interferes</p>
                <p><strong>Spatiotemporal:</strong> 3D part coordinates, version timestamps</p>
                <p><strong>Physical Constraints:</strong> Geometric interference detection, clearance requirements</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Current knowledge graph status
        st.markdown("---")
        st.markdown("### Current Knowledge Graph Status")
        
        # Calculate accepted facts
        accepted_facts = sum(1 for record in st.session_state.verification_history 
                           if record.get("verification_results", {}).get("decision") == "Accept")
        
        kg_col1, kg_col2, kg_col3, kg_col4 = st.columns(4)
        
        with kg_col1:
            st.metric("Entities (V)", len(st.session_state.kg.entities))
        with kg_col2:
            st.metric("Relationships (E)", len(st.session_state.kg.relationships))
        with kg_col3:
            st.metric("Facts Verified", st.session_state.verification_count)
        with kg_col4:
            st.metric("Accepted Facts", accepted_facts)
        
        # Show recent additions
        if st.session_state.verification_history:
            st.markdown("### Recent STKG Updates")
            
            recent_accepted = [r for r in st.session_state.verification_history[-5:] 
                             if r.get("verification_results", {}).get("decision") == "Accept"]
            
            if recent_accepted:
                for record in reversed(recent_accepted):
                    fact = record.get("candidate_fact", {})
                    subject = fact.get("subject_entity_id", "Unknown")
                    relation = fact.get("relationship_type", "Unknown")
                    obj = fact.get("object_entity_id", "Unknown")
                    
                    st.markdown(f"""
                    <div style='background-color: #f8fafc; padding: 10px; margin: 5px 0; border-left: 4px solid #10b981; border-radius: 4px;'>
                        <strong>✓ Added:</strong> {subject} → {relation} → {obj}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No accepted facts yet. Verify facts in the Verification Process tab to see STKG updates.")
        
        # Show ontology structure
        st.markdown("### Ontology Structure (O)")
        
        ontology_tabs = st.tabs(["Entity Classes", "Relationships", "Physical Constraints", "Domain Rules"])
        
        with ontology_tabs[0]:
            st.markdown("#### Entity Class Hierarchy")
            for name, entity_class in list(st.session_state.kg.ontology.entity_classes.items())[:10]:
                parent = entity_class.parent_class or "None"
                domain = entity_class.domain
                st.markdown(f"""
                <div style='background-color: #f8fafc; padding: 8px; margin: 4px 0; border-left: 3px solid #3b82f6; border-radius: 4px;'>
                    <strong>{name}</strong> (parent: {parent}, domain: {domain})<br>
                    <small>Required: {', '.join(entity_class.required_attributes[:3])}...</small>
                </div>
                """, unsafe_allow_html=True)
        
        with ontology_tabs[1]:
            st.markdown("#### Relationship Types")
            for name, rel_type in list(st.session_state.kg.ontology.relationship_types.items())[:8]:
                st.markdown(f"""
                <div style='background-color: #f8fafc; padding: 8px; margin: 4px 0; border-left: 3px solid #10b981; border-radius: 4px;'>
                    <strong>{name}</strong> - {rel_type.description}
                </div>
                """, unsafe_allow_html=True)
        
        with ontology_tabs[2]:
            st.markdown("#### Physical Constraints")
            for name, constraint in st.session_state.kg.ontology.physical_constraints.items():
                st.markdown(f"""
                <div style='background-color: #f8fafc; padding: 8px; margin: 4px 0; border-left: 3px solid #f59e0b; border-radius: 4px;'>
                    <strong>{name}</strong> - {constraint.description}
                </div>
                """, unsafe_allow_html=True)
        
        with ontology_tabs[3]:
            st.markdown("#### Domain Rules")
            for name, rule in st.session_state.kg.ontology.domain_rules.items():
                severity_colors = {"critical": "#ef4444", "high": "#f59e0b", "medium": "#3b82f6", "low": "#6b7280"}
                color = severity_colors.get(rule.violation_severity, "#6b7280")
                st.markdown(f"""
                <div style='background-color: #f8fafc; padding: 8px; margin: 4px 0; border-left: 3px solid {color}; border-radius: 4px;'>
                    <strong>{name}</strong> ({rule.domain}) - {rule.description}<br>
                    <small>Severity: {rule.violation_severity.upper()}</small>
                </div>
                """, unsafe_allow_html=True)

    with tab_verification:  # Verification Process Tab
        st.markdown("## ATLASky-AI Three-Stage Verification Pipeline")
        
        # Show global active domain configuration
        if st.session_state.current_domain_name:
            config = st.session_state.domain_adapter.current_domain
            st.markdown(f"""
            <div class="card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; margin-bottom: 20px;">
                <h4 style="margin: 0 0 10px 0; color: white;">🌐 Global Active Domain: <strong>{config.domain_name.upper()}</strong></h4>
                <p style="margin: 0; font-size: 14px; opacity: 0.9;">{config.domain_description}</p>
                <div style="margin-top: 10px; display: flex; gap: 20px; font-size: 12px;">
                    <span>📋 {len(config.ontology.entity_classes)} Entity Classes</span>
                    <span>📜 {len(config.standards.standards)} Standards</span>
                    <span>⚙️ {len(config.physics.max_velocities)} Transport Modes</span>
                    <span>🌐 {len(config.credibility.credibility_weights)} Credibility Sources</span>
                </div>
                <div style="margin-top: 8px; font-size: 11px; opacity: 0.8;">
                    Data generation and verification use {config.domain_name} domain parameters
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ No domain configuration active. Using default parameters. Select a domain from the sidebar to apply domain-specific verification.")
        
        st.markdown("---")

        # Show the three stages
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div style='text-align: center; padding: 15px; background-color: #f8fafc; border-radius: 8px; border: 2px solid #3b82f6;'>
                <div style='font-size: 24px; margin-bottom: 10px;'>📊</div>
                <h4 style='margin: 0; color: #1e40af;'>Stage 1</h4>
                <p style='margin: 5px 0 0 0; font-size: 14px; color: #64748b;'>Data Preprocessing</p>
                <p style='margin: 5px 0 0 0; font-size: 12px; color: #64748b;'>Normalize heterogeneous data</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 15px; background-color: #f8fafc; border-radius: 8px; border: 2px solid #f59e0b;'>
                <div style='font-size: 24px; margin-bottom: 10px;'>🤖</div>
                <h4 style='margin: 0; color: #92400e;'>Stage 2</h4>
                <p style='margin: 5px 0 0 0; font-size: 14px; color: #64748b;'>LLM Extraction</p>
                <p style='margin: 5px 0 0 0; font-size: 12px; color: #64748b;'>Generate candidate facts</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div style='text-align: center; padding: 15px; background-color: #f8fafc; border-radius: 8px; border: 2px solid #10b981;'>
                <div style='font-size: 24px; margin-bottom: 10px;'>🔍</div>
                <h4 style='margin: 0; color: #047857;'>Stage 3</h4>
                <p style='margin: 5px 0 0 0; font-size: 14px; color: #64748b;'>TruthFlow Verification</p>
                <p style='margin: 5px 0 0 0; font-size: 12px; color: #64748b;'>RMMVe + AAIC validation</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Main content area
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("## Candidate Fact Generation")
            
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
            if st.session_state.candidate_fact and "contains_performance_shift" in st.session_state.candidate_fact:
                shift_module = st.session_state.candidate_fact["contains_performance_shift"]
                st.markdown(format_shift_alert(shift_module), unsafe_allow_html=True)
            
            # Stage 1: Data Preprocessing (real pipeline)
            st.markdown("### Stage 1: Data Preprocessing")

            # Initialize preprocessing state
            if "preprocessed_data" not in st.session_state:
                st.session_state.preprocessed_data = None
            if "extracted_facts" not in st.session_state:
                st.session_state.extracted_facts = []

            st.markdown("""
            <div style='background-color: #eff6ff; padding: 12px; border-radius: 8px; border-left: 4px solid #3b82f6; margin: 10px 0;'>
                <strong>📊 Stage 1 Input:</strong> Upload a file OR enter text manually to process through the preprocessing pipeline.
                This stage normalizes text, aligns timestamps, and maps locations to coordinates.
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📊 Configure & Run Preprocessing", expanded=True):
                # Show if test fact was generated
                if "generated_raw_text" in st.session_state:
                    st.markdown(f"""
                    <div style='background-color: #dbeafe; padding: 10px; border-radius: 6px; margin-bottom: 10px;'>
                        <strong>🎲 Test Fact Generated:</strong> Quality level: <strong>{st.session_state.fact_quality}</strong>, Domain: <strong>{st.session_state.test_domain_selected}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Upload file option
                uploaded = st.file_uploader(
                    "📁 Upload Raw Data File (TXT, JSON, or PDF*)",
                    type=["json", "txt", "pdf"],
                    key="stage1_uploader",
                    help="Upload inspection reports, clinical notes, or any text document (*PDF requires additional dependencies)"
                )
                
                # Domain selection - use test domain if available
                default_domain_idx = 0
                if "test_domain_selected" in st.session_state:
                    domain_map = {"aerospace": 0, "healthcare": 1, "aviation": 2, "cad": 3}
                    default_domain_idx = domain_map.get(st.session_state.test_domain_selected, 0)
                
                default_domain = st.selectbox(
                    "🏭 Domain",
                    options=["aerospace", "healthcare", "aviation", "cad"],
                    index=default_domain_idx,
                    key="stage1_domain",
                    help="Domain determines ontology, facility maps, and terminology standards"
                )
                
                st.markdown("---")
                
                # Show domain-specific configuration
                loc_col, time_col = st.columns(2)
                
                with loc_col:
                    # Domain-specific default locations
                    default_locations = {
                        "aerospace": "Bay 7",
                        "healthcare": "MICU",
                        "aviation": "Runway 24L",
                        "cad": "Assembly Station 1"
                    }
                    default_location = st.text_input(
                        "📍 Location (symbolic)",
                        value=default_locations.get(default_domain, "Bay 7"),
                        key="stage1_location",
                        help="Symbolic location to be mapped to coordinates"
                    )
                
                with time_col:
                    default_timestamp = st.text_input(
                        "🕐 Timestamp",
                        value=datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                        key="stage1_timestamp",
                        help="Will be normalized to UTC ISO 8601"
                    )
                
                # Use generated raw text if available, otherwise use default
                if "generated_raw_text" in st.session_state:
                    initial_text = st.session_state.generated_raw_text
                else:
                    # Domain-specific default text with spelling errors to demonstrate correction
                    default_texts = {
                        "aerospace": "Instalation completed in bay 7. Blade Gamma measurment: 3.02 mm on leading edge. Tolerance check passed.",
                        "healthcare": "Patient P001234 tranfser from micu to Operating Room at 14:35. Transfer duration: 22 minutes.",
                        "aviation": "Aircraft N12345 experienced turbulence at FL350 during climb phase. Incident at 10:45 UTC.",
                        "cad": "Assembly clearance check: interference detected between Part A-023 and Housing B-104. Clearance gap: 0.15mm."
                    }
                    initial_text = default_texts.get(default_domain, default_texts["aerospace"])
                
                raw_text = st.text_area(
                    "📝 Raw Text Input",
                    value=initial_text,
                    height=120,
                    key="stage1_text",
                    help="Raw text to be preprocessed (spell corrected, terminology standardized, etc.)"
                )

                if st.button("▶️ Run Stage 1 Preprocessing", key="run_stage1", use_container_width=True):
                    with st.spinner("Running preprocessing pipeline..."):
                        raw_docs = []
                        source_info = ""
                        
                        # If user uploaded a file, try to parse a minimal JSON/text doc
                        if uploaded is not None:
                            source_info = f"Uploaded file: {uploaded.name}"
                            try:
                                if uploaded.name.lower().endswith(".json"):
                                    import json as _json
                                    payload = _json.loads(uploaded.read().decode("utf-8"))
                                    # Accept either a dict or list of dicts
                                    if isinstance(payload, dict):
                                        raw_docs = [payload]
                                    elif isinstance(payload, list):
                                        raw_docs = payload
                                elif uploaded.name.lower().endswith(".pdf"):
                                    st.warning("⚠️ PDF support requires PyPDF2. Using text extraction fallback.")
                                    source_info += " (PDF - text extraction not fully supported)"
                                    raw_docs = [{
                                        "document_id": f"uploaded_pdf_{uploaded.name}",
                                        "domain": default_domain,
                                        "format": "pdf",
                                        "content": {"text": f"[PDF extraction placeholder for {uploaded.name}]"},
                                        "metadata": {"timestamp": default_timestamp, "location": default_location}
                                    }]
                                else:
                                    txt = uploaded.read().decode("utf-8")
                                    raw_docs = [{
                                        "document_id": f"uploaded_{uploaded.name}",
                                        "domain": default_domain,
                                        "format": "text",
                                        "content": {"text": txt},
                                        "metadata": {"timestamp": default_timestamp, "location": default_location, "source_type": "uploaded_file"}
                                    }]
                            except Exception as e:
                                st.error(f"❌ Error reading file: {e}")
                                raw_docs = []
                        else:
                            source_info = "Manual text input"

                        # If no upload, build from UI fields
                        if not raw_docs:
                            raw_docs = [{
                                "document_id": "manual_doc_001",
                                "domain": default_domain,
                                "format": "text",
                                "content": {"text": raw_text},
                                "metadata": {
                                    "timestamp": default_timestamp,
                                    "location": default_location,
                                    "source_type": "manual_input"
                                }
                            }]

                        # Store raw input for comparison
                        st.session_state.raw_input = raw_docs[0]

                        # Run Stage 1 preprocessing
                        processed = PREPROCESSOR.preprocess_multimodal_data(raw_docs)
                        st.session_state.preprocessed_data = processed

                        st.success(f"✅ Stage 1 preprocessing completed! ({source_info})")

            # Show preprocessing results OUTSIDE the expander (clearer display)
            if st.session_state.preprocessed_data and "raw_input" in st.session_state:
                st.markdown("#### Preprocessing Results")
                
                # Show before/after comparison (use different variable names to avoid conflict)
                before_col, after_col = st.columns(2)
                
                with before_col:
                    st.markdown("**📥 BEFORE (Raw Input)**")
                    raw_doc = st.session_state.raw_input
                    raw_text = raw_doc.get("content", {}).get("text", "")
                    
                    st.code(raw_text[:200] + ("..." if len(raw_text) > 200 else ""), language="text")
                    
                    st.markdown("**Metadata:**")
                    st.json({
                        "domain": raw_doc.get("domain"),
                        "location": raw_doc.get("metadata", {}).get("location"),
                        "timestamp": raw_doc.get("metadata", {}).get("timestamp")
                    })
                
                with after_col:
                    st.markdown("**📤 AFTER (Preprocessed RD')**")
                    proc_doc = st.session_state.preprocessed_data.get("documents", [{}])[0]
                    normalized_text = proc_doc.get("content", {}).get("normalized_text", "")
                    
                    st.code(normalized_text[:200] + ("..." if len(normalized_text) > 200 else ""), language="text")
                    
                    st.markdown("**Spatiotemporal:**")
                    st_data = proc_doc.get("content", {}).get("spatiotemporal", {})
                    st.json({
                        "timestamp_utc": st_data.get("timestamp_utc"),
                        "location_symbol": st_data.get("location_symbol"),
                        "coordinates": st_data.get("coordinates")
                    })
                
                # Show preprocessing changes
                st.markdown("**🔧 Preprocessing Applied:**")
                changes = []
                
                # Detect spell corrections
                if "measurment" in raw_text and "measurement" in normalized_text:
                    changes.append("✓ Spell correction: 'measurment' → 'measurement'")
                if "Instalation" in raw_text and "Installation" in normalized_text:
                    changes.append("✓ Spell correction: 'Instalation' → 'Installation'")
                if "tranfser" in raw_text and "transfer" in normalized_text:
                    changes.append("✓ Spell correction: 'tranfser' → 'transfer'")
                    
                # Detect terminology standardization
                if "bay" in raw_text.lower() and "Bay" in normalized_text:
                    changes.append("✓ Terminology standardization: 'bay' → 'Bay'")
                if "micu" in raw_text.lower() and "MICU" in normalized_text:
                    changes.append("✓ Terminology standardization: 'micu' → 'MICU'")
                
                # Timestamp normalization
                if st_data.get("timestamp_utc"):
                    changes.append(f"✓ Temporal alignment: Normalized to UTC ISO 8601")
                
                # Spatial mapping
                if st_data.get("coordinates"):
                    coords = st_data["coordinates"]
                    changes.append(f"✓ Spatial mapping: '{st_data.get('location_symbol')}' → ({coords[0]}, {coords[1]}, {coords[2]})")
                
                if not changes:
                    changes.append("✓ Text cleaned and normalized")
                
                for change in changes:
                    st.markdown(f"<small>{change}</small>", unsafe_allow_html=True)
                
                with st.expander("📋 Complete RD' Output (JSON)", expanded=False):
                    st.json(st.session_state.preprocessed_data)

            # Show preprocessing status with clear indicators
            if not st.session_state.preprocessed_data:
                st.markdown("""
                <div style='background-color: #fee2e2; padding: 12px; border-radius: 8px; border-left: 4px solid #ef4444; margin: 10px 0;'>
                    <strong>⚠️ Stage 1 Not Completed:</strong> Upload a file or enter text above, then click "Run Stage 1 Preprocessing"
                </div>
                """, unsafe_allow_html=True)
            else:
                doc_count = len(st.session_state.preprocessed_data.get("documents", []))
                st.markdown(f"""
                <div style='background-color: #d1fae5; padding: 12px; border-radius: 8px; border-left: 4px solid #10b981; margin: 10px 0;'>
                    <strong>✅ Stage 1 Complete:</strong> {doc_count} document(s) preprocessed successfully<br>
                    <small>Ready to proceed to Stage 2 LLM Extraction</small>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### Stage 2: LLM Extraction")

            # Show prompt template
            with st.expander("📝 View Prompt Template (Listing 1)", expanded=False):
                selected_domain = st.selectbox("Domain for template", options=["aerospace", "healthcare", "aviation", "cad"], index=0, key="prompt_domain")
                prompt_template = LLM_EXTRACTOR.get_prompt_template(selected_domain)
                st.code(prompt_template, language="text")

            # LLM Extraction button - only enabled if Stage 1 is complete
            extraction_disabled = st.session_state.preprocessed_data is None
            
            if extraction_disabled:
                st.markdown("""
                <div style='background-color: #fee2e2; padding: 10px; border-radius: 6px; border-left: 4px solid #ef4444; margin: 10px 0;'>
                    <strong>⚠️ Stage 2 Blocked:</strong> Complete Stage 1 preprocessing first
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("▶️ Run Stage 2 LLM Extraction", key="run_stage2", disabled=extraction_disabled, use_container_width=True):
                with st.spinner("Running LLM extraction (D = L(RD'; P))..."):
                    # Extract facts using LLM from Stage 1 output
                    extracted_facts = LLM_EXTRACTOR.extract_knowledge(st.session_state.preprocessed_data)
                    st.session_state.extracted_facts = extracted_facts

                    st.success(f"✅ Stage 2 completed. Extracted {len(extracted_facts)} candidate facts from Stage 1 output.")
                    st.rerun()

            # Show extraction results with before/after data flow
            if st.session_state.extracted_facts:
                fact_count = len(st.session_state.extracted_facts)
                avg_conf = sum(f.get("llm_confidence", 0.8) for f in st.session_state.extracted_facts) / max(1, fact_count)
                conf_level = "High" if avg_conf >= 0.8 else "Medium" if avg_conf >= 0.6 else "Low"
                
                st.markdown(f"""
                <div style='background-color: #d1fae5; padding: 12px; border-radius: 8px; border-left: 4px solid #10b981; margin: 10px 0;'>
                    <strong>✅ Stage 2 Complete:</strong><br>
                    • Input: RD' from Stage 1 preprocessing<br>
                    • Extracted {fact_count} candidate fact(s) using domain-specialized prompts<br>
                    • Average LLM Confidence: {conf_level} ({avg_conf:.1f})<br>
                    • Output Structure: d_k = ⟨s, r, o, T(d_k), conf_k⟩<br>
                    <small>✓ Ready for Stage 3 TruthFlow verification</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Show extracted facts
                st.markdown("#### 📄 Extracted Candidate Facts")
                for i, fact in enumerate(st.session_state.extracted_facts):
                    conf_level = fact.get("llm_confidence_level", "medium")
                    conf_value = fact.get("llm_confidence", 0.8)
                    
                    # Color code by confidence
                    conf_color = "#10b981" if conf_value >= 0.9 else "#f59e0b" if conf_value >= 0.7 else "#ef4444"
                    
                    with st.expander(f"Fact {i+1}: {fact.get('subject_entity_id', 'Unknown')} → {fact.get('relationship_type', 'Unknown')}", expanded=False):
                        triple_col, st_col = st.columns(2)
                        
                        with triple_col:
                            st.markdown("**Triple Structure:**")
                            st.json({
                                "subject": fact.get("subject_entity_id"),
                                "relation": fact.get("relationship_type"),
                                "object": fact.get("object_entity_id")
                            })
                        
                        with st_col:
                            st.markdown("**Spatiotemporal:**")
                            st.json(fact.get("spatiotemporal", {}))
                        
                        st.markdown(f"""
                        <div style='background-color: {conf_color}20; padding: 8px; border-radius: 4px; margin-top: 8px;'>
                            <strong>LLM Confidence:</strong> {conf_level.upper()} ({conf_value:.1f})
                        </div>
                        """, unsafe_allow_html=True)
                        
            elif st.session_state.preprocessed_data:
                st.markdown("""
                <div style='background-color: #fef3c7; padding: 12px; border-radius: 8px; border-left: 4px solid #f59e0b; margin: 10px 0;'>
                    <strong>📋 Stage 1 Complete, Stage 2 Pending:</strong><br>
                    • Input ready: RD' from Stage 1<br>
                    • Click "Run Stage 2 LLM Extraction" to extract facts
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background-color: #fee2e2; padding: 12px; border-radius: 8px; border-left: 4px solid #ef4444; margin: 10px 0;'>
                    <strong>⚠️ Stage 2 Blocked:</strong> Complete Stage 1 preprocessing first
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### Stage 3: TruthFlow Verification")

            # Determine which facts to verify
            if st.session_state.extracted_facts:
                # Use facts from Stage 2 extraction
                facts_to_verify = st.session_state.extracted_facts
                fact_source = "Stage 2 LLM Extraction"
            elif st.session_state.candidate_fact:
                # Fallback to demo fact generation
                facts_to_verify = [st.session_state.candidate_fact]
                fact_source = "Demo Generator"
                
                # Show demo fact with quality
                llm_confidence = {"high_quality": 1.0, "medium_quality": 0.8, "spatial_issue": 0.6,
                                "external_ref": 0.7, "semantic_issue": 0.5, "low_quality": 0.3}.get(
                                    st.session_state.fact_quality, 0.5)
                confidence_level = "High" if llm_confidence >= 0.8 else "Medium" if llm_confidence >= 0.6 else "Low"

                st.markdown(f"""
                <div style='background-color: #fef3c7; padding: 12px; border-radius: 8px; border-left: 4px solid #f59e0b; margin: 10px 0;'>
                    <strong>Demo Mode:</strong> Using generated fact<br>
                    <strong>LLM Confidence:</strong> {confidence_level} ({llm_confidence:.1f})<br>
                    <small>Confidence weights module scores in verification</small>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("Fact Details", expanded=False):
                    st.markdown(create_fact_preview_card(st.session_state.candidate_fact), unsafe_allow_html=True)
            else:
                # No facts available
                facts_to_verify = []
                fact_source = "None"
                
                st.markdown("""
                <div style='background-color: #fee2e2; padding: 12px; border-radius: 8px; border-left: 4px solid #ef4444; margin: 10px 0;'>
                    <strong>⚠️ No Facts Available:</strong> Generate a test fact OR run Stage 1 & 2 to extract facts
                </div>
                """, unsafe_allow_html=True)
            
            # Verification control
            st.markdown("#### RMMVe + AAIC Verification")
            
            if facts_to_verify:
                st.markdown(f"""
            <div class="alert-info">
                    Click "Run TruthFlow Verification" to verify {len(facts_to_verify)} fact(s) from {fact_source}.
                    The system applies 5 verification modules with early termination, then adapts parameters
                    based on performance monitoring. Facts receive Accept/Review/Reject decisions.
            </div>
            """, unsafe_allow_html=True)
            
            verification_disabled = len(facts_to_verify) == 0
            
            if st.button("▶️ Run TruthFlow Verification", key="run_verify", disabled=verification_disabled, use_container_width=True):
                with st.spinner(f"Running TruthFlow verification on {len(facts_to_verify)} fact(s)..."):
                    # Verify all facts
                    for fact in facts_to_verify:
                        # Get LLM confidence from fact or use quality-based default
                        llm_confidence = fact.get("llm_confidence", 0.8)

                        # Run verification with LLM confidence
                        verification_results = st.session_state.rmmve.verify(
                            fact,
                            st.session_state.kg,
                            st.session_state.fact_quality,
                            llm_confidence
                        )

                        # Add LLM confidence to results for proper weighting
                        verification_results["llm_confidence"] = llm_confidence

                        # Implement proper three-way decision rule
                        total_confidence = verification_results["total_confidence"]
                        global_threshold = st.session_state.rmmve.global_threshold
                        epsilon = 0.1  # Review margin

                        if total_confidence >= global_threshold:
                            decision = "Accept"
                            decision_color = "#10b981"
                        elif total_confidence >= global_threshold - epsilon:
                            decision = "Review"
                            decision_color = "#f59e0b"
                        else:
                            decision = "Reject"
                            decision_color = "#ef4444"

                        verification_results["decision"] = decision
                        verification_results["decision_color"] = decision_color

                        # Add accepted facts to the knowledge graph
                        if decision == "Accept":
                            try:
                                # Extract entity information from fact
                                subject_id = fact.get("subject_entity_id")
                                object_id = fact.get("object_entity_id")
                                relationship_type = fact.get("relationship_type")
                                
                                # Add subject entity if not exists
                                if subject_id and subject_id not in st.session_state.kg.entities:
                                    entity_class = fact.get("entity_class", "PhysicalEntity")
                                    st.session_state.kg.add_entity(
                                        entity_id=subject_id,
                                        entity_class=entity_class,
                                        attributes=fact.get("attributes", {}),
                                        spatiotemporal=fact.get("spatiotemporal", {}),
                                        validate=False
                                    )
                                
                                # Add object entity if not exists (and it's not just a location)
                                if object_id and object_id not in st.session_state.kg.entities:
                                    object_class = fact.get("object_entity_class", "Entity")
                                    st.session_state.kg.add_entity(
                                        entity_id=object_id,
                                        entity_class=object_class,
                                        attributes={},
                                        spatiotemporal={},
                                        validate=False
                                    )
                                
                                # Add relationship
                                if subject_id and object_id and relationship_type:
                                    rel_id = f"R_{len(st.session_state.kg.relationships)}_{subject_id}_{relationship_type}"
                                    st.session_state.kg.add_relationship(
                                        rel_id=rel_id,
                                        subject_id=subject_id,
                                        relation_type=relationship_type,
                                        object_id=object_id,
                                        validate=False
                                    )
                                
                                verification_results["added_to_kg"] = True
                            except Exception as e:
                                verification_results["added_to_kg"] = False
                                verification_results["kg_error"] = str(e)
                        else:
                            verification_results["added_to_kg"] = False

                        # Record verification
                        verification_record = {
                            "timestamp": time.time(),
                            "candidate_fact": fact,
                            "llm_confidence": llm_confidence,
                            "verification_results": verification_results,
                            "fact_quality": st.session_state.fact_quality,
                            "fact_source": fact_source
                        }
                        
                        # Run AAIC
                        aaic_updates = st.session_state.aaic.update_all_modules()
                        verification_record["aaic_updates"] = aaic_updates
                        
                        # Add to history
                        st.session_state.verification_history.append(verification_record)
                        st.session_state.verification_count += 1

                    # Show success and refresh once after processing all facts
                    st.success(f"✅ Verified {len(facts_to_verify)} fact(s) using TruthFlow")
                    st.rerun()
        
        with col2:
            st.markdown("## 🎯 TruthFlow Verification Results")
            
            # Show global active domain during verification
            if st.session_state.current_domain_name:
                st.markdown(f"""
                <div style="background: #f3f4f6; padding: 10px; border-radius: 6px; margin-bottom: 15px; border-left: 4px solid #667eea;">
                    <strong>🌐 Global Domain Active: {st.session_state.current_domain_name.upper()}</strong>
                    <span style="color: #64748b; font-size: 12px; margin-left: 10px;">
                        Using domain-specific verification parameters and data generation
                    </span>
                </div>
                """, unsafe_allow_html=True)
            
            if st.session_state.verification_history:
                # Get latest verification
                latest = st.session_state.verification_history[-1]
                results = latest["verification_results"]
                llm_confidence = latest.get("llm_confidence", 0.5)
                
                # Display fact quality
                quality = results.get("fact_quality", "unknown")
                quality_color = QUALITY_COLOR_MAP.get(quality, "#6b7280")

                # Get decision and color from results
                decision = results.get("decision", "Reject")
                decision_color = results.get("decision_color", "#ef4444")
                
                # Handle case where decision might be boolean from old stored results
                if isinstance(decision, bool):
                    decision = "Accept" if decision else "Reject"
                    decision_color = "#10b981" if decision == "Accept" else "#ef4444"
                
                # Create a results header with fancy styling
                st.markdown(
                    f"""
                    <div style='display: flex; margin-bottom: 15px;'>
                        <div style='background-color: {quality_color}; padding: 12px; border-radius: 8px; 
                             color: white; flex: 1; margin-right: 10px;'>
                            <span style='font-size: 16px;'>Fact Quality</span><br>
                            <span style='font-size: 20px; font-weight: 500;'>{quality.replace('_', ' ').title()}</span>
                        </div>
                        <div style='background-color: {decision_color};
                                    padding: 12px; border-radius: 8px; color: white; flex: 1;'>
                            <span style='font-size: 16px;'>TruthFlow Decision</span><br>
                            <span style='font-size: 20px; font-weight: 500;'>{decision.upper() if isinstance(decision, str) else decision}</span>
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

                # Show LLM confidence
                confidence_level = "High" if llm_confidence >= 0.8 else "Medium" if llm_confidence >= 0.6 else "Low"
                st.markdown(f"""
                <div style='background-color: #dbeafe; padding: 8px; border-radius: 6px; margin-bottom: 15px; border-left: 4px solid #3b82f6;'>
                    <strong>LLM Confidence:</strong> {confidence_level} ({llm_confidence:.1f})
                </div>
                """, unsafe_allow_html=True)
                
                # Show STKG integration status
                if results.get("added_to_kg"):
                    st.markdown("""
                    <div style='background-color: #d1fae5; padding: 10px; border-radius: 6px; margin-bottom: 15px; border-left: 4px solid #10b981;'>
                        <strong>✅ Added to STKG:</strong> Fact integrated into knowledge graph<br>
                        <small>View updated graph in STKG Structure tab</small>
                    </div>
                    """, unsafe_allow_html=True)
                elif decision == "Reject":
                    st.markdown("""
                    <div style='background-color: #fee2e2; padding: 10px; border-radius: 6px; margin-bottom: 15px; border-left: 4px solid #ef4444;'>
                        <strong>❌ Not Added to STKG:</strong> Fact rejected by verification
                    </div>
                    """, unsafe_allow_html=True)
                elif decision == "Review":
                    st.markdown("""
                    <div style='background-color: #fef3c7; padding: 10px; border-radius: 6px; margin-bottom: 15px; border-left: 4px solid #f59e0b;'>
                        <strong>🔍 Pending Review:</strong> Fact requires human review before STKG integration
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show if this fact contained a performance shift
                if results.get("contains_shift"):
                    st.markdown(format_shift_alert(results['contains_shift']), unsafe_allow_html=True)
                
                # Display verification summary in a card
                st.markdown(
                    f"""
                    <div class="card">
                        <h3>TruthFlow Verification Summary</h3>
                        <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 10px;">
                            <div style="flex: 1; min-width: 150px;">
                                <div class="metric-label">Cumulative Confidence (C)</div>
                                <div class="metric-value" style="color: {decision_color};">
                                    {results['total_confidence']:.3f}
                                </div>
                                <small style="color: #64748b;">Weighted average of activated modules</small>
                            </div>
                            <div style="flex: 1; min-width: 150px;">
                                <div class="metric-label">Global Threshold (Θ)</div>
                                <div class="metric-value">{st.session_state.rmmve.global_threshold:.3f}</div>
                                <small style="color: #64748b;">Accept if C ≥ Θ</small>
                            </div>
                            <div style="flex: 1; min-width: 150px;">
                                <div class="metric-label">Review Margin (ε)</div>
                                <div class="metric-value">0.10</div>
                                <small style="color: #64748b;">Review if Θ-ε ≤ C < Θ</small>
                            </div>
                        </div>
                        <div style="margin-top: 10px; padding: 8px; background-color: #f8fafc; border-radius: 4px; border-left: 3px solid {decision_color};">
                            <strong>Decision Logic:</strong> {decision.upper()}<br>
                            <small>
                                {'Accept: C ≥ Θ' if decision == 'Accept' else
                                 'Review: Θ-ε ≤ C < Θ' if decision == 'Review' else
                                 'Reject: C < Θ-ε'}
                            </small>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Show RMMVe process details
                activated_modules = results.get("activated_modules", [])
                st.markdown(f"""
                <div style='background-color: #f0f9ff; padding: 12px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #0ea5e9;'>
                    <h4 style='margin: 0 0 8px 0; color: #0c4a6e;'>RMMVe Module Execution</h4>
                    <p style='margin: 0; font-size: 14px;'>
                        <strong>Activated Modules:</strong> {', '.join(activated_modules) if activated_modules else 'None'}<br>
                        <strong>Sequential Order:</strong> LOV → POV → MAV → WSV → ESV<br>
                        <strong>Termination:</strong> {'Early (confidence ≥ threshold)' if results.get('early_termination') else 'Complete (all modules)'}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
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
                                    <p style='margin-bottom: 5px;'>RMMVe stopped at <b>{module_name}</b> module</p>
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
                            <strong>⚠️ Complete verification.</strong> All 5 RMMVe modules were executed (no early termination).
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Show cumulative confidence calculation
                st.markdown("### Confidence Aggregation")
                activated_modules_info = []
                for module_name in activated_modules:
                    module_result = next((r for r in results.get("module_results", []) if r["module_name"] == module_name), {})
                    # Only include activated modules (those that passed threshold)
                    if module_result.get("activated", False):
                        confidence = module_result.get("confidence", 0)
                        weight = module_result.get("weight", 0)
                        weighted_conf = confidence * weight
                        activated_modules_info.append({
                            "name": module_name,
                            "confidence": confidence,
                            "weight": weight,
                            "weighted": weighted_conf
                        })

                if activated_modules_info:
                    # Create a table showing the calculation using Streamlit components instead of raw HTML
                    st.markdown("#### Cumulative Confidence Calculation")
                    with st.container():
                        # Create a clean table-like display
                        col1, col2, col3, col4 = st.columns([2, 2, 1, 2])

                        with col1:
                            st.markdown("**Module**")
                        with col2:
                            st.markdown("**Confidence**")
                        with col3:
                            st.markdown("**×**")
                        with col4:
                            st.markdown("**Weighted**")

                        st.markdown("---")

                        total_weighted = 0.0
                        total_weights = 0.0

                        for info in activated_modules_info:
                            weighted = info['confidence'] * info['weight']
                            total_weighted += weighted
                            total_weights += info['weight']

                            col1, col2, col3, col4 = st.columns([2, 2, 1, 2])
                            with col1:
                                st.markdown(f"**{info['name']}**")
                            with col2:
                                st.markdown(f"{info['confidence']:.3f}")
                            with col3:
                                st.markdown(f"× {info['weight']:.3f}")
                            with col4:
                                st.markdown(f"= {weighted:.3f}")

                        st.markdown("---")

                        cumulative_conf = total_weighted / total_weights if total_weights > 0 else 0

                        # Final calculation in a colored box
                        st.markdown(f"""
                        <div style='background-color: {decision_color}15; border-left: 4px solid {decision_color};
                                   padding: 12px; border-radius: 6px; margin-top: 10px;'>
                            <strong style='color: {decision_color};'>C = Σ(wᵢ × Sᵢ) / Σ(wᵢ)</strong><br>
                            <span style='font-family: monospace; font-size: 14px;'>
                                {total_weighted:.3f} / {total_weights:.3f} = {cumulative_conf:.3f}
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Create a grid of gauge charts for module performance
                st.markdown("### Module Performance (RMMVe)")
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
                st.markdown("""
                <div style='background-color: #f3f4f6; padding: 40px; border-radius: 8px; text-align: center; margin-top: 20px;'>
                    <div style='font-size: 48px; margin-bottom: 15px;'>📊</div>
                    <h3 style='color: #6b7280; margin-bottom: 10px;'>No Verification Results Yet</h3>
                    <p style='color: #9ca3af;'>Complete the verification pipeline on the left to see results here.</p>
                    </div>
                """, unsafe_allow_html=True)

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
                # Helper function to format decision for history
                def format_decision_for_history(decision):
                    """Format decision string for history display."""
                    if isinstance(decision, bool):
                        # Handle old boolean decisions
                        return "✅ Verified" if decision else "❌ Rejected"
                    elif isinstance(decision, str):
                        # Handle new string decisions
                        if decision == "Accept":
                            return "✅ Verified"
                        elif decision == "Review":
                            return "⚠️ Review"
                        elif decision == "Reject":
                            return "❌ Rejected"
                        else:
                            # Fallback for unknown decision strings
                            return "❓ Unknown"
                    else:
                        return "❌ Rejected"
                
                # Calculate statistics
                history_df = pd.DataFrame([
                    {
                        "Time": datetime.fromtimestamp(record["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
                        "Fact Quality": record["verification_results"].get("fact_quality", "unknown"),
                        "Contains Shift": record["verification_results"].get("contains_shift", "None"),
                        "Total Confidence": round(record["verification_results"]["total_confidence"], 4),
                        "Decision": format_decision_for_history(record['verification_results'].get('decision', 'Reject')),
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