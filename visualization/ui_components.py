import streamlit as st
import pandas as pd
from datetime import datetime

from models.constants import QUALITY_COLOR_MAP

def create_fact_preview_card(fact):
    """Create a formatted preview of a candidate fact as an HTML card."""
    fact_data = fact.get("spatiotemporal_inspection_data", {})
    report_id = fact_data.get("report_id", "N/A")
    
    preview = f"""
    <div class="card">
        <h3>Report ID: {report_id}</h3>
    """
    
    # Add inspection measurements
    if "inspection_data" in fact_data and "inspection_measurements" in fact_data["inspection_data"]:
        preview += "<h4>Measurements</h4>"
        
        for i, item in enumerate(fact_data["inspection_data"]["inspection_measurements"]):
            component_id = item.get("component_id", "N/A")
            measurement_id = item.get("measurement_id", "N/A")
            feature_name = item.get("feature_name", "N/A")
            nominal = item.get("nominal_value_mm", "N/A")
            actual = item.get("actual_value_mm", "N/A")
            deviation = item.get("deviation_mm", "N/A")
            status = item.get("status", "N/A")
            
            status_class = "status-verified" if status == "PASS" else "status-rejected"
            
            preview += f"""
            <div style="margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid #e5e7eb;">
                <div><strong>Component:</strong> {component_id}</div>
                <div><strong>ID:</strong> {measurement_id}</div>
                <div><strong>Feature:</strong> {feature_name}</div>
                <div><strong>Nominal:</strong> {nominal} mm</div>
                <div><strong>Actual:</strong> {actual} mm</div>
                <div><strong>Deviation:</strong> {deviation} mm</div>
                <div><strong>Status:</strong> <span class="{status_class}">{status}</span></div>
            </div>
            """
    
    # Add temporal data
    if "temporal_data" in fact_data:
        preview += "<h4>Temporal Data</h4>"
        preview += "<div style='margin-bottom: 12px;'>"
        
        for i, item in enumerate(fact_data["temporal_data"]):
            entity_id = item.get("entity_id", "N/A")
            timestamp = item.get("timestamp", "N/A")
            event_type = item.get("event_type_or_feature", "N/A")
            
            preview += f"<div>Entity: {entity_id}, Time: {timestamp}, Event: {event_type}</div>"
        
        preview += "</div>"
    
    # Add spatial data
    if "spatial_data" in fact_data:
        preview += "<h4>Spatial Data</h4>"
        preview += "<div style='margin-bottom: 12px;'>"
        
        for i, item in enumerate(fact_data["spatial_data"]):
            entity_id = item.get("entity_id", "N/A")
            coords = item.get("coordinates", {})
            x = coords.get("x_coord", "N/A")
            y = coords.get("y_coord", "N/A")
            z = coords.get("z_coord", "N/A")
            
            preview += f"<div>Entity: {entity_id}, Coordinates: ({x}, {y}, {z})</div>"
        
        preview += "</div>"
    
    # Add relationships
    if "relationships" in fact_data:
        preview += "<h4>Relationships</h4>"
        preview += "<div>"
        
        for i, rel in enumerate(fact_data["relationships"]):
            rel_id = rel.get("relationship_id", "N/A")
            subject = rel.get("subject_entity_id", "N/A")
            rel_type = rel.get("relationship_type", "N/A")
            object_id = rel.get("object_entity_id", "N/A")
            
            preview += f"<div>{rel_id}: {subject} --{rel_type}--> {object_id}</div>"
        
        preview += "</div>"
    
    preview += "</div>"
    return preview

def get_quality_badge(quality):
    """Return HTML for a quality badge."""
    quality_classes = {
        "high_quality": "quality-high",
        "medium_quality": "quality-medium",
        "spatial_issue": "quality-spatial",
        "external_ref": "quality-external",
        "semantic_issue": "quality-semantic",
        "low_quality": "quality-low"
    }
    
    quality_names = {
        "high_quality": "High Quality",
        "medium_quality": "Medium Quality",
        "spatial_issue": "Spatial Issue",
        "external_ref": "External Reference",
        "semantic_issue": "Semantic Issue",
        "low_quality": "Low Quality"
    }
    
    cls = quality_classes.get(quality, "")
    name = quality_names.get(quality, quality)
    
    return f'<span class="{cls}">{name}</span>'

def get_decision_badge(decision):
    """Return HTML for a decision badge."""
    if decision:
        return '<span class="status-verified">VERIFIED</span>'
    else:
        return '<span class="status-rejected">REJECTED</span>'

def format_shift_alert(shift_module):
    """Format an alert for performance shift."""
    if not shift_module:
        return ""
    
    return f"""
    <div class="alert-warning">
        <strong>⚠️ Performance Shift Detected:</strong> This fact contains a shift in the {shift_module} module
        that may trigger adaptation by the AAIC system.
    </div>
    """

def display_parameter_change(shift):
    """Display parameter changes using Streamlit native components instead of HTML."""
    if shift is None or (hasattr(shift, 'empty') and shift.empty):
        return
    
    # Convert to dictionary if it's a pandas Series
    if hasattr(shift, 'to_dict'):
        shift = shift.to_dict()
    
    # Create an expander with module name
    with st.expander(f"Shift: Module {shift['module']}", expanded=True):
        # Display metrics at the top
        cols = st.columns(3)
        with cols[0]:
            st.metric("Cumulative Sum", f"{shift['cumulative_sum']:.3f}")
        with cols[1]:
            st.metric("Performance", f"{shift['performance']*100:.1f}%")
        with cols[2]:
            timestamp = datetime.fromtimestamp(shift['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
            st.metric("Timestamp", timestamp)
        
        st.markdown("### Parameter Adjustments")
        
        # Create a dataframe for parameter changes
        param_data = {
            "Parameter": ["Weight (w)", "Threshold (θ)", "Alpha (α)"],
            "Old Value": [
                f"{shift['old_weight']:.4f}",
                f"{shift['old_threshold']:.4f}",
                f"{shift['old_alpha']:.4f}"
            ],
            "New Value": [
                f"{shift['new_weight']:.4f}",
                f"{shift['new_threshold']:.4f}",
                f"{shift['new_alpha']:.4f}"
            ]
        }
        
        # Calculate changes and format with arrows
        weight_pct = round((shift["weight_change"] / shift["old_weight"]) * 100, 1)
        threshold_pct = round((shift["threshold_change"] / shift["old_threshold"]) * 100, 1)
        alpha_pct = round((shift["alpha_change"] / shift["old_alpha"]) * 100, 1)
        
        param_data["Change"] = [
            f"{'↑' if weight_pct >= 0 else '↓'} {abs(weight_pct)}%",
            f"{'↑' if threshold_pct >= 0 else '↓'} {abs(threshold_pct)}%",
            f"{'↑' if alpha_pct >= 0 else '↓'} {abs(alpha_pct)}%"
        ]
        
        # Add descriptions
        param_data["Description"] = [
            "Module influence",
            "Early term. threshold",
            "Metric balance"
        ]
        
        # Display as table
        st.table(pd.DataFrame(param_data))
        
        st.info("Parameters adjusted according to AAIC Algorithm (Algorithm 1) from the paper.")

def create_gauge_grid(verification_results):
    """Create a grid of gauge charts for all modules."""
    if not verification_results["module_results"]:
        return None
    
    from .plots import create_module_performance_gauge
    
    gauges = []
    for result in verification_results["module_results"]:
        module_name = result["module_name"]
        confidence = result["confidence"]
        threshold = result["threshold"]
        
        gauge = create_module_performance_gauge(module_name, confidence, threshold)
        gauges.append(gauge)
    
    return gauges 