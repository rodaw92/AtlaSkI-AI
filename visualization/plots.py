import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_verification_results_plotly(verification_results):
    """Plot the verification results as an interactive bar chart using Plotly."""
    # Extract data for plotting
    module_names = []
    confidences = []
    thresholds = []
    
    for result in verification_results["module_results"]:
        module_names.append(result["module_name"])
        confidences.append(result["confidence"])
        thresholds.append(result["threshold"])
    
    # Create figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add confidence bars
    fig.add_trace(
        go.Bar(
            x=module_names,
            y=confidences,
            name="Confidence",
            marker_color='#3b82f6',
            text=[f'{conf:.2f}' for conf in confidences],
            textposition='auto',
        )
    )
    
    # Add threshold bars
    fig.add_trace(
        go.Bar(
            x=module_names,
            y=thresholds,
            name="Threshold",
            marker_color='#f43f5e',
            text=[f'{thr:.2f}' for thr in thresholds],
            textposition='auto',
        )
    )
    
    # Add global threshold line
    if "total_confidence" in verification_results:
        fig.add_trace(
            go.Scatter(
                x=module_names,
                y=[verification_results["total_confidence"]] * len(module_names),
                mode='lines',
                name='Total Confidence',
                line=dict(color='#1e3a8a', width=2, dash='dash'),
            )
        )
    
    # Customize layout
    fig.update_layout(
        title='Verification Results by Module',
        xaxis_title='Module',
        yaxis_title='Score',
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        height=400,
    )
    
    # Highlight early termination if applicable
    if verification_results.get("early_termination") and verification_results.get("early_termination_module"):
        early_term_module = verification_results["early_termination_module"]
        early_term_index = module_names.index(early_term_module)
        confidence = confidences[early_term_index]
        
        fig.add_annotation(
            x=module_names[early_term_index],
            y=confidence + 0.1,
            text="Early Termination",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#ef4444",
            font=dict(size=12, color="#ef4444", family="Arial, sans-serif"),
            align="center",
        )
    
    return fig

def plot_metrics_plotly(verification_results):
    """Plot the metrics used in each module's confidence calculation using Plotly."""
    # Extract data for plotting
    module_names = []
    metric1_values = []
    metric2_values = []
    metric_names = []
    
    metric_mappings = {
        "LOV": ("Precision", "Recall"),
        "POV": ("Accuracy", "Coverage"),
        "MAV": ("Consensus", "Reliability"),
        "WSV": ("Recall", "F1 Score"),
        "ESV": ("Similarity", "1-Anomaly Rate")
    }
    
    for result in verification_results["module_results"]:
        module_name = result["module_name"]
        module_names.append(module_name)
        metric1_values.append(result["metric1"])
        metric2_values.append(result["metric2"])
        metric_names.append(metric_mappings.get(module_name, ("Metric 1", "Metric 2")))
    
    # Create custom x labels with module name and metric name
    x_labels = []
    for module, (metric1, metric2) in zip(module_names, metric_names):
        x_labels.extend([f"{module}: {metric1}", f"{module}: {metric2}"])
    
    # Flatten metrics for grouped bar chart
    all_metrics = []
    for m1, m2 in zip(metric1_values, metric2_values):
        all_metrics.extend([m1, m2])
    
    # Create color mapping for better visualization
    colors = []
    for module in module_names:
        colors.extend(['#3b82f6', '#10b981'])  # Blue for Metric 1, Green for Metric 2
    
    # Create figure
    fig = go.Figure()
    
    # Add bars for all metrics
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=all_metrics,
            marker_color=colors,
            text=[f'{m:.2f}' for m in all_metrics],
            textposition='auto',
        )
    )
    
    # Customize layout
    fig.update_layout(
        title='Verification Metrics by Module',
        xaxis_title='Module and Metric',
        yaxis_title='Value',
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=10)
        ),
        margin=dict(l=20, r=20, t=60, b=100),
        height=450,
    )
    
    return fig

def plot_aaic_cumulative_sums_plotly(parameter_history_df):
    """Plot the CGR-CUSUM cumulative sums for each module using Plotly."""
    if parameter_history_df.empty:
        return None
    
    modules = parameter_history_df["Module"].unique()
    
    # Create dataframe for plotting by module iteration instead of timestamp
    plot_data = []
    for module in modules:
        module_data = parameter_history_df[parameter_history_df["Module"] == module]
        for i, row in enumerate(module_data.itertuples()):
            plot_data.append({
                "Module": row.Module,
                "Iteration": i + 1,
                "Cumulative Sum": row._7,  # Index of "Cumulative Sum" in the tuple
                "Timestamp": row.Timestamp,
                "Performance": row.Performance
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create figure
    fig = go.Figure()
    
    # Set color map for modules
    color_map = {
        "LOV": "#3b82f6",  # Blue
        "POV": "#10b981",  # Green
        "MAV": "#f59e0b",  # Orange
        "WSV": "#8b5cf6",  # Purple
        "ESV": "#06b6d4"   # Teal
    }
    
    # Add lines for each module
    for module in modules:
        module_data = plot_df[plot_df["Module"] == module]
        # Convert to lists to ensure Plotly compatibility
        x_values = module_data["Iteration"].tolist()
        y_values = module_data["Cumulative Sum"].tolist()
        custom_data = module_data["Performance"].tolist()
        
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines+markers',
                name=module,
                line=dict(color=color_map.get(module, "#000000"), width=2),
                hovertemplate=
                "<b>%{x}</b><br>" +
                "Module: " + module + "<br>" +
                "Cum. Sum: %{y:.3f}<br>" +
                "Performance: %{customdata:.3f}<extra></extra>",
                customdata=custom_data
            )
        )
    
    # Add threshold line
    h_value = 5.0  # Default value
    fig.add_trace(
        go.Scatter(
            x=[1, plot_df["Iteration"].max()],
            y=[h_value, h_value],
            mode='lines',
            name=f'Threshold (h={h_value})',
            line=dict(color='#ef4444', width=2, dash='dash'),
        )
    )
    
    # Customize layout
    fig.update_layout(
        title='CGR-CUSUM Monitoring',
        xaxis_title='Verification Iteration',
        yaxis_title='Cumulative Sum',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        height=400,
        hovermode="closest"
    )
    
    return fig

def plot_aaic_parameter_evolution_plotly(parameter_history_df, param_type):
    """Plot the evolution of module parameters over time using Plotly."""
    if parameter_history_df.empty:
        return None
    
    modules = parameter_history_df["Module"].unique()
    
    # Create dataframe for plotting by module iteration instead of timestamp
    plot_data = []
    for module in modules:
        module_data = parameter_history_df[parameter_history_df["Module"] == module]
        for i, row in enumerate(module_data.itertuples()):
            param_value = getattr(row, param_type)
            plot_data.append({
                "Module": row.Module,
                "Iteration": i + 1,
                "Value": param_value,
                "Timestamp": row.Timestamp,
                "Cumulative Sum": row._7  # Index of Cumulative Sum in tuple
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create figure
    fig = go.Figure()
    
    # Set color map for modules
    color_map = {
        "LOV": "#3b82f6",  # Blue
        "POV": "#10b981",  # Green
        "MAV": "#f59e0b",  # Orange
        "WSV": "#8b5cf6",  # Purple
        "ESV": "#06b6d4"   # Teal
    }
    
    # Add lines for each module
    for module in modules:
        module_data = plot_df[plot_df["Module"] == module]
        # Convert to list to ensure Plotly compatibility
        x_values = module_data["Iteration"].tolist()
        y_values = module_data["Value"].tolist()
        cum_sums = module_data["Cumulative Sum"].tolist()
        
        # Find points where parameter changes occurred (where cum_sum likely exceeded threshold)
        change_points = []
        change_values = []
        
        for i in range(1, len(y_values)):
            if abs(y_values[i] - y_values[i-1]) > 0.001:  # Detect significant parameter change
                change_points.append(x_values[i])
                change_values.append(y_values[i])
        
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines+markers',
                name=module,
                line=dict(color=color_map.get(module, "#000000"), width=2),
                hovertemplate=
                "<b>Iteration %{x}</b><br>" +
                "Module: " + module + "<br>" +
                f"{param_type}: %{{y:.3f}}<br>" +
                "Cum. Sum: %{customdata:.3f}<extra></extra>",
                customdata=cum_sums
            )
        )
        
        # Add markers for parameter change points
        if change_points:
            fig.add_trace(
                go.Scatter(
                    x=change_points,
                    y=change_values,
                    mode='markers',
                    marker=dict(
                        symbol='star',
                        size=12,
                        color='red',
                        line=dict(width=2, color='DarkSlateGrey')
                    ),
                    name=f'{module} Parameter Change',
                    hovertemplate=
                    "<b>Parameter Updated</b><br>" +
                    "Module: " + module + "<br>" +
                    f"New {param_type}: %{{y:.3f}}<extra></extra>"
                )
            )
    
    # Param-specific styling
    if param_type == "Weight":
        title = "Module Weights Evolution"
        y_title = "Weight (w)"
    elif param_type == "Threshold":
        title = "Module Thresholds Evolution"
        y_title = "Threshold (θ)"
    elif param_type == "Alpha":
        title = "Module Alpha Parameters Evolution"
        y_title = "Alpha (α)"
    elif param_type == "Performance":
        title = "Module Performance Evolution"
        y_title = "Performance"
    else:
        title = f"{param_type} Evolution"
        y_title = param_type
    
    # Customize layout
    fig.update_layout(
        title=title,
        xaxis_title='Verification Iteration',
        yaxis_title=y_title,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        height=400,
        hovermode="closest"
    )
    
    return fig

def plot_verification_history_plotly(history_df):
    """Plot verification history metrics using Plotly."""
    if history_df.empty:
        return None
    
    # Create figure for confidence over time
    fig = go.Figure()
    
    # Convert range to list for Plotly - range() objects aren't directly accepted
    x_values = list(range(1, len(history_df) + 1))
    
    fig.add_trace(
        go.Scatter(
            x=x_values,  # Using list instead of range object
            y=history_df["Total Confidence"],
            mode='lines+markers',
            name='Total Confidence',
            line=dict(color='#3b82f6', width=2),
            marker=dict(
                size=8,
                color=history_df["Decision"].apply(lambda x: '#10b981' if "Verified" in x else '#ef4444'),
                symbol=history_df["Early Termination"].apply(lambda x: 'circle' if x == "✓ Yes" else 'x')
            ),
            hovertemplate=
            "<b>Verification %{x}</b><br>" +
            "Confidence: %{y:.3f}<br>" +
            "Decision: %{customdata[0]}<br>" +
            "Early Term: %{customdata[1]}<br>" +
            "Fact Quality: %{customdata[2]}<extra></extra>",
            customdata=list(zip(
                history_df["Decision"],
                history_df["Early Termination"],
                history_df["Fact Quality"]
            ))
        )
    )
    
    # Add threshold line (using list for x values)
    fig.add_trace(
        go.Scatter(
            x=[1, len(history_df)],
            y=[0.65, 0.65],  # Default global threshold
            mode='lines',
            name='Global Threshold',
            line=dict(color='#6b7280', width=2, dash='dash'),
        )
    )
    
    # Customize layout
    fig.update_layout(
        title='Verification Confidence History',
        xaxis_title='Verification Number',
        yaxis_title='Total Confidence',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        height=400,
        hovermode="closest"
    )
    
    return fig

def plot_quality_distribution_plotly(history_df):
    """Plot fact quality distribution as a pie chart using Plotly."""
    if history_df.empty:
        return None
    
    # Count quality occurrences
    quality_counts = history_df["Fact Quality"].value_counts().reset_index()
    quality_counts.columns = ["Quality", "Count"]
    
    # Calculate percentages
    total = quality_counts["Count"].sum()
    quality_counts["Percentage"] = quality_counts["Count"] / total * 100
    
    # Create color map that matches the app's styling
    color_map = {
        "high_quality": "#10b981",  # Green
        "medium_quality": "#3b82f6",  # Blue
        "spatial_issue": "#f59e0b",  # Orange
        "external_ref": "#8b5cf6",  # Purple
        "semantic_issue": "#06b6d4",  # Teal
        "low_quality": "#ef4444"     # Red
    }
    
    colors = [color_map.get(q, "#6b7280") for q in quality_counts["Quality"]]
    
    # Create figure
    fig = go.Figure(
        data=[
            go.Pie(
                labels=quality_counts["Quality"],
                values=quality_counts["Percentage"],
                hole=0.4,
                marker=dict(colors=colors),
                textinfo='label+percent',
                insidetextorientation='radial',
                hovertemplate="<b>%{label}</b><br>Count: %{value:.1f}%<extra></extra>"
            )
        ]
    )
    
    # Customize layout
    fig.update_layout(
        title='Distribution of Fact Qualities',
        margin=dict(l=20, r=20, t=60, b=20),
        height=400,
    )
    
    return fig

def plot_early_term_by_module_plotly(history_df):
    """Plot early termination by module as a bar chart using Plotly."""
    if history_df.empty:
        return None
    
    # Get early termination counts by module
    term_counts = history_df["Term. Module"].value_counts().reset_index()
    term_counts.columns = ["Module", "Count"]
    term_counts = term_counts[term_counts["Module"] != "None"]
    
    if term_counts.empty:
        return None
    
    # Calculate percentages
    total = term_counts["Count"].sum()
    term_counts["Percentage"] = term_counts["Count"] / total * 100
    
    # Set color map for modules
    color_map = {
        "LOV": "#3b82f6",  # Blue
        "POV": "#10b981",  # Green
        "MAV": "#f59e0b",  # Orange
        "WSV": "#8b5cf6",  # Purple
        "ESV": "#06b6d4"   # Teal
    }
    
    colors = [color_map.get(m, "#6b7280") for m in term_counts["Module"]]
    
    # Create figure
    fig = go.Figure(
        data=[
            go.Bar(
                x=term_counts["Module"],
                y=term_counts["Percentage"],
                marker_color=colors,
                text=[f'{p:.1f}%' for p in term_counts["Percentage"]],
                textposition='auto',
                hovertemplate="<b>%{x}</b><br>Percentage: %{y:.1f}%<br>Count: %{customdata}<extra></extra>",
                customdata=term_counts["Count"]
            )
        ]
    )
    
    # Customize layout
    fig.update_layout(
        title='Distribution of Early Terminations by Module',
        xaxis_title='Module',
        yaxis_title='Percentage of Early Terminations',
        margin=dict(l=20, r=20, t=60, b=20),
        height=400,
    )
    
    return fig

def create_module_performance_gauge(module_name, confidence, threshold):
    """Create a gauge chart for module performance using Plotly."""
    # Set color based on confidence vs threshold
    if confidence >= threshold:
        color = "#10b981"  # Green for above threshold
    elif confidence >= 0.7 * threshold:
        color = "#f59e0b"  # Orange for close to threshold
    else:
        color = "#ef4444"  # Red for below threshold
    
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence,
            domain=dict(x=[0, 1], y=[0, 1]),
            title=dict(text=module_name, font=dict(size=16)),
            gauge=dict(
                axis=dict(range=[0, 1], tickwidth=1),
                bar=dict(color=color),
                bgcolor="white",
                borderwidth=2,
                bordercolor="gray",
                steps=[
                    dict(range=[0, 0.5], color="#fee2e2"),
                    dict(range=[0.5, 0.8], color="#fef3c7"),
                    dict(range=[0.8, 1.0], color="#d1fae5")
                ],
                threshold=dict(
                    line=dict(color="red", width=4),
                    thickness=0.75,
                    value=threshold
                )
            ),
            number=dict(
                font=dict(color=color),
                suffix=""
            )
        )
    )
    
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    
    return fig 