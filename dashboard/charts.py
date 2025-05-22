"""
Charts module for the Focus Monitor Dashboard.

This module provides functions for creating various data visualizations used
in the Focus Monitor Dashboard, including application usage pie charts,
browser usage bar charts, and activity category breakdowns.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_pie_chart(app_data: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a pie chart visualizing application usage breakdown.
    
    Args:
        app_data: List of dictionaries containing application usage data.
                 Each dictionary should have 'appName' and 'timeSpent' keys.
                
    Returns:
        A plotly Figure object with the pie chart visualization.
    """
    fig = go.Figure()
    
    # Handle empty data case
    if not app_data:
        fig.update_layout(title="App Usage Breakdown - No Data Available")
        return fig

    # Sort by time spent and limit to top 9 apps plus "Other"
    data_to_plot = sorted(app_data, key=lambda x: x.get("timeSpent", 0), reverse=True)
    if len(data_to_plot) > 10:
        top_apps = data_to_plot[:9]
        other_time = sum(app.get("timeSpent", 0) for app in data_to_plot[9:])
        if other_time > 0:
            top_apps.append({"appName": "Other Apps", "timeSpent": other_time})
        data_to_plot = top_apps

    # Prepare labels and values for the pie chart
    labels = [
        f"{app.get('appName', 'N/A')} ({app.get('timeSpent', 0) // 60}m)" 
        for app in data_to_plot
    ]
    values = [app.get("timeSpent", 0) for app in data_to_plot]

    # Handle case with no time data
    if not values or sum(values) == 0:
        fig.update_layout(title="App Usage Breakdown - No Time Spent Data")
        return fig

    # Create the pie chart
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            textinfo="percent+label",
            hoverinfo="label+value",
            textfont_size=12,
            pull=[0.05] * len(labels),
        )
    )
    
    # Set layout properties
    fig.update_layout(
        title_text="App Usage Breakdown", 
        height=500, 
        legend_title_text="Applications"
    )
    
    return fig


def create_browser_chart(app_data: List[Dict[str, Any]]) -> Optional[go.Figure]:
    """
    Create a bar chart showing browser usage details.
    
    Args:
        app_data: List of dictionaries containing application usage data.
                 Each dictionary should have 'appName' and 'timeSpent' keys.
                
    Returns:
        A plotly Figure object with the browser usage bar chart,
        or None if no browser data is available.
    """
    if not app_data:
        return None
        
    # Filter and process browser entries
    browser_entries = []
    browser_keywords = ["chrome", "msedge", "edge", "firefox"]
    
    for app in app_data:
        name_lower = app.get("appName", "").lower()
        
        # Check if this is a browser application
        if any(keyword in name_lower for keyword in browser_keywords):
            # Determine browser type
            if "msedge" in name_lower or "edge" in name_lower:
                browser_type = "MS Edge"
            elif "chrome" in name_lower:
                browser_type = "Google Chrome"
            else:
                browser_type = "Firefox"
                
            # Add to browser entries with browser type
            browser_entries.append({**app, "browserType": browser_type})

    # Return None if no browser data found
    if not browser_entries:
        return None

    # Create DataFrame and add minute representation
    df_browsers = pd.DataFrame(browser_entries).sort_values(
        by=["browserType", "timeSpent"], 
        ascending=[True, False]
    )
    df_browsers["timeSpentMinutesText"] = (df_browsers["timeSpent"] // 60).astype(str) + "m"

    # Define color mapping for browsers
    browser_colors = {
        "MS Edge": "#0078D4", 
        "Google Chrome": "#DB4437", 
        "Firefox": "#FF7139"
    }
    
    # Create the bar chart
    fig = px.bar(
        df_browsers,
        x="appName",
        y="timeSpent",
        text="timeSpentMinutesText",
        labels={
            "appName": "Browser Instance / Profile", 
            "timeSpent": "Time Spent (seconds)"
        },
        title="Browser Usage Details",
        color="browserType",
        color_discrete_map=browser_colors,
    )
    
    # Update layout and display properties
    fig.update_layout(
        xaxis_tickangle=-45, 
        height=450, 
        legend_title_text="Browser Type", 
        uniformtext_minsize=8, 
        uniformtext_mode="hide"
    )
    fig.update_traces(textposition="outside")
    
    return fig


def create_category_chart(
    buckets: List[Dict[str, Any]], 
    categories: List[Dict[str, str]]
) -> Optional[go.Figure]:
    """
    Create a pie chart showing time distribution by activity category.
    
    Args:
        buckets: List of time bucket dictionaries, each containing
                category_id, start, and end time information.
        categories: List of category dictionaries with id, name, and description.
                
    Returns:
        A plotly Figure object with the category distribution pie chart,
        or None if no valid data is available.
    """
    # Return None if we don't have the required data
    if not buckets or not categories:
        return None

    # Create mapping from category IDs to names
    cat_id_to_name = {
        cat.get("id", ""): cat.get("name", "Unknown") 
        for cat in categories
    }
    
    # Initialize with Uncategorized
    category_times = {"Uncategorized": 0}
    
    # Calculate time spent in each category
    for bucket in buckets:
        # Get category name from ID
        cat_id = bucket.get("category_id", "")
        cat_name = (
            cat_id_to_name.get(cat_id, "Uncategorized") 
            if cat_id else "Uncategorized"
        )
        
        # Parse start and end times
        start_time = pd.to_datetime(bucket.get("start", ""))
        end_time = pd.to_datetime(bucket.get("end", bucket.get("start", "")))
        
        # Calculate duration if times are valid
        if start_time is not None and end_time is not None:
            duration = (end_time - start_time).total_seconds()
            category_times[cat_name] = category_times.get(cat_name, 0) + duration

    # Filter out categories with zero time
    category_times = {k: v for k, v in category_times.items() if v > 0}
    
    # Return None if no category data remains
    if not category_times:
        return None

    # Prepare data for pie chart
    labels = [f"{cat} ({int(time // 60)}m)" for cat, time in category_times.items()]
    values = list(category_times.values())

    # Create the pie chart
    fig = go.Figure()
    fig.add_trace(
        go.Pie(
            labels=labels, 
            values=values, 
            hole=0.4, 
            textinfo="percent+label", 
            textfont_size=12
        )
    )
    
    # Set layout properties
    fig.update_layout(
        title_text="Time Distribution by Category", 
        height=450
    )
    
    return fig