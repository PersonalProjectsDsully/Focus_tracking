from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_pie_chart(app_data: List[Dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    if not app_data:
        fig.update_layout(title="App Usage Breakdown - No Data Available")
        return fig

    data_to_plot = sorted(app_data, key=lambda x: x.get("timeSpent", 0), reverse=True)
    if len(data_to_plot) > 10:
        top_apps = data_to_plot[:9]
        other_time = sum(app.get("timeSpent", 0) for app in data_to_plot[9:])
        if other_time > 0:
            top_apps.append({"appName": "Other Apps", "timeSpent": other_time})
        data_to_plot = top_apps

    labels = [f"{app.get('appName','N/A')} ({app.get('timeSpent',0)//60}m)" for app in data_to_plot]
    values = [app.get("timeSpent", 0) for app in data_to_plot]

    if not values or sum(values) == 0:
        fig.update_layout(title="App Usage Breakdown - No Time Spent Data")
        return fig

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
    fig.update_layout(title_text="App Usage Breakdown", height=500, legend_title_text="Applications")
    return fig


def create_browser_chart(app_data: List[Dict[str, Any]]) -> Optional[go.Figure]:
    if not app_data:
        return None
    browser_entries = []
    for app in app_data:
        name_lower = app.get("appName", "").lower()
        if any(b_keyword in name_lower for b_keyword in ["chrome", "msedge", "edge", "firefox"]):
            browser_type = (
                "MS Edge"
                if "msedge" in name_lower or "edge" in name_lower
                else ("Google Chrome" if "chrome" in name_lower else "Firefox")
            )
            browser_entries.append({**app, "browserType": browser_type})

    if not browser_entries:
        return None

    df_browsers = pd.DataFrame(browser_entries).sort_values(by=["browserType", "timeSpent"], ascending=[True, False])
    df_browsers["timeSpentMinutesText"] = (df_browsers["timeSpent"] // 60).astype(str) + "m"

    fig = px.bar(
        df_browsers,
        x="appName",
        y="timeSpent",
        text="timeSpentMinutesText",
        labels={"appName": "Browser Instance / Profile", "timeSpent": "Time Spent (seconds)"},
        title="Browser Usage Details",
        color="browserType",
        color_discrete_map={"MS Edge": "#0078D4", "Google Chrome": "#DB4437", "Firefox": "#FF7139"},
    )
    fig.update_layout(xaxis_tickangle=-45, height=450, legend_title_text="Browser Type", uniformtext_minsize=8, uniformtext_mode="hide")
    fig.update_traces(textposition="outside")
    return fig


def create_category_chart(buckets: List[Dict[str, Any]], categories: List[Dict[str, str]]) -> Optional[go.Figure]:
    if not buckets or not categories:
        return None

    cat_id_to_name = {cat.get("id", ""): cat.get("name", "Unknown") for cat in categories}
    category_times = {"Uncategorized": 0}
    for bucket in buckets:
        cat_id = bucket.get("category_id", "")
        cat_name = cat_id_to_name.get(cat_id, "Uncategorized") if cat_id else "Uncategorized"
        start_time = pd.to_datetime(bucket.get("start", ""))
        end_time = pd.to_datetime(bucket.get("end", bucket.get("start", "")))
        if start_time is not None and end_time is not None:
            duration = (end_time - start_time).total_seconds()
            category_times[cat_name] = category_times.get(cat_name, 0) + duration

    category_times = {k: v for k, v in category_times.items() if v > 0}
    if not category_times:
        return None

    labels = [f"{cat} ({int(time//60)}m)" for cat, time in category_times.items()]
    values = list(category_times.values())

    fig = go.Figure()
    fig.add_trace(
        go.Pie(labels=labels, values=values, hole=0.4, textinfo="percent+label", textfont_size=12)
    )
    fig.update_layout(title_text="Time Distribution by Category", height=450)
    return fig
