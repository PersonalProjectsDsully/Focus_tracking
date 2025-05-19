import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
import re
import os
from pathlib import Path
from datetime import datetime
import time
import subprocess
import sys
import psutil

# --- Configuration ---
LOGS_DIR = Path(__file__).parent / "focus_logs"
LABELS_FILE = LOGS_DIR / "activity_labels.json"
FEEDBACK_FILE = LOGS_DIR / "block_feedback.json"

# --- Tracker Control Functions ---
def is_tracker_running():
    pid = st.session_state.get("tracker_pid")
    if pid is None:
        return False
    return psutil.pid_exists(pid)


def start_tracker():
    """Launch the focus monitor as a background process."""
    if is_tracker_running():
        st.info("Focus tracker already running.")
        return
    script = str(Path(__file__).parent / "standalone_focus_monitor.py")
    process = subprocess.Popen([sys.executable, script])
    st.session_state["tracker_pid"] = process.pid
    st.success("Focus tracker started")


def stop_tracker():
    """Terminate the running focus monitor process."""
    pid = st.session_state.get("tracker_pid")
    if not pid or not psutil.pid_exists(pid):
        st.info("Focus tracker is not running.")
        st.session_state["tracker_pid"] = None
        return
    try:
        p = psutil.Process(pid)
        p.terminate()
        p.wait(timeout=5)
        st.success("Focus tracker stopped")
    except Exception as e:
        st.error(f"Failed to stop tracker: {e}")
    finally:
        st.session_state["tracker_pid"] = None

# --- Data Management Functions ---
def load_available_dates():
    """Get all dates that have either logs or summaries"""
    log_dates = set([
        f.name.split("_")[-1].replace(".jsonl", "")
        for f in LOGS_DIR.glob("focus_log_*.jsonl")
    ])
    summary_dates = set([
        f.name.split("_")[-1].replace(".json", "")
        for f in LOGS_DIR.glob("daily_summary_*.json")
    ])
    return sorted(log_dates.union(summary_dates), reverse=True)

def load_log_entries(date_str):
    """Load raw log entries for a given date"""
    file_path = LOGS_DIR / f"focus_log_{date_str}.jsonl"
    if not file_path.exists():
        return pd.DataFrame()
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                data.append(entry)
            except:
                continue
    return pd.DataFrame(data)

def load_daily_summary(date_str):
    """Load summary if exists, or generate on-the-fly from logs"""
    file_path = LOGS_DIR / f"daily_summary_{date_str}.json"
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        # If no summary exists, try to generate one from logs
        log_path = LOGS_DIR / f"focus_log_{date_str}.jsonl"
        if log_path.exists():
            st.info("No summary found for this date. Generating from raw logs...")
            return generate_summary_from_logs(date_str)
        else:
            st.warning(f"No data available for {date_str}")
            return None

def generate_summary_from_logs(date_str):
    """Generate a summary on-the-fly from log files (simplified version)"""
    # This is a simplified placeholder - in the actual implementation
    # this would generate a complete summary like the main script does
    logs_df = load_log_entries(date_str)
    if logs_df.empty:
        return None
        
    # Apply labels to the logs
    logs_df = apply_labels_to_logs(logs_df)
    
    # Basic summary structure
    summary = {
        "date": date_str,
        "totalTime": logs_df["duration"].sum(),
        "appBreakdown": []
    }
    
    # Group by app_name and create breakdown
    app_groups = logs_df.groupby("app_name").agg({
        "exe": "first",
        "duration": "sum",
        "title": lambda x: list(set(x))[:50]
    }).reset_index()
    
    # Create app breakdown entries
    for _, row in app_groups.iterrows():
        percentage = (row["duration"] / summary["totalTime"] * 100) if summary["totalTime"] > 0 else 0
        summary["appBreakdown"].append({
            "appName": row["app_name"],
            "exePath": row["exe"],
            "timeSpent": int(row["duration"]),
            "percentage": round(percentage, 2),
            "windowTitles": row["title"]
        })
    
    # Sort by time spent
    summary["appBreakdown"].sort(key=lambda x: x["timeSpent"], reverse=True)
    
    return summary

# --- Label Management Functions ---
def load_labels():
    """Load saved activity labels"""
    if not LABELS_FILE.exists():
        return {}
    
    try:
        with open(LABELS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading labels: {e}")
        return {}

def save_labels(labels):
    """Save activity labels to file"""
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(LABELS_FILE, 'w', encoding='utf-8') as f:
            json.dump(labels, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving labels: {e}")
        return False

def get_app_key(exe, title_pattern=None):
    """Create a unique key for an application or pattern"""
    if title_pattern:
        return f"{exe}::{title_pattern}"
    return exe

def apply_labels_to_logs(logs_df):
    """Apply custom labels to log entries based on saved labels"""
    if logs_df.empty:
        return logs_df
    
    # Load labels
    labels = load_labels()
    if not labels:
        # If no labels, just add the original_app_name column to match expected schema
        result_df = logs_df.copy()
        result_df['original_app_name'] = result_df['app_name']
        return result_df
    
    # Make a copy to avoid modifying the original
    df = logs_df.copy()
    
    # First, preserve the original app name
    df['original_app_name'] = df['app_name']
    
    # Create a new column for labeled app name
    df['labeled_app_name'] = df['app_name']
    
    # Apply exact title matches first (highest priority)
    for key, label in labels.get('exact_titles', {}).items():
        try:
            _, exe, title = key.split("::", 2)
            # Find exact matches for both exe and title
            exe_mask = df['exe'].str.lower() == exe.lower()
            title_mask = df['title'] == title  # Exact match for titles
            combined_mask = exe_mask & title_mask
            df.loc[combined_mask, 'labeled_app_name'] = label
        except Exception as e:
            # Skip invalid patterns
            print(f"Error with exact title match: {e}")
            continue
    
    # Apply exact executable matches next
    for exe, label in labels.get('exact_exe', {}).items():
        mask = df['exe'].str.lower().str.contains(exe.lower())
        # Only update those that haven't been labeled by exact title match
        mask = mask & (df['labeled_app_name'] == df['original_app_name'])
        df.loc[mask, 'labeled_app_name'] = label
    
    # Apply pattern matches last (lowest priority)
    for key, label in labels.get('patterns', {}).items():
        try:
            _, exe, pattern = key.split("::", 2)
            # Match both exe and title pattern
            exe_mask = df['exe'].str.lower().str.contains(exe.lower())
            title_mask = df['title'].str.lower().str.contains(pattern.lower())
            combined_mask = exe_mask & title_mask
            # Only update those that haven't been labeled by higher priority rules
            combined_mask = combined_mask & (df['labeled_app_name'] == df['original_app_name'])
            df.loc[combined_mask, 'labeled_app_name'] = label
        except Exception as e:
            # Skip invalid patterns
            print(f"Error with pattern match: {e}")
            continue
    
    # Update the app_name with the labeled version
    df['app_name'] = df['labeled_app_name']
    df.drop(columns=['labeled_app_name'], inplace=True)
    
    return df

def group_activities_for_labeling(logs_df):
    """Group similar activities for easier labeling, focusing on window titles"""
    if logs_df.empty:
        return pd.DataFrame()
    
    # Extract key information about window titles
    title_groups = []
    
    # First, get a count of each unique window title
    title_counts = logs_df['title'].value_counts().reset_index()
    title_counts.columns = ['title', 'count']
    
    # For each title, get app, duration, etc.
    for _, row in title_counts.iterrows():
        title = row['title']
        count = row['count']
        
        # Get all logs with this title
        title_logs = logs_df[logs_df['title'] == title]
        
        # Get the main app for this title - safely handling empty cases
        try:
            mode_result = title_logs['app_name'].mode()
            main_app = mode_result.iloc[0] if not mode_result.empty else "Unknown"
        except:
            main_app = title_logs['app_name'].iloc[0] if not title_logs.empty else "Unknown"
            
        # Same for exe
        try:
            mode_result = title_logs['exe'].mode()
            main_exe = mode_result.iloc[0] if not mode_result.empty else "Unknown"
        except:
            main_exe = title_logs['exe'].iloc[0] if not title_logs.empty else "Unknown"
        
        # Calculate total duration for this title
        total_duration = title_logs['duration'].sum()
        
        title_groups.append({
            'title': title,
            'app_name': main_app,
            'exe': main_exe,
            'count': count,
            'duration': total_duration
        })
    
    # Convert to DataFrame
    grouped = pd.DataFrame(title_groups)
    
    # Add percentage of total time
    total_time = logs_df['duration'].sum()
    if total_time > 0:
        grouped['percentage'] = (grouped['duration'] / total_time * 100).round(1)
    else:
        grouped['percentage'] = 0
    
    # Sort by duration
    return grouped.sort_values('duration', ascending=False)

# --- Feedback Management Functions ---
def load_block_feedback():
    """Load saved feedback for time blocks"""
    if not FEEDBACK_FILE.exists():
        return {}
    try:
        with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading feedback: {e}")
        return {}


def save_block_feedback(feedback):
    """Save feedback mapping to file"""
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f:
            json.dump(feedback, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving feedback: {e}")
        return False


def load_time_buckets(date_str):
    """Load 5-minute time bucket summaries for a given date"""
    buckets = []
    for path in LOGS_DIR.glob('time_buckets_*.json'):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for b in data:
                    start = b.get('start', '')
                    if start.split('T')[0] == date_str:
                        buckets.append(b)
        except Exception:
            continue
    buckets.sort(key=lambda x: x.get('start'))
    return buckets

# --- Visualization Functions ---
def create_pie_chart(app_data):
    """Create a plotly pie chart from app data"""
    if len(app_data) > 10:
        # If more than 10 apps, group smaller ones as "Other"
        top_apps = app_data[:10]
        other_time = sum(app["timeSpent"] for app in app_data[10:])
        other_percentage = sum(app["percentage"] for app in app_data[10:])
        
        if other_time > 0:
            top_apps.append({
                "appName": "Other Apps",
                "timeSpent": other_time,
                "percentage": other_percentage
            })
    else:
        top_apps = app_data
    
    labels = [f"{app['appName']} ({app['timeSpent'] // 60}m)" for app in top_apps]
    values = [app["timeSpent"] for app in top_apps]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3,
        textinfo='percent',
        hoverinfo='label+value',
        textfont_size=14
    )])
    
    fig.update_layout(
        title="App Usage Breakdown",
        height=500
    )
    
    return fig

def create_browser_chart(app_data):
    """Create a chart focusing on browser usage"""
    # Extract browser data
    browser_data = []
    for app in app_data:
        app_name_lower = app["appName"].lower()
        
        # Check if this is a browser
        is_chrome = "chrome" in app_name_lower
        is_edge = "msedge" in app_name_lower or "edge" in app_name_lower
        is_firefox = "firefox" in app_name_lower
        
        if is_chrome or is_edge or is_firefox:
            # Group by browser type for color coding
            if is_edge:
                browser_type = "Microsoft Edge"
            elif is_chrome:
                browser_type = "Google Chrome"
            else:
                browser_type = "Firefox"
            
            # Add to the browser data
            browser_data.append({
                "appName": app["appName"],
                "timeSpent": app["timeSpent"],
                "percentage": app["percentage"],
                "browserType": browser_type
            })
    
    if not browser_data:
        return None
    
    # Sort by browser type then by time spent
    browser_data.sort(key=lambda x: (x["browserType"], -x["timeSpent"]))
    
    # Create the visualization
    labels = [app["appName"] for app in browser_data]
    values = [app["timeSpent"] for app in browser_data]
    browser_types = [app["browserType"] for app in browser_data]
    minutes = [f"{app['timeSpent'] // 60}m" for app in browser_data]
    
    fig = px.bar(
        x=labels, 
        y=values,
        text=minutes,
        labels={"x": "Browser", "y": "Time (seconds)"},
        title="Browser Usage",
        color=browser_types,
        color_discrete_map={
            "Microsoft Edge": "#0078D7",
            "Google Chrome": "#4285F4",
            "Firefox": "#FF9500"
        }
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        legend_title="Browser Type"
    )
    
    return fig

def check_for_chart_image(date_str):
    """Check if a pre-generated chart image exists"""
    chart_path = LOGS_DIR / f"usage_chart_{date_str}.png"
    return chart_path if chart_path.exists() else None

# --- Label Editor Page ---
def display_label_editor():
    st.title("🏷 Label Editor")
    
    # Select date for labeling
    available_dates = load_available_dates()
    if not available_dates:
        st.error("No data found in focus_logs directory.")
        st.warning("Focus monitor must be running to collect data.")
        st.stop()
    
    selected_date = st.selectbox("Select a date to label activities", available_dates)
    
    # Load logs for the selected date
    logs_df = load_log_entries(selected_date)
    if logs_df.empty:
        st.warning(f"No log entries available for {selected_date}")
        st.stop()
    
    # Group activities by window title for easier labeling
    try:
        title_groups = group_activities_for_labeling(logs_df)
    except Exception as e:
        st.error(f"Error grouping activities: {e}")
        title_groups = pd.DataFrame()
        
    if title_groups.empty:
        st.warning("No valid window titles found to label")
        return
    
    # Load existing labels
    labels = load_labels()
    
    # Display current labels
    st.subheader("Current Custom Labels")
    
    if not labels.get('exact_exe', {}) and not labels.get('patterns', {}) and not labels.get('exact_titles', {}):
        st.info("No custom labels defined yet. Use the forms below to create labels.")
    else:
        # Create tabs for different label types
        label_tab1, label_tab2, label_tab3 = st.tabs(["Window Title Labels", "Application Labels", "Pattern Labels"])
        
        # Tab 1: Exact window title labels
        with label_tab1:
            if not labels.get('exact_titles', {}):
                st.info("No window title labels defined yet.")
            else:
                title_labels = []
                for key, label in labels.get('exact_titles', {}).items():
                    try:
                        _, exe, title = key.split("::", 2)
                        title_labels.append({
                            "Application": os.path.basename(exe),
                            "Window Title": title[:50] + ("..." if len(title) > 50 else ""),
                            "Label": label
                        })
                    except:
                        continue
                
                if title_labels:
                    st.dataframe(pd.DataFrame(title_labels), use_container_width=True)
        
        # Tab 2: Application labels
        with label_tab2:
            if not labels.get('exact_exe', {}):
                st.info("No application labels defined yet.")
            else:
                exe_labels_df = pd.DataFrame([
                    {"Application": os.path.basename(exe), "Label": label}
                    for exe, label in labels.get('exact_exe', {}).items()
                ])
                st.dataframe(exe_labels_df, use_container_width=True)
        
        # Tab 3: Pattern-based labels
        with label_tab3:
            if not labels.get('patterns', {}):
                st.info("No pattern labels defined yet.")
            else:
                pattern_labels = []
                for key, label in labels.get('patterns', {}).items():
                    try:
                        _, exe, pattern = key.split("::", 2)
                        pattern_labels.append({
                            "Application": os.path.basename(exe),
                            "Window Title Pattern": pattern,
                            "Label": label
                        })
                    except:
                        continue
                
                if pattern_labels:
                    st.dataframe(pd.DataFrame(pattern_labels), use_container_width=True)
    
    # Ensure all required label types exist
    if 'exact_exe' not in labels:
        labels['exact_exe'] = {}
    if 'patterns' not in labels:
        labels['patterns'] = {}
    if 'exact_titles' not in labels:
        labels['exact_titles'] = {}
    
    # Display window titles for labeling
    st.subheader("Available Window Titles")
    
    # Show hint about labeling
    st.info("💡 Tip: Label specific window titles to categorize your activities more precisely.")
    
    # Create a dataframe for display
    title_df = pd.DataFrame({
        "Window Title": title_groups['title'],
        "Application": title_groups['app_name'],
        "Duration": title_groups['duration'].apply(lambda x: f"{x // 60} minutes"),
        "Count": title_groups['count'],
        "Percentage": title_groups['percentage'].apply(lambda x: f"{x}%")
    })
    
    # Display the titles with a filter
    title_filter = st.text_input("Filter window titles", "")
    if title_filter:
        filtered_df = title_df[title_df["Window Title"].str.contains(title_filter, case=False) | 
                              title_df["Application"].str.contains(title_filter, case=False)]
        st.dataframe(filtered_df, use_container_width=True)
    else:
        st.dataframe(title_df, use_container_width=True)
    
    # Create tabs for different label types
    tab1, tab2, tab3 = st.tabs(["Label Specific Window Title", "Label Application", "Label by Pattern"])
    
    # Tab 1: Label Specific Window Title
    with tab1:
        st.write("Create labels for specific window titles")
        
        # Get titles from the logs
        titles = title_groups['title'].tolist()
        
        if not titles:
            st.warning("No window titles available to label")
        else:
            # First, let users filter titles (optional)
            title_filter_for_selection = st.text_input("Filter titles for selection", "", key="title_filter_selection")
            
            # Create a filtered list of titles
            if title_filter_for_selection:
                filtered_titles = [t for t in titles if title_filter_for_selection.lower() in t.lower()]
                if not filtered_titles:
                    st.warning("No titles match your filter")
                    filtered_titles = titles
            else:
                filtered_titles = titles
            
            # Show duration info for each title
            title_info = {}
            for title in filtered_titles:
                row = title_groups.loc[title_groups['title'] == title].iloc[0]
                duration_mins = row['duration'] // 60
                title_info[title] = f"({duration_mins} mins)"
            
            # Create a multi-select for titles
            selected_titles = st.multiselect(
                "Select window titles to label",
                options=filtered_titles,
                format_func=lambda t: f"{t[:80]}... {title_info.get(t, '')}" if len(t) > 80 else f"{t} {title_info.get(t, '')}"
            )
            
            if selected_titles:
                # Calculate combined time for selected titles
                combined_time = 0
                for title in selected_titles:
                    row = title_groups.loc[title_groups['title'] == title].iloc[0]
                    combined_time += row['duration']
                
                # ENHANCEMENT: Create a more prominent time display with metrics and colors
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        f"✅ Selected {len(selected_titles)} window titles",
                        f"{combined_time // 60} minutes"
                    )
                with col2:
                    st.metric(
                        "Total Hours", 
                        f"{(combined_time / 3600):.2f} hours"
                    )
                
                # Add a divider to make the section stand out
                st.markdown("---")
                
                # Show form for labeling
                with st.form("title_label_form"):
                    # Check if we already have a common label for these titles
                    common_label = None
                    for title in selected_titles:
                        selected_exe = title_groups.loc[title_groups['title'] == title, 'exe'].iloc[0]
                        title_key = f"exact::{selected_exe}::{title}"
                        if title_key in labels.get('exact_titles', {}):
                            if common_label is None:
                                common_label = labels['exact_titles'][title_key]
                            elif common_label != labels['exact_titles'][title_key]:
                                common_label = ""  # Different labels found
                    
                    # Input for the label
                    title_label = st.text_input(
                        f"Label for the {len(selected_titles)} selected window titles ({combined_time // 60} min total)", 
                        value=common_label or ""
                    )
                    
                    submit_title = st.form_submit_button("Apply Label to Selected Titles")
                    
                    if submit_title:
                        if title_label.strip():
                            # Apply the same label to all selected titles
                            for title in selected_titles:
                                selected_exe = title_groups.loc[title_groups['title'] == title, 'exe'].iloc[0]
                                title_key = f"exact::{selected_exe}::{title}"
                                labels['exact_titles'][title_key] = title_label.strip()
                            
                            save_labels(labels)
                            st.success(f"Saved label '{title_label}' for {len(selected_titles)} window titles")
                            time.sleep(1)
                            st.rerun()
                        else:
                            # Remove labels if empty
                            removed_count = 0
                            for title in selected_titles:
                                selected_exe = title_groups.loc[title_groups['title'] == title, 'exe'].iloc[0]
                                title_key = f"exact::{selected_exe}::{title}"
                                if title_key in labels['exact_titles']:
                                    del labels['exact_titles'][title_key]
                                    removed_count += 1
                            
                            if removed_count > 0:
                                save_labels(labels)
                                st.success(f"Removed labels from {removed_count} window titles")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.warning("No labels to remove")
            else:
                st.info("Select one or more window titles to label them")
    
    # Tab 2: Application Label
    with tab2:
        with st.form("app_label_form"):
            st.write("Create a label for an entire application")
            
            # Get executables from the logs
            exes = sorted(logs_df['exe'].unique())
            exe_options = [os.path.basename(exe) for exe in exes]
            
            if not exe_options:
                st.warning("No applications available to label")
                submit_app = st.form_submit_button("Save Application Label", disabled=True)
            else:
                exe_index = st.selectbox(
                    "Select Application", 
                    options=range(len(exe_options)),
                    format_func=lambda i: exe_options[i]
                )
                
                selected_exe = exe_options[exe_index]
                
                app_label = st.text_input("Label for this application", 
                                        value=labels['exact_exe'].get(selected_exe, ""))
                
                submit_app = st.form_submit_button("Save Application Label")
                
                if submit_app:
                    if app_label.strip():
                        labels['exact_exe'][selected_exe] = app_label.strip()
                        save_labels(labels)
                        st.success(f"Saved label '{app_label}' for {selected_exe}")
                        time.sleep(1)
                        st.rerun()
                    else:
                        # Remove label if empty
                        if selected_exe in labels['exact_exe']:
                            del labels['exact_exe'][selected_exe]
                            save_labels(labels)
                            st.success(f"Removed label for {selected_exe}")
                            time.sleep(1)
                            st.rerun()
    
    # Tab 3: Pattern-Based Label
    with tab3:
        with st.form("pattern_label_form"):
            st.write("Create a label for activities matching a specific pattern")
            
            # Get executables from the logs
            if not exe_options:
                st.warning("No applications available for pattern matching")
                submit_pattern = st.form_submit_button("Save Pattern Label", disabled=True)
            else:
                pattern_exe_index = st.selectbox(
                    "Select Application (for pattern)", 
                    options=range(len(exe_options)),
                    format_func=lambda i: exe_options[i],
                    key="pattern_exe"
                )
                
                pattern_exe = exe_options[pattern_exe_index]
                
                title_pattern = st.text_input("Window Title Pattern (case insensitive)")
                pattern_label = st.text_input("Pattern Label")
                
                # Create key for lookup
                pattern_key = f"pattern::{pattern_exe}::{title_pattern}"
                
                submit_pattern = st.form_submit_button("Save Pattern Label")
                
                if submit_pattern:
                    if pattern_label.strip() and title_pattern.strip():
                        labels['patterns'][pattern_key] = pattern_label.strip()
                        save_labels(labels)
                        st.success(f"Saved pattern label '{pattern_label}' for {pattern_exe} with pattern '{title_pattern}'")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Both pattern and label must be provided")
    
    # Delete labels section
    st.subheader("Delete Labels")
    
    tab_del1, tab_del2, tab_del3 = st.tabs(["Delete Window Title Label", "Delete Application Label", "Delete Pattern Label"])
    
    # Tab 1: Delete Window Title Label
    with tab_del1:
        if not labels.get('exact_titles', {}):
            st.info("No window title labels to delete")
        else:
            title_keys = []
            title_display = {}
            
            for key in labels.get('exact_titles', {}).keys():
                try:
                    _, exe, title = key.split("::", 2)
                    display = f"{os.path.basename(exe)} - '{title[:50]}...': {labels['exact_titles'][key]}"
                    title_keys.append(key)
                    title_display[key] = display
                except:
                    continue
            
            if title_keys:
                title_to_delete = st.selectbox("Select window title label to delete",
                                             options=title_keys,
                                             format_func=lambda x: title_display[x])
                
                if st.button("Delete Window Title Label"):
                    del labels['exact_titles'][title_to_delete]
                    save_labels(labels)
                    st.success(f"Deleted window title label")
                    time.sleep(1)
                    st.rerun()
            else:
                st.info("No valid window title labels to delete")
    
    # Tab 2: Delete Application Label
    with tab_del2:
        if not labels.get('exact_exe', {}):
            st.info("No application labels to delete")
        else:
            label_to_delete = st.selectbox("Select label to delete",
                                         options=list(labels['exact_exe'].keys()),
                                         format_func=lambda x: f"{x}: {labels['exact_exe'][x]}")
            
            if st.button("Delete Application Label"):
                del labels['exact_exe'][label_to_delete]
                save_labels(labels)
                st.success(f"Deleted label for {label_to_delete}")
                time.sleep(1)
                st.rerun()
    
    # Tab 3: Delete Pattern Label
    with tab_del3:
        if not labels.get('patterns', {}):
            st.info("No pattern labels to delete")
        else:
            pattern_keys = []
            pattern_display = {}
            
            for key in labels.get('patterns', {}).keys():
                try:
                    _, exe, pattern = key.split("::", 2)
                    display = f"{exe} - '{pattern}': {labels['patterns'][key]}"
                    pattern_keys.append(key)
                    pattern_display[key] = display
                except:
                    continue
            
            if pattern_keys:
                pattern_to_delete = st.selectbox("Select pattern to delete",
                                               options=pattern_keys,
                                               format_func=lambda x: pattern_display[x])
                
                if st.button("Delete Pattern Label"):
                    del labels['patterns'][pattern_to_delete]
                    save_labels(labels)
                    st.success(f"Deleted pattern label")
                    time.sleep(1)
                    st.rerun()
            else:
                st.info("No valid pattern labels to delete")
    
    # Preview labeled data
    st.subheader("Preview With Labels Applied")
    
    try:
        # Apply labels to the logs
        labeled_logs = apply_labels_to_logs(logs_df)
        
        # Group by labeled app name
        labeled_groups = labeled_logs.groupby('app_name').agg({
            'duration': 'sum',
            'original_app_name': 'first',  # Show original name
            'timestamp': 'count'  # Count entries
        }).reset_index()
        
        # Calculate percentage
        total_time = labeled_logs['duration'].sum()
        if total_time > 0:
            labeled_groups['percentage'] = (labeled_groups['duration'] / total_time * 100).round(1)
        else:
            labeled_groups['percentage'] = 0
        
        # Sort by duration
        labeled_groups = labeled_groups.sort_values('duration', ascending=False)
        
        # Display the grouped data
        if not labeled_groups.empty:
            st.dataframe({
                "Labeled Activity": labeled_groups['app_name'],
                "Original Name": labeled_groups['original_app_name'],
                "Duration": labeled_groups['duration'].apply(lambda x: f"{x // 60} minutes"),
                "Percentage": labeled_groups['percentage'].apply(lambda x: f"{x}%"),
                "Log Entries": labeled_groups['timestamp']
            }, use_container_width=True)
            
            # Create a pie chart of the labeled data
            fig = px.pie(
                labeled_groups,
                values='duration',
                names='app_name',
                title="Preview of Labeled Activities",
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for preview")
    except Exception as e:
        st.error(f"Error generating preview: {e}")
        st.info("Try labeling some activities first.")
        
    # ENHANCEMENT: Add Label Summary Table with improved styling and metrics
    st.subheader("📊 Label Summary")
    st.markdown("### Time Spent by Category")

    try:
        # Apply labels to get a full labeled dataset
        labeled_data = apply_labels_to_logs(logs_df)
        
        # Get unique labels from all sources
        all_labels = set()
        
        # From window titles
        for label in labels.get('exact_titles', {}).values():
            all_labels.add(label)
            
        # From applications
        for label in labels.get('exact_exe', {}).values():
            all_labels.add(label)
            
        # From patterns
        for label in labels.get('patterns', {}).values():
            all_labels.add(label)
            
        # Add unlabeled category
        all_labels.add("Unlabeled")
        
        # Create summary table
        label_summary = []
        
        for label_name in all_labels:
            if label_name == "Unlabeled":
                # For unlabeled, find entries where app_name == original_app_name
                unlabeled_mask = labeled_data['app_name'] == labeled_data['original_app_name']
                total_time = labeled_data.loc[unlabeled_mask, 'duration'].sum()
                entry_count = unlabeled_mask.sum()
            else:
                # For labeled entries, find where app_name matches the label
                label_mask = labeled_data['app_name'] == label_name
                total_time = labeled_data.loc[label_mask, 'duration'].sum()
                entry_count = label_mask.sum()
            
            # Skip empty labels
            if total_time == 0:
                continue
                
            # Calculate percentage
            if labeled_data['duration'].sum() > 0:
                percentage = (total_time / labeled_data['duration'].sum() * 100).round(1)
            else:
                percentage = 0
                
            label_summary.append({
                "Label": label_name,
                "Total Time (min)": total_time // 60,
                "Total Time (hr)": round(total_time / 3600, 2),
                "Percentage": percentage,
                "Log Entries": entry_count
            })
            
        # Convert to dataframe and sort
        if label_summary:
            summary_df = pd.DataFrame(label_summary)
            summary_df = summary_df.sort_values("Total Time (min)", ascending=False)
            
            # Calculate grand total
            grand_total_mins = summary_df["Total Time (min)"].sum()
            grand_total_hrs = summary_df["Total Time (hr)"].sum()
            
            # Create metrics for a quick overview
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Tracked Time", f"{grand_total_mins} minutes")
            with col2:
                st.metric("Total Hours", f"{grand_total_hrs:.2f} hours")
            with col3:
                st.metric("Number of Labels", f"{len(summary_df) - ('Unlabeled' in summary_df['Label'].values)}")
            
            # Display as styled table with conditionally formatted bars for percentages
            st.write("**Detailed Time Breakdown by Label:**")
            
            # Create a styled dataframe with a bar chart for percentages
            styled_df = summary_df.style.bar(subset=['Percentage'], color='#5fba7d', vmax=100)
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Create a pie chart of the label summary
            fig = px.pie(
                summary_df,
                values="Total Time (min)",
                names="Label",
                title="Time Distribution by Label",
                hole=0.3,
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            # Improve the pie chart layout
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                legend_title_text='Activity Labels',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add export options
            st.download_button(
                label="Download Summary as CSV",
                data=summary_df.to_csv(index=False).encode('utf-8'),
                file_name=f"label_summary_{selected_date}.csv",
                mime="text/csv"
            )
        else:
            st.info("No labels have been applied yet. Create some labels to see a summary.")
    except Exception as e:
        st.error(f"Error generating label summary: {e}")
        st.info("Try labeling some activities first.")

# --- Dashboard Page ---
def display_dashboard():
    st.title("📊 Focus Monitor Dashboard")
    
    available_dates = load_available_dates()
    if not available_dates:
        st.error("No data found in focus_logs directory.")
        st.warning("Focus monitor must be running to collect data.")
        st.stop()
    
    selected_date = st.selectbox("Select a date", available_dates)
    
    # Load or generate summary
    summary = load_daily_summary(selected_date)
    if not summary:
        st.warning(f"Could not load or generate summary for {selected_date}")
        st.stop()
    
    # --- Summary Metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("🕒 Total Focus Time", f"{summary['totalTime'] // 60} min")
    if 'focusScore' in summary:
        col2.metric("🎯 Focus Score", f"{summary['focusScore']} / 100")
    if 'meetingTime' in summary:
        col3.metric("📞 Meeting Time", f"{summary['meetingTime'] // 60} min")
    
    # --- Pie Chart ---
    st.subheader("🧠 Time Distribution by Application")
    app_data = summary["appBreakdown"]
    if app_data:
        # Create interactive chart with plotly
        fig = create_pie_chart(app_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show top apps in a table
        app_df = pd.DataFrame([{
            "Application": app["appName"],
            "Time (minutes)": app["timeSpent"] // 60,
            "Percentage": f"{app['percentage']:.1f}%"
        } for app in app_data[:10]])  # Show top 10 apps
        
        st.dataframe(app_df, use_container_width=True)
    else:
        st.info("No application usage data available for this date.")
    
    # --- Browser Usage Section ---
    st.subheader("🌐 Browser Usage Analysis")
    browser_fig = create_browser_chart(app_data)
    if browser_fig:
        st.plotly_chart(browser_fig, use_container_width=True)
        
        # Create a filtered table for just browsers
        browser_apps = [app for app in app_data if "msedge" in app["appName"].lower() 
                        or "chrome" in app["appName"].lower() 
                        or "firefox" in app["appName"].lower()]
        
        if browser_apps:
            # Check if we have user-labeled browser sessions
            has_labeled_sessions = any(" - " in app["appName"] for app in browser_apps)
            
            browser_df = pd.DataFrame([{
                "Browser Session": app["appName"],
                "Time (minutes)": app["timeSpent"] // 60,
                "Percentage of Total": f"{app['percentage']:.1f}%"
            } for app in browser_apps])
            
            st.dataframe(browser_df, use_container_width=True)
            
            st.info("💡 Pro tip: You can create custom labels for any activities using the Labeling tab above.")
    else:
        st.info("No browser usage detected for this date.")
   
    # --- Log Table ---
    with st.expander("📄 Raw Focus Log"):
        log_df = load_log_entries(selected_date)
        if not log_df.empty:
            # Convert timestamp to datetime and add friendly duration
            log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])
            log_df["duration_min"] = (log_df["duration"] / 60).round(1)
            
            # Use app_name which includes profile info
            display_df = log_df[["timestamp", "app_name", "title", "duration_min"]].rename(
                columns={"timestamp": "Time", "app_name": "Application", "title": "Window Title", "duration_min": "Duration (min)"}
            ).sort_values("Time", ascending=False)
            
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No log entries available for this date.")

    # --- Block Feedback Section ---
    buckets = load_time_buckets(selected_date)
    if buckets:
        st.subheader("📝 Add Feedback for 5-Minute Blocks")
        option_map = {b['start']: f"{b['start'][11:16]} UTC" for b in buckets}
        selected_blocks = st.multiselect(
            "Select block start times",
            options=list(option_map.keys()),
            format_func=lambda x: option_map[x]
        )
        feedback_text = st.text_area("Feedback")
        if st.button("Submit Feedback"):
            if selected_blocks and feedback_text.strip():
                feedback = load_block_feedback()
                for ts in selected_blocks:
                    feedback[ts] = feedback_text.strip()
                if save_block_feedback(feedback):
                    st.success("Feedback saved")
                    time.sleep(1)
                    st.rerun()
            else:
                st.warning("Select block(s) and enter feedback text")
        existing = load_block_feedback()
        if existing:
            st.markdown("#### Existing Feedback")
            feedback_rows = [
                {"Block Start": k, "Feedback": v}
                for k, v in existing.items() if k in option_map
            ]
            if feedback_rows:
                st.dataframe(pd.DataFrame(feedback_rows), use_container_width=True)
    
    # --- Additional Info ---
    st.markdown("---")
    st.markdown("""
    ### About Focus Monitor
    This dashboard displays data collected by the Focus Monitor agent. The agent tracks your active windows and applications to help you understand your computer usage patterns.
    
    **Activity Labeling**: You can now assign custom labels to any activities using the Labeling tab at the top of this page. This allows you to categorize your time in ways that are meaningful to you.
    
    To generate data:
    1. Make sure the Focus Monitor agent is running
    2. Use your computer normally
    3. Return to this dashboard to view your usage statistics
    """)

# --- Main App ---
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Focus Monitor Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Removed sidebar tracker controls. Manually run the tracker using
    # `python standalone_focus_monitor.py` before launching the dashboard.
    
    # Create tabs for Dashboard and Label Editor
    tab1, tab2 = st.tabs(["📊 Dashboard", "🏷 Label Editor"])
    
    with tab1:
        display_dashboard()
        
    with tab2:
        display_label_editor()

if __name__ == "__main__":
    main()
