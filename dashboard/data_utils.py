"""
Data Utilities for Focus Monitor Dashboard - Updated with Browser Profile Daily Aggregation

This module provides functions for loading, processing, and saving data used in the
Focus Monitor Dashboard, including support for daily browser profile aggregations.

Key Updates:
- Added functions to load and manage daily browser profile aggregations
- Browser profile activities are excluded from 5-minute buckets
- Browser activities are aggregated daily by category
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Define constants for file paths
LOGS_DIR = Path(__file__).resolve().parent.parent / "focus_logs"
LABELS_FILE = LOGS_DIR / "activity_labels.json"
FEEDBACK_FILE = LOGS_DIR / "block_feedback.json"
CATEGORIES_FILE = LOGS_DIR / "activity_categories.json"

# Helper to check Streamlit context safely
def _is_streamlit_running() -> bool:
    """Checks if the code is running in a Streamlit context."""
    return st.session_state.get('streamlit_running', False)


def load_available_dates() -> List[str]:
    """Return a sorted list of dates that have log or summary files."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    log_dates = {
        f.name.split("_")[-1].replace(".jsonl", "") 
        for f in LOGS_DIR.glob("focus_log_*.jsonl")
    }
    summary_dates = {
        f.name.split("_")[-1].replace(".json", "") 
        for f in LOGS_DIR.glob("daily_summary_*.json")
    }
    browser_dates = {
        f.name.split("_")[-1].replace(".json", "")
        for f in LOGS_DIR.glob("daily_browser_activities_*.json")
    }
    
    all_dates = log_dates.union(summary_dates).union(browser_dates)
    return sorted(list(all_dates), reverse=True)


def load_log_entries(date_str: str) -> pd.DataFrame:
    """Load log entries for a specific date into a DataFrame."""
    file_path = LOGS_DIR / f"focus_log_{date_str}.jsonl"
    if not file_path.exists():
        return pd.DataFrame()
    data: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON line {line_num} in {file_path.name}: {line.strip()[:100]}...")
                continue
    return pd.DataFrame(data)


def load_daily_browser_activities(date_str: str) -> List[Dict[str, Any]]:
    """
    Load daily browser profile activities for a specific date.
    
    Args:
        date_str: Date string in format 'YYYY-MM-DD'
        
    Returns:
        List[Dict]: List of browser activity aggregations for the date
    """
    file_path = LOGS_DIR / f"daily_browser_activities_{date_str}.json"
    if not file_path.exists():
        return []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        print(f"Error: Could not decode browser activities file {file_path.name}")
        return []
    except Exception as e:
        print(f"Error loading browser activities from {file_path.name}: {e}")
        return []


def load_daily_summary(date_str: str) -> Optional[Dict[str, Any]]:
    """
    Load or generate a daily summary for a specific date.
    Now includes browser profile daily aggregations in the summary.
    """
    file_path = LOGS_DIR / f"daily_summary_{date_str}.json"
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
                if "date" in summary_data and "appBreakdown" in summary_data:
                    # Add browser activities to the summary
                    browser_activities = load_daily_browser_activities(date_str)
                    if browser_activities:
                        summary_data["browserProfileActivities"] = browser_activities
                    return summary_data
                else:
                    if _is_streamlit_running():
                        st.warning(f"Summary file {file_path.name} seems incomplete. Will attempt to regenerate.")
                    else:
                        print(f"Warning: Summary file {file_path.name} seems incomplete.")
        except json.JSONDecodeError:
            if _is_streamlit_running():
                st.error(f"Error decoding summary file {file_path.name}. Will attempt to regenerate.")
            else:
                print(f"Error: Error decoding summary file {file_path.name}.")
        except Exception as e:
            if _is_streamlit_running():
                st.error(f"Unexpected error loading summary {file_path.name}: {e}. Will attempt to regenerate.")
            else:
                print(f"Error: Unexpected error loading summary {file_path.name}: {e}.")

    if _is_streamlit_running():
        st.info(f"Generating daily summary for {date_str} from raw logs...")
    else:
        print(f"Info: Generating daily summary for {date_str} from raw logs...")
    return generate_summary_from_logs(date_str)


def generate_summary_from_logs(date_str: str) -> Optional[Dict[str, Any]]:
    """
    Generate a daily summary by processing raw log entries for a specific date.
    Now excludes browser profile activities from app breakdown and includes them separately.
    """
    logs_df = load_log_entries(date_str)
    
    if logs_df.empty:
        if _is_streamlit_running():
            st.warning(f"No log entries found for {date_str}. Cannot generate full summary.")
        else:
            print(f"Warning: No log entries found for {date_str}.")
            
        empty_summary: Dict[str, Any] = {
            "date": date_str, "totalTime": 0, "appBreakdown": [],
            "focusScore": 0, "distractionEvents": 0, "meetingTime": 0,
            "productiveApps": [], "distractionApps": [],
            "browserProfileActivities": []
        }
        
        # Check for browser activities even if no regular logs
        browser_activities = load_daily_browser_activities(date_str)
        if browser_activities:
            empty_summary["browserProfileActivities"] = browser_activities
            total_browser_time = sum(activity.get("total_duration", 0) for activity in browser_activities)
            empty_summary["totalTime"] = total_browser_time
            
        summary_file_path = LOGS_DIR / f"daily_summary_{date_str}.json"
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            with open(summary_file_path, "w", encoding="utf-8") as f:
                json.dump(empty_summary, f, indent=2)
        except Exception as e_save:
            if _is_streamlit_running():
                st.error(f"Error saving summary for {date_str}: {e_save}")
            else:
                print(f"Error: Error saving summary for {date_str}: {e_save}")
        return empty_summary

    # Apply labels to logs
    logs_df_labeled = apply_labels_to_logs(logs_df.copy()) 

    if "duration" not in logs_df_labeled.columns or logs_df_labeled["duration"].isnull().all():
        if _is_streamlit_running():
            st.error(f"Log data for {date_str} is missing valid 'duration' information after labeling.")
        else:
            print(f"Error: Log data for {date_str} is missing valid 'duration' information.")
        return None 

    # Separate browser profile activities from regular activities
    # Check for detected_profile_category column existence and valid values
    if "detected_profile_category" in logs_df_labeled.columns:
        has_profile_category = (logs_df_labeled["detected_profile_category"].notna() & 
                               (logs_df_labeled["detected_profile_category"] != ""))
        browser_profile_logs = logs_df_labeled[has_profile_category]
        regular_logs = logs_df_labeled[~has_profile_category]
    else:
        # No browser profile data, all logs are regular
        browser_profile_logs = pd.DataFrame()
        regular_logs = logs_df_labeled

    # Calculate total duration including browser activities
    total_duration_all = logs_df_labeled["duration"].sum()
    regular_duration = regular_logs["duration"].sum() if not regular_logs.empty else 0
    browser_duration = browser_profile_logs["duration"].sum() if not browser_profile_logs.empty else 0

    summary: Dict[str, Any] = {
        "date": date_str, 
        "totalTime": round(total_duration_all),  # Include all activities in total
        "appBreakdown": [],
        "focusScore": 0,
        "distractionEvents": len(logs_df_labeled), 
        "meetingTime": 0,
        "productiveApps": [], 
        "distractionApps": [],
        "browserProfileActivities": []
    }
    
    # Process regular (non-browser-profile) activities for app breakdown
    if not regular_logs.empty and regular_duration > 0:
        if "app_name" not in regular_logs.columns: 
            regular_logs["app_name"] = "Unknown Application"
        if "exe" not in regular_logs.columns:
            regular_logs["exe"] = "unknown.exe"

        agg_dict = {
            "exe_path": ("exe", "first"),
            "time_spent": ("duration", "sum"),
        }

        if "title" in regular_logs.columns:
            agg_dict["window_titles"] = ("title", lambda x: sorted(list(set(str(t).strip() for t in x if pd.notna(t) and str(t).strip()))[:50]))
        else:
            agg_dict["window_titles"] = ("app_name", lambda x: []) 

        if "ocr_text" in regular_logs.columns:
            agg_dict["ocr_texts"] = ("ocr_text", lambda x: sorted(list(set(str(t).strip() for t in x if pd.notna(t) and str(t).strip()))[:20]))
        else:
            agg_dict["ocr_texts"] = ("app_name", lambda x: [])

        if "screenshot_path" in regular_logs.columns:
            agg_dict["screenshot_paths"] = ("screenshot_path", lambda x: sorted(list(set(str(t).strip() for t in x if pd.notna(t) and str(t).strip()))[:20]))
        else:
            agg_dict["screenshot_paths"] = ("app_name", lambda x: [])
        
        app_groups = (
            regular_logs.groupby("app_name")
            .agg(**agg_dict) 
            .reset_index()
        )

        app_breakdown_list = []
        for _, row in app_groups.iterrows():
            # Calculate percentage against total time (including browser activities)
            percentage = (row["time_spent"] / total_duration_all * 100) if total_duration_all > 0 else 0
            
            titles_list = row.get("window_titles", []) 
            if not isinstance(titles_list, list): 
                 titles_list = list(titles_list) if pd.notna(titles_list) else []

            app_breakdown_list.append({
                "appName": row["app_name"], 
                "exePath": row.get("exe_path", "unknown.exe"),
                "timeSpent": int(row["time_spent"]),
                "percentage": round(percentage, 2),
                "windowTitles": titles_list, 
                "ocrTexts": row.get("ocr_texts", []), 
                "screenshotPaths": row.get("screenshot_paths", []), 
            })
        
        summary["appBreakdown"] = sorted(app_breakdown_list, key=lambda x: x["timeSpent"], reverse=True)

    # Load and include browser profile activities
    browser_activities = load_daily_browser_activities(date_str)
    if browser_activities:
        # Convert browser activities to app breakdown format for compatibility
        for browser_activity in browser_activities:
            # Calculate percentage against total time
            percentage = (browser_activity.get("total_duration", 0) / total_duration_all * 100) if total_duration_all > 0 else 0
            
            # Find a representative app name from the activity
            app_names = browser_activity.get("app_names", set())
            if isinstance(app_names, list):
                app_names = set(app_names)
            representative_name = next(iter(app_names)) if app_names else f"Browser Profile ({browser_activity.get('category_id', 'Unknown')})"
            
            browser_app_entry = {
                "appName": representative_name,
                "exePath": "browser_profile_aggregated",
                "timeSpent": int(browser_activity.get("total_duration", 0)),
                "percentage": round(percentage, 2),
                "windowTitles": list(browser_activity.get("titles", [])),
                "ocrTexts": list(browser_activity.get("ocr_text", [])),
                "screenshotPaths": [],
                "isBrowserProfile": True,  # Flag to identify browser profile activities
                "categoryId": browser_activity.get("category_id", ""),
                "activityCount": browser_activity.get("activity_count", 0)
            }
            
            summary["appBreakdown"].append(browser_app_entry)
        
        # Re-sort app breakdown to include browser activities
        summary["appBreakdown"] = sorted(summary["appBreakdown"], key=lambda x: x["timeSpent"], reverse=True)
        summary["browserProfileActivities"] = browser_activities
    
    # Save summary to file
    summary_file_path = LOGS_DIR / f"daily_summary_{date_str}.json"
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(summary_file_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        if _is_streamlit_running():
            st.success(f"Daily summary for {date_str} generated/updated and saved")
        else:
            print(f"Daily summary for {date_str} generated/updated and saved")
    except Exception as e:
        if _is_streamlit_running():
            st.error(f"Error saving daily summary file: {e}")
        else:
            print(f"Error: Error saving daily summary file: {e}")
        return None

    return summary


# ----- Label Management Functions -----

def load_labels() -> Dict[str, Any]:
    """Load activity labels from the labels file."""
    if not LABELS_FILE.exists():
        return {}
    try:
        with open(LABELS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        if _is_streamlit_running(): st.error(f"Error loading labels: {e}")
        else: print(f"Error loading labels: {e}")
        return {}

def save_labels(labels: Dict[str, Any]) -> bool:
    """Save activity labels to the labels file."""
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(LABELS_FILE, "w", encoding="utf-8") as f:
            json.dump(labels, f, indent=2)
        return True
    except Exception as e:
        if _is_streamlit_running(): st.error(f"Error saving labels: {e}")
        else: print(f"Error saving labels: {e}")
        return False

def apply_labels_to_logs(logs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply activity labels to log entries.
    Updates 'app_name' in the DataFrame based on label matching rules.
    """
    if logs_df.empty:
        return logs_df

    labels = load_labels()
    df = logs_df.copy() 
    
    default_cols_config = {"app_name": "Unknown Application", "exe": "unknown.exe", "title": ""}
    for col_name, default_value in default_cols_config.items():
        if col_name not in df.columns:
            df[col_name] = default_value
        df[col_name] = df[col_name].fillna(default_value).astype(str)

    df["original_app_name"] = df["app_name"] 
    df["labeled_app_name"] = df["app_name"] 

    if not labels: 
        df.drop(columns=["labeled_app_name"], inplace=True, errors="ignore") 
        return df

    # Apply exact title labels
    for key, label_val in labels.get("exact_titles", {}).items():
        try:
            _, exe_from_key, title_from_key = key.split("::", 2)
            mask = (df["exe"].str.lower() == exe_from_key.lower()) & \
                   (df["title"] == title_from_key) 
            df.loc[mask, "labeled_app_name"] = label_val
        except ValueError: 
            print(f"Warning: Skipping malformed exact_title key: {key}")
        except Exception as e: 
            print(f"Error applying exact title label for '{key}': {e}")

    # Apply exact exe labels
    for exe_basename_key, label_val in labels.get("exact_exe", {}).items():
        mask = (df["exe"].apply(lambda x: os.path.basename(x).lower()) == exe_basename_key.lower()) & \
               (df["labeled_app_name"] == df["original_app_name"]) 
        df.loc[mask, "labeled_app_name"] = label_val
    
    # Apply pattern labels
    for key, label_val in labels.get("patterns", {}).items():
        try:
            _, exe_basename_key, pattern_from_key = key.split("::", 2)
            mask = (
                df["exe"].apply(lambda x: os.path.basename(x).lower()).str.contains(exe_basename_key.lower(), regex=False, na=False) &
                df["title"].str.lower().str.contains(pattern_from_key.lower(), regex=False, na=False) &
                (df["labeled_app_name"] == df["original_app_name"]) 
            )
            df.loc[mask, "labeled_app_name"] = label_val
        except ValueError: 
            print(f"Warning: Skipping malformed pattern key: {key}")
        except Exception as e: 
            print(f"Error applying pattern label for '{key}': {e}")
            
    df["app_name"] = df["labeled_app_name"]
    df.drop(columns=["labeled_app_name"], inplace=True, errors="ignore") 
    
    return df


def group_activities_for_labeling(logs_df: pd.DataFrame) -> pd.DataFrame:
    """Group activities by window title for easier labeling."""
    if logs_df.empty: return pd.DataFrame()

    df_copy = logs_df.copy()
    if "original_app_name" not in df_copy.columns and "app_name" in df_copy.columns:
        df_copy["original_app_name"] = df_copy["app_name"] 

    required_cols = {"title": "", "original_app_name": "Unknown", "exe": "unknown.exe", "duration": 0}
    for col, default_val in required_cols.items():
        if col not in df_copy.columns: df_copy[col] = default_val
        df_copy[col] = df_copy[col].fillna(default_val)
    
    df_copy["title"] = df_copy["title"].astype(str)

    title_groups_data: List[Dict[str, Any]] = []
    for title_val, group in df_copy.groupby("title"):
        app_name_mode = group["original_app_name"].mode()
        exe_mode = group["exe"].mode()
        
        title_groups_data.append({
            "title": title_val,
            "app_name": app_name_mode[0] if not app_name_mode.empty else "Unknown", 
            "exe": exe_mode[0] if not exe_mode.empty else "unknown.exe", 
            "count": len(group), "duration": group["duration"].sum(),
        })

    grouped_df = pd.DataFrame(title_groups_data)
    if not grouped_df.empty:
        total_log_duration = df_copy["duration"].sum()
        grouped_df["percentage"] = ((grouped_df["duration"] / total_log_duration * 100).round(1) if total_log_duration > 0 else 0.0)
        return grouped_df.sort_values("duration", ascending=False)
        
    return pd.DataFrame()


# ----- Feedback Management Functions -----

def load_block_feedback() -> Dict[str, str]:
    if not FEEDBACK_FILE.exists(): return {}
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except Exception as e:
        if _is_streamlit_running(): st.error(f"Error loading personal notes: {e}")
        else: print(f"Error loading personal notes: {e}")
        return {}

def save_block_feedback(feedback_data: Dict[str, str]) -> bool:
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f: json.dump(feedback_data, f, indent=2)
        return True
    except Exception as e:
        if _is_streamlit_running(): st.error(f"Error saving personal notes: {e}")
        else: print(f"Error saving personal notes: {e}")
        return False


# ----- Time Bucket Helper Functions -----

def load_time_buckets_for_date(date_str: str) -> List[Dict[str, Any]]:
    """Load time buckets for a date. Note: Browser profile activities are now excluded from buckets."""
    buckets_with_tags: List[Dict[str, Any]] = []
    for path in LOGS_DIR.glob("time_buckets_*.json"):
        try:
            load_this_file = date_str in path.name 
            if not load_this_file:
                pass

            session_tag = path.name.replace("time_buckets_", "").replace(".json", "")
            with open(path, "r", encoding="utf-8") as f: data = json.load(f)
            for b_data in data:
                if b_data.get("start", "")[:10] == date_str:
                    b_copy = b_data.copy(); b_copy["session_tag"] = session_tag 
                    buckets_with_tags.append(b_copy)
        except Exception as e:
            if _is_streamlit_running(): st.error(f"Error reading bucket file {path.name}: {e}")
            else: print(f"Error reading bucket file {path.name}: {e}")
    
    buckets_with_tags.sort(key=lambda x: x.get("start", ""))
    return buckets_with_tags


def update_bucket_summary_in_file(session_tag: str, bucket_start_iso: str, new_summary: str) -> bool:
    bucket_file_path = LOGS_DIR / f"time_buckets_{session_tag}.json"
    if not bucket_file_path.exists():
        msg = f"Target summary file for update not found: {bucket_file_path.name}"
        if _is_streamlit_running(): 
            st.error(msg)
        else: 
            print(msg)
        return False
    try:
        with open(bucket_file_path, "r", encoding="utf-8") as f:
            all_buckets_data = json.load(f)
        
        found_bucket = False
        for bucket_data in all_buckets_data:
            if bucket_data.get("start") == bucket_start_iso:
                bucket_data["summary"] = new_summary
                found_bucket = True; break
        
        if not found_bucket:
            msg = f"Bucket {bucket_start_iso} not found in {bucket_file_path.name}"
            if _is_streamlit_running(): 
                st.error(msg)
            else: 
                print(msg)
            return False
            
        with open(bucket_file_path, "w", encoding="utf-8") as f:
            json.dump(all_buckets_data, f, indent=2)
        return True
    except Exception as e:
        msg = f"Error updating summary in {bucket_file_path.name}: {e}"
        if _is_streamlit_running(): 
            st.error(msg)
        else: 
            print(msg)
        return False


# ----- Category Management Functions -----

def load_categories() -> List[Dict[str, str]]:
    if not CATEGORIES_FILE.exists(): return []
    try:
        with open(CATEGORIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f); return data.get("categories", []) 
    except Exception as e: print(f"Error loading categories: {e}"); return []

def save_categories(categories: List[Dict[str, str]]) -> bool:
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(CATEGORIES_FILE, "w", encoding="utf-8") as f:
            json.dump({"categories": categories}, f, indent=2) 
        return True
    except Exception as e:
        if _is_streamlit_running(): st.error(f"Error saving categories: {e}")
        else: print(f"Error saving categories: {e}")
        return False

def update_bucket_category_in_file(session_tag: str, bucket_start_iso: str, new_category_id: str) -> bool:
    bucket_file_path = LOGS_DIR / f"time_buckets_{session_tag}.json"
    if not bucket_file_path.exists():
        msg = f"Target category file for update not found: {bucket_file_path.name}"
        if _is_streamlit_running(): 
            st.error(msg)
        else: 
            print(msg)
        return False
    try:
        with open(bucket_file_path, "r", encoding="utf-8") as f:
            all_buckets_data = json.load(f)
        
        found_bucket = False
        for bucket_data in all_buckets_data:
            if bucket_data.get("start") == bucket_start_iso:
                bucket_data["category_id"] = new_category_id
                found_bucket = True; break
        
        if not found_bucket:
            msg = f"Bucket {bucket_start_iso} not found in {bucket_file_path.name}"
            if _is_streamlit_running(): 
                st.error(msg)
            else: 
                print(msg)
            return False
            
        with open(bucket_file_path, "w", encoding="utf-8") as f:
            json.dump(all_buckets_data, f, indent=2)
        return True
    except Exception as e:
        msg = f"Error updating category in {bucket_file_path.name}: {e}"
        if _is_streamlit_running(): 
            st.error(msg)
        else: 
            print(msg)
        return False

def generate_time_buckets_from_logs(date_str: str) -> bool:
    """
    Generate time buckets from logs for a specific date.
    Note: Browser profile activities will be excluded from buckets as they are handled daily.
    """
    is_st_true = _is_streamlit_running() 

    log_path = LOGS_DIR / f"focus_log_{date_str}.jsonl"
    if not log_path.exists():
        msg=f"Focus log for {date_str} not found."
        if is_st_true: 
            st.error(msg)
        else: 
            print(msg)
        return False

    logs_df = load_log_entries(date_str)
    if logs_df.empty:
        msg=f"No log entries found for {date_str}."
        if is_st_true: 
            st.error(msg)
        else: 
            print(msg)
        return False
    
    # Filter out browser profile activities (they have detected_profile_category)
    if "detected_profile_category" in logs_df.columns:
        has_profile_category = (logs_df["detected_profile_category"].notna() & 
                               (logs_df["detected_profile_category"] != ""))
        non_browser_logs = logs_df[~has_profile_category]
    else:
        # No browser profile column, all logs are non-browser
        non_browser_logs = logs_df
    
    if non_browser_logs.empty:
        msg=f"No non-browser-profile activities found for {date_str} to create time buckets."
        if is_st_true: 
            st.info(msg)
        else: 
            print(msg)
        return False
    
    if "timestamp" not in non_browser_logs.columns:
        msg="Log data missing timestamp column."
        if is_st_true: 
            st.error(msg)
        else: 
            print(msg)
        return False

    try:
        non_browser_logs["timestamp"] = pd.to_datetime(non_browser_logs["timestamp"], errors='coerce')
        non_browser_logs.dropna(subset=["timestamp"], inplace=True) 
        if non_browser_logs.empty:
             msg="No valid timestamps in non-browser log data after conversion."
             if is_st_true: 
                 st.error(msg)
             else: 
                 print(msg)
             return False
    except Exception as e_ts:
        msg=f"Error converting timestamps: {e_ts}"
        if is_st_true: 
            st.error(msg)
        else: 
            print(msg)
        return False
        
    non_browser_logs = non_browser_logs.sort_values("timestamp")
    min_time = non_browser_logs["timestamp"].min()
    max_time = non_browser_logs["timestamp"].max()
    bucket_duration_minutes = 5 
    bucket_size_timedelta = pd.Timedelta(minutes=bucket_duration_minutes)
    start_time_aligned = min_time.floor(f"{bucket_duration_minutes}min")
    
    bucket_start_timestamps: List[pd.Timestamp] = []
    current_bucket_start = start_time_aligned
    while current_bucket_start <= max_time: 
        bucket_start_timestamps.append(current_bucket_start)
        current_bucket_start += bucket_size_timedelta

    if not bucket_start_timestamps:
        msg = f"Not enough time range in non-browser logs for {date_str} to create {bucket_duration_minutes}-min buckets."
        if is_st_true: 
            st.warning(msg)
        else: 
            print(msg)
        return False

    generated_buckets_list: List[Dict[str, Any]] = []
    
    from .llm_utils import generate_summary_from_raw_with_llm  

    for bucket_dt_start in bucket_start_timestamps:
        bucket_dt_end = bucket_dt_start + bucket_size_timedelta
        current_bucket_logs_df = non_browser_logs[(non_browser_logs["timestamp"] >= bucket_dt_start) & (non_browser_logs["timestamp"] < bucket_dt_end)]
        
        if not current_bucket_logs_df.empty:
            window_titles_list: List[str] = []
            if "title" in current_bucket_logs_df.columns:
                window_titles_list = current_bucket_logs_df["title"].dropna().astype(str).str.strip().unique().tolist()
                window_titles_list = [t for t in window_titles_list if t] 
            ocr_text_snippets: List[str] = []
            if "ocr_text" in current_bucket_logs_df.columns:
                ocr_text_snippets = current_bucket_logs_df["ocr_text"].dropna().astype(str).str.strip().unique().tolist()
                ocr_text_snippets = [o for o in ocr_text_snippets if o]
            
            bucket_entry: Dict[str, Any] = {
                "start": bucket_dt_start.isoformat(), "end": bucket_dt_end.isoformat(),
                "titles": window_titles_list, "ocr_text": ocr_text_snippets,
                "summary": "", "category_id": "" 
            }
            
            llm_summary, llm_category_id, _suggested_category, _prompt = generate_summary_from_raw_with_llm(
                window_titles_list, ocr_text_snippets, allow_suggestions=True, return_prompt=False
            )
            if llm_summary: bucket_entry["summary"] = llm_summary
            if llm_category_id: bucket_entry["category_id"] = llm_category_id
            generated_buckets_list.append(bucket_entry)

    if not generated_buckets_list:
        msg = f"No non-browser activity found within defined time buckets for {date_str}."
        if is_st_true: 
            st.warning(msg)
        else: 
            print(msg)
        return False 

    bucket_file_path = LOGS_DIR / f"time_buckets_{date_str}.json" 
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(bucket_file_path, "w", encoding="utf-8") as f:
            json.dump(generated_buckets_list, f, indent=2)
        msg = f"Time buckets for {date_str} generated ({len(generated_buckets_list)} buckets, browser profiles excluded)"
        if is_st_true: 
            st.success(msg)
        else: 
            print(msg)
        return True
    except Exception as e_save_buckets:
        msg = f"Error saving time buckets to {bucket_file_path.name}: {e_save_buckets}"
        if is_st_true: 
            st.error(msg)
        else: 
            print(msg)
        return False

# Browser Profile Management (existing functions remain the same)
BROWSER_PROFILES_FILE = LOGS_DIR / "browser_profiles.json"

def load_browser_profiles() -> List[Dict[str, Any]]:
    """Load browser profiles from the profiles file."""
    if not BROWSER_PROFILES_FILE.exists():
        return [
            {
                "name": "Edge - Personal",
                "exe_pattern": "msedge.exe",
                "color_rgb": [7, 43, 71],
                "color_tolerance": 15,
                "search_rect": [9, 3, 30, 30],
                "category_id": "personal_browsing",
                "enabled": True
            }
        ]
    try:
        with open(BROWSER_PROFILES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("profiles", [])
    except Exception as e:
        if _is_streamlit_running():
            st.error(f"Error loading browser profiles: {e}")
        else:
            print(f"Error loading browser profiles: {e}")
        return []

def save_browser_profiles(profiles: List[Dict[str, Any]]) -> bool:
    """Save browser profiles to the profiles file."""
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(BROWSER_PROFILES_FILE, "w", encoding="utf-8") as f:
            json.dump({"profiles": profiles}, f, indent=2)
        return True
    except Exception as e:
        if _is_streamlit_running():
            st.error(f"Error saving browser profiles: {e}")
        else:
            print(f"Error saving browser profiles: {e}")
        return False

def get_available_exe_patterns() -> List[str]:
    """Get list of common browser executable patterns."""
    return [
        "msedge.exe",
        "chrome.exe", 
        "firefox.exe",
        "brave.exe",
        "opera.exe",
        "vivaldi.exe",
        "safari.exe"
    ]

def validate_browser_profile(profile: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate a browser profile configuration."""
    required_fields = ["name", "exe_pattern", "color_rgb", "search_rect", "category_id"]
    
    for field in required_fields:
        if field not in profile:
            return False, f"Missing required field: {field}"
    
    color_rgb = profile["color_rgb"]
    if not isinstance(color_rgb, (list, tuple)) or len(color_rgb) != 3:
        return False, "color_rgb must be a list/tuple of 3 values [R, G, B]"
    
    for color_val in color_rgb:
        if not isinstance(color_val, int) or not (0 <= color_val <= 255):
            return False, "Each RGB value must be an integer between 0 and 255"
    
    search_rect = profile["search_rect"]
    if not isinstance(search_rect, (list, tuple)) or len(search_rect) != 4:
        return False, "search_rect must be a list/tuple of 4 values [x, y, width, height]"
    
    for rect_val in search_rect:
        if not isinstance(rect_val, int) or rect_val < 0:
            return False, "Each search_rect value must be a non-negative integer"
    
    if "color_tolerance" in profile:
        tolerance = profile["color_tolerance"]
        if not isinstance(tolerance, int) or not (0 <= tolerance <= 255):
            return False, "color_tolerance must be an integer between 0 and 255"
    
    if not isinstance(profile["exe_pattern"], str) or not profile["exe_pattern"].strip():
        return False, "exe_pattern must be a non-empty string"
    
    if not isinstance(profile["name"], str) or not profile["name"].strip():
        return False, "name must be a non-empty string"
    
    return True, ""

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert RGB tuple to hex color string."""
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

def get_default_search_rect_for_browser(exe_pattern: str) -> List[int]:
    """Get default search rectangle based on browser type."""
    defaults = {
        "msedge.exe": [9, 3, 30, 30],
        "chrome.exe": [40, 8, 25, 25],     
        "firefox.exe": [45, 10, 20, 20],   
        "brave.exe": [40, 8, 25, 25],      
        "opera.exe": [35, 8, 25, 25],      
        "vivaldi.exe": [40, 8, 25, 25],    
    }
    return defaults.get(exe_pattern, [10, 5, 25, 25])