"""
Data Utilities for Focus Monitor Dashboard.

This module provides functions for loading, processing, and saving data used in the
Focus Monitor Dashboard. It handles operations such as:
- Loading and saving activity logs and summaries
- Managing activity labels and categories
- Processing time buckets and feedback data
- Generating summaries from raw logs

All data is stored in the focus_logs directory within the parent directory.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st # Keep for type hinting and potential direct use if careful

# Define constants for file paths
LOGS_DIR = Path(__file__).resolve().parent.parent / "focus_logs"
LABELS_FILE = LOGS_DIR / "activity_labels.json"
FEEDBACK_FILE = LOGS_DIR / "block_feedback.json"
CATEGORIES_FILE = LOGS_DIR / "activity_categories.json"

# Helper to check Streamlit context safely
def _is_streamlit_running() -> bool:
    """Checks if the code is running in a Streamlit context by checking session_state."""
    # This assumes 'streamlit_running' is set to True at the start of the Streamlit app script.
    return st.session_state.get('streamlit_running', False)


def load_available_dates() -> List[str]:
    """
    Return a sorted list of dates that have log or summary files.
    
    Returns:
        List[str]: Sorted list of dates in descending order (newest first).
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    log_dates = {
        f.name.split("_")[-1].replace(".jsonl", "") 
        for f in LOGS_DIR.glob("focus_log_*.jsonl")
    }
    summary_dates = {
        f.name.split("_")[-1].replace(".json", "") 
        for f in LOGS_DIR.glob("daily_summary_*.json")
    }
    return sorted(list(log_dates.union(summary_dates)), reverse=True)


def load_log_entries(date_str: str) -> pd.DataFrame:
    """
    Load log entries for a specific date into a DataFrame.
    
    Args:
        date_str: Date string in format 'YYYY-MM-DD'
        
    Returns:
        pd.DataFrame: DataFrame containing log entries, or empty DataFrame if no data.
    """
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


def load_daily_summary(date_str: str) -> Optional[Dict[str, Any]]:
    """
    Load or generate a daily summary for a specific date.
    
    Tries to load pre-generated summary first, falls back to generating
    from raw logs if summary file doesn't exist or is corrupted.
    
    Args:
        date_str: Date string in format 'YYYY-MM-DD'
        
    Returns:
        Dict[str, Any]: Summary data, or None if no data available and generation fails.
    """
    file_path = LOGS_DIR / f"daily_summary_{date_str}.json"
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
                if "date" in summary_data and "appBreakdown" in summary_data:
                    return summary_data
                else:
                    if _is_streamlit_running():
                        st.warning(f"Summary file {file_path.name} seems incomplete. Will attempt to regenerate.")
                    else:
                        print(f"Warning: Summary file {file_path.name} seems incomplete. Will attempt to regenerate.")
        except json.JSONDecodeError:
            if _is_streamlit_running():
                st.error(f"Error decoding summary file {file_path.name}. Will attempt to regenerate.")
            else:
                print(f"Error: Error decoding summary file {file_path.name}. Will attempt to regenerate.")
        except Exception as e:
            if _is_streamlit_running():
                st.error(f"Unexpected error loading summary {file_path.name}: {e}. Will attempt to regenerate.")
            else:
                print(f"Error: Unexpected error loading summary {file_path.name}: {e}. Will attempt to regenerate.")

    if _is_streamlit_running():
        st.info(f"Generating daily summary for {date_str} from raw logs...")
    else:
        print(f"Info: Generating daily summary for {date_str} from raw logs...")
    return generate_summary_from_logs(date_str)


def generate_summary_from_logs(date_str: str) -> Optional[Dict[str, Any]]:
    """
    Generate a daily summary by processing raw log entries for a specific date.
    This summary includes app breakdowns with labeled app names and basic metrics.
    Saves the summary to 'daily_summary_{date_str}.json'.

    Args:
        date_str: Date string in format 'YYYY-MM-DD'

    Returns:
        Dict[str, Any]: Generated summary, or None if no log data or critical error.
    """
    logs_df = load_log_entries(date_str)
    
    if logs_df.empty:
        if _is_streamlit_running():
            st.warning(f"No log entries found for {date_str}. Cannot generate full summary.")
        else:
            print(f"Warning: No log entries found for {date_str}. Cannot generate full summary.")
            
        empty_summary: Dict[str, Any] = {
            "date": date_str, "totalTime": 0, "appBreakdown": [],
            "focusScore": 0, "distractionEvents": 0, "meetingTime": 0,
            "productiveApps": [], "distractionApps": []
        }
        summary_file_path = LOGS_DIR / f"daily_summary_{date_str}.json"
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            with open(summary_file_path, "w", encoding="utf-8") as f:
                json.dump(empty_summary, f, indent=2)
            if _is_streamlit_running():
                st.info(f"Empty daily summary saved for {date_str} as no logs were found.")
            else:
                print(f"Info: Empty daily summary saved for {date_str} as no logs were found.")
        except Exception as e_save:
            if _is_streamlit_running():
                st.error(f"Error saving empty daily summary for {date_str}: {e_save}")
            else:
                print(f"Error: Error saving empty daily summary for {date_str}: {e_save}")
        return empty_summary

    logs_df_labeled = apply_labels_to_logs(logs_df.copy()) 

    if "duration" not in logs_df_labeled.columns or logs_df_labeled["duration"].isnull().all():
        if _is_streamlit_running():
            st.error(f"Log data for {date_str} is missing valid 'duration' information after labeling.")
        else:
            print(f"Error: Log data for {date_str} is missing valid 'duration' information after labeling.")
        return None 

    total_duration = logs_df_labeled["duration"].sum()

    summary: Dict[str, Any] = {
        "date": date_str, 
        "totalTime": round(total_duration), 
        "appBreakdown": [],
        "focusScore": 0,
        "distractionEvents": len(logs_df_labeled), 
        "meetingTime": 0,
        "productiveApps": [], 
        "distractionApps": [], 
    }
    
    if total_duration > 0:
        if "app_name" not in logs_df_labeled.columns: 
            logs_df_labeled["app_name"] = "Unknown Application"
        if "exe" not in logs_df_labeled.columns:
            logs_df_labeled["exe"] = "unknown.exe"

        agg_dict = {
            "exe_path": ("exe", "first"),
            "time_spent": ("duration", "sum"),
        }

        if "title" in logs_df_labeled.columns:
            agg_dict["window_titles"] = ("title", lambda x: sorted(list(set(str(t).strip() for t in x if pd.notna(t) and str(t).strip()))[:50]))
        else:
            agg_dict["window_titles"] = ("app_name", lambda x: []) 

        if "ocr_text" in logs_df_labeled.columns:
            agg_dict["ocr_texts"] = ("ocr_text", lambda x: sorted(list(set(str(t).strip() for t in x if pd.notna(t) and str(t).strip()))[:20]))
        else:
            agg_dict["ocr_texts"] = ("app_name", lambda x: [])

        if "screenshot_path" in logs_df_labeled.columns:
            agg_dict["screenshot_paths"] = ("screenshot_path", lambda x: sorted(list(set(str(t).strip() for t in x if pd.notna(t) and str(t).strip()))[:20]))
        else:
            agg_dict["screenshot_paths"] = ("app_name", lambda x: [])
        
        app_groups = (
            logs_df_labeled.groupby("app_name")
            .agg(**agg_dict) 
            .reset_index()
        )

        app_breakdown_list = []
        for _, row in app_groups.iterrows():
            percentage = (row["time_spent"] / total_duration * 100) if total_duration > 0 else 0
            
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
        summary["appBreakdown"].sort(key=lambda x: x["timeSpent"], reverse=True)
    
    summary_file_path = LOGS_DIR / f"daily_summary_{date_str}.json"
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(summary_file_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        if _is_streamlit_running():
            st.success(f"Daily summary for {date_str} generated/updated and saved to {summary_file_path.name}")
        else:
            print(f"Daily summary for {date_str} generated/updated and saved to {summary_file_path.name}")
    except Exception as e:
        if _is_streamlit_running():
            st.error(f"Error saving daily summary file {summary_file_path.name}: {e}")
        else:
            print(f"Error: Error saving daily summary file {summary_file_path.name}: {e}")
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
    Adds 'original_app_name' to preserve the pre-labeling app name.
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

    for key, label_val in labels.get("exact_titles", {}).items():
        try:
            _, exe_from_key, title_from_key = key.split("::", 2)
            mask = (df["exe"].str.lower() == exe_from_key.lower()) & \
                   (df["title"] == title_from_key) 
            df.loc[mask, "labeled_app_name"] = label_val
        except ValueError: print(f"Warning: Skipping malformed exact_title key: {key}")
        except Exception as e: print(f"Error applying exact title label for '{key}': {e}")

    for exe_basename_key, label_val in labels.get("exact_exe", {}).items():
        mask = (df["exe"].apply(lambda x: os.path.basename(x).lower()) == exe_basename_key.lower()) & \
               (df["labeled_app_name"] == df["original_app_name"]) 
        df.loc[mask, "labeled_app_name"] = label_val
    
    for key, label_val in labels.get("patterns", {}).items():
        try:
            _, exe_basename_key, pattern_from_key = key.split("::", 2)
            mask = (
                df["exe"].apply(lambda x: os.path.basename(x).lower()).str.contains(exe_basename_key.lower(), regex=False, na=False) &
                df["title"].str.lower().str.contains(pattern_from_key.lower(), regex=False, na=False) &
                (df["labeled_app_name"] == df["original_app_name"]) 
            )
            df.loc[mask, "labeled_app_name"] = label_val
        except ValueError: print(f"Warning: Skipping malformed pattern key: {key}")
        except Exception as e: print(f"Error applying pattern label for '{key}': {e}")
            
    df["app_name"] = df["labeled_app_name"]
    df.drop(columns=["labeled_app_name"], inplace=True, errors="ignore") 
    
    return df


def group_activities_for_labeling(logs_df: pd.DataFrame) -> pd.DataFrame:
    """Group activities by window title for easier labeling, using original app names."""
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
    buckets_with_tags: List[Dict[str, Any]] = []
    for path in LOGS_DIR.glob("time_buckets_*.json"):
        try:
            # A simple check: if the date string is in the filename, it's a candidate.
            # More robustly, one might always load and then filter by the 'start' time of buckets.
            # For now, this heuristic assumes filenames might contain the date or be session-specific.
            load_this_file = date_str in path.name 
            if not load_this_file: # If date not in name, check if it's a generic session file
                # This could be refined, e.g. by checking file modification time if relevant
                # For now, we assume if date_str is not in name, we might still need to check its contents
                # if the file naming isn't strictly 'time_buckets_YYYY-MM-DD.json'
                pass # Let's assume we always check contents if not explicitly dated by name

            session_tag = path.name.replace("time_buckets_", "").replace(".json", "")
            with open(path, "r", encoding="utf-8") as f: data = json.load(f)
            for b_data in data:
                if b_data.get("start", "")[:10] == date_str: # Filter by bucket's start date
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
        # Read current data
        with open(bucket_file_path, "r", encoding="utf-8") as f:
            all_buckets_data = json.load(f)
        
        found_bucket = False
        for bucket_data in all_buckets_data:
            if bucket_data.get("start") == bucket_start_iso:
                bucket_data["summary"] = new_summary
                found_bucket = True; break
        
        if not found_bucket:
            msg = f"Bucket {bucket_start_iso} not found in {bucket_file_path.name} for summary update."
            if _is_streamlit_running(): 
                st.error(msg)
            else: 
                print(msg)
            return False
            
        # Write updated data back
        with open(bucket_file_path, "w", encoding="utf-8") as f:
            json.dump(all_buckets_data, f, indent=2)
        return True
    except Exception as e:
        msg = f"Error updating official summary in {bucket_file_path.name}: {e}"
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
    except Exception as e: print(f"Error loading categories from {CATEGORIES_FILE}: {e}"); return []

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
        with open(bucket_file_path, "r", encoding="utf-8") as f: # Read first
            all_buckets_data = json.load(f)
        
        found_bucket = False
        for bucket_data in all_buckets_data:
            if bucket_data.get("start") == bucket_start_iso:
                bucket_data["category_id"] = new_category_id
                found_bucket = True; break
        
        if not found_bucket:
            msg = f"Bucket {bucket_start_iso} not found in {bucket_file_path.name} for category update."
            if _is_streamlit_running(): 
                st.error(msg)
            else: 
                print(msg)
            return False
            
        with open(bucket_file_path, "w", encoding="utf-8") as f: # Write updated
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
    if "timestamp" not in logs_df.columns:
        msg="Log data missing timestamp column."
        if is_st_true: 
            st.error(msg)
        else: 
            print(msg)
        return False

    try:
        logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"], errors='coerce')
        logs_df.dropna(subset=["timestamp"], inplace=True) 
        if logs_df.empty:
             msg="No valid timestamps in log data after conversion."
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
        
    logs_df = logs_df.sort_values("timestamp")
    min_time = logs_df["timestamp"].min(); max_time = logs_df["timestamp"].max()
    bucket_duration_minutes = 5 
    bucket_size_timedelta = pd.Timedelta(minutes=bucket_duration_minutes)
    start_time_aligned = min_time.floor(f"{bucket_duration_minutes}min")
    
    bucket_start_timestamps: List[pd.Timestamp] = []
    current_bucket_start = start_time_aligned
    while current_bucket_start <= max_time: 
        bucket_start_timestamps.append(current_bucket_start)
        current_bucket_start += bucket_size_timedelta

    if not bucket_start_timestamps:
        msg = f"Not enough time range in logs for {date_str} to create {bucket_duration_minutes}-min buckets."
        if is_st_true: 
            st.warning(msg)
        else: 
            print(msg)
        return False

    generated_buckets_list: List[Dict[str, Any]] = []
    
    # This import is fine here as generate_time_buckets_from_logs is typically called from dashboard context
    # and llm_utils does not import data_utils directly.
    from .llm_utils import generate_summary_from_raw_with_llm  

    for bucket_dt_start in bucket_start_timestamps:
        bucket_dt_end = bucket_dt_start + bucket_size_timedelta
        current_bucket_logs_df = logs_df[(logs_df["timestamp"] >= bucket_dt_start) & (logs_df["timestamp"] < bucket_dt_end)]
        
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
            # The fourth return value (prompt) is ignored here.
            llm_summary, llm_category_id, _suggested_category, _prompt = generate_summary_from_raw_with_llm(
                window_titles_list, ocr_text_snippets, allow_suggestions=True, return_prompt=False
            )
            if llm_summary: bucket_entry["summary"] = llm_summary
            if llm_category_id: bucket_entry["category_id"] = llm_category_id
            generated_buckets_list.append(bucket_entry)

    if not generated_buckets_list:
        msg = f"No activity found within defined time buckets for {date_str}."
        if is_st_true: 
            st.warning(msg)
        else: 
            print(msg)
        # Decide if an empty bucket file should be written or just return False
        # For now, returning False as no useful buckets were made.
        return False 

    # For retroactive generation, a file per date is clear.
    # For live agent generation, session_tag based naming is in standalone_focus_monitor.
    bucket_file_path = LOGS_DIR / f"time_buckets_{date_str}.json" 
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(bucket_file_path, "w", encoding="utf-8") as f:
            json.dump(generated_buckets_list, f, indent=2)
        msg = f"Time buckets for {date_str} generated ({len(generated_buckets_list)} buckets) and saved to {bucket_file_path.name}"
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

BROWSER_PROFILES_FILE = LOGS_DIR / "browser_profiles.json"

def load_browser_profiles() -> List[Dict[str, Any]]:
    """Load browser profiles from the profiles file."""
    if not BROWSER_PROFILES_FILE.exists():
        # Return default profile if file doesn't exist
        return [
            {
                "name": "Edge - Personal",
                "exe_pattern": "msedge.exe",
                "color_rgb": [7, 43, 71],  # Store as list for JSON compatibility
                "color_tolerance": 15,
                "search_rect": [9, 3, 30, 30],  # [x_offset, y_offset, width, height]
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
    """
    Validate a browser profile configuration.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ["name", "exe_pattern", "color_rgb", "search_rect", "category_id"]
    
    for field in required_fields:
        if field not in profile:
            return False, f"Missing required field: {field}"
    
    # Validate color_rgb format
    color_rgb = profile["color_rgb"]
    if not isinstance(color_rgb, (list, tuple)) or len(color_rgb) != 3:
        return False, "color_rgb must be a list/tuple of 3 values [R, G, B]"
    
    for color_val in color_rgb:
        if not isinstance(color_val, int) or not (0 <= color_val <= 255):
            return False, "Each RGB value must be an integer between 0 and 255"
    
    # Validate search_rect format
    search_rect = profile["search_rect"]
    if not isinstance(search_rect, (list, tuple)) or len(search_rect) != 4:
        return False, "search_rect must be a list/tuple of 4 values [x, y, width, height]"
    
    for rect_val in search_rect:
        if not isinstance(rect_val, int) or rect_val < 0:
            return False, "Each search_rect value must be a non-negative integer"
    
    # Validate tolerance if present
    if "color_tolerance" in profile:
        tolerance = profile["color_tolerance"]
        if not isinstance(tolerance, int) or not (0 <= tolerance <= 255):
            return False, "color_tolerance must be an integer between 0 and 255"
    
    # Validate exe_pattern
    if not isinstance(profile["exe_pattern"], str) or not profile["exe_pattern"].strip():
        return False, "exe_pattern must be a non-empty string"
    
    # Validate name
    if not isinstance(profile["name"], str) or not profile["name"].strip():
        return False, "name must be a non-empty string"
    
    return True, ""

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color string to RGB tuple."""
    # Remove # if present
    hex_color = hex_color.lstrip('#')
    # Convert to RGB
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert RGB tuple to hex color string."""
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

def get_default_search_rect_for_browser(exe_pattern: str) -> List[int]:
    """Get default search rectangle based on browser type."""
    # These are starting points - users should adjust based on their setup
    defaults = {
        "msedge.exe": [9, 3, 30, 30],      # Edge profile indicator location
        "chrome.exe": [40, 8, 25, 25],     # Chrome profile avatar location  
        "firefox.exe": [45, 10, 20, 20],   # Firefox profile area
        "brave.exe": [40, 8, 25, 25],      # Similar to Chrome
        "opera.exe": [35, 8, 25, 25],      # Opera profile area
        "vivaldi.exe": [40, 8, 25, 25],    # Similar to Chrome
    }
    return defaults.get(exe_pattern, [10, 5, 25, 25])  # Generic default