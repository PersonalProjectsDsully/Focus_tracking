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
import requests 
from typing import Optional, Dict, Any, List

# --- Configuration ---
LOGS_DIR = Path(__file__).parent / "focus_logs"
LABELS_FILE = LOGS_DIR / "activity_labels.json"
FEEDBACK_FILE = LOGS_DIR / "block_feedback.json" # For personal notes

LLM_API_URL = "http://localhost:11434/api/generate"  # Ollama API endpoint
LLM_MODEL = "llama3.1:8b"  # Specify your desired model


# --- Tracker Control Functions ---
def is_tracker_running():
    pid = st.session_state.get("tracker_pid")
    if pid is None:
        return False
    return psutil.pid_exists(pid)

def start_tracker():
    if is_tracker_running():
        st.info("Focus tracker already running.")
        return
    script = str(Path(__file__).parent / "standalone_focus_monitor.py")
    try:
        process = subprocess.Popen([sys.executable, script])
        st.session_state["tracker_pid"] = process.pid
        st.success("Focus tracker started")
    except Exception as e:
        st.error(f"Failed to start tracker: {e}")

def stop_tracker():
    pid = st.session_state.get("tracker_pid")
    if not pid or not psutil.pid_exists(pid):
        st.info("Focus tracker is not running.")
        st.session_state["tracker_pid"] = None
        return
    try:
        p = psutil.Process(pid)
        p.terminate()
        p.wait(timeout=5) # Give it a moment to terminate
        st.success("Focus tracker stopped")
    except psutil.NoSuchProcess:
        st.info("Focus tracker process not found (already stopped?).")
    except Exception as e:
        st.error(f"Failed to stop tracker: {e}")
    finally:
        st.session_state["tracker_pid"] = None

# --- Data Management Functions ---
def load_available_dates():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_dates = set([
        f.name.split("_")[-1].replace(".jsonl", "")
        for f in LOGS_DIR.glob("focus_log_*.jsonl")
    ])
    summary_dates = set([
        f.name.split("_")[-1].replace(".json", "")
        for f in LOGS_DIR.glob("daily_summary_*.json")
    ])
    all_dates = sorted(list(log_dates.union(summary_dates)), reverse=True)
    return all_dates

def load_log_entries(date_str: str) -> pd.DataFrame:
    file_path = LOGS_DIR / f"focus_log_{date_str}.jsonl"
    if not file_path.exists():
        return pd.DataFrame()
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                data.append(entry)
            except json.JSONDecodeError:
                # Optionally log this error
                continue
    return pd.DataFrame(data)

def load_daily_summary(date_str: str) -> Optional[Dict[str, Any]]:
    file_path = LOGS_DIR / f"daily_summary_{date_str}.json"
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            st.error(f"Error decoding summary file for {date_str}. Generating from logs if possible.")
    
    log_path = LOGS_DIR / f"focus_log_{date_str}.jsonl"
    if log_path.exists():
        st.info(f"No pre-generated summary for {date_str}. Generating from raw logs...")
        return generate_summary_from_logs(date_str)
    else:
        st.warning(f"No log data available to generate summary for {date_str}")
        return None

def generate_summary_from_logs(date_str: str) -> Optional[Dict[str, Any]]:
    logs_df = load_log_entries(date_str)
    if logs_df.empty:
        return None
        
    logs_df = apply_labels_to_logs(logs_df) # Apply custom labels first
    
    total_duration = logs_df.get("duration", pd.Series(dtype='float')).sum()
    summary = {
        "date": date_str,
        "totalTime": total_duration,
        "appBreakdown": []
    }
    if total_duration == 0:
        return summary

    # Ensure 'app_name' exists before grouping; default if not
    if 'app_name' not in logs_df.columns:
        logs_df['app_name'] = 'Unknown Application'
    if 'exe' not in logs_df.columns:
        logs_df['exe'] = 'unknown.exe'


    app_groups = logs_df.groupby("app_name").agg(
        exe_path=("exe", "first"),
        time_spent=("duration", "sum"),
        window_titles=("title", lambda x: list(set(str(t) for t in x if pd.notna(t)))[:50]) if 'title' in logs_df.columns else ([],)
    ).reset_index()
    
    for _, row in app_groups.iterrows():
        percentage = (row["time_spent"] / total_duration * 100) if total_duration > 0 else 0
        summary["appBreakdown"].append({
            "appName": row["app_name"],
            "exePath": row.get("exe_path", "unknown.exe"),
            "timeSpent": int(row["time_spent"]),
            "percentage": round(percentage, 2),
            "windowTitles": row.get("window_titles", []) if isinstance(row.get("window_titles"), list) else [str(row.get("window_titles"))]
        })
    
    summary["appBreakdown"].sort(key=lambda x: x["timeSpent"], reverse=True)
    return summary

# --- Label Management Functions ---
def load_labels() -> Dict[str, Any]:
    if not LABELS_FILE.exists():
        return {}
    try:
        with open(LABELS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading labels: {e}")
        return {}

def save_labels(labels: Dict[str, Any]) -> bool:
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(LABELS_FILE, 'w', encoding='utf-8') as f:
            json.dump(labels, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving labels: {e}")
        return False

def apply_labels_to_logs(logs_df: pd.DataFrame) -> pd.DataFrame:
    if logs_df.empty:
        return logs_df
    
    labels = load_labels()
    if not labels: # No custom labels defined
        df_copy = logs_df.copy()
        df_copy['original_app_name'] = df_copy.get('app_name', "Unknown Application")
        return df_copy
    
    df = logs_df.copy()
    
    # Ensure essential columns exist and handle NaN by converting to string
    for col, default_val in [('app_name', "Unknown Application"), ('exe', "unknown.exe"), ('title', "")]:
        if col not in df.columns:
            df[col] = default_val
        df[col] = df[col].fillna(default_val).astype(str)


    df['original_app_name'] = df['app_name'] # Preserve original before modification
    df['labeled_app_name'] = df['app_name']  # Start with current app_name, then override
    
    # Apply exact titles: key is "exact::FULL_EXE_PATH::TITLE_STRING"
    for key, label_val in labels.get('exact_titles', {}).items():
        try:
            _, exe_from_key, title_from_key = key.split("::", 2)
            mask = (df['exe'].str.lower() == exe_from_key.lower()) & \
                   (df['title'] == title_from_key) # Case-sensitive title match
            df.loc[mask, 'labeled_app_name'] = label_val
        except ValueError: continue # Skip malformed keys
        except Exception as e: print(f"Error applying exact title label for '{key}': {e}")

    # Apply exact exe (basename): key is "BASENAME.exe"
    for exe_basename_key, label_val in labels.get('exact_exe', {}).items():
        # Apply only if not already labeled by a more specific rule (exact_title)
        mask = (df['exe'].apply(lambda x: os.path.basename(x).lower()) == exe_basename_key.lower()) & \
               (df['labeled_app_name'] == df['original_app_name']) 
        df.loc[mask, 'labeled_app_name'] = label_val
        
    # Apply patterns (basename): key is "pattern::BASENAME.exe::TITLE_PATTERN"
    for key, label_val in labels.get('patterns', {}).items():
        try:
            _, exe_basename_key, pattern_from_key = key.split("::", 2)
            # Apply only if not already labeled by a more specific rule
            mask = (df['exe'].apply(lambda x: os.path.basename(x).lower()).str.contains(exe_basename_key.lower(), regex=False)) & \
                   (df['title'].str.lower().str.contains(pattern_from_key.lower(), regex=False)) & \
                   (df['labeled_app_name'] == df['original_app_name'])
            df.loc[mask, 'labeled_app_name'] = label_val
        except ValueError: continue
        except Exception as e: print(f"Error applying pattern label for '{key}': {e}")
            
    df['app_name'] = df['labeled_app_name'] # Final app_name is the labeled one
    df.drop(columns=['labeled_app_name'], inplace=True, errors='ignore')
    return df

def group_activities_for_labeling(logs_df: pd.DataFrame) -> pd.DataFrame:
    if logs_df.empty or 'title' not in logs_df.columns:
        return pd.DataFrame()
    
    # Ensure necessary columns exist and handle NaNs
    for col, default_val in [('app_name', "Unknown Application"), ('exe', "unknown.exe"), ('duration', 0)]:
        if col not in logs_df.columns:
            logs_df[col] = default_val
        logs_df[col] = logs_df[col].fillna(default_val)

    logs_df['title'] = logs_df['title'].astype(str) # Ensure title is string for grouping
    
    # Group by title, then find mode for app_name and exe, sum duration
    title_groups_data = []
    for title_val, group in logs_df.groupby('title'):
        app_name_mode = group['app_name'].mode()
        exe_mode = group['exe'].mode()
        title_groups_data.append({
            'title': title_val,
            'app_name': app_name_mode[0] if not app_name_mode.empty else "Unknown Application",
            'exe': exe_mode[0] if not exe_mode.empty else "unknown.exe", # Store full path
            'count': len(group),
            'duration': group['duration'].sum()
        })
        
    grouped_df = pd.DataFrame(title_groups_data)
    if not grouped_df.empty:
        total_log_duration = logs_df['duration'].sum()
        grouped_df['percentage'] = (grouped_df['duration'] / total_log_duration * 100).round(1) if total_log_duration > 0 else 0.0
        return grouped_df.sort_values('duration', ascending=False)
    return pd.DataFrame()


# --- Feedback Management Functions (for block_feedback.json - Personal Notes) ---
def load_block_feedback() -> Dict[str, str]:
    if not FEEDBACK_FILE.exists(): return {}
    try:
        with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f: return json.load(f)
    except Exception as e: st.error(f"Error loading personal notes (feedback file): {e}"); return {}

def save_block_feedback(feedback_data: Dict[str, str]) -> bool:
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f: json.dump(feedback_data, f, indent=2)
        return True
    except Exception as e: st.error(f"Error saving personal notes (feedback file): {e}"); return False

# --- Time Bucket Management Functions (for time_buckets_*.json files - Official Summaries) ---
def load_time_buckets_for_date(date_str: str) -> List[Dict[str, Any]]:
    buckets_with_tags = []
    for path in LOGS_DIR.glob("time_buckets_*.json"):
        try:
            session_tag = path.name.replace("time_buckets_", "").replace(".json", "")
            with open(path, "r", encoding="utf-8") as f: data = json.load(f)
            for b in data:
                if b.get("start", "")[:10] == date_str:
                    b_copy = b.copy()
                    b_copy['session_tag'] = session_tag # Crucial for identifying source file
                    buckets_with_tags.append(b_copy)
        except Exception as e: st.error(f"Error reading bucket file {path.name}: {e}")
    buckets_with_tags.sort(key=lambda x: x.get("start", ""))
    return buckets_with_tags

def update_bucket_summary_in_file(session_tag: str, bucket_start_iso: str, new_summary: str) -> bool:
    bucket_file_path = LOGS_DIR / f"time_buckets_{session_tag}.json"
    if not bucket_file_path.exists(): st.error(f"Official summary file not found: {bucket_file_path.name}"); return False
    try:
        with open(bucket_file_path, "r", encoding="utf-8") as f: all_buckets_data = json.load(f)
        found_bucket = False
        for bucket_data in all_buckets_data:
            if bucket_data.get("start") == bucket_start_iso:
                bucket_data["summary"] = new_summary; found_bucket = True; break
        if not found_bucket: st.error(f"Bucket {bucket_start_iso} not found in {bucket_file_path.name}"); return False
        with open(bucket_file_path, "w", encoding="utf-8") as f: json.dump(all_buckets_data, f, indent=2)
        return True
    except Exception as e: st.error(f"Error updating official summary in {bucket_file_path.name}: {e}"); return False

# --- LLM Interaction Functions ---
def _call_llm_api(prompt: str, operation_name: str) -> Optional[str]:
    """Generic LLM API call helper."""
    payload = {"model": LLM_MODEL, "prompt": prompt, "stream": False}
    try:
        st.info(f"Sending request to LLM ({LLM_MODEL}) for {operation_name}...")
        response = requests.post(LLM_API_URL, json=payload, timeout=60) # Timeout of 60s
        response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
        data = response.json()
        llm_response_text = data.get("response", "").strip()
        if not llm_response_text and prompt: # Warn if prompt was non-empty but response is
             st.warning(f"LLM returned an empty response for {operation_name}.")
        st.success(f"LLM {operation_name} successful.")
        return llm_response_text
    except requests.exceptions.Timeout: 
        st.error(f"LLM API request timed out ({LLM_MODEL} at {LLM_API_URL}) during {operation_name}.")
    except requests.exceptions.RequestException as e: 
        st.error(f"LLM API communication error ({LLM_MODEL} at {LLM_API_URL}) during {operation_name}: {e}")
    except Exception as e: 
        st.error(f"An unexpected error occurred during LLM {operation_name}: {e}")
    return None


def refine_summary_with_llm(original_summary: str, user_feedback: str) -> Optional[str]:
    prompt = f"""Current summary of activity:
"{original_summary}"

User's feedback or additional details:
"{user_feedback}"

Based on the current summary and the user's input, please provide an improved and concise summary.
If the user's input suggests a complete rewrite, then generate that new summary.
Focus on integrating the user's points accurately.
Output only the new summary text.
"""
    return _call_llm_api(prompt, "refinement")

def generate_summary_from_raw_with_llm(bucket_titles: List[str], bucket_ocr_texts: List[str]) -> Optional[str]:
    titles_to_send = [str(t) for t in bucket_titles if t and str(t).strip()] if bucket_titles else []
    ocr_to_send = [str(o) for o in bucket_ocr_texts if o and str(o).strip()] if bucket_ocr_texts else []
    
    text_parts = list(set(titles_to_send + ocr_to_send)) # Unique raw data points
    if not text_parts:
        st.info("No raw titles or OCR text available in this bucket to generate a summary.")
        return "" # Return empty string to signify no summary could be made from raw

    prompt_text = "Please provide a concise summary of computer activity based on the following window titles and detected text fragments. Focus on the primary tasks or topics.\n\n"
    if titles_to_send:
        prompt_text += "Window Titles:\n" + "\n".join([f"- \"{t}\"" for t in titles_to_send]) + "\n\n"
    if ocr_to_send:
        prompt_text += "Detected Text (OCR Snippets):\n" + "\n".join([f"- \"{o}\"" for o in ocr_to_send]) + "\n\n"
    prompt_text += "Output only the summary text."
    return _call_llm_api(prompt_text, "raw data summarization")


# --- Visualization Functions ---
def create_pie_chart(app_data: List[Dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    if not app_data: 
        fig.update_layout(title="App Usage Breakdown - No Data Available")
        return fig

    # Group small slices into "Other"
    data_to_plot = sorted(app_data, key=lambda x: x.get("timeSpent", 0), reverse=True)
    if len(data_to_plot) > 10:
        top_apps = data_to_plot[:9] # Show top 9
        other_time = sum(app.get("timeSpent", 0) for app in data_to_plot[9:])
        if other_time > 0:
            top_apps.append({"appName": "Other Apps", "timeSpent": other_time})
        data_to_plot = top_apps
    
    labels = [f"{app.get('appName','N/A')} ({app.get('timeSpent',0)//60}m)" for app in data_to_plot]
    values = [app.get('timeSpent',0) for app in data_to_plot]
    
    if not values or sum(values) == 0: # Handle case where all values are zero
        fig.update_layout(title="App Usage Breakdown - No Time Spent Data")
        return fig

    fig.add_trace(go.Pie(labels=labels, values=values, hole=0.3, textinfo='percent+label', 
                         hoverinfo='label+value', textfont_size=12, pull=[0.05]*len(labels)))
    fig.update_layout(title_text="App Usage Breakdown", height=500, legend_title_text="Applications")
    return fig

def create_browser_chart(app_data: List[Dict[str, Any]]) -> Optional[go.Figure]:
    if not app_data: return None
    browser_entries = []
    for app in app_data:
        name_lower = app.get("appName","").lower()
        if any(b_keyword in name_lower for b_keyword in ["chrome", "msedge", "edge", "firefox"]):
            browser_type = "MS Edge" if "msedge" in name_lower or "edge" in name_lower else \
                           ("Google Chrome" if "chrome" in name_lower else "Firefox")
            browser_entries.append({**app, "browserType": browser_type}) # Add type for coloring
    
    if not browser_entries: return None
    
    df_browsers = pd.DataFrame(browser_entries).sort_values(by=["browserType", "timeSpent"], ascending=[True, False])
    df_browsers["timeSpentMinutesText"] = (df_browsers["timeSpent"] // 60).astype(str) + "m"

    fig = px.bar(df_browsers, x="appName", y="timeSpent", 
                 text="timeSpentMinutesText",
                 labels={"appName": "Browser Instance / Profile", "timeSpent": "Time Spent (seconds)"}, 
                 title="Browser Usage Details", color="browserType",
                 color_discrete_map={"MS Edge": "#0078D4", "Google Chrome": "#DB4437", "Firefox": "#FF7139"})
    fig.update_layout(xaxis_tickangle=-45, height=450, legend_title_text="Browser Type", uniformtext_minsize=8, uniformtext_mode='hide')
    fig.update_traces(textposition='outside')
    return fig


# --- Time Bucket Summaries Page ---
def display_time_bucket_summaries():
    st.title("📝 5-Minute Summaries & Feedback")
    dates = load_available_dates()
    if not dates: st.error("No data found in focus_logs directory."); st.stop()
    selected_date = st.selectbox("Select a date", dates, key="summaries_date_select")
    
    official_buckets = load_time_buckets_for_date(selected_date)
    personal_notes_data = load_block_feedback() 

    if not official_buckets: st.info(f"No 5-minute summary blocks found for {selected_date}."); return

    for idx, bucket_data in enumerate(official_buckets):
        bucket_start_iso = bucket_data["start"]
        session_tag = bucket_data.get("session_tag", "unknown_session")
        
        start_dt = pd.to_datetime(bucket_start_iso)
        end_dt = pd.to_datetime(bucket_data.get("end", bucket_start_iso))
        time_range_display = f"{start_dt.strftime('%H:%M')} - {end_dt.strftime('%H:%M')} (UTC {start_dt.strftime('%Y-%m-%d')})"
        
        current_official_summary_text = bucket_data.get("summary", "") # This is from time_buckets_*.json
        current_personal_note_text = personal_notes_data.get(bucket_start_iso, "")

        with st.expander(time_range_display):
            st.markdown("**Current Official Summary (from log file):**")
            st.caption(current_official_summary_text or "_No official summary available for this block._")
            
            st.markdown("**Your Input Text:**")
            user_input_text = st.text_area(
                "Use this text area to: (1) Save a personal note, (2) Provide feedback for the LLM to refine the summary, or (3) Write your own summary to replace the current one.", 
                value=current_personal_note_text, 
                key=f"user_text_area_{idx}_{bucket_start_iso}",
                height=100 # Adjust height as needed
            )

            # --- Personal Note Actions ---
            st.markdown("---")
            st.write("**Manage Personal Note (Saved Separately):**")
            note_cols = st.columns(2)
            if note_cols[0].button("Save Input as Personal Note", key=f"save_personal_note_{idx}_{bucket_start_iso}", help="Saves the text in the 'Your Input Text' area above as a private note for this block. This does NOT change the Official Summary."):
                personal_notes_data[bucket_start_iso] = user_input_text.strip()
                if save_block_feedback(personal_notes_data): st.success("Personal note saved!")
                else: st.error("Failed to save personal note.")
            
            if note_cols[1].button("Delete Personal Note", key=f"delete_personal_note_{idx}_{bucket_start_iso}", help="Deletes the private note associated with this block."):
                if bucket_start_iso in personal_notes_data:
                    del personal_notes_data[bucket_start_iso]
                    if save_block_feedback(personal_notes_data): 
                        st.success("Personal note deleted!"); time.sleep(0.5); st.rerun()
                    else: st.error("Failed to delete personal note.")
                else: st.info("No personal note to delete for this block.")

            # --- Official Summary Actions (Modifies time_buckets_*.json) ---
            st.markdown("---")
            st.write("**Manage Official Summary (Modifies Log File):**")
            official_summary_action_cols = st.columns(3)
            
            with official_summary_action_cols[0]:
                if st.button("Set as Official Summary", key=f"set_official_summary_from_input_{idx}_{bucket_start_iso}", help="Replaces the 'Current Official Summary' above with the text from the 'Your Input Text' area. This directly modifies the summary in the log file."):
                    if user_input_text.strip():
                        if update_bucket_summary_in_file(session_tag, bucket_start_iso, user_input_text.strip()):
                            st.success("Official summary updated with your input!"); time.sleep(1); st.rerun()
                        else: st.error("Failed to update official summary with your input.")
                    else: st.warning("Text area is empty. Please enter text to set as the official summary.")
            
            with official_summary_action_cols[1]:
                if st.button("Refine LLM Summary", key=f"refine_llm_summary_btn_{idx}_{bucket_start_iso}", help="Uses the 'Current Official Summary' AND the text from 'Your Input Text' (as feedback) to ask the LLM to generate an improved summary. This new summary will replace the current one in the log file."):
                    if user_input_text.strip(): 
                        refined_summary_text = refine_summary_with_llm(current_official_summary_text, user_input_text.strip())
                        if refined_summary_text is not None:
                            if update_bucket_summary_in_file(session_tag, bucket_start_iso, refined_summary_text):
                                st.success("LLM summary refined and updated!"); time.sleep(1); st.rerun()
                            else: st.error("Failed to save LLM-refined official summary.")
                    else: st.warning("Please enter some feedback in the 'Your Input Text' area to help refine the summary.")

            with official_summary_action_cols[2]:
                if st.button("Re-Generate Original LLM Summary", key=f"regenerate_original_llm_summary_btn_{idx}_{bucket_start_iso}", help="Asks the LLM to create a brand new summary based on the raw window titles and OCR text recorded for this block. This will replace the 'Current Official Summary' in the log file."):
                    raw_titles_list = bucket_data.get("titles", [])
                    raw_ocr_list = bucket_data.get("ocr_text", [])
                    regenerated_summary_text = generate_summary_from_raw_with_llm(raw_titles_list, raw_ocr_list)
                    if regenerated_summary_text is not None: 
                        if update_bucket_summary_in_file(session_tag, bucket_start_iso, regenerated_summary_text):
                            st.success("Original LLM summary re-generated and updated!"); time.sleep(1); st.rerun()
                        else: st.error("Failed to save LLM re-generated official summary.")


# --- Label Editor Page ---
def display_label_editor():
    st.title("🏷 Activity Label Editor")
    available_dates = load_available_dates()
    if not available_dates: st.error("No data found in focus_logs directory."); st.stop()
    selected_date = st.selectbox("Select a date for labeling activities", available_dates, key="label_editor_date_select")
    
    logs_df_for_labeling = load_log_entries(selected_date)
    if logs_df_for_labeling.empty: st.warning(f"No log entries found for {selected_date} to label."); st.stop()
    
    # title_groups will contain 'exe' as full path
    title_groups_for_labeling = group_activities_for_labeling(logs_df_for_labeling) 
    
    # For forms needing basenames (App Label, Pattern Label)
    all_exe_basenames_in_logs = sorted(list(set(
        os.path.basename(str(exe)) for exe in logs_df_for_labeling['exe'].dropna().unique() if 'exe' in logs_df_for_labeling
    )))

    current_labels = load_labels()
    for k_label_type in ['exact_titles', 'exact_exe', 'patterns']: # Ensure keys exist
        current_labels.setdefault(k_label_type, {})

    st.subheader("Current Custom Labels")
    if not any(current_labels.get(k) for k in ['exact_titles', 'exact_exe', 'patterns']):
        st.info("No custom labels defined yet. Use the forms below to create labels.")
    else:
        ltab1, ltab2, ltab3 = st.tabs(["Window Title Labels", "Application Labels (Basename)", "Pattern Labels (Basename)"])
        with ltab1: # Exact Titles (Key: exact::FULL_EXE_PATH::TITLE)
            data = [{"Key": k, "Label": v, "App": k.split("::")[1], "Title": k.split("::")[2][:70]+"..."} 
                    for k,v in current_labels.get('exact_titles',{}).items() if k.count("::")==2]
            if data: st.dataframe(pd.DataFrame(data)[["App", "Title", "Label"]], use_container_width=True)
            else: st.info("No window title labels defined.")
        with ltab2: # Exact Exe (Key: BASENAME.exe)
            data = [{"App Basename": k, "Label": v} for k,v in current_labels.get('exact_exe',{}).items()]
            if data: st.dataframe(pd.DataFrame(data), use_container_width=True)
            else: st.info("No application basename labels defined.")
        with ltab3: # Patterns (Key: pattern::BASENAME.exe::PATTERN_STRING)
            data = [{"App Basename": k.split("::")[1], "Title Pattern": k.split("::")[2], "Label": v} 
                    for k,v in current_labels.get('patterns',{}).items() if k.count("::")==2]
            if data: st.dataframe(pd.DataFrame(data), use_container_width=True)
            else: st.info("No pattern labels defined.")
    
    if not title_groups_for_labeling.empty:
        st.subheader("Available Window Titles for Labeling (from Grouped Activities)")
        st.dataframe(title_groups_for_labeling[["title", "app_name", "exe", "duration"]].head(20), use_container_width=True, height=300) # Show top 20
    else:
        st.info("No grouped window title activities for this date. You can still create Application or Pattern labels based on raw log data.")

    form_tab_title, form_tab_app, form_tab_pattern = st.tabs([
        "Create: Label Specific Window Title", 
        "Create: Label Application (by Basename)", 
        "Create: Label by Pattern (App Basename + Title Pattern)"
    ])

    with form_tab_title: # Label by specific title (uses full EXE path from title_groups)
        st.markdown("Labels specific `Window Title` + `Full Executable Path` from grouped activities above.")
        if title_groups_for_labeling.empty:
            st.warning("No grouped activities available to select specific titles.")
        else:
            with st.form("form_specific_title_label"):
                # Let user pick from title_groups
                titles_available = title_groups_for_labeling['title'].tolist()
                selected_titles = st.multiselect("Select Window Title(s) from grouped activities", options=titles_available, key="sel_spec_titles")
                new_label_for_titles = st.text_input("Enter Label for selected title(s)", key="label_spec_titles")
                submitted_spec_title = st.form_submit_button("Save Title Label(s)")
                if submitted_spec_title and selected_titles and new_label_for_titles.strip():
                    for title_to_label in selected_titles:
                        # Find the full exe path associated with this title from title_groups
                        exe_for_title = title_groups_for_labeling[title_groups_for_labeling['title'] == title_to_label]['exe'].iloc[0]
                        label_key = f"exact::{exe_for_title}::{title_to_label}"
                        current_labels['exact_titles'][label_key] = new_label_for_titles.strip()
                    if save_labels(current_labels): st.success("Specific title label(s) saved!"); time.sleep(1); st.rerun()
                elif submitted_spec_title: st.warning("Please select title(s) and enter a label.")
    
    with form_tab_app: # Label by app basename
        st.markdown("Labels an entire application based on its `Executable Basename` (e.g., `chrome.exe`).")
        if not all_exe_basenames_in_logs:
            st.warning("No executable basenames found in logs for this date.")
        else:
            with st.form("form_app_basename_label"):
                selected_app_basename = st.selectbox("Select Application Basename", options=all_exe_basenames_in_logs, key="sel_app_basename")
                new_label_for_app = st.text_input(f"Enter Label for '{selected_app_basename}'", 
                                                  value=current_labels['exact_exe'].get(selected_app_basename, ""), key="label_app_basename")
                submitted_app_label = st.form_submit_button("Save Application Label")
                if submitted_app_label:
                    if new_label_for_app.strip():
                        current_labels['exact_exe'][selected_app_basename] = new_label_for_app.strip()
                    elif selected_app_basename in current_labels['exact_exe']: # If label cleared, remove it
                        del current_labels['exact_exe'][selected_app_basename]
                    if save_labels(current_labels): st.success("Application label updated!"); time.sleep(1); st.rerun()
    
    with form_tab_pattern: # Label by pattern (app basename + title pattern)
        st.markdown("Labels activities matching an `App Basename` AND a `Window Title Pattern` (case-insensitive).")
        if not all_exe_basenames_in_logs:
            st.warning("No executable basenames found for pattern matching.")
        else:
            with st.form("form_pattern_label"):
                selected_app_basename_patt = st.selectbox("Select Application Basename for Pattern", options=all_exe_basenames_in_logs, key="sel_app_basename_patt")
                title_pattern_for_label = st.text_input("Enter Window Title Pattern (e.g., 'youtube', 'project x')", key="patt_title_input")
                new_label_for_pattern = st.text_input("Enter Label for this pattern", key="label_patt_input")
                submitted_pattern_label = st.form_submit_button("Save Pattern Label")
                if submitted_pattern_label and selected_app_basename_patt and title_pattern_for_label.strip() and new_label_for_pattern.strip():
                    label_key = f"pattern::{selected_app_basename_patt}::{title_pattern_for_label.strip()}"
                    current_labels['patterns'][label_key] = new_label_for_pattern.strip()
                    if save_labels(current_labels): st.success("Pattern label saved!"); time.sleep(1); st.rerun()
                elif submitted_pattern_label: st.warning("Please select an app, enter a title pattern, and a label.")

    st.subheader("Delete Labels")
    del_ltab1, del_ltab2, del_ltab3 = st.tabs(["Delete Window Title Label", "Delete App Label", "Delete Pattern Label"])
    with del_ltab1:
        keys_to_del = list(current_labels.get('exact_titles', {}).keys())
        if keys_to_del:
            sel_key = st.selectbox("Select Specific Title Label to Delete", keys_to_del, 
                                   format_func=lambda k: f"{k.split('::')[1].split(os.sep)[-1]} - '{k.split('::')[2][:30]}...' -> {current_labels['exact_titles'][k]}", 
                                   key="del_sel_exact_title")
            if st.button("Delete Selected Title Label", key="del_btn_exact_title"):
                if sel_key in current_labels['exact_titles']: del current_labels['exact_titles'][sel_key]
                if save_labels(current_labels): st.success("Label deleted."); time.sleep(1); st.rerun()
        else: st.info("No specific title labels to delete.")
    # ... (similar deletion UI for exact_exe and patterns) ...

    st.subheader("Preview With Labels Applied")
    if not logs_df_for_labeling.empty:
        labeled_logs_preview = apply_labels_to_logs(logs_df_for_labeling.copy()) # Important: use a copy
        if not labeled_logs_preview.empty and 'app_name' in labeled_logs_preview.columns:
            preview_agg = labeled_logs_preview.groupby('app_name').agg(
                total_duration_sec=('duration', 'sum'),
                original_names=('original_app_name', lambda x: list(set(x))[:3]), # Show a few original names
                log_entries=('timestamp', 'count') if 'timestamp' in labeled_logs_preview else ('duration', 'count')
            ).reset_index().sort_values('total_duration_sec', ascending=False)
            preview_agg['Total Duration (min)'] = (preview_agg['total_duration_sec'] / 60).round(1)
            st.dataframe(preview_agg[['app_name', 'Total Duration (min)', 'log_entries', 'original_names']], use_container_width=True)
            if not preview_agg.empty and preview_agg['total_duration_sec'].sum() > 0:
                fig_preview = px.pie(preview_agg, values='total_duration_sec', names='app_name', title="Labeled Activity Preview", hole=0.3)
                st.plotly_chart(fig_preview, use_container_width=True)
    
    st.subheader("📊 Overall Label Summary for Selected Date")
    if not logs_df_for_labeling.empty:
        labeled_logs_summary_data = apply_labels_to_logs(logs_df_for_labeling.copy())
        if not labeled_logs_summary_data.empty and 'app_name' in labeled_logs_summary_data.columns:
            # Group by the final 'app_name' which is the label itself or "Unlabeled" (original_app_name if no label matched)
            # For "Unlabeled", we need to sum where app_name == original_app_name
            
            summary_list = []
            # Labeled items
            labeled_items = labeled_logs_summary_data[labeled_logs_summary_data['app_name'] != labeled_logs_summary_data['original_app_name']]
            if not labeled_items.empty:
                labeled_agg = labeled_items.groupby('app_name')['duration'].sum().reset_index()
                for _, row in labeled_agg.iterrows():
                    summary_list.append({'Label Category': row['app_name'], 'Total Time (sec)': row['duration']})
            
            # Unlabeled items
            unlabeled_items = labeled_logs_summary_data[labeled_logs_summary_data['app_name'] == labeled_logs_summary_data['original_app_name']]
            if not unlabeled_items.empty:
                 # Could further group unlabeled by original_app_name if desired, or just sum all as "Unlabeled"
                total_unlabeled_time = unlabeled_items['duration'].sum()
                if total_unlabeled_time > 0:
                     summary_list.append({'Label Category': 'Uncategorized (Original Names)', 'Total Time (sec)': total_unlabeled_time})

            if summary_list:
                df_label_summary = pd.DataFrame(summary_list).sort_values('Total Time (sec)', ascending=False)
                df_label_summary['Total Time (min)'] = (df_label_summary['Total Time (sec)'] / 60).round(1)
                df_label_summary['Percentage'] = (df_label_summary['Total Time (sec)'] / df_label_summary['Total Time (sec)'].sum() * 100).round(1)
                st.dataframe(df_label_summary[['Label Category', 'Total Time (min)', 'Percentage']], use_container_width=True)
                if df_label_summary['Total Time (sec)'].sum() > 0:
                    fig_summary = px.pie(df_label_summary, values='Total Time (sec)', names='Label Category', title="Time Distribution by Label Category", hole=0.4)
                    st.plotly_chart(fig_summary, use_container_width=True)
            else:
                st.info("No data to summarize by label for this date after applying labels.")


# --- Dashboard Page ---
def display_dashboard():
    st.title("📊 Focus Monitor Dashboard")
    available_dates = load_available_dates()
    if not available_dates: st.error("No data found in focus_logs directory. Is the monitor running?"); st.stop()
    selected_date = st.selectbox("Select a date to view", available_dates, key="dashboard_date_select")
    
    daily_summary_data = load_daily_summary(selected_date) # This now applies labels via generate_summary_from_logs
    if not daily_summary_data: st.warning(f"Could not load or generate a summary for {selected_date}."); st.stop()

    # --- Summary Metrics ---
    d_col1, d_col2, d_col3 = st.columns(3)
    d_col1.metric("🕒 Total Tracked Time", f"{daily_summary_data.get('totalTime', 0) // 60} min")
    # Add Focus Score and Meeting Time if they are part of your daily_summary_data structure from standalone_focus_monitor.py
    # For now, they are not in the simplified generate_summary_from_logs
    
    st.subheader("🧠 Time Distribution by Application/Activity (Post-Labeling)")
    app_breakdown_data = daily_summary_data.get("appBreakdown", [])
    if app_breakdown_data:
        fig_pie_main = create_pie_chart(app_breakdown_data)
        st.plotly_chart(fig_pie_main, use_container_width=True)
        
        df_app_top = pd.DataFrame([{
            "Activity/Application": app.get("appName", "N/A"),
            "Time (min)": app.get("timeSpent", 0) // 60,
            "Percentage": f"{app.get('percentage', 0):.1f}%"
        } for app in app_breakdown_data[:10]]) # Show top 10 labeled activities
        st.dataframe(df_app_top, use_container_width=True)
    else:
        st.info("No application usage data available for this date in the summary.")
    
    st.subheader("🌐 Browser Usage Analysis (Labeled)")
    fig_browser_main = create_browser_chart(app_breakdown_data) # Uses labeled appName
    if fig_browser_main:
        st.plotly_chart(fig_browser_main, use_container_width=True)
    else:
        st.info("No browser usage detected or included in the summary for this date.")
   
    with st.expander("📄 View Raw Focus Log Entries (Unlabeled Original Data)"):
        raw_log_df = load_log_entries(selected_date) # Load raw, pre-labeling
        if not raw_log_df.empty:
            # Create a copy to avoid modifying the original DataFrame from cache
            df_to_display = raw_log_df.copy()

            # Define the desired final column names and their original sources
            # Ensure 'timestamp' is the primary source for the 'Time' column
            final_columns_map = {
                "Time": "timestamp", # This will be formatted
                "Original App Name": "app_name",
                "Original Window Title": "title",
                "Duration (s)": "duration"
            }
            
            # Columns that must exist in raw_log_df for this section to work
            required_raw_cols = ["timestamp", "app_name", "title", "duration"]
            
            if all(col in df_to_display.columns for col in required_raw_cols):
                # Format the 'timestamp' to a readable 'Time' string IN A NEW COLUMN
                df_to_display['Formatted Time'] = pd.to_datetime(df_to_display["timestamp"], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

                # Select and rename:
                # Create a new DataFrame with only the columns we want, renaming them.
                # This avoids duplicate column issues.
                display_data = {}
                if 'Formatted Time' in df_to_display.columns: # Check if conversion was successful
                    display_data['Time'] = df_to_display['Formatted Time']
                if 'app_name' in df_to_display.columns:
                    display_data['Original App Name'] = df_to_display['app_name']
                if 'title' in df_to_display.columns:
                    display_data['Original Window Title'] = df_to_display['title']
                if 'duration' in df_to_display.columns:
                    display_data['Duration (s)'] = df_to_display['duration']

                final_display_df = pd.DataFrame(display_data)

                # Sort by the 'Time' column (which is now uniquely named and formatted)
                if 'Time' in final_display_df.columns:
                    final_display_df = final_display_df.sort_values("Time", ascending=False)
                
                st.dataframe(final_display_df, use_container_width=True, height=400)
            else:
                missing_cols = [col for col in required_raw_cols if col not in df_to_display.columns]
                st.info(f"Essential columns missing in raw log for display: {', '.join(missing_cols)}")
        else:
            st.info("No raw log entries available for this date.")


    # Simplified feedback view on dashboard - directs to Summaries tab
    st.subheader("📝 Quick View: Recent Personal Notes for Blocks")
    dashboard_personal_notes = load_block_feedback()
    notes_for_selected_date = {k:v for k,v in dashboard_personal_notes.items() if k.startswith(selected_date)}
    if notes_for_selected_date:
        st.caption("Last 3 personal notes for this date (full management on '📝 Summaries' tab):")
        for ts, note_text in list(notes_for_selected_date.items())[-3:]: # Show most recent 3
             st.caption(f"_{pd.to_datetime(ts).strftime('%H:%M')}_: {note_text[:70]}{'...' if len(note_text) > 70 else ''}")
    else:
        st.info("No personal notes recorded for this date yet. Use the '📝 Summaries' tab to add notes and manage official summaries.")
    
    st.markdown("---")
    st.markdown("""
    ### About Focus Monitor
    This dashboard displays data collected by the Focus Monitor agent. The agent tracks your active windows and applications to help you understand your computer usage patterns.
    - **Labels**: Use the '🏷 Activity Label Editor' tab to categorize your time.
    - **Summaries**: Use the '📝 Summaries' tab to review, note, and refine 5-minute block summaries.
    
    To generate data:
    1. Ensure the Focus Monitor agent (`standalone_focus_monitor.py`) is running in the background.
    2. Use your computer normally. Data is logged to the `focus_logs` directory.
    3. Refresh this dashboard to view updated statistics and summaries.
    """)


# --- Main App ---
def main():
    st.set_page_config(page_title="Focus Monitor Dashboard", layout="wide", initial_sidebar_state="expanded")
    
    # Sidebar for tracker control (optional, can be removed if tracker is managed externally)
    # st.sidebar.title("Tracker Control")
    # if st.sidebar.button("Start Tracker", key="start_tracker_btn"):
    #     start_tracker()
    # if st.sidebar.button("Stop Tracker", key="stop_tracker_btn"):
    #     stop_tracker()
    # st.sidebar.caption(f"Tracker Running: {is_tracker_running()}")
    # st.sidebar.markdown("---")


    tab_dashboard, tab_label_editor, tab_summaries = st.tabs([
        "📊 Dashboard", 
        "🏷 Activity Label Editor", 
        "📝 5-Min Summaries & Feedback"
    ])

    with tab_dashboard:
        display_dashboard()
    with tab_label_editor:
        display_label_editor()
    with tab_summaries:
        display_time_bucket_summaries()

if __name__ == "__main__":
    main()