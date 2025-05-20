# Utility functions and constants for the Streamlit dashboard.

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
import requests
import streamlit as st

# --- Configuration ---
LOGS_DIR = Path(__file__).parent / "focus_logs"
LABELS_FILE = LOGS_DIR / "activity_labels.json"
FEEDBACK_FILE = LOGS_DIR / "block_feedback.json"  # For personal notes
CATEGORIES_FILE = LOGS_DIR / "activity_categories.json"  # For user-defined categories

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
        p.wait(timeout=5)  # Give it a moment to terminate
        st.success("Focus tracker stopped")
    except psutil.NoSuchProcess:
        st.info("Focus tracker process not found (already stopped?).")
    except Exception as e:
        st.error(f"Failed to stop tracker: {e}")
    finally:
        st.session_state["tracker_pid"] = None


def llm_test_page():
    """A dedicated page for testing LLM category suggestions"""
    st.title("🧪 LLM Category Suggestion Test Tool")

    # Check if categories exist
    categories = load_categories()
    if not categories:
        st.error("No categories have been defined. Please create categories first.")
        st.info("Go to the Activity Categories Manager tab to create categories.")
        return

    st.write(
        "This tool helps diagnose issues with category suggestions by directly testing the LLM's responses."
    )

    # Display current categories
    with st.expander("View Current Categories"):
        for cat in categories:
            st.write(
                f"**{cat.get('name')}** ({cat.get('id')}): {cat.get('description')}"
            )

    # Input area for test data
    st.subheader("Test Data")
    test_method = st.radio(
        "Choose test method",
        ["Sample Titles", "Custom Titles", "Raw Log Data"],
        help="Select how you want to provide test data",
    )

    test_titles = []

    if test_method == "Sample Titles":
        st.write("Using sample titles:")
        sample_options = [
            "Programming sample (VS Code, GitHub)",
            "Meeting sample (Zoom, Calendar)",
            "Web browsing sample (Chrome, browsing)",
            "Gaming sample (Steam, game windows)",
        ]
        selected_sample = st.selectbox("Select a sample dataset", sample_options)

        if selected_sample == "Programming sample (VS Code, GitHub)":
            test_titles = [
                "VS Code - focus_monitor.py",
                "GitHub - Issues - Focus Tracking",
                "Stack Overflow - Python multithreading question",
            ]
        elif selected_sample == "Meeting sample (Zoom, Calendar)":
            test_titles = [
                "Zoom Meeting - Weekly Planning",
                "Google Calendar - Meeting Schedule",
                "Slack - #team-updates channel",
            ]
        elif selected_sample == "Web browsing sample (Chrome, browsing)":
            test_titles = [
                "Chrome - Reddit - r/programming",
                "Chrome - YouTube - Python Tutorial",
                "Chrome - Amazon - Shopping Cart",
            ]
        elif selected_sample == "Gaming sample (Steam, game windows)":
            test_titles = [
                "Steam - Library",
                "EscapeFromTarkov",
                "Discord - Gaming Server",
            ]

        st.write("Window titles:")
        for title in test_titles:
            st.write(f"- {title}")

    elif test_method == "Custom Titles":
        custom_titles = st.text_area(
            "Enter window titles (one per line)",
            placeholder="VS Code - my_project.py\nChrome - Google Search\nSlack - #general",
        )
        if custom_titles:
            test_titles = [t.strip() for t in custom_titles.split("\n") if t.strip()]

    elif test_method == "Raw Log Data":
        # Load available dates
        available_dates = load_available_dates()
        if not available_dates:
            st.error("No log data found.")
            return

        selected_date = st.selectbox("Select date", available_dates)
        logs_df = load_log_entries(selected_date)

        if logs_df.empty:
            st.error(f"No log entries found for {selected_date}")
            return

        if "title" not in logs_df.columns:
            st.error("Log data doesn't contain window title information")
            return

        # Show sample of log data
        st.write("Sample of log data:")
        st.dataframe(logs_df[["title", "app_name"]].head(10))

        # Get unique titles
        unique_titles = logs_df["title"].dropna().unique().tolist()
        test_titles = unique_titles[:10]  # Use first 10 unique titles

        st.write(f"Using {len(test_titles)} unique window titles from the logs")

    # Test button
    if test_titles and st.button("Test LLM Category Suggestion", type="primary"):
        with st.spinner("Sending request to LLM... This may take a few moments."):
            # Generate the prompt
            prompt_text = "Please provide a concise summary of computer activity based on the following window titles and detected text fragments. Focus on the primary tasks or topics.\n\n"
            prompt_text += (
                "Window Titles:\n"
                + "\n".join([f'- "{t}"' for t in test_titles])
                + "\n\n"
            )

            prompt_text += "Based on the activity, please categorize it into ONE of the following categories:\n\n"
            for cat in categories:
                prompt_text += (
                    f"- {cat.get('name')} ({cat.get('id')}): {cat.get('description')}\n"
                )

            prompt_text += "\nIf none of these categories fit well, you can suggest a new category instead."
            prompt_text += "\n\nFirst, provide a concise summary of the activity."
            prompt_text += "\nThen, on a new line after 'CATEGORY:', provide ONLY the category ID that best matches the activity, or 'none' if no categories fit well."
            prompt_text += "\nIf you answered 'none', then on a new line after 'SUGGESTION:', provide a suggested new category name and short description in the format 'name | description'."

            # Show the full prompt
            with st.expander("View full prompt sent to LLM"):
                st.code(prompt_text)

            # Call LLM API
            llm_response = _call_llm_api(prompt_text, "test category suggestion")

            # Display raw response
            st.subheader("Raw LLM Response")
            st.code(llm_response)

            # Process response
            st.subheader("Processed Response")

            # Parse the response
            try:
                parts = llm_response.split("CATEGORY:")
                if len(parts) < 2:
                    st.error("❌ LLM did not include 'CATEGORY:' tag in response.")
                    st.warning(
                        "The LLM should respond with a summary followed by 'CATEGORY:' tag."
                    )
                    return

                summary_text = parts[0].strip()
                category_text = parts[1].strip()

                st.write("**Summary:**")
                st.info(summary_text)

                # Get category ID (first word after CATEGORY:)
                category_id = category_text.split()[0] if category_text.split() else ""

                # Check for suggestions
                suggested_category = ""
                if "SUGGESTION:" in category_text:
                    suggestion_parts = category_text.split("SUGGESTION:")
                    if len(suggestion_parts) > 1:
                        category_text = suggestion_parts[0].strip()
                        suggested_category = suggestion_parts[1].strip()

                        # Update category_id
                        category_id = (
                            category_text.split()[0] if category_text.split() else ""
                        )

                # Check if the category is "none"
                if category_id.lower() == "none":
                    st.write(
                        "**Category:** None (LLM suggests creating a new category)"
                    )
                    category_id = ""
                else:
                    # Verify if category ID exists
                    valid_cat_ids = [cat.get("id", "") for cat in categories]
                    if category_id in valid_cat_ids:
                        cat_name = next(
                            (
                                cat.get("name", "Unknown")
                                for cat in categories
                                if cat.get("id") == category_id
                            ),
                            "Unknown",
                        )
                        st.write(f"**Category:** {cat_name} ({category_id})")
                        st.success("✅ Category ID is valid")
                    else:
                        st.error(f"❌ Invalid Category ID: '{category_id}'")
                        st.warning(
                            "The category ID doesn't match any existing categories."
                        )

                # Display suggestion if present
                if suggested_category:
                    st.write("**Suggested New Category:**")

                    # Check format
                    if "|" in suggested_category:
                        name, desc = suggested_category.split("|", 1)
                        st.info(f"Name: {name.strip()}")
                        st.info(f"Description: {desc.strip()}")
                        st.success(
                            "✅ Suggestion format is correct (contains name | description)"
                        )

                        # Show how it would be added
                        suggested_id = name.strip().lower().replace(" ", "_")
                        st.write("Would be added as:")
                        st.code(
                            f"""{{
  "id": "{suggested_id}",
  "name": "{name.strip()}",
  "description": "{desc.strip()}"
}}""",
                            language="json",
                        )

                        # Option to add it
                        if st.button("Add This Category Now"):
                            # Create the new category
                            new_category = {
                                "id": suggested_id,
                                "name": name.strip(),
                                "description": desc.strip(),
                            }

                            # Check if ID already exists
                            existing_ids = [cat.get("id", "") for cat in categories]
                            if suggested_id in existing_ids:
                                st.error(
                                    f"A category with ID '{suggested_id}' already exists."
                                )
                            else:
                                # Add it to categories
                                categories.append(new_category)
                                if save_categories(categories):
                                    st.success(
                                        f"Created new category '{name.strip()}'!"
                                    )
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("Failed to save the new category.")
                    else:
                        st.warning(
                            "⚠️ Suggestion format is incorrect (missing '|' separator)"
                        )
                        st.info(f"Raw suggestion: {suggested_category}")

            except Exception as e:
                st.error(f"Error processing LLM response: {e}")
                st.info(
                    "This indicates a problem with the response format or parsing logic."
                )

    # Help section
    with st.expander("Troubleshooting Tips"):
        st.write(
            """
        ### Troubleshooting LLM Category Suggestions
        
        If category suggestions aren't working correctly:
        
        1. **Check response format**: The LLM should respond with:
           - A summary text
           - A "CATEGORY:" tag followed by a category ID or "none"
           - A "SUGGESTION:" tag (if "none" was given) with a new category name and description
        
        2. **Verify category IDs**: Make sure the LLM is using the exact category IDs you defined
        
        3. **Format issues**: The suggestion should use the format "name | description"
        
        4. **API connection**: Ensure your LLM API (Ollama) is running and accessible
        
        5. **Prompt clarity**: The prompt should clearly explain the format requirements
        """
        )


# --- Data Management Functions ---
def load_available_dates():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_dates = set(
        [
            f.name.split("_")[-1].replace(".jsonl", "")
            for f in LOGS_DIR.glob("focus_log_*.jsonl")
        ]
    )
    summary_dates = set(
        [
            f.name.split("_")[-1].replace(".json", "")
            for f in LOGS_DIR.glob("daily_summary_*.json")
        ]
    )
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
            st.error(
                f"Error decoding summary file for {date_str}. Generating from logs if possible."
            )

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

    logs_df = apply_labels_to_logs(logs_df)  # Apply custom labels first

    total_duration = logs_df.get("duration", pd.Series(dtype="float")).sum()
    summary = {"date": date_str, "totalTime": total_duration, "appBreakdown": []}
    if total_duration == 0:
        return summary

    # Ensure 'app_name' exists before grouping; default if not
    if "app_name" not in logs_df.columns:
        logs_df["app_name"] = "Unknown Application"
    if "exe" not in logs_df.columns:
        logs_df["exe"] = "unknown.exe"

    app_groups = (
        logs_df.groupby("app_name")
        .agg(
            exe_path=("exe", "first"),
            time_spent=("duration", "sum"),
            window_titles=(
                ("title", lambda x: list(set(str(t) for t in x if pd.notna(t)))[:50])
                if "title" in logs_df.columns
                else ([],)
            ),
        )
        .reset_index()
    )

    for _, row in app_groups.iterrows():
        percentage = (
            (row["time_spent"] / total_duration * 100) if total_duration > 0 else 0
        )
        summary["appBreakdown"].append(
            {
                "appName": row["app_name"],
                "exePath": row.get("exe_path", "unknown.exe"),
                "timeSpent": int(row["time_spent"]),
                "percentage": round(percentage, 2),
                "windowTitles": (
                    row.get("window_titles", [])
                    if isinstance(row.get("window_titles"), list)
                    else [str(row.get("window_titles"))]
                ),
            }
        )

    summary["appBreakdown"].sort(key=lambda x: x["timeSpent"], reverse=True)
    return summary


# --- Label Management Functions ---
def load_labels() -> Dict[str, Any]:
    if not LABELS_FILE.exists():
        return {}
    try:
        with open(LABELS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading labels: {e}")
        return {}


def save_labels(labels: Dict[str, Any]) -> bool:
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(LABELS_FILE, "w", encoding="utf-8") as f:
            json.dump(labels, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving labels: {e}")
        return False


def apply_labels_to_logs(logs_df: pd.DataFrame) -> pd.DataFrame:
    if logs_df.empty:
        return logs_df

    labels = load_labels()
    if not labels:  # No custom labels defined
        df_copy = logs_df.copy()
        df_copy["original_app_name"] = df_copy.get("app_name", "Unknown Application")
        return df_copy

    df = logs_df.copy()

    # Ensure essential columns exist and handle NaN by converting to string
    for col, default_val in [
        ("app_name", "Unknown Application"),
        ("exe", "unknown.exe"),
        ("title", ""),
    ]:
        if col not in df.columns:
            df[col] = default_val
        df[col] = df[col].fillna(default_val).astype(str)

    df["original_app_name"] = df["app_name"]  # Preserve original before modification
    df["labeled_app_name"] = df[
        "app_name"
    ]  # Start with current app_name, then override

    # Apply exact titles: key is "exact::FULL_EXE_PATH::TITLE_STRING"
    for key, label_val in labels.get("exact_titles", {}).items():
        try:
            _, exe_from_key, title_from_key = key.split("::", 2)
            mask = (df["exe"].str.lower() == exe_from_key.lower()) & (
                df["title"] == title_from_key
            )  # Case-sensitive title match
            df.loc[mask, "labeled_app_name"] = label_val
        except ValueError:
            continue  # Skip malformed keys
        except Exception as e:
            print(f"Error applying exact title label for '{key}': {e}")

    # Apply exact exe (basename): key is "BASENAME.exe"
    for exe_basename_key, label_val in labels.get("exact_exe", {}).items():
        # Apply only if not already labeled by a more specific rule (exact_title)
        mask = (
            df["exe"].apply(lambda x: os.path.basename(x).lower())
            == exe_basename_key.lower()
        ) & (df["labeled_app_name"] == df["original_app_name"])
        df.loc[mask, "labeled_app_name"] = label_val

    # Apply patterns (basename): key is "pattern::BASENAME.exe::TITLE_PATTERN"
    for key, label_val in labels.get("patterns", {}).items():
        try:
            _, exe_basename_key, pattern_from_key = key.split("::", 2)
            # Apply only if not already labeled by a more specific rule
            mask = (
                (
                    df["exe"]
                    .apply(lambda x: os.path.basename(x).lower())
                    .str.contains(exe_basename_key.lower(), regex=False)
                )
                & (
                    df["title"]
                    .str.lower()
                    .str.contains(pattern_from_key.lower(), regex=False)
                )
                & (df["labeled_app_name"] == df["original_app_name"])
            )
            df.loc[mask, "labeled_app_name"] = label_val
        except ValueError:
            continue
        except Exception as e:
            print(f"Error applying pattern label for '{key}': {e}")

    df["app_name"] = df["labeled_app_name"]  # Final app_name is the labeled one
    df.drop(columns=["labeled_app_name"], inplace=True, errors="ignore")
    return df


def group_activities_for_labeling(logs_df: pd.DataFrame) -> pd.DataFrame:
    if logs_df.empty or "title" not in logs_df.columns:
        return pd.DataFrame()

    # Ensure necessary columns exist and handle NaNs
    for col, default_val in [
        ("app_name", "Unknown Application"),
        ("exe", "unknown.exe"),
        ("duration", 0),
    ]:
        if col not in logs_df.columns:
            logs_df[col] = default_val
        logs_df[col] = logs_df[col].fillna(default_val)

    logs_df["title"] = logs_df["title"].astype(
        str
    )  # Ensure title is string for grouping

    # Group by title, then find mode for app_name and exe, sum duration
    title_groups_data = []
    for title_val, group in logs_df.groupby("title"):
        app_name_mode = group["app_name"].mode()
        exe_mode = group["exe"].mode()
        title_groups_data.append(
            {
                "title": title_val,
                "app_name": (
                    app_name_mode[0]
                    if not app_name_mode.empty
                    else "Unknown Application"
                ),
                "exe": (
                    exe_mode[0] if not exe_mode.empty else "unknown.exe"
                ),  # Store full path
                "count": len(group),
                "duration": group["duration"].sum(),
            }
        )

    grouped_df = pd.DataFrame(title_groups_data)
    if not grouped_df.empty:
        total_log_duration = logs_df["duration"].sum()
        grouped_df["percentage"] = (
            (grouped_df["duration"] / total_log_duration * 100).round(1)
            if total_log_duration > 0
            else 0.0
        )
        return grouped_df.sort_values("duration", ascending=False)
    return pd.DataFrame()


# --- Feedback Management Functions (for block_feedback.json - Personal Notes) ---
def load_block_feedback() -> Dict[str, str]:
    if not FEEDBACK_FILE.exists():
        return {}
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading personal notes (feedback file): {e}")
        return {}


def save_block_feedback(feedback_data: Dict[str, str]) -> bool:
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(feedback_data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving personal notes (feedback file): {e}")
        return False


# --- Time Bucket Management Functions (for time_buckets_*.json files - Official Summaries) ---
def load_time_buckets_for_date(date_str: str) -> List[Dict[str, Any]]:
    buckets_with_tags = []
    for path in LOGS_DIR.glob("time_buckets_*.json"):
        try:
            session_tag = path.name.replace("time_buckets_", "").replace(".json", "")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for b in data:
                if b.get("start", "")[:10] == date_str:
                    b_copy = b.copy()
                    b_copy["session_tag"] = (
                        session_tag  # Crucial for identifying source file
                    )
                    buckets_with_tags.append(b_copy)
        except Exception as e:
            st.error(f"Error reading bucket file {path.name}: {e}")
    buckets_with_tags.sort(key=lambda x: x.get("start", ""))
    return buckets_with_tags


def update_bucket_summary_in_file(
    session_tag: str, bucket_start_iso: str, new_summary: str
) -> bool:
    bucket_file_path = LOGS_DIR / f"time_buckets_{session_tag}.json"
    if not bucket_file_path.exists():
        st.error(f"Official summary file not found: {bucket_file_path.name}")
        return False
    try:
        with open(bucket_file_path, "r", encoding="utf-8") as f:
            all_buckets_data = json.load(f)
        found_bucket = False
        for bucket_data in all_buckets_data:
            if bucket_data.get("start") == bucket_start_iso:
                bucket_data["summary"] = new_summary
                found_bucket = True
                break
        if not found_bucket:
            st.error(f"Bucket {bucket_start_iso} not found in {bucket_file_path.name}")
            return False
        with open(bucket_file_path, "w", encoding="utf-8") as f:
            json.dump(all_buckets_data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error updating official summary in {bucket_file_path.name}: {e}")
        return False


# --- Category Management Functions ---
def load_categories():
    """Load user-defined activity categories from JSON file."""
    categories_file = Path(__file__).parent / "focus_logs" / "activity_categories.json"
    if not categories_file.exists():
        return []
    try:
        with open(categories_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("categories", [])
    except Exception as e:
        print(f"Error loading categories: {e}")
        return []


def save_categories(categories: List[Dict[str, str]]) -> bool:
    """Save user-defined activity categories to JSON file."""
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(CATEGORIES_FILE, "w", encoding="utf-8") as f:
            json.dump({"categories": categories}, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving categories: {e}")
        return False


def update_bucket_category_in_file(
    session_tag: str, bucket_start_iso: str, new_category_id: str
) -> bool:
    """Update the category_id field for a specific time bucket in its file."""
    bucket_file_path = LOGS_DIR / f"time_buckets_{session_tag}.json"
    if not bucket_file_path.exists():
        st.error(f"Official summary file not found: {bucket_file_path.name}")
        return False
    try:
        with open(bucket_file_path, "r", encoding="utf-8") as f:
            all_buckets_data = json.load(f)
        found_bucket = False
        for bucket_data in all_buckets_data:
            if bucket_data.get("start") == bucket_start_iso:
                bucket_data["category_id"] = new_category_id
                found_bucket = True
                break
        if not found_bucket:
            st.error(f"Bucket {bucket_start_iso} not found in {bucket_file_path.name}")
            return False
        with open(bucket_file_path, "w", encoding="utf-8") as f:
            json.dump(all_buckets_data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error updating bucket category in {bucket_file_path.name}: {e}")
        return False


# --- LLM Interaction Functions ---
def _call_llm_api(prompt: str, operation_name: str = "LLM call") -> Optional[str]:
    """Generic LLM API call helper."""
    LLM_API_URL = "http://localhost:11434/api/generate"
    LLM_MODEL = "llama3.1:8b"

    payload = {"model": LLM_MODEL, "prompt": prompt, "stream": False}
    try:
        print(f"Sending request to LLM ({LLM_MODEL}) for {operation_name}...")
        response = requests.post(LLM_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        llm_response_text = data.get("response", "").strip()
        if not llm_response_text and prompt:
            print(f"LLM returned an empty response for {operation_name}.")
        else:
            print(f"LLM {operation_name} successful.")
        return llm_response_text
    except requests.exceptions.Timeout:
        print(f"LLM API request timed out during {operation_name}.")
    except requests.exceptions.RequestException as e:
        print(f"LLM API communication error during {operation_name}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during LLM {operation_name}: {e}")
    return None


