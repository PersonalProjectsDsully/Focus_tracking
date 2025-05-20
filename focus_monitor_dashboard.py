import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from focus_tracker.dashboard_utils import *

# --- Add these retroactive processing functions ---


def display_retroactive_processor():
    """Tab for retroactively processing historical focus logs."""
    st.title("⏮️ Historical Data Processor")

    # Check available dates with raw logs
    all_dates = load_available_dates()

    # Find dates with logs but missing time buckets or summaries
    incomplete_dates = []

    for date_str in all_dates:
        log_path = LOGS_DIR / f"focus_log_{date_str}.jsonl"
        summary_path = LOGS_DIR / f"daily_summary_{date_str}.json"

        # Check if time buckets exist for this date
        time_buckets = load_time_buckets_for_date(date_str)

        if log_path.exists() and (not summary_path.exists() or not time_buckets):
            incomplete_dates.append(
                {
                    "date": date_str,
                    "has_summary": summary_path.exists(),
                    "has_buckets": bool(time_buckets),
                    "bucket_count": len(time_buckets) if time_buckets else 0,
                }
            )

    if not incomplete_dates:
        st.success(
            "All focus logs have been processed! No historical data needs processing."
        )
        st.info(
            "If you've just added categories and want to apply them to existing time buckets, use the 'Recategorize Existing Data' section below."
        )
    else:
        st.warning(
            f"Found {len(incomplete_dates)} dates with focus logs that need processing."
        )

        # Display dates needing processing
        df_incomplete = pd.DataFrame(incomplete_dates)
        st.dataframe(df_incomplete, use_container_width=True)

        # Process selected date
        selected_date = st.selectbox(
            "Select a date to process",
            options=[d["date"] for d in incomplete_dates],
            key="retroactive_date_select",
        )

        selected_info = next(
            (d for d in incomplete_dates if d["date"] == selected_date), None
        )

        if selected_info:
            col1, col2 = st.columns(2)

            # Generate summary if needed
            if not selected_info["has_summary"]:
                if col1.button("Generate Daily Summary", key="gen_daily_summary_btn"):
                    with st.spinner(f"Generating daily summary for {selected_date}..."):
                        summary = generate_summary_from_logs(selected_date)
                        if summary:
                            # Save summary
                            LOGS_DIR.mkdir(parents=True, exist_ok=True)
                            summary_path = (
                                LOGS_DIR / f"daily_summary_{selected_date}.json"
                            )
                            with open(summary_path, "w", encoding="utf-8") as f:
                                json.dump(summary, f, indent=2)
                            st.success(
                                f"Daily summary for {selected_date} generated and saved!"
                            )
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"Failed to generate summary for {selected_date}")

            # Generate time buckets if needed
            if not selected_info["has_buckets"] or selected_info["bucket_count"] == 0:
                if col2.button("Generate Time Buckets", key="gen_time_buckets_btn"):
                    with st.spinner(f"Generating time buckets for {selected_date}..."):
                        success = generate_time_buckets_from_logs(selected_date)
                        if success:
                            st.success(
                                f"Time buckets for {selected_date} generated and saved!"
                            )
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(
                                f"Failed to generate time buckets for {selected_date}"
                            )

    # Section for recategorizing existing time buckets
    st.markdown("---")
    st.subheader("Recategorize Existing Data")
    st.info(
        "This will apply your current categories to existing time buckets using the LLM."
    )

    # Get all dates with time buckets
    dates_with_buckets = []
    for date_str in all_dates:
        buckets = load_time_buckets_for_date(date_str)
        if buckets:
            # Count buckets with and without categories
            with_category = sum(1 for b in buckets if b.get("category_id", ""))
            total_buckets = len(buckets)

            dates_with_buckets.append(
                {
                    "date": date_str,
                    "total_buckets": total_buckets,
                    "categorized_buckets": with_category,
                    "needs_categorization": with_category < total_buckets,
                }
            )

    if not dates_with_buckets:
        st.warning("No time bucket data found.")
    else:
        # Display dates with their categorization status
        df_dates = pd.DataFrame(dates_with_buckets)
        df_dates["Categorization %"] = (
            df_dates["categorized_buckets"] / df_dates["total_buckets"] * 100
        ).round(1)
        st.dataframe(df_dates, use_container_width=True)

        # Select date to recategorize
        recategorize_date = st.selectbox(
            "Select a date to recategorize",
            options=[d["date"] for d in dates_with_buckets],
            key="recategorize_date_select",
        )

        # Get buckets for the selected date
        buckets_to_recategorize = load_time_buckets_for_date(recategorize_date)

        # Count uncategorized buckets
        uncategorized_count = sum(
            1 for b in buckets_to_recategorize if not b.get("category_id", "")
        )

        # Options for recategorization
        recategorize_all = st.checkbox(
            "Recategorize all buckets (even those already categorized)", value=False
        )

        if st.button("Apply Categories to Time Buckets", key="apply_categories_btn"):
            # Load categories
            categories = load_categories()
            if not categories:
                st.error("No categories defined. Please create categories first.")
            else:
                bucket_count = len(buckets_to_recategorize)

                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                processed = 0
                success_count = 0

                # Process each bucket
                for bucket in buckets_to_recategorize:
                    # Skip already categorized buckets unless recategorize_all is checked
                    if bucket.get("category_id", "") and not recategorize_all:
                        processed += 1
                        progress_bar.progress(processed / bucket_count)
                        continue

                    bucket_start_iso = bucket.get("start", "")
                    session_tag = bucket.get("session_tag", "")

                    status_text.text(
                        f"Processing bucket {processed+1}/{bucket_count}: {bucket_start_iso}"
                    )

                    # Get summary and raw data
                    summary = bucket.get("summary", "")
                    raw_titles = bucket.get("titles", [])
                    raw_ocr = bucket.get("ocr_text", [])

                    if summary:
                        # Use the summary to determine category
                        # Fix: Unpack 3 values instead of 2
                        _, category_id, _ = refine_summary_with_llm(
                            summary, "Please categorize this activity.", ""
                        )
                    elif raw_titles or raw_ocr:
                        # Generate new summary and category from raw data
                        # Fix: Unpack 3 values instead of 2
                        summary, category_id, _ = generate_summary_from_raw_with_llm(
                            raw_titles, raw_ocr
                        )
                    else:
                        # No data to categorize
                        category_id = ""

                    if category_id:
                        # Update category in the file
                        if update_bucket_category_in_file(
                            session_tag, bucket_start_iso, category_id
                        ):
                            success_count += 1

                    processed += 1
                    progress_bar.progress(processed / bucket_count)

                if success_count > 0:
                    st.success(
                        f"Successfully categorized {success_count} buckets for {recategorize_date}!"
                    )
                else:
                    st.warning(
                        "No buckets were categorized. They may already have categories or lack data."
                    )

                time.sleep(1)
                st.rerun()


def generate_time_buckets_from_logs(date_str: str) -> bool:
    """Generate time buckets from focus logs for a specific date."""
    log_path = LOGS_DIR / f"focus_log_{date_str}.jsonl"
    if not log_path.exists():
        st.error(f"Focus log for {date_str} not found.")
        return False

    # Load raw logs
    logs_df = load_log_entries(date_str)
    if logs_df.empty:
        st.error(f"No log entries found for {date_str}.")
        return False

    # Ensure timestamp column exists and convert to datetime
    if "timestamp" not in logs_df.columns:
        st.error("Log data missing timestamp column.")
        return False

    logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"])

    # Sort by timestamp
    logs_df = logs_df.sort_values("timestamp")

    # Get min and max timestamp for the day
    min_time = logs_df["timestamp"].min()
    max_time = logs_df["timestamp"].max()

    # Create 5-minute buckets
    bucket_size = pd.Timedelta(minutes=5)

    # Generate bucket start times (floor to 5-minute intervals)
    start_time = min_time.floor("5min")
    end_time = max_time.ceil("5min")

    bucket_starts = []
    current = start_time
    while current <= end_time:
        bucket_starts.append(current)
        current += bucket_size

    # Create empty buckets
    buckets = []
    session_tag = datetime.now().strftime("%Y%m%dT%H%M%SZ")  # Unique session tag

    for i in range(len(bucket_starts) - 1):
        bucket_start = bucket_starts[i]
        bucket_end = bucket_starts[i + 1]

        # Filter logs for this bucket
        bucket_logs = logs_df[
            (logs_df["timestamp"] >= bucket_start) & (logs_df["timestamp"] < bucket_end)
        ]

        if not bucket_logs.empty:
            # Collect window titles
            titles = []
            if "title" in bucket_logs.columns:
                titles = bucket_logs["title"].dropna().unique().tolist()

            # Create bucket
            bucket = {
                "start": bucket_start.isoformat(),
                "end": bucket_end.isoformat(),
                "titles": titles,
                "ocr_text": [],  # Empty as historical logs might not have OCR
            }

            # Generate summary and category with LLM
            # Fix: Unpack 3 values instead of 2
            summary, category_id, _ = generate_summary_from_raw_with_llm(titles, [])
            if summary:
                bucket["summary"] = summary
            if category_id:
                bucket["category_id"] = category_id

            buckets.append(bucket)

    if not buckets:
        st.warning(f"No activity buckets created for {date_str}.")
        return False

    # Save buckets to file
    bucket_file_path = LOGS_DIR / f"time_buckets_{session_tag}.json"
    try:
        with open(bucket_file_path, "w", encoding="utf-8") as f:
            json.dump(buckets, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving time buckets: {e}")
        return False


def process_activity_data(
    bucket_titles: List[str],
    bucket_ocr_texts: List[str],
    allow_suggestions: bool = True,
    update_bucket: bool = False,
    bucket_data: Optional[Dict[str, Any]] = None,
    bucket_file_path: Optional[str] = None,
    bucket_index: Optional[int] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Unified function to generate summaries/categories from activity data and optionally update bucket files.

    This function combines the functionality of generate_summary_from_raw_with_llm and categorize_new_time_bucket.

    Args:
        bucket_titles: List of window titles
        bucket_ocr_texts: List of OCR text snippets
        allow_suggestions: Whether to allow the LLM to suggest new categories
        update_bucket: Whether to update a bucket file with the generated data
        bucket_data: The bucket data to update (required if update_bucket=True)
        bucket_file_path: Path to the bucket file (required if update_bucket=True)
        bucket_index: Index of bucket in the file (required if update_bucket=True)

    Returns:
        Tuple of (summary_text, category_id, suggested_category)
    """
    # Validate and clean inputs
    titles_to_send = (
        [str(t) for t in bucket_titles if t and str(t).strip()] if bucket_titles else []
    )
    ocr_to_send = (
        [str(o) for o in bucket_ocr_texts if o and str(o).strip()]
        if bucket_ocr_texts
        else []
    )

    text_parts = list(set(titles_to_send + ocr_to_send))  # Unique raw data points
    if not text_parts:
        print("No raw titles or OCR text available to generate a summary")
        return "", "", ""  # Return empty strings for summary, category, and suggestion

    # Check if update_bucket is True but required params are missing
    if update_bucket and (
        bucket_data is None or bucket_file_path is None or bucket_index is None
    ):
        print("Warning: Cannot update bucket - missing required parameters")
        update_bucket = False

    # Get available categories
    categories = load_categories()

    # Build the prompt for the LLM
    prompt_text = "Please provide a concise summary of computer activity based on the following window titles and detected text fragments. Focus on the primary tasks or topics.\n\n"

    if titles_to_send:
        prompt_text += (
            "Window Titles:\n"
            + "\n".join([f'- "{t}"' for t in titles_to_send])
            + "\n\n"
        )
    if ocr_to_send:
        prompt_text += (
            "Detected Text (OCR Snippets):\n"
            + "\n".join([f'- "{o}"' for o in ocr_to_send])
            + "\n\n"
        )

    # Add category classification if categories exist
    if categories:
        prompt_text += "Based on the activity, please categorize it into ONE of the following categories:\n\n"
        for cat in categories:
            prompt_text += (
                f"- {cat.get('name')} ({cat.get('id')}): {cat.get('description')}\n"
            )

        if allow_suggestions:
            prompt_text += "\nIf none of these categories fit well, you can suggest a new category instead."
            prompt_text += "\n\nFirst, provide a concise summary of the activity."
            prompt_text += "\nThen, on a new line after 'CATEGORY:', provide ONLY the category ID that best matches the activity, or 'none' if no categories fit well."
            prompt_text += "\nIf you answered 'none', then on a new line after 'SUGGESTION:', provide a suggested new category name and short description in the format 'name | description'."
        else:
            prompt_text += "\nFirst, provide a concise summary of the activity. Then, on a new line after 'CATEGORY:', provide ONLY the category ID that best matches the activity."
    else:
        prompt_text += "Output only the summary text."

    # Call LLM API
    llm_response = _call_llm_api(prompt_text, "activity processing")

    if not llm_response:
        return "", "", ""

    # Parse response to extract summary, category, and suggestion
    summary_text = llm_response
    category_id = ""
    suggested_category = ""

    if categories:
        parts = llm_response.split("CATEGORY:")
        if len(parts) > 1:
            summary_text = parts[0].strip()
            category_text = parts[1].strip()

            # Check for suggestion
            suggestion_parts = category_text.split("SUGGESTION:")
            if len(suggestion_parts) > 1:
                category_text = suggestion_parts[0].strip()
                suggested_category = suggestion_parts[1].strip()

            # Extract just the category ID (assuming it's the first word after "CATEGORY:")
            category_id = category_text.split()[0] if category_text.split() else ""

            # Handle case where LLM says 'none'
            if category_id.lower() == "none":
                category_id = ""
            else:
                # Verify category ID exists
                valid_cat_ids = [cat.get("id", "") for cat in categories]
                if category_id not in valid_cat_ids:
                    category_id = ""  # Reset if not valid

    # Update bucket if requested
    if (
        update_bucket
        and bucket_data is not None
        and bucket_file_path is not None
        and bucket_index is not None
    ):
        try:
            # Update the bucket data
            if summary_text:
                bucket_data["summary"] = summary_text
            if category_id:
                bucket_data["category_id"] = category_id

            # Read the file
            with open(bucket_file_path, "r", encoding="utf-8") as f:
                all_buckets = json.load(f)

            # Update the bucket at the specified index
            if 0 <= bucket_index < len(all_buckets):
                all_buckets[bucket_index] = bucket_data

                # Write back to the file
                with open(bucket_file_path, "w", encoding="utf-8") as f:
                    json.dump(all_buckets, f, indent=2)
                print(f"Successfully updated bucket at index {bucket_index}")
            else:
                print(
                    f"Error: bucket index {bucket_index} out of range (0-{len(all_buckets)-1})"
                )
        except Exception as e:
            print(f"Error updating bucket: {e}")

    return summary_text, category_id, suggested_category


def generate_summary_and_category(bucket_data):
    """Generate a summary and category for the current bucket."""
    bucket_titles = bucket_data.get("titles", [])
    bucket_ocr_texts = bucket_data.get("ocr_text", [])

    # Skip processing if no data
    if not bucket_titles and not bucket_ocr_texts:
        return None, None

    # Load categories
    categories = load_categories()

    # Prepare prompt
    prompt_text = "Please provide a concise summary of computer activity based on the following window titles and detected text fragments. Focus on the primary tasks or topics.\n\n"
    if bucket_titles:
        prompt_text += (
            "Window Titles:\n" + "\n".join([f'- "{t}"' for t in bucket_titles]) + "\n\n"
        )
    if bucket_ocr_texts:
        prompt_text += (
            "Detected Text (OCR Snippets):\n"
            + "\n".join([f'- "{o}"' for o in bucket_ocr_texts])
            + "\n\n"
        )

    # Add category options if available
    if categories:
        prompt_text += "Based on the activity, please categorize it into ONE of the following categories:\n\n"
        for cat in categories:
            prompt_text += (
                f"- {cat.get('name')} ({cat.get('id')}): {cat.get('description')}\n"
            )
        prompt_text += "\nFirst, provide a concise summary of the activity.\nThen, on a new line after 'CATEGORY:', provide ONLY the category ID that best matches the activity."
    else:
        prompt_text += "Output only the summary text."

    # Call LLM API
    llm_response = _call_llm_api(prompt_text, "bucket summarization")
    if not llm_response:
        return None, None

    # Parse response
    summary_text = llm_response
    category_id = None

    if categories:
        # Split on "CATEGORY:" if present
        parts = llm_response.split("CATEGORY:")
        if len(parts) > 1:
            summary_text = parts[0].strip()
            category_text = parts[1].strip()

            # Get first word as category ID
            if category_text:
                category_id = (
                    category_text.split()[0] if category_text.split() else None
                )

                # Validate category ID
                valid_ids = [cat.get("id", "") for cat in categories]
                if category_id not in valid_ids:
                    category_id = None  # Reset if invalid

    return summary_text, category_id


def generate_summary_from_raw_with_llm(
    bucket_titles: List[str],
    bucket_ocr_texts: List[str],
    allow_suggestions: bool = True,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Generate a summary, category, and optionally suggest a new category if needed.

    Args:
        bucket_titles: List of window titles
        bucket_ocr_texts: List of OCR text snippets
        allow_suggestions: Whether to allow the LLM to suggest new categories

    Returns:
        Tuple of (summary_text, category_id, suggested_category)
    """
    titles_to_send = (
        [str(t) for t in bucket_titles if t and str(t).strip()] if bucket_titles else []
    )
    ocr_to_send = (
        [str(o) for o in bucket_ocr_texts if o and str(o).strip()]
        if bucket_ocr_texts
        else []
    )

    text_parts = list(set(titles_to_send + ocr_to_send))  # Unique raw data points
    if not text_parts:
        print("No raw titles or OCR text available in this bucket")
        return "", "", ""  # Return empty strings for summary, category, and suggestion

    # Get available categories
    categories = load_categories()

    prompt_text = "Please provide a concise summary of computer activity based on the following window titles and detected text fragments. Focus on the primary tasks or topics.\n\n"

    if titles_to_send:
        prompt_text += (
            "Window Titles:\n"
            + "\n".join([f'- "{t}"' for t in titles_to_send])
            + "\n\n"
        )
    if ocr_to_send:
        prompt_text += (
            "Detected Text (OCR Snippets):\n"
            + "\n".join([f'- "{o}"' for o in ocr_to_send])
            + "\n\n"
        )

    # Add category classification if categories exist
    if categories:
        prompt_text += "Based on the activity, please categorize it into ONE of the following categories:\n\n"
        for cat in categories:
            prompt_text += (
                f"- {cat.get('name')} ({cat.get('id')}): {cat.get('description')}\n"
            )

        if allow_suggestions:
            prompt_text += "\nIf none of these categories fit well, you can suggest a new category instead."
            prompt_text += "\n\nFirst, provide a concise summary of the activity."
            prompt_text += "\nThen, on a new line after 'CATEGORY:', provide ONLY the category ID that best matches the activity, or 'none' if no categories fit well."
            prompt_text += "\nIf you answered 'none', then on a new line after 'SUGGESTION:', provide a suggested new category name and short description in the format 'name | description'."
        else:
            prompt_text += "\nFirst, provide a concise summary of the activity. Then, on a new line after 'CATEGORY:', provide ONLY the category ID that best matches the activity."
    else:
        prompt_text += "Output only the summary text."

    llm_response = _call_llm_api(prompt_text, "raw data summarization")

    if not llm_response:
        return "", "", ""

    # Parse response to extract summary, category, and suggestion
    summary_text = llm_response
    category_id = ""
    suggested_category = ""

    if categories:
        parts = llm_response.split("CATEGORY:")
        if len(parts) > 1:
            summary_text = parts[0].strip()
            category_text = parts[1].strip()

            # Check for suggestion
            suggestion_parts = category_text.split("SUGGESTION:")
            if len(suggestion_parts) > 1:
                category_text = suggestion_parts[0].strip()
                suggested_category = suggestion_parts[1].strip()

            # Extract just the category ID (assuming it's the first word after "CATEGORY:")
            category_id = category_text.split()[0] if category_text.split() else ""

            # Handle case where LLM says 'none'
            if category_id.lower() == "none":
                category_id = ""
            else:
                # Verify category ID exists
                valid_cat_ids = [cat.get("id", "") for cat in categories]
                if category_id not in valid_cat_ids:
                    category_id = ""  # Reset if not valid

    return summary_text, category_id, suggested_category


def refine_summary_with_llm(
    original_summary: str,
    user_feedback: str,
    current_category_id: str = "",
    allow_suggestions: bool = True,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Refine an existing summary based on user feedback, preserving category if possible.

    Args:
        original_summary: The current summary text
        user_feedback: User's feedback or input
        current_category_id: Current category ID (if any)
        allow_suggestions: Whether to allow the LLM to suggest new categories

    Returns:
        Tuple of (refined_summary, category_id, suggested_category)
    """
    categories = load_categories()
    current_category_name = "Uncategorized"

    if current_category_id:
        cat_match = next(
            (
                cat.get("name", "Unknown")
                for cat in categories
                if cat.get("id") == current_category_id
            ),
            "Uncategorized",
        )
        current_category_name = cat_match

    prompt = f"""Current summary of activity:
"{original_summary}"

User's feedback or additional details:
"{user_feedback}"

Based on the current summary and the user's input, please provide an improved and concise summary.
If the user's input suggests a complete rewrite, then generate that new summary.
Focus on integrating the user's points accurately.
"""

    if categories:
        prompt += f"\nThe current activity is categorized as: {current_category_name} ({current_category_id})\n\n"
        prompt += "Available categories:\n"
        for cat in categories:
            prompt += (
                f"- {cat.get('name')} ({cat.get('id')}): {cat.get('description')}\n"
            )

        if allow_suggestions:
            prompt += "\nIf none of these categories fit well, you can suggest a new category instead."
            prompt += "\n\nFirst, provide the updated summary."
            prompt += f"\nThen, on a new line after 'CATEGORY:', provide the category ID that best matches (either keep '{current_category_id}' or suggest a different one if appropriate), or 'none' if no categories fit well."
            prompt += "\nIf you answered 'none', then on a new line after 'SUGGESTION:', provide a suggested new category name and short description in the format 'name | description'."
        else:
            prompt += "\nFirst, provide the updated summary. Then, on a new line after 'CATEGORY:', provide the category ID that best matches (either keep the current one or suggest a different one if appropriate)."
    else:
        prompt += "\nOutput only the new summary text."

    llm_response = _call_llm_api(prompt, "refinement")

    if not llm_response:
        return None, current_category_id, ""  # Maintain current category if LLM fails

    # Parse response to extract summary, category, and suggestion
    refined_summary = llm_response
    category_id = current_category_id
    suggested_category = ""

    if categories:
        parts = llm_response.split("CATEGORY:")
        if len(parts) > 1:
            refined_summary = parts[0].strip()
            category_text = parts[1].strip()

            # Check for suggestion
            suggestion_parts = category_text.split("SUGGESTION:")
            if len(suggestion_parts) > 1:
                category_text = suggestion_parts[0].strip()
                suggested_category = suggestion_parts[1].strip()

            # Extract category ID
            category_id = (
                category_text.split()[0]
                if category_text.split()
                else current_category_id
            )

            # Handle case where LLM says 'none'
            if category_id.lower() == "none":
                category_id = ""
            else:
                # Verify category ID exists
                valid_cat_ids = [cat.get("id", "") for cat in categories]
                if (
                    category_id not in valid_cat_ids
                    and category_id != current_category_id
                ):
                    category_id = current_category_id  # Keep current if invalid

    return refined_summary, category_id, suggested_category


# --- Visualization Functions ---
def create_pie_chart(app_data: List[Dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    if not app_data:
        fig.update_layout(title="App Usage Breakdown - No Data Available")
        return fig

    # Group small slices into "Other"
    data_to_plot = sorted(app_data, key=lambda x: x.get("timeSpent", 0), reverse=True)
    if len(data_to_plot) > 10:
        top_apps = data_to_plot[:9]  # Show top 9
        other_time = sum(app.get("timeSpent", 0) for app in data_to_plot[9:])
        if other_time > 0:
            top_apps.append({"appName": "Other Apps", "timeSpent": other_time})
        data_to_plot = top_apps

    labels = [
        f"{app.get('appName','N/A')} ({app.get('timeSpent',0)//60}m)"
        for app in data_to_plot
    ]
    values = [app.get("timeSpent", 0) for app in data_to_plot]

    if not values or sum(values) == 0:  # Handle case where all values are zero
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
    fig.update_layout(
        title_text="App Usage Breakdown", height=500, legend_title_text="Applications"
    )
    return fig


def create_browser_chart(app_data: List[Dict[str, Any]]) -> Optional[go.Figure]:
    if not app_data:
        return None
    browser_entries = []
    for app in app_data:
        name_lower = app.get("appName", "").lower()
        if any(
            b_keyword in name_lower
            for b_keyword in ["chrome", "msedge", "edge", "firefox"]
        ):
            browser_type = (
                "MS Edge"
                if "msedge" in name_lower or "edge" in name_lower
                else ("Google Chrome" if "chrome" in name_lower else "Firefox")
            )
            browser_entries.append(
                {**app, "browserType": browser_type}
            )  # Add type for coloring

    if not browser_entries:
        return None

    df_browsers = pd.DataFrame(browser_entries).sort_values(
        by=["browserType", "timeSpent"], ascending=[True, False]
    )
    df_browsers["timeSpentMinutesText"] = (df_browsers["timeSpent"] // 60).astype(
        str
    ) + "m"

    fig = px.bar(
        df_browsers,
        x="appName",
        y="timeSpent",
        text="timeSpentMinutesText",
        labels={
            "appName": "Browser Instance / Profile",
            "timeSpent": "Time Spent (seconds)",
        },
        title="Browser Usage Details",
        color="browserType",
        color_discrete_map={
            "MS Edge": "#0078D4",
            "Google Chrome": "#DB4437",
            "Firefox": "#FF7139",
        },
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        height=450,
        legend_title_text="Browser Type",
        uniformtext_minsize=8,
        uniformtext_mode="hide",
    )
    fig.update_traces(textposition="outside")
    return fig


def create_category_chart(
    buckets: List[Dict[str, Any]], categories: List[Dict[str, str]]
) -> Optional[go.Figure]:
    """Create a pie chart showing time distribution by category."""
    if not buckets or not categories:
        return None

    # Create a mapping of category ID to name
    cat_id_to_name = {
        cat.get("id", ""): cat.get("name", "Unknown") for cat in categories
    }

    # Count time per category
    category_times = {"Uncategorized": 0}
    for bucket in buckets:
        cat_id = bucket.get("category_id", "")
        cat_name = (
            cat_id_to_name.get(cat_id, "Uncategorized") if cat_id else "Uncategorized"
        )

        # Calculate duration
        start_time = pd.to_datetime(bucket.get("start", ""))
        end_time = pd.to_datetime(bucket.get("end", bucket.get("start", "")))
        if start_time and end_time:
            duration = (end_time - start_time).total_seconds()
            category_times[cat_name] = category_times.get(cat_name, 0) + duration

    # Remove categories with zero time
    category_times = {k: v for k, v in category_times.items() if v > 0}

    if not category_times:
        return None

    # Create data for the chart
    labels = [f"{cat} ({int(time//60)}m)" for cat, time in category_times.items()]
    values = list(category_times.values())

    fig = go.Figure()
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            textinfo="percent+label",
            textfont_size=12,
        )
    )
    fig.update_layout(title_text="Time Distribution by Category", height=450)
    return fig


# --- Activity Categories Manager Page ---
def display_category_manager():
    st.title("🏆 Activity Categories Manager")

    # Load existing categories
    categories = load_categories()

    # Display existing categories
    if categories:
        st.subheader("Current Categories")
        for i, category in enumerate(categories):
            with st.expander(f"{category.get('name', 'Unnamed Category')}"):
                st.write(f"**ID:** {category.get('id', '')}")
                st.write(f"**Description:** {category.get('description', '')}")
    else:
        st.info("No categories defined yet. Use the form below to create categories.")

    # Add new category form
    st.subheader("Add New Category")
    with st.form("add_category_form"):
        cat_name = st.text_input(
            "Category Name", placeholder="e.g., Coding, Meetings, Research"
        )
        cat_id = st.text_input(
            "Category ID (lowercase, no spaces)",
            value="" if not cat_name else cat_name.lower().replace(" ", "_"),
            placeholder="e.g., coding, meetings, research",
        )
        cat_desc = st.text_area(
            "Category Description",
            placeholder="Describe what activities fall under this category...",
            help="Provide details to help the LLM correctly categorize activities",
        )
        submitted = st.form_submit_button("Add Category")

        if submitted:
            if not cat_name or not cat_id or not cat_desc:
                st.warning("All fields are required")
            else:
                # Check for duplicate IDs
                if any(cat.get("id") == cat_id for cat in categories):
                    st.error(f"Category ID '{cat_id}' already exists")
                else:
                    categories.append(
                        {"id": cat_id, "name": cat_name, "description": cat_desc}
                    )
                    if save_categories(categories):
                        st.success(f"Category '{cat_name}' added")
                        time.sleep(1)
                        st.rerun()

    # Edit/delete categories
    if categories:
        st.subheader("Edit or Delete Categories")

        # Select category to edit/delete
        selected_cat_name = st.selectbox(
            "Select Category",
            options=[
                cat.get("name", f"Unnamed ({cat.get('id', 'unknown')})")
                for cat in categories
            ],
            key="edit_delete_category_select",
        )

        selected_cat_index = next(
            (
                i
                for i, cat in enumerate(categories)
                if cat.get("name") == selected_cat_name
            ),
            None,
        )

        if selected_cat_index is not None:
            selected_cat = categories[selected_cat_index]

            # Edit form
            with st.form("edit_category_form"):
                edit_name = st.text_input(
                    "Category Name", value=selected_cat.get("name", "")
                )
                edit_id = st.text_input("Category ID", value=selected_cat.get("id", ""))
                edit_desc = st.text_area(
                    "Category Description", value=selected_cat.get("description", "")
                )

                col1, col2 = st.columns(2)
                update_btn = col1.form_submit_button("Update Category")
                delete_btn = col2.form_submit_button("Delete Category")

                if update_btn:
                    if not edit_name or not edit_id or not edit_desc:
                        st.warning("All fields are required")
                    else:
                        # Check for duplicate IDs (excluding the current category)
                        if edit_id != selected_cat.get("id") and any(
                            cat.get("id") == edit_id for cat in categories
                        ):
                            st.error(f"Category ID '{edit_id}' already exists")
                        else:
                            categories[selected_cat_index] = {
                                "id": edit_id,
                                "name": edit_name,
                                "description": edit_desc,
                            }
                            if save_categories(categories):
                                st.success(f"Category '{edit_name}' updated")
                                time.sleep(1)
                                st.rerun()

                if delete_btn:
                    categories.pop(selected_cat_index)
                    if save_categories(categories):
                        st.success(f"Category '{selected_cat_name}' deleted")
                        time.sleep(1)
                        st.rerun()


def save_current_bucket():
    global current_bucket, buckets, bucket_start_time

    if current_bucket["titles"] or current_bucket["ocr_text"]:
        # Set the end time for the bucket
        current_bucket["end"] = datetime.now().isoformat()

        # Generate summary and category with simplified function
        summary_text, category_id = generate_summary_and_category(current_bucket)

        # Add summary and category to bucket
        if summary_text:
            current_bucket["summary"] = summary_text
        if category_id:
            current_bucket["category_id"] = category_id

        # Add to list of buckets
        buckets.append(current_bucket)

        # Save buckets to file
        save_buckets_to_file()

    # Reset for next bucket
    bucket_start_time = datetime.now()
    current_bucket = {
        "start": bucket_start_time.isoformat(),
        "titles": [],
        "ocr_text": [],
    }


# --- Time Bucket Summaries Page ---
def display_time_bucket_summaries():
    st.title("📝 5-Minute Summaries & Feedback")
    dates = load_available_dates()
    if not dates:
        st.error("No data found in focus_logs directory.")
        st.stop()
    selected_date = st.selectbox("Select a date", dates, key="summaries_date_select")

    official_buckets = load_time_buckets_for_date(selected_date)
    personal_notes_data = load_block_feedback()
    categories = load_categories()  # Load categories

    if not official_buckets:
        st.info(f"No 5-minute summary blocks found for {selected_date}.")
        return

    # Add tab for uncategorized entries
    show_uncategorized = st.checkbox("Show only uncategorized entries", value=False)

    # Create category filter if categories exist
    category_filter = "All Categories"
    if categories and not show_uncategorized:
        category_options = ["All Categories", "Uncategorized"] + [
            cat.get("name", "Unknown") for cat in categories
        ]
        category_filter = st.selectbox(
            "Filter by category", category_options, key="category_filter"
        )

    # Count buckets by category
    if categories:
        cat_counts = {"Uncategorized": 0}
        for cat in categories:
            cat_counts[cat.get("name", "Unknown")] = 0

        for bucket in official_buckets:
            cat_id = bucket.get("category_id", "")
            if cat_id:
                cat_name = next(
                    (
                        cat.get("name", "Unknown")
                        for cat in categories
                        if cat.get("id") == cat_id
                    ),
                    "Uncategorized",
                )
                cat_counts[cat_name] = cat_counts.get(cat_name, 0) + 1
            else:
                cat_counts["Uncategorized"] = cat_counts.get("Uncategorized", 0) + 1

        # Display category counts
        st.subheader("Summary Blocks by Category")
        cols = st.columns(len(cat_counts))
        for i, (cat_name, count) in enumerate(cat_counts.items()):
            cols[i].metric(cat_name, count)

        # Show warning if there are uncategorized entries
        if cat_counts["Uncategorized"] > 0:
            st.warning(
                f"You have {cat_counts['Uncategorized']} uncategorized entries. Use the checkbox above to focus on them."
            )

        st.markdown("---")

    for idx, bucket_data in enumerate(official_buckets):
        bucket_start_iso = bucket_data["start"]
        session_tag = bucket_data.get("session_tag", "unknown_session")

        start_dt = pd.to_datetime(bucket_start_iso)
        end_dt = pd.to_datetime(bucket_data.get("end", bucket_start_iso))
        time_range_display = f"{start_dt.strftime('%H:%M')} - {end_dt.strftime('%H:%M')} (UTC {start_dt.strftime('%Y-%m-%d')})"

        current_official_summary_text = bucket_data.get("summary", "")
        current_personal_note_text = personal_notes_data.get(bucket_start_iso, "")
        current_category_id = bucket_data.get("category_id", "")

        # Find current category name from ID
        current_category_name = "Uncategorized"
        if current_category_id:
            cat_match = next(
                (
                    cat.get("name", "Unknown")
                    for cat in categories
                    if cat.get("id") == current_category_id
                ),
                "Uncategorized",
            )
            current_category_name = cat_match

        # Apply category filter and uncategorized filter
        if show_uncategorized:
            if current_category_id:  # Skip categorized entries
                continue
        elif category_filter != "All Categories":
            if (
                category_filter == "Uncategorized"
                and current_category_name != "Uncategorized"
            ):
                continue
            elif (
                category_filter != "Uncategorized"
                and current_category_name != category_filter
            ):
                continue

        # Add category to expander title
        expander_title = f"{time_range_display}"
        if categories:
            expander_title += f" - [{current_category_name}]"
            if not current_category_id:
                expander_title += " ⚠️"  # Add warning icon for uncategorized

        with st.expander(expander_title):
            # Display category selection if categories exist
            if categories:
                st.markdown("**Current Category:**")

                # Show suggestion UI if uncategorized
                suggested_category_name = ""
                suggested_category_desc = ""

                if not current_category_id:
                    raw_titles = bucket_data.get("titles", [])
                    raw_ocr = bucket_data.get("ocr_text", [])

                    # Button to get suggestion
                    if st.button(
                        "Get Category Suggestion",
                        key=f"suggest_cat_btn_{idx}_{bucket_start_iso}",
                    ):
                        with st.spinner("Asking LLM for category suggestion..."):
                            _, _, suggestion = generate_summary_from_raw_with_llm(
                                raw_titles, raw_ocr
                            )
                            if suggestion:
                                try:
                                    suggestion_parts = suggestion.split("|", 1)
                                    suggested_category_name = suggestion_parts[
                                        0
                                    ].strip()
                                    suggested_category_desc = (
                                        suggestion_parts[1].strip()
                                        if len(suggestion_parts) > 1
                                        else ""
                                    )
                                    st.session_state[
                                        f"suggested_cat_name_{idx}_{bucket_start_iso}"
                                    ] = suggested_category_name
                                    st.session_state[
                                        f"suggested_cat_desc_{idx}_{bucket_start_iso}"
                                    ] = suggested_category_desc
                                except:
                                    st.error("Couldn't parse the suggestion properly.")

                # Display suggestion if available
                suggestion_key_name = f"suggested_cat_name_{idx}_{bucket_start_iso}"
                suggestion_key_desc = f"suggested_cat_desc_{idx}_{bucket_start_iso}"

                if (
                    suggestion_key_name in st.session_state
                    and st.session_state[suggestion_key_name]
                ):
                    suggested_category_name = st.session_state[suggestion_key_name]
                    suggested_category_desc = st.session_state.get(
                        suggestion_key_desc, ""
                    )

                    st.info(
                        f"**Suggested New Category:** {suggested_category_name}\n\n{suggested_category_desc}"
                    )

                    # Add buttons to create this category and apply it
                    col1, col2 = st.columns(2)

                    if col1.button(
                        "Create & Apply This Category",
                        key=f"create_apply_cat_btn_{idx}_{bucket_start_iso}",
                    ):
                        # Generate a valid ID from the name
                        suggested_id = suggested_category_name.lower().replace(" ", "_")

                        # Check if this ID already exists
                        existing_ids = [cat.get("id", "") for cat in categories]
                        if suggested_id in existing_ids:
                            st.error(
                                f"A category with ID '{suggested_id}' already exists. Please edit it manually."
                            )
                        else:
                            # Create the new category
                            new_category = {
                                "id": suggested_id,
                                "name": suggested_category_name,
                                "description": suggested_category_desc,
                            }

                            # Add it to categories
                            categories.append(new_category)
                            if save_categories(categories):
                                # Apply the new category to this bucket
                                if update_bucket_category_in_file(
                                    session_tag, bucket_start_iso, suggested_id
                                ):
                                    st.success(
                                        f"Created new category '{suggested_category_name}' and applied it!"
                                    )
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error(
                                        "Failed to apply the new category to this bucket."
                                    )
                            else:
                                st.error("Failed to save the new category.")

                    if col2.button(
                        "Discard Suggestion",
                        key=f"discard_suggestion_btn_{idx}_{bucket_start_iso}",
                    ):
                        # Clear suggestion from session state
                        if suggestion_key_name in st.session_state:
                            del st.session_state[suggestion_key_name]
                        if suggestion_key_desc in st.session_state:
                            del st.session_state[suggestion_key_desc]
                        st.success("Suggestion discarded.")
                        time.sleep(0.5)
                        st.rerun()

                # Standard category selection
                category_options = ["Uncategorized"] + [
                    cat.get("name", "Unknown") for cat in categories
                ]
                category_ids = [""] + [cat.get("id", "") for cat in categories]

                # Find index of current category in the options
                selected_cat_index = 0  # Default to "Uncategorized"
                if current_category_id:
                    try:
                        selected_cat_index = category_ids.index(current_category_id)
                    except ValueError:
                        pass  # Keep default if not found

                selected_category = st.selectbox(
                    "Select category for this activity block",
                    options=category_options,
                    index=selected_cat_index,
                    key=f"category_select_{idx}_{bucket_start_iso}",
                )

                # Get the ID for the selected category name
                selected_cat_id = ""
                if selected_category != "Uncategorized":
                    selected_cat_index = category_options.index(selected_category)
                    if selected_cat_index > 0:  # Skip "Uncategorized"
                        selected_cat_id = category_ids[
                            selected_cat_index - 1
                        ]  # Adjust index for the offset

                # Update category if changed
                if selected_cat_id != current_category_id:
                    if st.button(
                        "Update Category",
                        key=f"update_cat_btn_{idx}_{bucket_start_iso}",
                    ):
                        if update_bucket_category_in_file(
                            session_tag, bucket_start_iso, selected_cat_id
                        ):
                            st.success(f"Category updated to '{selected_category}'")
                            time.sleep(1)
                            st.rerun()

            # Show the rest of the bucket details (summary, notes, actions)
            st.markdown("**Current Official Summary (from log file):**")
            st.caption(
                current_official_summary_text
                or "_No official summary available for this block._"
            )

            st.markdown("**Your Input Text:**")
            user_input_text = st.text_area(
                "Use this text area to: (1) Save a personal note, (2) Provide feedback for the LLM to refine the summary, or (3) Write your own summary to replace the current one.",
                value=current_personal_note_text,
                key=f"user_text_area_{idx}_{bucket_start_iso}",
                height=100,
            )

            # --- Personal Note Actions ---
            st.markdown("---")
            st.write("**Manage Personal Note (Saved Separately):**")
            note_cols = st.columns(2)
            if note_cols[0].button(
                "Save Input as Personal Note",
                key=f"save_personal_note_{idx}_{bucket_start_iso}",
                help="Saves the text in the 'Your Input Text' area above as a private note for this block. This does NOT change the Official Summary.",
            ):
                personal_notes_data[bucket_start_iso] = user_input_text.strip()
                if save_block_feedback(personal_notes_data):
                    st.success("Personal note saved!")
                else:
                    st.error("Failed to save personal note.")

            if note_cols[1].button(
                "Delete Personal Note",
                key=f"delete_personal_note_{idx}_{bucket_start_iso}",
                help="Deletes the private note associated with this block.",
            ):
                if bucket_start_iso in personal_notes_data:
                    del personal_notes_data[bucket_start_iso]
                    if save_block_feedback(personal_notes_data):
                        st.success("Personal note deleted!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("Failed to delete personal note.")
                else:
                    st.info("No personal note to delete for this block.")

            # --- Official Summary Actions (Modifies time_buckets_*.json) ---
            st.markdown("---")
            st.write("**Manage Official Summary (Modifies Log File):**")
            official_summary_action_cols = st.columns(3)

            with official_summary_action_cols[0]:
                if st.button(
                    "Set as Official Summary",
                    key=f"set_official_summary_from_input_{idx}_{bucket_start_iso}",
                    help="Replaces the 'Current Official Summary' above with the text from the 'Your Input Text' area. This directly modifies the summary in the log file.",
                ):
                    if user_input_text.strip():
                        if update_bucket_summary_in_file(
                            session_tag, bucket_start_iso, user_input_text.strip()
                        ):
                            st.success("Official summary updated with your input!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(
                                "Failed to update official summary with your input."
                            )
                    else:
                        st.warning(
                            "Text area is empty. Please enter text to set as the official summary."
                        )

            with official_summary_action_cols[1]:
                if st.button(
                    "Refine LLM Summary",
                    key=f"refine_llm_summary_btn_{idx}_{bucket_start_iso}",
                    help="Uses the 'Current Official Summary' AND the text from 'Your Input Text' (as feedback) to ask the LLM to generate an improved summary. This new summary will replace the current one in the log file.",
                ):
                    if user_input_text.strip():
                        (
                            refined_summary_text,
                            refined_category_id,
                            suggested_category,
                        ) = refine_summary_with_llm(
                            current_official_summary_text,
                            user_input_text.strip(),
                            current_category_id,
                        )
                        if refined_summary_text is not None:
                            # Update summary
                            if update_bucket_summary_in_file(
                                session_tag, bucket_start_iso, refined_summary_text
                            ):
                                # Update category if it changed
                                category_updated = False
                                if refined_category_id != current_category_id:
                                    category_updated = update_bucket_category_in_file(
                                        session_tag,
                                        bucket_start_iso,
                                        refined_category_id,
                                    )

                                # Store suggestion if provided
                                if suggested_category:
                                    try:
                                        suggestion_parts = suggested_category.split(
                                            "|", 1
                                        )
                                        suggested_category_name = suggestion_parts[
                                            0
                                        ].strip()
                                        suggested_category_desc = (
                                            suggestion_parts[1].strip()
                                            if len(suggestion_parts) > 1
                                            else ""
                                        )
                                        st.session_state[
                                            f"suggested_cat_name_{idx}_{bucket_start_iso}"
                                        ] = suggested_category_name
                                        st.session_state[
                                            f"suggested_cat_desc_{idx}_{bucket_start_iso}"
                                        ] = suggested_category_desc
                                    except:
                                        pass  # Ignore parsing errors for suggestions

                                success_msg = "LLM summary refined and updated!"
                                if category_updated:
                                    success_msg += " Category was also updated."
                                if suggested_category:
                                    success_msg += " A new category was suggested."

                                st.success(success_msg)
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Failed to save LLM-refined official summary.")
                    else:
                        st.warning(
                            "Please enter some feedback in the 'Your Input Text' area to help refine the summary."
                        )

            with official_summary_action_cols[2]:
                if st.button(
                    "Re-Generate Original LLM Summary",
                    key=f"regenerate_original_llm_summary_btn_{idx}_{bucket_start_iso}",
                    help="Asks the LLM to create a brand new summary based on the raw window titles and OCR text recorded for this block. This will replace the 'Current Official Summary' in the log file.",
                ):
                    raw_titles_list = bucket_data.get("titles", [])
                    raw_ocr_list = bucket_data.get("ocr_text", [])
                    (
                        regenerated_summary_text,
                        regenerated_category_id,
                        suggested_category,
                    ) = generate_summary_from_raw_with_llm(
                        raw_titles_list, raw_ocr_list
                    )
                    if regenerated_summary_text is not None:
                        # Update summary
                        if update_bucket_summary_in_file(
                            session_tag, bucket_start_iso, regenerated_summary_text
                        ):
                            # Update category
                            category_updated = False
                            if regenerated_category_id:
                                category_updated = update_bucket_category_in_file(
                                    session_tag,
                                    bucket_start_iso,
                                    regenerated_category_id,
                                )

                            # Store suggestion if provided
                            if suggested_category:
                                try:
                                    suggestion_parts = suggested_category.split("|", 1)
                                    suggested_category_name = suggestion_parts[
                                        0
                                    ].strip()
                                    suggested_category_desc = (
                                        suggestion_parts[1].strip()
                                        if len(suggestion_parts) > 1
                                        else ""
                                    )
                                    st.session_state[
                                        f"suggested_cat_name_{idx}_{bucket_start_iso}"
                                    ] = suggested_category_name
                                    st.session_state[
                                        f"suggested_cat_desc_{idx}_{bucket_start_iso}"
                                    ] = suggested_category_desc
                                except:
                                    pass  # Ignore parsing errors for suggestions

                            success_msg = (
                                "Original LLM summary re-generated and updated!"
                            )
                            if category_updated:
                                success_msg += " Category was also updated."
                            if suggested_category:
                                success_msg += " A new category was suggested."

                            st.success(success_msg)
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(
                                "Failed to save LLM re-generated official summary."
                            )

    # Display notice if all items are filtered out
    if show_uncategorized and all(
        bucket.get("category_id", "") for bucket in official_buckets
    ):
        st.info("All entries for this date have been categorized! 🎉")


# --- Label Editor Page ---
def display_label_editor():
    st.title("🏷 Activity Label Editor")
    available_dates = load_available_dates()
    if not available_dates:
        st.error("No data found in focus_logs directory.")
        st.stop()
    selected_date = st.selectbox(
        "Select a date for labeling activities",
        available_dates,
        key="label_editor_date_select",
    )

    logs_df_for_labeling = load_log_entries(selected_date)
    if logs_df_for_labeling.empty:
        st.warning(f"No log entries found for {selected_date} to label.")
        st.stop()

    # title_groups will contain 'exe' as full path
    title_groups_for_labeling = group_activities_for_labeling(logs_df_for_labeling)

    # For forms needing basenames (App Label, Pattern Label)
    all_exe_basenames_in_logs = sorted(
        list(
            set(
                os.path.basename(str(exe))
                for exe in logs_df_for_labeling["exe"].dropna().unique()
                if "exe" in logs_df_for_labeling
            )
        )
    )

    current_labels = load_labels()
    for k_label_type in ["exact_titles", "exact_exe", "patterns"]:  # Ensure keys exist
        current_labels.setdefault(k_label_type, {})

    st.subheader("Current Custom Labels")
    if not any(
        current_labels.get(k) for k in ["exact_titles", "exact_exe", "patterns"]
    ):
        st.info("No custom labels defined yet. Use the forms below to create labels.")
    else:
        ltab1, ltab2, ltab3 = st.tabs(
            [
                "Window Title Labels",
                "Application Labels (Basename)",
                "Pattern Labels (Basename)",
            ]
        )
        with ltab1:  # Exact Titles (Key: exact::FULL_EXE_PATH::TITLE)
            data = [
                {
                    "Key": k,
                    "Label": v,
                    "App": k.split("::")[1],
                    "Title": k.split("::")[2][:70] + "...",
                }
                for k, v in current_labels.get("exact_titles", {}).items()
                if k.count("::") == 2
            ]
            if data:
                st.dataframe(
                    pd.DataFrame(data)[["App", "Title", "Label"]],
                    use_container_width=True,
                )
            else:
                st.info("No window title labels defined.")
        with ltab2:  # Exact Exe (Key: BASENAME.exe)
            data = [
                {"App Basename": k, "Label": v}
                for k, v in current_labels.get("exact_exe", {}).items()
            ]
            if data:
                st.dataframe(pd.DataFrame(data), use_container_width=True)
            else:
                st.info("No application basename labels defined.")
        with ltab3:  # Patterns (Key: pattern::BASENAME.exe::PATTERN_STRING)
            data = [
                {
                    "App Basename": k.split("::")[1],
                    "Title Pattern": k.split("::")[2],
                    "Label": v,
                }
                for k, v in current_labels.get("patterns", {}).items()
                if k.count("::") == 2
            ]
            if data:
                st.dataframe(pd.DataFrame(data), use_container_width=True)
            else:
                st.info("No pattern labels defined.")

    if not title_groups_for_labeling.empty:
        st.subheader("Available Window Titles for Labeling (from Grouped Activities)")
        st.dataframe(
            title_groups_for_labeling[["title", "app_name", "exe", "duration"]].head(
                20
            ),
            use_container_width=True,
            height=300,
        )  # Show top 20
    else:
        st.info(
            "No grouped window title activities for this date. You can still create Application or Pattern labels based on raw log data."
        )

    form_tab_title, form_tab_app, form_tab_pattern = st.tabs(
        [
            "Create: Label Specific Window Title",
            "Create: Label Application (by Basename)",
            "Create: Label by Pattern (App Basename + Title Pattern)",
        ]
    )

    with (
        form_tab_title
    ):  # Label by specific title (uses full EXE path from title_groups)
        st.markdown(
            "Labels specific `Window Title` + `Full Executable Path` from grouped activities above."
        )
        if title_groups_for_labeling.empty:
            st.warning("No grouped activities available to select specific titles.")
        else:
            with st.form("form_specific_title_label"):
                # Let user pick from title_groups
                titles_available = title_groups_for_labeling["title"].tolist()
                selected_titles = st.multiselect(
                    "Select Window Title(s) from grouped activities",
                    options=titles_available,
                    key="sel_spec_titles",
                )
                new_label_for_titles = st.text_input(
                    "Enter Label for selected title(s)", key="label_spec_titles"
                )
                submitted_spec_title = st.form_submit_button("Save Title Label(s)")
                if (
                    submitted_spec_title
                    and selected_titles
                    and new_label_for_titles.strip()
                ):
                    for title_to_label in selected_titles:
                        # Find the full exe path associated with this title from title_groups
                        exe_for_title = title_groups_for_labeling[
                            title_groups_for_labeling["title"] == title_to_label
                        ]["exe"].iloc[0]
                        label_key = f"exact::{exe_for_title}::{title_to_label}"
                        current_labels["exact_titles"][
                            label_key
                        ] = new_label_for_titles.strip()
                    if save_labels(current_labels):
                        st.success("Specific title label(s) saved!")
                        time.sleep(1)
                        st.rerun()
                elif submitted_spec_title:
                    st.warning("Please select title(s) and enter a label.")

    with form_tab_app:  # Label by app basename
        st.markdown(
            "Labels an entire application based on its `Executable Basename` (e.g., `chrome.exe`)."
        )
        if not all_exe_basenames_in_logs:
            st.warning("No executable basenames found in logs for this date.")
        else:
            with st.form("form_app_basename_label"):
                selected_app_basename = st.selectbox(
                    "Select Application Basename",
                    options=all_exe_basenames_in_logs,
                    key="sel_app_basename",
                )
                new_label_for_app = st.text_input(
                    f"Enter Label for '{selected_app_basename}'",
                    value=current_labels["exact_exe"].get(selected_app_basename, ""),
                    key="label_app_basename",
                )
                submitted_app_label = st.form_submit_button("Save Application Label")
                if submitted_app_label:
                    if new_label_for_app.strip():
                        current_labels["exact_exe"][
                            selected_app_basename
                        ] = new_label_for_app.strip()
                    elif (
                        selected_app_basename in current_labels["exact_exe"]
                    ):  # If label cleared, remove it
                        del current_labels["exact_exe"][selected_app_basename]
                    if save_labels(current_labels):
                        st.success("Application label updated!")
                        time.sleep(1)
                        st.rerun()

    with form_tab_pattern:  # Label by pattern (app basename + title pattern)
        st.markdown(
            "Labels activities matching an `App Basename` AND a `Window Title Pattern` (case-insensitive)."
        )
        if not all_exe_basenames_in_logs:
            st.warning("No executable basenames found for pattern matching.")
        else:
            with st.form("form_pattern_label"):
                selected_app_basename_patt = st.selectbox(
                    "Select Application Basename for Pattern",
                    options=all_exe_basenames_in_logs,
                    key="sel_app_basename_patt",
                )
                title_pattern_for_label = st.text_input(
                    "Enter Window Title Pattern (e.g., 'youtube', 'project x')",
                    key="patt_title_input",
                )
                new_label_for_pattern = st.text_input(
                    "Enter Label for this pattern", key="label_patt_input"
                )
                submitted_pattern_label = st.form_submit_button("Save Pattern Label")
                if (
                    submitted_pattern_label
                    and selected_app_basename_patt
                    and title_pattern_for_label.strip()
                    and new_label_for_pattern.strip()
                ):
                    label_key = f"pattern::{selected_app_basename_patt}::{title_pattern_for_label.strip()}"
                    current_labels["patterns"][
                        label_key
                    ] = new_label_for_pattern.strip()
                    if save_labels(current_labels):
                        st.success("Pattern label saved!")
                        time.sleep(1)
                        st.rerun()
                elif submitted_pattern_label:
                    st.warning(
                        "Please select an app, enter a title pattern, and a label."
                    )

    st.subheader("Delete Labels")
    del_ltab1, del_ltab2, del_ltab3 = st.tabs(
        ["Delete Window Title Label", "Delete App Label", "Delete Pattern Label"]
    )
    with del_ltab1:
        keys_to_del = list(current_labels.get("exact_titles", {}).keys())
        if keys_to_del:
            sel_key = st.selectbox(
                "Select Specific Title Label to Delete",
                keys_to_del,
                format_func=lambda k: f"{k.split('::')[1].split(os.sep)[-1]} - '{k.split('::')[2][:30]}...' -> {current_labels['exact_titles'][k]}",
                key="del_sel_exact_title",
            )
            if st.button("Delete Selected Title Label", key="del_btn_exact_title"):
                if sel_key in current_labels["exact_titles"]:
                    del current_labels["exact_titles"][sel_key]
                if save_labels(current_labels):
                    st.success("Label deleted.")
                    time.sleep(1)
                    st.rerun()
        else:
            st.info("No specific title labels to delete.")
    # ... (similar deletion UI for exact_exe and patterns) ...

    st.subheader("Preview With Labels Applied")
    if not logs_df_for_labeling.empty:
        labeled_logs_preview = apply_labels_to_logs(
            logs_df_for_labeling.copy()
        )  # Important: use a copy
        if (
            not labeled_logs_preview.empty
            and "app_name" in labeled_logs_preview.columns
        ):
            preview_agg = (
                labeled_logs_preview.groupby("app_name")
                .agg(
                    total_duration_sec=("duration", "sum"),
                    original_names=(
                        "original_app_name",
                        lambda x: list(set(x))[:3],
                    ),  # Show a few original names
                    log_entries=(
                        ("timestamp", "count")
                        if "timestamp" in labeled_logs_preview
                        else ("duration", "count")
                    ),
                )
                .reset_index()
                .sort_values("total_duration_sec", ascending=False)
            )
            preview_agg["Total Duration (min)"] = (
                preview_agg["total_duration_sec"] / 60
            ).round(1)
            st.dataframe(
                preview_agg[
                    [
                        "app_name",
                        "Total Duration (min)",
                        "log_entries",
                        "original_names",
                    ]
                ],
                use_container_width=True,
            )
            if not preview_agg.empty and preview_agg["total_duration_sec"].sum() > 0:
                fig_preview = px.pie(
                    preview_agg,
                    values="total_duration_sec",
                    names="app_name",
                    title="Labeled Activity Preview",
                    hole=0.3,
                )
                st.plotly_chart(fig_preview, use_container_width=True)

    st.subheader("📊 Overall Label Summary for Selected Date")
    if not logs_df_for_labeling.empty:
        labeled_logs_summary_data = apply_labels_to_logs(logs_df_for_labeling.copy())
        if (
            not labeled_logs_summary_data.empty
            and "app_name" in labeled_logs_summary_data.columns
        ):
            # Group by the final 'app_name' which is the label itself or "Unlabeled" (original_app_name if no label matched)
            # For "Unlabeled", we need to sum where app_name == original_app_name

            summary_list = []
            # Labeled items
            labeled_items = labeled_logs_summary_data[
                labeled_logs_summary_data["app_name"]
                != labeled_logs_summary_data["original_app_name"]
            ]
            if not labeled_items.empty:
                labeled_agg = (
                    labeled_items.groupby("app_name")["duration"].sum().reset_index()
                )
                for _, row in labeled_agg.iterrows():
                    summary_list.append(
                        {
                            "Label Category": row["app_name"],
                            "Total Time (sec)": row["duration"],
                        }
                    )

            # Unlabeled items
            unlabeled_items = labeled_logs_summary_data[
                labeled_logs_summary_data["app_name"]
                == labeled_logs_summary_data["original_app_name"]
            ]
            if not unlabeled_items.empty:
                # Could further group unlabeled by original_app_name if desired, or just sum all as "Unlabeled"
                total_unlabeled_time = unlabeled_items["duration"].sum()
                if total_unlabeled_time > 0:
                    summary_list.append(
                        {
                            "Label Category": "Uncategorized (Original Names)",
                            "Total Time (sec)": total_unlabeled_time,
                        }
                    )

            if summary_list:
                df_label_summary = pd.DataFrame(summary_list).sort_values(
                    "Total Time (sec)", ascending=False
                )
                df_label_summary["Total Time (min)"] = (
                    df_label_summary["Total Time (sec)"] / 60
                ).round(1)
                df_label_summary["Percentage"] = (
                    df_label_summary["Total Time (sec)"]
                    / df_label_summary["Total Time (sec)"].sum()
                    * 100
                ).round(1)
                st.dataframe(
                    df_label_summary[
                        ["Label Category", "Total Time (min)", "Percentage"]
                    ],
                    use_container_width=True,
                )
                if df_label_summary["Total Time (sec)"].sum() > 0:
                    fig_summary = px.pie(
                        df_label_summary,
                        values="Total Time (sec)",
                        names="Label Category",
                        title="Time Distribution by Label Category",
                        hole=0.4,
                    )
                    st.plotly_chart(fig_summary, use_container_width=True)
            else:
                st.info(
                    "No data to summarize by label for this date after applying labels."
                )


# --- Dashboard Page ---
def display_dashboard():
    st.title("📊 Focus Monitor Dashboard")
    available_dates = load_available_dates()
    if not available_dates:
        st.error("No data found in focus_logs directory. Is the monitor running?")
        st.stop()
    selected_date = st.selectbox(
        "Select a date to view", available_dates, key="dashboard_date_select"
    )

    daily_summary_data = load_daily_summary(
        selected_date
    )  # This now applies labels via generate_summary_from_logs
    if not daily_summary_data:
        st.warning(f"Could not load or generate a summary for {selected_date}.")
        st.stop()

    # --- Summary Metrics ---
    d_col1, d_col2, d_col3 = st.columns(3)
    d_col1.metric(
        "🕒 Total Tracked Time", f"{daily_summary_data.get('totalTime', 0) // 60} min"
    )

    # --- NEW: Category Analysis ---
    categories = load_categories()
    all_buckets = load_time_buckets_for_date(selected_date)

    if categories and all_buckets:
        st.subheader("🏆 Activity by Category")

        # Create category chart
        fig_category = create_category_chart(all_buckets, categories)
        if fig_category:
            st.plotly_chart(fig_category, use_container_width=True)

            # Create a detailed breakdown table
            cat_id_to_name = {
                cat.get("id", ""): cat.get("name", "Unknown") for cat in categories
            }
            category_data = []

            for bucket in all_buckets:
                cat_id = bucket.get("category_id", "")
                cat_name = (
                    cat_id_to_name.get(cat_id, "Uncategorized")
                    if cat_id
                    else "Uncategorized"
                )

                start_time = pd.to_datetime(bucket.get("start", ""))
                end_time = pd.to_datetime(bucket.get("end", bucket.get("start", "")))

                if start_time and end_time:
                    duration_seconds = (end_time - start_time).total_seconds()
                    category_data.append(
                        {
                            "Category": cat_name,
                            "Start Time": start_time.strftime("%H:%M"),
                            "End Time": end_time.strftime("%H:%M"),
                            "Duration (min)": round(duration_seconds / 60, 1),
                            "Summary": bucket.get("summary", "")[:50]
                            + (
                                "..."
                                if bucket.get("summary", "")
                                and len(bucket.get("summary", "")) > 50
                                else ""
                            ),
                        }
                    )

            if category_data:
                df_categories = pd.DataFrame(category_data)
                # Group by category and calculate total time
                category_totals = (
                    df_categories.groupby("Category")["Duration (min)"]
                    .sum()
                    .reset_index()
                )
                category_totals["Percentage"] = (
                    category_totals["Duration (min)"]
                    / category_totals["Duration (min)"].sum()
                    * 100
                ).round(1)
                category_totals = category_totals.sort_values(
                    "Duration (min)", ascending=False
                )

                st.subheader("Category Time Distribution")
                st.dataframe(category_totals, use_container_width=True)

                with st.expander("View Detailed Category Timeline"):
                    st.dataframe(
                        df_categories.sort_values("Start Time"),
                        use_container_width=True,
                    )
        else:
            st.info("No category data available for visualization.")
    elif categories:
        st.info("No time bucket data available for category analysis.")
    else:
        st.info(
            "No categories defined. Go to the 'Activity Categories Manager' tab to create categories."
        )

    st.subheader("🧠 Time Distribution by Application/Activity (Post-Labeling)")
    app_breakdown_data = daily_summary_data.get("appBreakdown", [])
    if app_breakdown_data:
        fig_pie_main = create_pie_chart(app_breakdown_data)
        st.plotly_chart(fig_pie_main, use_container_width=True)

        df_app_top = pd.DataFrame(
            [
                {
                    "Activity/Application": app.get("appName", "N/A"),
                    "Time (min)": app.get("timeSpent", 0) // 60,
                    "Percentage": f"{app.get('percentage', 0):.1f}%",
                }
                for app in app_breakdown_data[:10]
            ]
        )  # Show top 10 labeled activities
        st.dataframe(df_app_top, use_container_width=True)
    else:
        st.info("No application usage data available for this date in the summary.")

    st.subheader("🌐 Browser Usage Analysis (Labeled)")
    fig_browser_main = create_browser_chart(app_breakdown_data)  # Uses labeled appName
    if fig_browser_main:
        st.plotly_chart(fig_browser_main, use_container_width=True)
    else:
        st.info("No browser usage detected or included in the summary for this date.")

    with st.expander("📄 View Raw Focus Log Entries (Unlabeled Original Data)"):
        raw_log_df = load_log_entries(selected_date)  # Load raw, pre-labeling
        if not raw_log_df.empty:
            # Create a copy to avoid modifying the original DataFrame from cache
            df_to_display = raw_log_df.copy()

            # Define the desired final column names and their original sources
            # Ensure 'timestamp' is the primary source for the 'Time' column
            final_columns_map = {
                "Time": "timestamp",  # This will be formatted
                "Original App Name": "app_name",
                "Original Window Title": "title",
                "Duration (s)": "duration",
            }

            # Columns that must exist in raw_log_df for this section to work
            required_raw_cols = ["timestamp", "app_name", "title", "duration"]

            if all(col in df_to_display.columns for col in required_raw_cols):
                # Format the 'timestamp' to a readable 'Time' string IN A NEW COLUMN
                df_to_display["Formatted Time"] = pd.to_datetime(
                    df_to_display["timestamp"], errors="coerce"
                ).dt.strftime("%Y-%m-%d %H:%M:%S")

                # Select and rename:
                # Create a new DataFrame with only the columns we want, renaming them.
                # This avoids duplicate column issues.
                display_data = {}
                if (
                    "Formatted Time" in df_to_display.columns
                ):  # Check if conversion was successful
                    display_data["Time"] = df_to_display["Formatted Time"]
                if "app_name" in df_to_display.columns:
                    display_data["Original App Name"] = df_to_display["app_name"]
                if "title" in df_to_display.columns:
                    display_data["Original Window Title"] = df_to_display["title"]
                if "duration" in df_to_display.columns:
                    display_data["Duration (s)"] = df_to_display["duration"]

                final_display_df = pd.DataFrame(display_data)

                # Sort by the 'Time' column (which is now uniquely named and formatted)
                if "Time" in final_display_df.columns:
                    final_display_df = final_display_df.sort_values(
                        "Time", ascending=False
                    )

                st.dataframe(final_display_df, use_container_width=True, height=400)
            else:
                missing_cols = [
                    col for col in required_raw_cols if col not in df_to_display.columns
                ]
                st.info(
                    f"Essential columns missing in raw log for display: {', '.join(missing_cols)}"
                )
        else:
            st.info("No raw log entries available for this date.")

    # Simplified feedback view on dashboard - directs to Summaries tab
    st.subheader("📝 Quick View: Recent Personal Notes for Blocks")
    dashboard_personal_notes = load_block_feedback()
    notes_for_selected_date = {
        k: v for k, v in dashboard_personal_notes.items() if k.startswith(selected_date)
    }
    if notes_for_selected_date:
        st.caption(
            "Last 3 personal notes for this date (full management on '📝 Summaries' tab):"
        )
        for ts, note_text in list(notes_for_selected_date.items())[
            -3:
        ]:  # Show most recent 3
            st.caption(
                f"_{pd.to_datetime(ts).strftime('%H:%M')}_: {note_text[:70]}{'...' if len(note_text) > 70 else ''}"
            )
    else:
        st.info(
            "No personal notes recorded for this date yet. Use the '📝 Summaries' tab to add notes and manage official summaries."
        )

    st.markdown("---")
    st.markdown(
        """
    ### About Focus Monitor
    This dashboard displays data collected by the Focus Monitor agent. The agent tracks your active windows and applications to help you understand your computer usage patterns.
    - **Labels**: Use the '🏷 Activity Label Editor' tab to categorize your time.
    - **Summaries**: Use the '📝 Summaries' tab to review, note, and refine 5-minute block summaries.
    - **Categories**: Use the '🏆 Activity Categories Manager' tab to define high-level activity categories.
    
    To generate data:
    1. Ensure the Focus Monitor agent (`standalone_focus_monitor.py`) is running in the background.
    2. Use your computer normally. Data is logged to the `focus_logs` directory.
    3. Refresh this dashboard to view updated statistics and summaries.
    """
    )


# --- Main App ---
def main():
    st.set_page_config(
        page_title="Focus Monitor Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Add a dedicated test page
    (
        tab_dashboard,
        tab_label_editor,
        tab_summaries,
        tab_categories,
        tab_processor,
        tab_test,
    ) = st.tabs(
        [
            "📊 Dashboard",
            "🏷 Activity Label Editor",
            "📝 5-Min Summaries & Feedback",
            "🏆 Activity Categories Manager",
            "⏮️ Historical Processor",
            "🧪 LLM Test",
        ]
    )

    with tab_dashboard:
        display_dashboard()
    with tab_label_editor:
        display_label_editor()
    with tab_summaries:
        display_time_bucket_summaries()
    with tab_categories:
        display_category_manager()
    with tab_processor:
        display_retroactive_processor()
    with tab_test:
        llm_test_page()  # Add the new test page


if __name__ == "__main__":
    main()
