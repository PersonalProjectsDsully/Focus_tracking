import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

LOGS_DIR = Path(__file__).resolve().parent.parent / "focus_logs"
LABELS_FILE = LOGS_DIR / "activity_labels.json"
FEEDBACK_FILE = LOGS_DIR / "block_feedback.json"
CATEGORIES_FILE = LOGS_DIR / "activity_categories.json"


def load_available_dates() -> List[str]:
    """Return sorted list of dates with log or summary files."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_dates = {f.name.split("_")[-1].replace(".jsonl", "") for f in LOGS_DIR.glob("focus_log_*.jsonl")}
    summary_dates = {f.name.split("_")[-1].replace(".json", "") for f in LOGS_DIR.glob("daily_summary_*.json")}
    return sorted(list(log_dates.union(summary_dates)), reverse=True)


def load_log_entries(date_str: str) -> pd.DataFrame:
    file_path = LOGS_DIR / f"focus_log_{date_str}.jsonl"
    if not file_path.exists():
        return pd.DataFrame()
    data: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
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
    st.warning(f"No log data available to generate summary for {date_str}")
    return None

def generate_summary_from_logs(date_str: str) -> Optional[Dict[str, Any]]:
    logs_df = load_log_entries(date_str)
    if logs_df.empty:
        return None

    logs_df = apply_labels_to_logs(logs_df)
    total_duration = logs_df.get("duration", pd.Series(dtype="float")).sum()
    summary: Dict[str, Any] = {"date": date_str, "totalTime": total_duration, "appBreakdown": []}
    if total_duration == 0:
        return summary

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
        percentage = (row["time_spent"] / total_duration * 100) if total_duration > 0 else 0
        summary["appBreakdown"].append(
            {
                "appName": row["app_name"],
                "exePath": row.get("exe_path", "unknown.exe"),
                "timeSpent": int(row["time_spent"]),
                "percentage": round(percentage, 2),
                "windowTitles": (
                    row.get("window_titles", []) if isinstance(row.get("window_titles"), list) else [str(row.get("window_titles"))]
                ),
            }
        )

    summary["appBreakdown"].sort(key=lambda x: x["timeSpent"], reverse=True)
    return summary


# Label management

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
    if not labels:
        df_copy = logs_df.copy()
        df_copy["original_app_name"] = df_copy.get("app_name", "Unknown Application")
        return df_copy

    df = logs_df.copy()
    for col, default_val in [("app_name", "Unknown Application"), ("exe", "unknown.exe"), ("title", "")]:
        if col not in df.columns:
            df[col] = default_val
        df[col] = df[col].fillna(default_val).astype(str)

    df["original_app_name"] = df["app_name"]
    df["labeled_app_name"] = df["app_name"]

    for key, label_val in labels.get("exact_titles", {}).items():
        try:
            _, exe_from_key, title_from_key = key.split("::", 2)
            mask = (df["exe"].str.lower() == exe_from_key.lower()) & (df["title"] == title_from_key)
            df.loc[mask, "labeled_app_name"] = label_val
        except ValueError:
            continue
        except Exception as e:
            print(f"Error applying exact title label for '{key}': {e}")

    for exe_basename_key, label_val in labels.get("exact_exe", {}).items():
        mask = (
            df["exe"].apply(lambda x: os.path.basename(x).lower()) == exe_basename_key.lower()
        ) & (df["labeled_app_name"] == df["original_app_name"])
        df.loc[mask, "labeled_app_name"] = label_val

    for key, label_val in labels.get("patterns", {}).items():
        try:
            _, exe_basename_key, pattern_from_key = key.split("::", 2)
            mask = (
                df["exe"].apply(lambda x: os.path.basename(x).lower()).str.contains(exe_basename_key.lower(), regex=False)
                & df["title"].str.lower().str.contains(pattern_from_key.lower(), regex=False)
                & (df["labeled_app_name"] == df["original_app_name"])
            )
            df.loc[mask, "labeled_app_name"] = label_val
        except ValueError:
            continue
        except Exception as e:
            print(f"Error applying pattern label for '{key}': {e}")

    df["app_name"] = df["labeled_app_name"]
    df.drop(columns=["labeled_app_name"], inplace=True, errors="ignore")
    return df


def group_activities_for_labeling(logs_df: pd.DataFrame) -> pd.DataFrame:
    if logs_df.empty or "title" not in logs_df.columns:
        return pd.DataFrame()

    for col, default_val in [("app_name", "Unknown Application"), ("exe", "unknown.exe"), ("duration", 0)]:
        if col not in logs_df.columns:
            logs_df[col] = default_val
        logs_df[col] = logs_df[col].fillna(default_val)

    logs_df["title"] = logs_df["title"].astype(str)

    title_groups_data: List[Dict[str, Any]] = []
    for title_val, group in logs_df.groupby("title"):
        app_name_mode = group["app_name"].mode()
        exe_mode = group["exe"].mode()
        title_groups_data.append(
            {
                "title": title_val,
                "app_name": app_name_mode[0] if not app_name_mode.empty else "Unknown Application",
                "exe": exe_mode[0] if not exe_mode.empty else "unknown.exe",
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

# Feedback management

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


# Time bucket helpers

def load_time_buckets_for_date(date_str: str) -> List[Dict[str, Any]]:
    buckets_with_tags: List[Dict[str, Any]] = []
    for path in LOGS_DIR.glob("time_buckets_*.json"):
        try:
            session_tag = path.name.replace("time_buckets_", "").replace(".json", "")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for b in data:
                if b.get("start", "")[:10] == date_str:
                    b_copy = b.copy()
                    b_copy["session_tag"] = session_tag
                    buckets_with_tags.append(b_copy)
        except Exception as e:
            st.error(f"Error reading bucket file {path.name}: {e}")
    buckets_with_tags.sort(key=lambda x: x.get("start", ""))
    return buckets_with_tags


def update_bucket_summary_in_file(session_tag: str, bucket_start_iso: str, new_summary: str) -> bool:
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


# Category helpers

def load_categories() -> List[Dict[str, str]]:
    if not CATEGORIES_FILE.exists():
        return []
    try:
        with open(CATEGORIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("categories", [])
    except Exception as e:
        print(f"Error loading categories: {e}")
        return []


def save_categories(categories: List[Dict[str, str]]) -> bool:
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(CATEGORIES_FILE, "w", encoding="utf-8") as f:
            json.dump({"categories": categories}, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving categories: {e}")
        return False


def update_bucket_category_in_file(session_tag: str, bucket_start_iso: str, new_category_id: str) -> bool:
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

def generate_time_buckets_from_logs(date_str: str) -> bool:
    log_path = LOGS_DIR / f"focus_log_{date_str}.jsonl"
    if not log_path.exists():
        st.error(f"Focus log for {date_str} not found.")
        return False

    logs_df = load_log_entries(date_str)
    if logs_df.empty:
        st.error(f"No log entries found for {date_str}.")
        return False

    if "timestamp" not in logs_df.columns:
        st.error("Log data missing timestamp column.")
        return False

    logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"])
    logs_df = logs_df.sort_values("timestamp")

    min_time = logs_df["timestamp"].min()
    max_time = logs_df["timestamp"].max()

    bucket_size = pd.Timedelta(minutes=5)
    start_time = min_time.floor("5min")
    end_time = max_time.ceil("5min")

    bucket_starts: List[pd.Timestamp] = []
    current = start_time
    while current <= end_time:
        bucket_starts.append(current)
        current += bucket_size

    buckets: List[Dict[str, Any]] = []
    session_tag = datetime.now().strftime("%Y%m%dT%H%M%SZ")

    from .llm_utils import generate_summary_from_raw_with_llm

    for i in range(len(bucket_starts) - 1):
        bucket_start = bucket_starts[i]
        bucket_end = bucket_starts[i + 1]
        bucket_logs = logs_df[(logs_df["timestamp"] >= bucket_start) & (logs_df["timestamp"] < bucket_end)]
        if not bucket_logs.empty:
            titles: List[str] = []
            if "title" in bucket_logs.columns:
                titles = bucket_logs["title"].dropna().unique().tolist()
            bucket: Dict[str, Any] = {
                "start": bucket_start.isoformat(),
                "end": bucket_end.isoformat(),
                "titles": titles,
                "ocr_text": [],
            }
            summary, category_id, _ = generate_summary_from_raw_with_llm(titles, [])
            if summary:
                bucket["summary"] = summary
            if category_id:
                bucket["category_id"] = category_id
            buckets.append(bucket)

    if not buckets:
        st.warning(f"No activity buckets created for {date_str}.")
        return False

    bucket_file_path = LOGS_DIR / f"time_buckets_{session_tag}.json"
    try:
        with open(bucket_file_path, "w", encoding="utf-8") as f:
            json.dump(buckets, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving time buckets: {e}")
        return False
