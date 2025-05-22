import streamlit as st
st.session_state['streamlit_running'] = True
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
from dashboard.data_utils import (
     load_browser_profiles, save_browser_profiles, get_available_exe_patterns,
     validate_browser_profile, hex_to_rgb, rgb_to_hex, get_default_search_rect_for_browser
)

# Import all utility functions from dashboard modules
from dashboard.data_utils import (
    load_available_dates, load_log_entries, load_daily_summary, generate_summary_from_logs,
    load_labels, save_labels, apply_labels_to_logs, group_activities_for_labeling,
    load_block_feedback, save_block_feedback, load_time_buckets_for_date,
    update_bucket_summary_in_file, load_categories, save_categories,
    update_bucket_category_in_file, generate_time_buckets_from_logs,
    # Add these new imports:
    load_browser_profiles, save_browser_profiles, get_available_exe_patterns,
    validate_browser_profile, hex_to_rgb, rgb_to_hex, get_default_search_rect_for_browser,
    LOGS_DIR, LABELS_FILE, FEEDBACK_FILE, CATEGORIES_FILE
)
from dashboard.llm_utils import (
    _call_llm_api, generate_summary_and_category, generate_summary_from_raw_with_llm,
    refine_summary_with_llm, get_available_ollama_models, 
    get_available_openai_models, test_llm_connection, DEFAULT_LLM_API_URL, DEFAULT_LLM_MODEL, 
    DEFAULT_OPENAI_MODEL, get_openai_models_from_api
) # Removed process_activity_data import
from dashboard.charts import (
    create_pie_chart, create_browser_chart, create_category_chart
)

# --- Tracker Control Functions ---
def is_tracker_running():
    pid = st.session_state.get("tracker_pid")
    if pid is None: return False
    try:
        return psutil.pid_exists(pid) and "standalone_focus_monitor.py" in " ".join(psutil.Process(pid).cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    
def start_tracker():
    if is_tracker_running():
        st.info("Focus tracker already running.")
        return
    # Ensure script path is robust
    script_path = Path(__file__).resolve().parent / "standalone_focus_monitor.py"
    if not script_path.exists():
        st.error(f"Tracker script not found at {script_path}")
        # Try to find it in the current working directory as a fallback for some execution contexts
        script_path_cwd = Path.cwd() / "standalone_focus_monitor.py"
        if not script_path_cwd.exists():
            st.error(f"Also not found at {script_path_cwd}")
            return
        script_path = script_path_cwd
        
    try:
        # Start detached if possible, or manage process carefully
        process = subprocess.Popen([sys.executable, str(script_path)])
        st.session_state["tracker_pid"] = process.pid
        st.success(f"Focus tracker started (PID: {process.pid})")
        time.sleep(1) # Give it a moment to stabilize
    except Exception as e:
        st.error(f"Failed to start tracker: {e}")


def stop_tracker():
    pid = st.session_state.get("tracker_pid")
    if not pid or not is_tracker_running(): # Use is_tracker_running for robust check
        st.info("Focus tracker is not running or PID is stale.")
        st.session_state["tracker_pid"] = None
        return
    try:
        p = psutil.Process(pid)
        p.terminate() # SIGTERM
        p.wait(timeout=5) 
        st.success("Focus tracker stop signal sent.")
    except psutil.NoSuchProcess:
        st.info("Focus tracker process not found (already stopped?).")
    except psutil.TimeoutExpired:
        st.warning("Tracker did not terminate gracefully, attempting to kill.")
        try:
            p.kill() # SIGKILL
            p.wait(timeout=2)
            st.success("Focus tracker killed.")
        except Exception as e_kill:
            st.error(f"Failed to kill tracker: {e_kill}")
    except Exception as e:
        st.error(f"Failed to stop tracker: {e}")
    finally:
        st.session_state["tracker_pid"] = None

def display_browser_profile_manager():
    st.title("🌐 Browser Profile Manager")
    st.markdown("""
    Configure browser profiles for automatic categorization based on visual color detection.
    The focus monitor will detect these colors in browser windows and automatically categorize activities.
    """)
    
    # Load existing profiles and categories
    profiles = load_browser_profiles()
    categories = load_categories()
    
    if not categories:
        st.error("No activity categories defined. Please create categories first in the 'Activity Categories Manager' tab.")
        st.stop()
    
    # Display existing profiles
    if profiles:
        st.subheader("Current Browser Profiles")
        
        # Create a more visual display of profiles
        for i, profile in enumerate(profiles):
            enabled_icon = "✅" if profile.get("enabled", True) else "❌"
            color_rgb = tuple(profile.get("color_rgb", [128, 128, 128]))
            color_hex = rgb_to_hex(color_rgb)
            
            with st.expander(f"{enabled_icon} {profile.get('name', 'Unnamed Profile')} - {profile.get('exe_pattern', 'Unknown')}"):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**Browser:** {profile.get('exe_pattern', 'N/A')}")
                    st.write(f"**Category:** {profile.get('category_id', 'N/A')}")
                    st.write(f"**Color:** RGB{color_rgb} ({color_hex})")
                    st.write(f"**Search Area:** {profile.get('search_rect', 'N/A')} (x, y, width, height)")
                    st.write(f"**Tolerance:** {profile.get('color_tolerance', 15)}")
                
                with col2:
                    # Show color swatch
                    st.markdown(
                        f'<div style="width: 60px; height: 60px; background-color: {color_hex}; '
                        f'border: 2px solid #ccc; border-radius: 8px; margin: 10px 0;"></div>',
                        unsafe_allow_html=True
                    )
                
                with col3:
                    st.write(f"**Status:** {'Enabled' if profile.get('enabled', True) else 'Disabled'}")
    else:
        st.info("No browser profiles defined yet. Use the form below to create profiles.")
    
    # Add new profile form
    st.subheader("Add New Browser Profile")
    
    with st.form("add_browser_profile_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            profile_name = st.text_input(
                "Profile Name", 
                placeholder="e.g., Edge - Work, Chrome - Personal"
            )
            
            exe_pattern = st.selectbox(
                "Browser Executable",
                options=get_available_exe_patterns(),
                help="Select the browser executable pattern to match"
            )
            
            # Category selection
            category_options = [cat.get("id", "") for cat in categories]
            category_names = [cat.get("name", "Unknown") for cat in categories]
            category_display = [f"{name} ({id})" for name, id in zip(category_names, category_options)]
            
            selected_category_idx = st.selectbox(
                "Activity Category",
                options=range(len(category_display)),
                format_func=lambda x: category_display[x],
                help="Category to assign when this profile is detected"
            )
            selected_category_id = category_options[selected_category_idx]
        
        with col2:
            # Color selection - offer both hex and RGB input
            color_input_method = st.radio(
                "Color Input Method",
                ["Color Picker", "RGB Values", "Hex Code"],
                horizontal=True
            )
            
            if color_input_method == "Color Picker":
                # Streamlit color picker returns hex
                color_hex = st.color_picker("Profile Color", "#072B47")
                color_rgb = hex_to_rgb(color_hex)
                st.write(f"RGB: {color_rgb}")
                
            elif color_input_method == "RGB Values":
                rgb_col1, rgb_col2, rgb_col3 = st.columns(3)
                with rgb_col1:
                    r_val = st.number_input("Red", min_value=0, max_value=255, value=7)
                with rgb_col2:
                    g_val = st.number_input("Green", min_value=0, max_value=255, value=43)
                with rgb_col3:
                    b_val = st.number_input("Blue", min_value=0, max_value=255, value=71)
                color_rgb = (r_val, g_val, b_val)
                color_hex = rgb_to_hex(color_rgb)
                st.write(f"Hex: {color_hex}")
                
            else:  # Hex Code
                hex_input = st.text_input("Hex Color Code", value="#072B47", placeholder="#RRGGBB")
                try:
                    color_rgb = hex_to_rgb(hex_input)
                    color_hex = hex_input
                    st.write(f"RGB: {color_rgb}")
                except ValueError:
                    st.error("Invalid hex color code")
                    color_rgb = (7, 43, 71)
                    color_hex = "#072B47"
            
            # Show color preview
            st.markdown(
                f'<div style="width: 80px; height: 80px; background-color: {color_hex}; '
                f'border: 2px solid #ccc; border-radius: 8px; margin: 10px 0;"></div>',
                unsafe_allow_html=True
            )
        
        # Advanced settings
        st.subheader("Detection Settings")
        adv_col1, adv_col2 = st.columns(2)
        
        with adv_col1:
            # Search rectangle
            st.write("**Search Rectangle (pixels from top-left of window)**")
            default_rect = get_default_search_rect_for_browser(exe_pattern)
            
            rect_col1, rect_col2, rect_col3, rect_col4 = st.columns(4)
            with rect_col1:
                x_offset = st.number_input("X Offset", min_value=0, value=default_rect[0])
            with rect_col2:
                y_offset = st.number_input("Y Offset", min_value=0, value=default_rect[1])
            with rect_col3:
                width = st.number_input("Width", min_value=1, value=default_rect[2])
            with rect_col4:
                height = st.number_input("Height", min_value=1, value=default_rect[3])
            
            search_rect = [x_offset, y_offset, width, height]
        
        with adv_col2:
            color_tolerance = st.slider(
                "Color Tolerance",
                min_value=0, max_value=50, value=15,
                help="How much the detected color can vary from the target color (0 = exact match)"
            )
            
            enabled = st.checkbox("Enable Profile", value=True)
        
        # Form submission
        col1, col2 = st.columns([1, 4])
        with col1:
            submitted = st.form_submit_button("Add Profile", type="primary")
        with col2:
            if st.form_submit_button("🎯 Test Color Detection", help="This will be implemented to help test color detection"):
                st.info("Color detection testing feature coming soon!")
        
        if submitted:
            if not profile_name or not exe_pattern:
                st.error("Profile name and browser executable are required")
            else:
                # Check for duplicate names
                if any(p.get("name") == profile_name for p in profiles):
                    st.error(f"Profile name '{profile_name}' already exists")
                else:
                    new_profile = {
                        "name": profile_name,
                        "exe_pattern": exe_pattern,
                        "color_rgb": list(color_rgb),  # Convert tuple to list for JSON
                        "color_tolerance": color_tolerance,
                        "search_rect": search_rect,
                        "category_id": selected_category_id,
                        "enabled": enabled
                    }
                    
                    # Validate the profile
                    is_valid, error_msg = validate_browser_profile(new_profile)
                    if not is_valid:
                        st.error(f"Profile validation failed: {error_msg}")
                    else:
                        profiles.append(new_profile)
                        if save_browser_profiles(profiles):
                            st.success(f"Browser profile '{profile_name}' added successfully!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Failed to save browser profile")

    # Edit/Delete existing profiles
    if profiles:
        st.subheader("Edit or Delete Profiles")
        
        # Select profile to edit
        profile_names = [p.get("name", f"Unnamed ({i})") for i, p in enumerate(profiles)]
        selected_profile_name = st.selectbox(
            "Select Profile to Edit/Delete",
            options=profile_names,
            key="edit_delete_profile_select"
        )
        
        selected_profile_idx = profile_names.index(selected_profile_name)
        selected_profile = profiles[selected_profile_idx]
        
        # Edit form
        with st.form("edit_browser_profile_form"):
            edit_col1, edit_col2 = st.columns(2)
            
            with edit_col1:
                edit_name = st.text_input("Profile Name", value=selected_profile.get("name", ""))
                edit_exe = st.selectbox(
                    "Browser Executable",
                    options=get_available_exe_patterns(),
                    index=get_available_exe_patterns().index(selected_profile.get("exe_pattern", "msedge.exe"))
                    if selected_profile.get("exe_pattern") in get_available_exe_patterns() else 0
                )
                
                # Category selection for editing
                current_category_id = selected_profile.get("category_id", "")
                try:
                    current_category_idx = category_options.index(current_category_id)
                except ValueError:
                    current_category_idx = 0
                
                edit_category_idx = st.selectbox(
                    "Activity Category",
                    options=range(len(category_display)),
                    format_func=lambda x: category_display[x],
                    index=current_category_idx
                )
                edit_category_id = category_options[edit_category_idx]
            
            with edit_col2:
                # Color editing
                current_color_rgb = tuple(selected_profile.get("color_rgb", [7, 43, 71]))
                current_color_hex = rgb_to_hex(current_color_rgb)
                
                edit_color_hex = st.color_picker("Profile Color", current_color_hex)
                edit_color_rgb = hex_to_rgb(edit_color_hex)
                
                # Show color preview
                st.markdown(
                    f'<div style="width: 60px; height: 60px; background-color: {edit_color_hex}; '
                    f'border: 2px solid #ccc; border-radius: 8px; margin: 10px 0;"></div>',
                    unsafe_allow_html=True
                )
            
            # Advanced settings for editing
            current_rect = selected_profile.get("search_rect", [9, 3, 30, 30])
            current_tolerance = selected_profile.get("color_tolerance", 15)
            current_enabled = selected_profile.get("enabled", True)
            
            st.write("**Detection Settings**")
            edit_adv_col1, edit_adv_col2 = st.columns(2)
            
            with edit_adv_col1:
                st.write("Search Rectangle:")
                edit_rect_cols = st.columns(4)
                with edit_rect_cols[0]:
                    edit_x = st.number_input("X", min_value=0, value=current_rect[0], key="edit_x")
                with edit_rect_cols[1]:
                    edit_y = st.number_input("Y", min_value=0, value=current_rect[1], key="edit_y")
                with edit_rect_cols[2]:
                    edit_w = st.number_input("W", min_value=1, value=current_rect[2], key="edit_w")
                with edit_rect_cols[3]:
                    edit_h = st.number_input("H", min_value=1, value=current_rect[3], key="edit_h")
                edit_search_rect = [edit_x, edit_y, edit_w, edit_h]
            
            with edit_adv_col2:
                edit_tolerance = st.slider(
                    "Color Tolerance",
                    min_value=0, max_value=50, value=current_tolerance,
                    key="edit_tolerance"
                )
                edit_enabled = st.checkbox("Enable Profile", value=current_enabled, key="edit_enabled")
            
            # Form buttons
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                update_btn = st.form_submit_button("Update Profile")
            with btn_col2:
                delete_btn = st.form_submit_button("Delete Profile", type="secondary")
            with btn_col3:
                toggle_btn = st.form_submit_button(
                    "Disable" if current_enabled else "Enable",
                    help="Quick toggle to enable/disable profile"
                )
            
            if update_btn:
                if not edit_name or not edit_exe:
                    st.error("Profile name and browser executable are required")
                else:
                    # Check for duplicate names (excluding current profile)
                    if edit_name != selected_profile.get("name") and any(p.get("name") == edit_name for p in profiles):
                        st.error(f"Profile name '{edit_name}' already exists")
                    else:
                        updated_profile = {
                            "name": edit_name,
                            "exe_pattern": edit_exe,
                            "color_rgb": list(edit_color_rgb),
                            "color_tolerance": edit_tolerance,
                            "search_rect": edit_search_rect,
                            "category_id": edit_category_id,
                            "enabled": edit_enabled
                        }
                        
                        # Validate the updated profile
                        is_valid, error_msg = validate_browser_profile(updated_profile)
                        if not is_valid:
                            st.error(f"Profile validation failed: {error_msg}")
                        else:
                            profiles[selected_profile_idx] = updated_profile
                            if save_browser_profiles(profiles):
                                st.success(f"Profile '{edit_name}' updated successfully!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Failed to save updated profile")
            
            if delete_btn:
                profiles.pop(selected_profile_idx)
                if save_browser_profiles(profiles):
                    st.success(f"Profile '{selected_profile_name}' deleted successfully!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Failed to delete profile")
            
            if toggle_btn:
                profiles[selected_profile_idx]["enabled"] = not current_enabled
                if save_browser_profiles(profiles):
                    status = "enabled" if not current_enabled else "disabled"
                    st.success(f"Profile '{selected_profile_name}' {status}!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Failed to update profile status")
    
    # Help section
    st.subheader("ℹ️ How to Use Browser Profiles")
    with st.expander("Setup Instructions"):
        st.markdown("""
        **Setting up browser profiles for automatic detection:**
        
        1. **Identify the target color**: Open your browser with the specific profile you want to detect
        2. **Find the profile indicator**: Look for colored elements that are unique to this profile (e.g., profile avatar, theme colors)
        3. **Determine coordinates**: The search rectangle defines where to look for the color (x, y from top-left, width, height)
        4. **Get the exact color**: Use a color picker tool to get the RGB values of the profile indicator
        5. **Set tolerance**: Higher tolerance allows for slight color variations (recommended: 10-20)
        6. **Test the setup**: The focus monitor will automatically detect and categorize activities
        
        **Tips:**
        - Profile avatars and theme colors work best for detection
        - Avoid areas that change frequently (like webpage content)
        - Start with higher tolerance and adjust down if getting false positives
        - Different browsers have profile indicators in different locations
        """)
    
    with st.expander("Troubleshooting"):
        st.markdown("""
        **Common issues and solutions:**
        
        - **Profile not detected**: Check if the search rectangle covers the profile indicator area
        - **Wrong categorization**: Verify the RGB color values match exactly
        - **False positives**: Reduce color tolerance or adjust search area
        - **Multi-monitor issues**: Color detection works across multiple monitors
        - **Browser updates**: Profile indicator locations may change with browser updates
        
        **Getting RGB values:**
        - Windows: Use built-in Snipping Tool with color picker
        - Online tools: Use web-based color picker tools
        - Browser extensions: Color picker browser extensions
        """)

def display_retroactive_processor():
    st.title("⏮️ Historical Data Processor")
    dates = load_available_dates()
    if not dates: st.error("No data found in focus_logs directory."); st.stop()
    
    selected_date = st.selectbox("Select a date to process", dates, key="processor_date_select")
    if not selected_date: st.info("Please select a date."); st.stop()

    log_file = LOGS_DIR / f"focus_log_{selected_date}.jsonl"
    summary_file = LOGS_DIR / f"daily_summary_{selected_date}.json"
    # For time buckets, the filename might vary. We check if ANY buckets exist for the date.
    existing_buckets_for_date = load_time_buckets_for_date(selected_date)
    
    st.subheader("Available Data Status")
    col1, col2 = st.columns(2)
    col1.metric("Focus Log File", "✅ Found" if log_file.exists() else "❌ Missing")
    col1.metric("Daily Summary File", "✅ Found" if summary_file.exists() else "❌ Missing")
    col2.metric("Time Buckets for Date", f"{len(existing_buckets_for_date)} found" if existing_buckets_for_date else "❌ None Found")
    
    # Check how many buckets have LLM summaries
    if existing_buckets_for_date:
        buckets_with_summaries = sum(1 for bucket in existing_buckets_for_date if bucket.get("summary", "").strip())
        buckets_without_summaries = len(existing_buckets_for_date) - buckets_with_summaries
        col2.metric("Buckets with LLM Summaries", f"{buckets_with_summaries}/{len(existing_buckets_for_date)}")
        if buckets_without_summaries > 0:
            st.warning(f"⚠️ {buckets_without_summaries} time buckets are missing LLM summaries (collected by agent but not yet processed)")

    st.subheader("Processing Actions")
    
    # Daily Summary Processing
    st.markdown("#### Daily Summary Processing")
    if st.button("🔄 (Re)Generate Daily Summary from Logs", key="proc_gen_summary_btn", help="Parses raw logs, applies labels, and creates/overwrites the daily summary file."):
        with st.spinner("Generating daily summary..."):
            summary_data = generate_summary_from_logs(selected_date) # This now saves the file
            if summary_data: st.success(f"Daily summary for {selected_date} processed."); st.json(summary_data, expanded=False)
            else: st.error("Failed to process daily summary.")
            st.rerun()

    # Time Bucket Processing
    st.markdown("#### Time Bucket Processing")
    
    # Option 1: Generate new time buckets from logs (overwrites existing)
    if st.button("🔄 (Re)Generate 5-Min Time Buckets & LLM Summaries", key="proc_gen_buckets_btn", help="Divides logs into 5-min chunks, calls LLM for summaries/categories, and saves a new time_buckets_*.json file. This will overwrite existing buckets."):
        with st.spinner("Generating time buckets and LLM summaries... This can take time."):
            success = generate_time_buckets_from_logs(selected_date) # This now saves the file
            if success: st.success(f"Time buckets for {selected_date} processed.")
            else: st.error("Failed to process time buckets.")
            st.rerun() # Rerun to refresh bucket count and sample display
    
    # Option 2: Process LLM summaries for existing buckets that don't have them
    if existing_buckets_for_date and buckets_without_summaries > 0:
        st.markdown("#### LLM Processing for Existing Buckets")
        st.info(f"Found {buckets_without_summaries} time buckets without LLM summaries. You can generate summaries for these existing buckets without recreating them.")
        
        if st.button("🤖 Generate LLM Summaries for Existing Buckets", key="proc_gen_llm_summaries_btn", help="Processes existing time buckets to add LLM-generated summaries and categories where missing."):
            with st.spinner(f"Generating LLM summaries for {buckets_without_summaries} buckets... This may take some time."):
                success_count = 0
                error_count = 0
                
                # Group buckets by session tag for efficient file updates
                buckets_by_session = {}
                for bucket in existing_buckets_for_date:
                    if not bucket.get("summary", "").strip():  # Only process buckets without summaries
                        session_tag = bucket.get("session_tag", "unknown_session")
                        if session_tag not in buckets_by_session:
                            buckets_by_session[session_tag] = []
                        buckets_by_session[session_tag].append(bucket)
                
                progress_bar = st.progress(0)
                total_buckets = sum(len(buckets) for buckets in buckets_by_session.values())
                processed_buckets = 0
                
                for session_tag, session_buckets in buckets_by_session.items():
                    for bucket in session_buckets:
                        try:
                            # Extract data for LLM processing
                            titles_list = bucket.get("titles", [])
                            ocr_text_list = bucket.get("ocr_text", [])
                            
                            if titles_list or ocr_text_list:
                                # Generate summary using user's selected LLM provider/model
                                summary_text, category_id, _suggested_category, _ = generate_summary_from_raw_with_llm(
                                    titles_list, ocr_text_list, allow_suggestions=False, return_prompt=False
                                )
                                
                                # Update the bucket in the file
                                bucket_start_iso = bucket.get("start", "")
                                if summary_text and bucket_start_iso:
                                    if update_bucket_summary_in_file(session_tag, bucket_start_iso, summary_text):
                                        success_count += 1
                                        # Also update category if one was identified
                                        if category_id:
                                            update_bucket_category_in_file(session_tag, bucket_start_iso, category_id)
                                    else:
                                        error_count += 1
                                        st.error(f"Failed to update summary for bucket {bucket_start_iso}")
                                else:
                                    error_count += 1
                                    st.warning(f"LLM returned empty summary for bucket {bucket_start_iso}")
                            else:
                                # No content to summarize
                                error_count += 1
                                
                        except Exception as e:
                            error_count += 1
                            st.error(f"Error processing bucket: {e}")
                        
                        processed_buckets += 1
                        progress_bar.progress(processed_buckets / total_buckets)
                
                progress_bar.empty()
                
                if success_count > 0:
                    st.success(f"✅ Successfully generated {success_count} LLM summaries")
                if error_count > 0:
                    st.warning(f"⚠️ {error_count} buckets had errors or no content to process")
                
                if success_count > 0:
                    st.rerun()  # Refresh to show updated bucket counts
    
    # Display sample of existing buckets
    if existing_buckets_for_date:
        st.subheader("Sample of Existing Time Buckets (Max 5 Shown)")
        categories_map = {cat.get("id"): cat.get("name", "Unknown") for cat in load_categories()}
        for i, bucket in enumerate(existing_buckets_for_date[:5]):
            start_dt = pd.to_datetime(bucket.get("start")).strftime("%H:%M")
            end_dt = pd.to_datetime(bucket.get("end")).strftime("%H:%M")
            cat_name = categories_map.get(bucket.get("category_id", ""), "Uncategorized")
            
            # Show if bucket has LLM summary or not
            has_summary = bool(bucket.get("summary", "").strip())
            summary_indicator = "🤖" if has_summary else "⚪"
            
            with st.expander(f"{summary_indicator} Bucket {start_dt}-{end_dt} [{cat_name}] (Session: {bucket.get('session_tag', 'N/A')})"):
                if has_summary:
                    st.write("**LLM Summary:**", bucket.get("summary", "_N/A_"))
                else:
                    st.write("**LLM Summary:** _No summary generated yet_")
                    st.caption("Raw data available - use 'Generate LLM Summaries' button above to process")
                
                st.caption(f"Titles: {len(bucket.get('titles',[]))}, OCR: {len(bucket.get('ocr_text',[]))}")
                
                # Show raw data in expandable section
                if bucket.get('titles') or bucket.get('ocr_text'):
                    with st.expander("View Raw Data"):
                        if bucket.get('titles'):
                            st.write("**Window Titles:**")
                            for title in bucket.get('titles', [])[:10]:  # Show first 10
                                st.text(f"• {title}")
                            if len(bucket.get('titles', [])) > 10:
                                st.caption(f"... and {len(bucket.get('titles', [])) - 10} more")
                        
                        if bucket.get('ocr_text'):
                            st.write("**OCR Text:**")
                            for ocr in bucket.get('ocr_text', [])[:5]:  # Show first 5
                                st.text(f"• {ocr[:100]}{'...' if len(ocr) > 100 else ''}")
                            if len(bucket.get('ocr_text', [])) > 5:
                                st.caption(f"... and {len(bucket.get('ocr_text', [])) - 5} more")


def llm_test_page():
    st.title("🧪 LLM Processing Test Tool")
    categories = load_categories()
    if not categories:
        st.error("No categories defined. Please create them in 'Activity Categories Manager'."); return

    st.write("Test LLM summarization and categorization with custom inputs.")
    with st.expander("View Current Categories"):
        for cat in categories: st.write(f"**{cat.get('name')}** ({cat.get('id')}): {cat.get('description')}")

    st.subheader("Test Input")
    test_titles_str = st.text_area("Window Titles (one per line):", height=100, placeholder="VS Code - my_project.py\nChrome - Google Search")
    test_ocr_str = st.text_area("OCR Text Snippets (one per line, optional):", placeholder="Important text from screen...")

    
    test_titles = [t.strip() for t in test_titles_str.split("\n") if t.strip()]
    test_ocr = [o.strip() for o in test_ocr_str.split("\n") if o.strip()]

    allow_suggestions_test = st.checkbox("Allow LLM to suggest new categories", True)

    if (test_titles or test_ocr) and st.button("🚀 Run LLM Test", type="primary"):
        with st.spinner("Querying LLM..."):
            # Use the refactored llm_utils function, requesting the prompt back
            summary, cat_id, suggestion, gen_prompt = generate_summary_from_raw_with_llm(
                test_titles, test_ocr, allow_suggestions=allow_suggestions_test, return_prompt=True
            )
        
        st.subheader("LLM Interaction Details")
        if gen_prompt:
            with st.expander("Full Prompt Sent to LLM"): st.code(gen_prompt, language='text')
        
        # For raw response, ideally _call_llm_api would be called directly or its response passed through.
        # Here, we reconstruct what the LLM *should* have output if it followed the format.
        st.subheader("LLM's Parsed Output:")
        st.markdown(f"**Generated Summary:**")
        st.info(summary or "_No summary returned or parsed._")
        
        st.markdown(f"**Identified Category ID:**")
        if cat_id:
            cat_name = next((c.get("name") for c in categories if c.get("id") == cat_id), f"Unknown ID: {cat_id}")
            st.success(f"{cat_name} ({cat_id})")
        elif allow_suggestions_test and not cat_id and suggestion: # Implies 'none' was chosen
             st.warning("None (LLM suggests a new category below)")
        else:
            st.warning("_No valid category ID identified._")

        if allow_suggestions_test and suggestion:
            st.markdown(f"**Suggested New Category:**")
            if "|" in suggestion:
                sugg_name, sugg_desc = suggestion.split("|", 1)
                st.success(f"Name: `{sugg_name.strip()}` | Description: `{sugg_desc.strip()}`")
            else:
                st.error(f"Suggestion format error. Raw: `{suggestion}` (Expected 'name | description')")
        elif allow_suggestions_test:
            st.markdown("**Suggested New Category:** _None_")


# --- Data Management Functions ---
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

# --- Time Bucket Summaries Page ---
def display_time_bucket_summaries():
    st.title("📝 5-Minute Summaries & Feedback")
    st.caption("Note: Browser profile activities are excluded from 5-minute buckets and aggregated daily instead")
    
    dates = load_available_dates()
    if not dates: 
        st.error("No data in focus_logs directory.")
        st.stop()
    
    selected_date = st.selectbox("Select a date", dates, key="summaries_date_select")

    official_buckets = load_time_buckets_for_date(selected_date)
    personal_notes_data = load_block_feedback()
    categories = load_categories()
    cat_map = {cat.get("id"): cat for cat in categories}

    # Show browser profile activities for context
    browser_activities = load_daily_browser_activities(selected_date)
    if browser_activities:
        st.subheader("🌐 Browser Profile Activities (Daily Aggregated)")
        st.caption("These activities are automatically categorized and excluded from 5-minute buckets below")
        
        browser_summary_data = []
        for activity in browser_activities:
            cat_name = cat_map.get(activity.get("category_id", ""), {}).get("name", "Uncategorized")
            duration_minutes = round(activity.get("total_duration", 0) / 60, 1)
            
            browser_summary_data.append({
                "Category": cat_name,
                "Duration (min)": duration_minutes,
                "Activities": activity.get("activity_count", 0),
                "Unique Titles": len(activity.get("titles", []))
            })
        
        if browser_summary_data:
            df_browser_summary = pd.DataFrame(browser_summary_data)
            st.dataframe(df_browser_summary, use_container_width=True)
        
        with st.expander("View Detailed Browser Profile Activities"):
            for activity in browser_activities:
                cat_name = cat_map.get(activity.get("category_id", ""), {}).get("name", "Uncategorized")
                duration_minutes = round(activity.get("total_duration", 0) / 60, 1)
                
                st.markdown(f"**{cat_name}** - {duration_minutes} minutes")
                
                if activity.get("titles"):
                    st.markdown("*Sample Window Titles:*")
                    for title in list(activity["titles"])[:5]:  # Show first 5
                        st.text(f"  • {title[:60]}{'...' if len(title) > 60 else ''}")
                    if len(activity["titles"]) > 5:
                        st.caption(f"... and {len(activity['titles']) - 5} more")
                
                st.markdown("---")
        
        st.markdown("---")
    
    if not official_buckets:
        if browser_activities:
            st.info(f"No 5-minute summary blocks found for {selected_date}, but browser profile activities are available above.")
        else:
            st.info(f"No 5-minute summary blocks found for {selected_date}.")
        return

    st.subheader("📊 5-Minute Activity Buckets (Non-Browser-Profile Activities)")
    st.caption(f"Found {len(official_buckets)} five-minute blocks containing regular activities")

    show_uncategorized_only = st.checkbox("Filter: Show only uncategorized entries", False)
    
    # Category filter
    active_category_filter = "All Categories"
    if categories and not show_uncategorized_only:
        cat_options = ["All Categories", "Uncategorized"] + [cat.get("name") for cat in categories]
        active_category_filter = st.selectbox("Filter by category", cat_options, key="sum_cat_filter")

    # Bucket counts by category
    if categories:
        counts = {"Uncategorized": 0, **{cat.get("name"): 0 for cat in categories}}
        for b in official_buckets:
            cat_name = cat_map.get(b.get("category_id"), {}).get("name", "Uncategorized")
            counts[cat_name] = counts.get(cat_name, 0) + 1
        
        st.subheader("5-Minute Blocks by Category")
        cols = st.columns(min(len(counts), 5))
        for i, (name, count) in enumerate(counts.items()):
            if count > 0: 
                cols[i % 5].metric(name, count)
        
        if counts["Uncategorized"] > 0:
             st.warning(f"{counts['Uncategorized']} 5-minute blocks are uncategorized. Use filter above to focus on them.")
        st.markdown("---")

    # Display individual buckets
    displayed_count = 0
    for idx, bucket in enumerate(official_buckets):
        bucket_start_iso = bucket["start"]
        session_tag = bucket.get("session_tag", "unknown_session")
        current_cat_id = bucket.get("category_id", "")
        current_cat_name = cat_map.get(current_cat_id, {}).get("name", "Uncategorized")

        # Apply filters
        if show_uncategorized_only and current_cat_id: 
            continue
        if not show_uncategorized_only and active_category_filter != "All Categories":
            if active_category_filter == "Uncategorized" and current_cat_name != "Uncategorized": 
                continue
            if active_category_filter != "Uncategorized" and current_cat_name != active_category_filter: 
                continue
        
        displayed_count += 1
        start_dt_fmt = pd.to_datetime(bucket_start_iso).strftime('%H:%M')
        exp_title = f"{start_dt_fmt} (Session: {session_tag}) - [{current_cat_name}]"
        if not current_cat_id: 
            exp_title += " ⚠️"

        with st.expander(exp_title):
            # Category selection
            if categories:
                st.markdown("**Activity Category:**")
                cat_names_options = ["Uncategorized"] + [c.get("name") for c in categories]
                cat_ids_options = [""] + [c.get("id") for c in categories]
                try: 
                    current_sel_idx = cat_ids_options.index(current_cat_id)
                except ValueError: 
                    current_sel_idx = 0

                new_cat_name_sel = st.selectbox(
                    "Set Category:", 
                    cat_names_options, 
                    index=current_sel_idx, 
                    key=f"cat_sel_{idx}_{bucket_start_iso}"
                )
                new_cat_id_sel = cat_ids_options[cat_names_options.index(new_cat_name_sel)]

                if new_cat_id_sel != current_cat_id:
                    if st.button("Update Category", key=f"upd_cat_btn_{idx}_{bucket_start_iso}"):
                        if update_bucket_category_in_file(session_tag, bucket_start_iso, new_cat_id_sel):
                            st.success("Category updated!")
                            time.sleep(1)
                            st.rerun()
            
            st.markdown("**Official LLM Summary:**")
            st.caption(bucket.get("summary", "_No official summary._"))

            # Show what activities are in this bucket
            if bucket.get("titles") or bucket.get("apps"):
                st.markdown("**Activities in this 5-minute block:**")
                if bucket.get("apps"):
                    st.write("*Applications:*", ", ".join(list(bucket.get("apps", []))))
                if bucket.get("titles"):
                    st.write("*Window Titles:*")
                    for title in list(bucket.get("titles", []))[:3]:  # Show first 3
                        st.text(f"  • {title[:50]}{'...' if len(title) > 50 else ''}")
                    if len(bucket.get("titles", [])) > 3:
                        st.caption(f"... and {len(bucket.get('titles', [])) - 3} more titles")

            # LLM Actions
            st.markdown("**LLM Summary Actions:**")
            llm_action_cols = st.columns(2)
            user_feedback_for_refine = st.text_area(
                "Your feedback/details for LLM refinement:", 
                key=f"feedback_llm_{idx}_{bucket_start_iso}", 
                height=75
            )

            if llm_action_cols[0].button("Refine LLM Summary", key=f"refine_btn_{idx}_{bucket_start_iso}"):
                if user_feedback_for_refine.strip():
                    with st.spinner("Refining summary with LLM..."):
                        refined_sum, new_cid, sugg_cat, _ = refine_summary_with_llm(
                            bucket.get("summary",""), user_feedback_for_refine, current_cat_id, True)
                        if refined_sum is not None:
                            update_bucket_summary_in_file(session_tag, bucket_start_iso, refined_sum)
                            if new_cid != current_cat_id: 
                                update_bucket_category_in_file(session_tag, bucket_start_iso, new_cid)
                            st.success("Summary refined!")
                            time.sleep(1)
                            st.rerun()
                        else: 
                            st.error("LLM refinement failed.")
                else: 
                    st.warning("Provide feedback text to refine.")

            if llm_action_cols[1].button("Re-Generate Original LLM Summary", key=f"regen_btn_{idx}_{bucket_start_iso}"):
                with st.spinner("Re-generating summary with LLM..."):
                    new_sum, new_cid, sugg_cat, _ = generate_summary_from_raw_with_llm(
                        bucket.get("titles",[]), bucket.get("ocr_text",[]), True)
                    if new_sum is not None:
                        update_bucket_summary_in_file(session_tag, bucket_start_iso, new_sum)
                        if new_cid: 
                            update_bucket_category_in_file(session_tag, bucket_start_iso, new_cid)
                        st.success("Summary re-generated!")
                        time.sleep(1)
                        st.rerun()
                    else: 
                        st.error("LLM re-generation failed.")
            
            # Personal Notes
            st.markdown("**Personal Note (Private):**")
            personal_note_key = f"note_{idx}_{bucket_start_iso}"
            current_personal_note = personal_notes_data.get(bucket_start_iso, "")
            new_personal_note = st.text_area(
                "Your private note for this 5-minute block:", 
                value=current_personal_note, 
                key=personal_note_key, 
                height=75
            )
            
            note_cols = st.columns(2)
            if note_cols[0].button("Save Personal Note", key=f"save_note_{idx}_{bucket_start_iso}"):
                personal_notes_data[bucket_start_iso] = new_personal_note.strip()
                if save_block_feedback(personal_notes_data): 
                    st.success("Personal note saved!")
                    time.sleep(0.5)
                    st.rerun()
            
            if current_personal_note and note_cols[1].button("Delete Personal Note", key=f"del_note_{idx}_{bucket_start_iso}"):
                if bucket_start_iso in personal_notes_data: 
                    del personal_notes_data[bucket_start_iso]
                if save_block_feedback(personal_notes_data): 
                    st.success("Personal note deleted!")
                    time.sleep(0.5)
                    st.rerun()

    if displayed_count == 0:
        st.info(f"No 5-minute blocks match the current filter criteria.")
    
    # Summary info
    st.markdown("---")
    st.info(f"""
    **Summary for {selected_date}:**
    - 🌐 Browser profile activities: {len(browser_activities)} categories (daily aggregated)
    - 📊 5-minute activity blocks: {len(official_buckets)} blocks (regular activities only)
    - 📝 Personal notes: {len([k for k in personal_notes_data.keys() if k.startswith(selected_date)])} notes
    """)
    
    st.markdown("""
    **How this works:**
    - Browser activities that match configured profiles are automatically categorized and aggregated for the entire day
    - Regular activities (non-browser-profile) are organized into 5-minute blocks for detailed analysis and note-taking
    - This separation allows for both automatic categorization and detailed manual review
    """)



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
    with del_ltab2:
        keys_to_del = list(current_labels.get("exact_exe", {}).keys())
        if keys_to_del:
            sel_key = st.selectbox(
                "Select Application Label to Delete",
                keys_to_del,
                format_func=lambda k: f"{k} -> {current_labels['exact_exe'][k]}",
                key="del_sel_exact_exe",
            )
            if st.button("Delete Selected App Label", key="del_btn_exact_exe"):
                if sel_key in current_labels["exact_exe"]:
                    del current_labels["exact_exe"][sel_key]
                if save_labels(current_labels):
                    st.success("Label deleted.")
                    time.sleep(1)
                    st.rerun()
        else:
            st.info("No application labels to delete.")
    with del_ltab3:
        keys_to_del = list(current_labels.get("patterns", {}).keys())
        if keys_to_del:
            sel_key = st.selectbox(
                "Select Pattern Label to Delete",
                keys_to_del,
                format_func=lambda k: f"{k.split('::')[1]} - '{k.split('::')[2]}' -> {current_labels['patterns'][k]}",
                key="del_sel_pattern",
            )
            if st.button("Delete Selected Pattern Label", key="del_btn_pattern"):
                if sel_key in current_labels["patterns"]:
                    del current_labels["patterns"][sel_key]
                if save_labels(current_labels):
                    st.success("Label deleted.")
                    time.sleep(1)
                    st.rerun()
        else:
            st.info("No pattern labels to delete.")

    st.subheader("Preview With Labels Applied")
    if not logs_df_for_labeling.empty:
        labeled_logs_preview = apply_labels_to_logs(logs_df_for_labeling.copy())
        if not labeled_logs_preview.empty and "app_name" in labeled_logs_preview.columns:
            preview_agg = (
                labeled_logs_preview.groupby("app_name")
                .agg(
                    total_duration_sec=("duration", "sum"),
                    original_names=("original_app_name", lambda x: list(set(x))[:3]),
                    log_entries=("timestamp", "count") if "timestamp" in labeled_logs_preview else ("duration", "count"),
                )
                .reset_index()
                .sort_values("total_duration_sec", ascending=False)
            )
            preview_agg["Total Duration (min)"] = (preview_agg["total_duration_sec"] / 60).round(1)
            st.dataframe(
                preview_agg[["app_name", "Total Duration (min)", "log_entries", "original_names"]],
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

    daily_summary_data = load_daily_summary(selected_date)
    if not daily_summary_data:
        st.warning(f"Could not load or generate a summary for {selected_date}.")
        st.stop()

    # --- Summary Metrics ---
    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
    
    total_time_minutes = daily_summary_data.get('totalTime', 0) // 60
    d_col1.metric("🕒 Total Tracked Time", f"{total_time_minutes} min")
    
    # Calculate browser profile time
    browser_activities = daily_summary_data.get("browserProfileActivities", [])
    browser_time_minutes = sum(activity.get("total_duration", 0) for activity in browser_activities) // 60
    d_col2.metric("🌐 Browser Profile Time", f"{browser_time_minutes} min")
    
    # Calculate non-browser time
    non_browser_time_minutes = total_time_minutes - browser_time_minutes
    d_col3.metric("💻 Regular Activity Time", f"{non_browser_time_minutes} min")
    
    # Browser profile count
    d_col4.metric("🎯 Browser Profiles Used", len(browser_activities))

    # --- NEW: Browser Profile Daily Aggregation Display ---
    if browser_activities:
        st.subheader("🌐 Daily Browser Profile Activities")
        
        # Load categories for display
        categories = load_categories()
        cat_id_to_name = {cat.get("id", ""): cat.get("name", "Unknown") for cat in categories}
        
        # Create a visualization of browser activities
        browser_viz_data = []
        for activity in browser_activities:
            category_name = cat_id_to_name.get(activity.get("category_id", ""), "Uncategorized")
            duration_minutes = activity.get("total_duration", 0) / 60
            
            browser_viz_data.append({
                "Category": category_name,
                "Duration (min)": round(duration_minutes, 1),
                "Activity Count": activity.get("activity_count", 0),
                "App Names": ", ".join(list(activity.get("app_names", [])))[:50] + "..." if len(", ".join(list(activity.get("app_names", [])))) > 50 else ", ".join(list(activity.get("app_names", []))),
                "Sample Titles": len(activity.get("titles", []))
            })
        
        if browser_viz_data:
            df_browser = pd.DataFrame(browser_viz_data)
            
            # Create pie chart for browser profile time distribution
            if len(browser_viz_data) > 1:
                fig_browser_pie = px.pie(
                    df_browser,
                    values="Duration (min)",
                    names="Category",
                    title="Browser Profile Time Distribution",
                    hole=0.4
                )
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.plotly_chart(fig_browser_pie, use_container_width=True)
                
                with col2:
                    st.dataframe(df_browser, use_container_width=True)
            else:
                st.dataframe(df_browser, use_container_width=True)
            
            # Detailed browser activities
            with st.expander("🔍 Detailed Browser Profile Activities"):
                for i, activity in enumerate(browser_activities):
                    category_name = cat_id_to_name.get(activity.get("category_id", ""), "Uncategorized")
                    duration_minutes = round(activity.get("total_duration", 0) / 60, 1)
                    
                    st.markdown(f"**{category_name}** - {duration_minutes} minutes ({activity.get('activity_count', 0)} activities)")
                    
                    if activity.get("titles"):
                        titles_sample = list(activity["titles"])[:10]  # Show first 10 titles
                        st.markdown("*Sample Window Titles:*")
                        for title in titles_sample:
                            st.text(f"  • {title[:80]}{'...' if len(title) > 80 else ''}")
                        if len(activity["titles"]) > 10:
                            st.caption(f"... and {len(activity['titles']) - 10} more titles")
                    
                    if activity.get("app_names"):
                        st.markdown(f"*App Names:* {', '.join(list(activity['app_names']))}")
                    
                    st.markdown("---")
    else:
        st.info("No browser profile activities detected for this date.")

    # --- Category Analysis for 5-Minute Buckets ---
    categories = load_categories()
    all_buckets = load_time_buckets_for_date(selected_date)

    if categories and all_buckets:
        st.subheader("🏆 5-Minute Activity Buckets by Category")
        st.caption("Note: Browser profile activities are excluded from time buckets and shown separately above")

        # Create category chart using the imported function
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

                st.subheader("5-Minute Bucket Category Time Distribution")
                st.dataframe(category_totals, use_container_width=True)

                with st.expander("View Detailed 5-Minute Bucket Timeline"):
                    st.dataframe(
                        df_categories.sort_values("Start Time"),
                        use_container_width=True,
                    )
        else:
            st.info("No 5-minute bucket category data available for visualization.")
    elif categories:
        st.info("No 5-minute bucket data available for category analysis.")
    else:
        st.info(
            "No categories defined. Go to the 'Activity Categories Manager' tab to create categories."
        )

    # --- Application Breakdown (Now includes Browser Profiles) ---
    st.subheader("🧠 Time Distribution by Application/Activity")
    st.caption("This includes both regular activities and browser profile aggregations")
    
    app_breakdown_data = daily_summary_data.get("appBreakdown", [])
    if app_breakdown_data:
        # Separate browser profile activities from regular activities for display
        browser_profile_apps = [app for app in app_breakdown_data if app.get("isBrowserProfile", False)]
        regular_apps = [app for app in app_breakdown_data if not app.get("isBrowserProfile", False)]
        
        # Create pie chart with both types
        fig_pie_main = create_pie_chart(app_breakdown_data)
        st.plotly_chart(fig_pie_main, use_container_width=True)

        # Create detailed breakdown table
        df_app_top = pd.DataFrame(
            [
                {
                    "Activity/Application": app.get("appName", "N/A"),
                    "Type": "Browser Profile" if app.get("isBrowserProfile", False) else "Regular Activity",
                    "Time (min)": app.get("timeSpent", 0) // 60,
                    "Percentage": f"{app.get('percentage', 0):.1f}%",
                    "Category": app.get("categoryId", "N/A") if app.get("isBrowserProfile") else "N/A",
                    "Activity Count": app.get("activityCount", "N/A") if app.get("isBrowserProfile") else "N/A"
                }
                for app in app_breakdown_data[:15]  # Show top 15
            ]
        )
        st.dataframe(df_app_top, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        col1.metric("Regular Activities", len(regular_apps))
        col2.metric("Browser Profile Activities", len(browser_profile_apps))
        col3.metric("Total Activities", len(app_breakdown_data))
        
    else:
        st.info("No application usage data available for this date in the summary.")

    # --- Browser Usage Analysis (Regular Browser Usage - not profiles) ---
    st.subheader("🌐 Regular Browser Usage Analysis")
    st.caption("This shows browser usage that was not automatically categorized by profiles")
    
    # Filter out browser profile activities to show only regular browser usage
    regular_browser_data = [app for app in app_breakdown_data if not app.get("isBrowserProfile", False)]
    fig_browser_main = create_browser_chart(regular_browser_data)
    if fig_browser_main:
        st.plotly_chart(fig_browser_main, use_container_width=True)
    else:
        st.info("No regular browser usage detected (all browser activity was categorized by profiles).")

    # --- Raw Log Entries ---
    with st.expander("📄 View Raw Focus Log Entries (All Activities)"):
        raw_log_df = load_log_entries(selected_date)
        if not raw_log_df.empty:
            df_to_display = raw_log_df.copy()

            required_raw_cols = ["timestamp", "app_name", "title", "duration"]

            if all(col in df_to_display.columns for col in required_raw_cols):
                df_to_display["Formatted Time"] = pd.to_datetime(
                    df_to_display["timestamp"], errors="coerce"
                ).dt.strftime("%Y-%m-%d %H:%M:%S")

                display_data = {}
                if "Formatted Time" in df_to_display.columns:
                    display_data["Time"] = df_to_display["Formatted Time"]
                if "app_name" in df_to_display.columns:
                    display_data["App Name"] = df_to_display["app_name"]
                if "title" in df_to_display.columns:
                    display_data["Window Title"] = df_to_display["title"]
                if "duration" in df_to_display.columns:
                    display_data["Duration (s)"] = df_to_display["duration"]
                if "detected_profile_category" in df_to_display.columns:
                    display_data["Profile Category"] = df_to_display["detected_profile_category"].fillna("N/A")

                final_display_df = pd.DataFrame(display_data)

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

    # --- Personal Notes Quick View ---
    st.subheader("📝 Quick View: Recent Personal Notes for 5-Minute Blocks")
    dashboard_personal_notes = load_block_feedback()
    notes_for_selected_date = {
        k: v for k, v in dashboard_personal_notes.items() if k.startswith(selected_date)
    }
    if notes_for_selected_date:
        st.caption(
            "Last 3 personal notes for this date (full management on '📝 Summaries' tab):"
        )
        for ts, note_text in list(notes_for_selected_date.items())[-3:]:
            st.caption(
                f"_{pd.to_datetime(ts).strftime('%H:%M')}_: {note_text[:70]}{'...' if len(note_text) > 70 else ''}"
            )
    else:
        st.info(
            "No personal notes recorded for this date yet. Use the '📝 Summaries' tab to add notes."
        )

    st.markdown("---")
    st.markdown(
        """
    ### About Focus Monitor with Browser Profile Daily Aggregation
    This dashboard displays data collected by the Focus Monitor agent with new browser profile daily aggregation:
    
    **Key Features:**
    - **Browser Profile Detection**: Automatically categorizes browser windows based on visual color detection
    - **Daily Aggregation**: Browser profile activities are grouped by category for the entire day
    - **5-Minute Buckets**: Regular (non-browser-profile) activities are still organized in 5-minute time buckets
    - **Unified View**: See both regular activities and browser profile aggregations in one place
    
    **How it works:**
    1. The agent detects browser profile colors and automatically categorizes matching windows
    2. Browser profile activities accumulate daily by category (no 5-minute splitting)
    3. Regular activities continue to be organized in 5-minute buckets for detailed analysis
    4. The dashboard shows both views: daily browser aggregations and detailed 5-minute buckets
    
    **Navigation:**
    - **Labels**: Use the '🏷 Activity Label Editor' tab to categorize regular activities
    - **Browser Profiles**: Use the '🌐 Browser Profiles' tab to configure automatic browser categorization
    - **Summaries**: Use the '📝 Summaries' tab to review 5-minute block summaries (browser profiles excluded)
    - **Categories**: Use the '🏆 Activity Categories Manager' tab to define activity categories
    """
    )


# --- Control Panel Sidebar ---
def display_control_panel():
    st.sidebar.title("Focus Monitor Controls")
    
    # Tracker control
    st.sidebar.subheader("Tracker Status")
    if is_tracker_running():
        st.sidebar.success("✅ Focus Tracker Running")
        if st.sidebar.button("Stop Tracker"):
            stop_tracker()
            st.rerun()
    else:
        st.sidebar.warning("❌ Focus Tracker Not Running")
        if st.sidebar.button("Start Tracker"):
            start_tracker()
            st.rerun()
    
    # LLM Provider Selection
    st.sidebar.subheader("LLM Provider Settings")
    
    # Initialize session state defaults if not set
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = "ollama"
    if "ollama_api_url" not in st.session_state:
        st.session_state.ollama_api_url = DEFAULT_LLM_API_URL
    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = DEFAULT_LLM_MODEL
    if "openai_model" not in st.session_state:
        st.session_state.openai_model = DEFAULT_OPENAI_MODEL
    if "openai_models_fetched" not in st.session_state:
        st.session_state.openai_models_fetched = False
    if "available_openai_models" not in st.session_state:
        st.session_state.available_openai_models = get_available_openai_models()  # Default models
    if "ollama_models_fetched" not in st.session_state:
        st.session_state.ollama_models_fetched = False
    if "available_ollama_models" not in st.session_state:
        st.session_state.available_ollama_models = []
    if "ollama_model_display_info" not in st.session_state:
        st.session_state.ollama_model_display_info = {}
    
    # LLM provider selection
    provider_options = ["ollama", "openai"]
    selected_provider = st.sidebar.radio(
        "Select LLM Provider", 
        provider_options,
        index=provider_options.index(st.session_state.llm_provider),
        help="Choose between local Ollama models or OpenAI API",
        horizontal=True
    )
    
    # Update session state if changed
    if selected_provider != st.session_state.llm_provider:
        st.session_state.llm_provider = selected_provider
        st.rerun()
    
    # Provider-specific settings
    if selected_provider == "ollama":
        # Ollama settings
        ollama_api_url = st.sidebar.text_input(
            "Ollama API URL",
            value=st.session_state.ollama_api_url,
            help="URL of the Ollama API server (default: http://localhost:11434/api/generate)"
        )
        
        # Update API URL in session state if changed
        api_url_changed = False
        if ollama_api_url != st.session_state.ollama_api_url:
            st.session_state.ollama_api_url = ollama_api_url
            api_url_changed = True
            # Reset models_fetched flag when URL changes
            st.session_state.ollama_models_fetched = False
        
        # Fetch models if not fetched yet or URL changed
        if not st.session_state.ollama_models_fetched or api_url_changed:
            with st.sidebar.status("Fetching Ollama models..."):
                model_names, model_display_info = get_available_ollama_models(ollama_api_url)
                st.session_state.available_ollama_models = model_names
                st.session_state.ollama_model_display_info = model_display_info
                st.session_state.ollama_models_fetched = True
            
            # Set default model if current one isn't available
            if (st.session_state.ollama_model not in model_names and model_names):
                st.session_state.ollama_model = model_names[0]
        
        # Model selection with format display names
        if st.session_state.available_ollama_models:
            # Create formatted model display names
            format_model_name = lambda model: (
                f"{model} - {st.session_state.ollama_model_display_info[model]}" 
                if model in st.session_state.ollama_model_display_info 
                else model
            )
            
            formatted_model_names = [
                format_model_name(model) for model in st.session_state.available_ollama_models
            ]
            
            # Find index of current model
            current_index = 0
            for i, model_name in enumerate(st.session_state.available_ollama_models):
                if model_name == st.session_state.ollama_model:
                    current_index = i
                    break
            
            # Display the dropdown with formatted names
            selected_formatted_model = st.sidebar.selectbox(
                "Ollama Model",
                options=formatted_model_names,
                index=current_index,
                help="Select an Ollama model to use for summaries and categorization"
            )
            
            # Extract the actual model name from the formatted name
            selected_model = selected_formatted_model.split(" - ")[0] if " - " in selected_formatted_model else selected_formatted_model
            
            # Update model if changed
            if selected_model != st.session_state.ollama_model:
                st.session_state.ollama_model = selected_model
        else:
            st.sidebar.warning("No Ollama models found. Is Ollama running?")
        
        # Button to manually refresh models
        if st.sidebar.button("Refresh Ollama Models"):
            with st.sidebar.status("Refreshing models..."):
                model_names, model_display_info = get_available_ollama_models(ollama_api_url)
                st.session_state.available_ollama_models = model_names
                st.session_state.ollama_model_display_info = model_display_info
                st.session_state.ollama_models_fetched = True
                st.sidebar.success(f"Found {len(model_names)} available Ollama models")
                st.rerun()
            
    else:
        # OpenAI settings
        openai_api_key = st.sidebar.text_input(
            "OpenAI API Key",
            value=st.session_state.get("openai_api_key", ""),
            type="password",
            help="Your OpenAI API key (required for using OpenAI models)"
        )
        
        # Update API key in session state if changed
        api_key_changed = False
        if openai_api_key != st.session_state.get("openai_api_key", ""):
            st.session_state.openai_api_key = openai_api_key
            api_key_changed = True
            # Reset the models_fetched flag when API key changes
            st.session_state.openai_models_fetched = False
        
        # Fetch models from API if key provided and not fetched yet or key changed
        if openai_api_key and (not st.session_state.openai_models_fetched or api_key_changed):
            with st.sidebar.status("Fetching available models..."):
                # Try to fetch models from API
                api_models = get_openai_models_from_api(openai_api_key)
                
                if api_models:
                    st.session_state.available_openai_models = api_models
                    st.session_state.openai_models_fetched = True
                    # Success message will auto-dismiss
                else:
                    # If API call fails, use default models
                    if api_key_changed:  # Only show error on explicit key change
                        st.sidebar.error("Could not fetch models with this API key. Using default list.")
                        st.session_state.available_openai_models = get_available_openai_models()
        
        # Available OpenAI models - either from API or defaults
        openai_models = st.session_state.available_openai_models
        
        if openai_models:
            # Model selection
            openai_model = st.sidebar.selectbox(
                "OpenAI Model",
                options=openai_models,
                index=0 if st.session_state.openai_model not in openai_models else openai_models.index(st.session_state.openai_model),
                help="Select an OpenAI model to use for summaries and categorization"
            )
            
            # Update session state
            if openai_model != st.session_state.openai_model:
                st.session_state.openai_model = openai_model
        else:
            st.sidebar.warning("No models available. Please enter a valid API key.")
            
        # Button to manually refresh models
        if openai_api_key and st.sidebar.button("Refresh OpenAI Models"):
            with st.sidebar.status("Refreshing models..."):
                api_models = get_openai_models_from_api(openai_api_key)
                if api_models:
                    st.session_state.available_openai_models = api_models
                    st.session_state.openai_models_fetched = True
                    st.sidebar.success(f"Found {len(api_models)} available models")
                    st.rerun()
                else:
                    st.sidebar.error("Failed to refresh models. Check your API key and connection.")
            
        # Warning about costs
        st.sidebar.warning("⚠️ Using OpenAI models will incur API costs based on your usage.")
    
    # Test LLM connection
    if st.sidebar.button("Test LLM Connection"):
        with st.sidebar.status("Testing connection..."):
            success, message = test_llm_connection()
        
        if success:
            st.sidebar.success(f"✅ {message}")
        else:
            st.sidebar.error(f"❌ {message}")
    
    # Quick data info
    st.sidebar.subheader("Data Status")
    dates = load_available_dates()
    if dates:
        st.sidebar.info(f"📊 {len(dates)} days with data")
        st.sidebar.text(f"Latest: {dates[0] if dates else 'None'}")
    else:
        st.sidebar.warning("No log data found")
    
    # Path info
    st.sidebar.subheader("Storage Locations")
    st.sidebar.info(f"Logs Directory: {LOGS_DIR}")

# --- Main App ---
def main():
    st.set_page_config(page_title="Focus Monitor", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
    display_control_panel()
    tab_names = [
        "📊 Dashboard", 
        "🏷 Activity Label Editor", 
        "📝 5-Min Summaries", 
        "🏆 Categories", 
        "🌐 Browser Profiles",  # New tab
        "⏮️ Historical Processor", 
        "🧪 LLM Test"
    ]
    tabs = st.tabs(tab_names)
    
    with tabs[0]: display_dashboard()
    with tabs[1]: display_label_editor()
    with tabs[2]: display_time_bucket_summaries()
    with tabs[3]: display_category_manager()
    with tabs[4]: display_browser_profile_manager()  # New tab content
    with tabs[5]: display_retroactive_processor()
    with tabs[6]: llm_test_page()

if __name__ == "__main__":
    main()