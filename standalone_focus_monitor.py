#!/usr/bin/env python3
"""
Focus Monitor Agent (Window Title Tracking Version) - Modified for Browser Profile Daily Aggregation

This script tracks the title and executable of the currently focused window
to generate focus activity logs and daily summaries. It can capture screenshots
for certain applications and perform OCR on the images.
Runs completely standalone and saves all files to the script's directory.
Generates live summaries for the current day's activity.
Detects and groups browser profiles for Microsoft Edge.

Key Changes for Browser Profile Daily Aggregation:
- Browser profile activities are excluded from 5-minute buckets
- Browser profile activities are aggregated daily by category
- 5-minute buckets only contain non-browser-profile activities
- All other functionality preserved (OCR, focus scoring, classifications, etc.)

NOTE: LLM processing has been moved to the dashboard for better user control.
Browser profile configuration is now managed through the dashboard UI.
The agent now focuses purely on data collection and profile detection.
"""

import argparse
import asyncio
import datetime
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import warnings
import numpy as np  # Add this import for array operations with image data

# Filter out specific Streamlit warnings
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*Session state does not function.*")

# Attempt to import Tesseract and Pillow for OCR; fail gracefully if not present.
try:
    import pytesseract
    from PIL import ImageGrab, Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: pytesseract or Pillow not found. OCR features will be disabled.")
    print("Install with: pip install pytesseract Pillow")

# Attempt to import core Windows API, requests; these are critical.
try:
    import requests 
    import win32api
    import win32con 
    import win32gui
    import win32process
    # WMI is optional for advanced Chrome profile detection, can be removed if not used/working.
    # import wmi 
except ImportError:
    print("Critical packages (pywin32, requests) not found. Please install them.")
    print("pip install pywin32 requests")
    sys.exit(1)

# Get the directory where this script is located and add parent to path for dashboard imports
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# --- Conditional import for dashboard utilities ---
# Only import data_utils for basic functionality, not LLM utils
DASHBOARD_UTILS_AVAILABLE = False
try:
    from dashboard.data_utils import generate_summary_from_logs as generate_dashboard_summary_file
    DASHBOARD_UTILS_AVAILABLE = True
    print("INFO: Dashboard data utilities loaded successfully.")
except ImportError as e:
    DASHBOARD_UTILS_AVAILABLE = False
    print(f"WARNING: Dashboard data utilities not found or import error ({e}).")
    print("INFO: Agent will use fallback mechanisms for summaries.")
    print(f"INFO: Python sys.path: {sys.path}")

# Define dummy function if dashboard utils are not available
if not DASHBOARD_UTILS_AVAILABLE:
    def generate_dashboard_summary_file(date_str: str) -> Optional[Dict[str, Any]]:
        print(f"Fallback: generate_dashboard_summary_file called for {date_str} (dashboard_utils not available).")
        return None 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(SCRIPT_DIR / "focus_monitor.log", encoding="utf-8"),
        logging.StreamHandler(), # Also print to console
    ],
)
logger = logging.getLogger("focus_monitor_agent") # More specific logger name
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.state.session_state_proxy").setLevel(logging.ERROR)

# --- Configuration (Agent's own classification rules) ---
PRODUCTIVE_EXES = {"code.exe", "pycharm.exe", "idea.exe", "webstorm.exe", "goland.exe", "clion.exe", "word.exe", "excel.exe", "powerpnt.exe", "outlook.exe", "chrome.exe", "firefox.exe", "msedge.exe", "cmd.exe", "powershell.exe", "wt.exe", "explorer.exe", "obsidian.exe"}
DISTRACTION_EXES = {"steam.exe", "epicgameslauncher.exe", "origin.exe", "spotify.exe", "discord.exe", "slack.exe", "netflix.exe", "whatsapp.exe", "telegram.exe"} # Note: Slack can be dual-use
DISTRACTION_TITLE_KEYWORDS = {"youtube", "facebook", "twitter", "reddit", "netflix", "hulu", "twitch", "instagram", "9gag", "game", "play", "tiktok", "pinterest"}
MEETING_EXES = {"teams.exe", "zoom.exe", "webex.exe", "skype.exe"} # Note: Slack also used for meetings
MEETING_TITLE_KEYWORDS = {"meet", "meeting", "call", "webinar", "huddle", "zoom meeting", "microsoft teams meeting", "google meet"}
OCR_APPS = {"notepad.exe", "wordpad.exe", "acrord32.exe", "sumatrapdf.exe"} # Apps for which to attempt OCR

# --- Browser Profile Detection Patterns ---
EDGE_PROFILE_PATTERN = re.compile(r"Microsoft Edge(?:\s*-\s*(.+))?$")
CHROME_PROFILE_PATTERN = re.compile(r".*Google Chrome(?: - (?!New Tab|Settings|History|Downloads|Extensions)(.+))?$") # Improved to avoid common non-profile window parts
FIREFOX_PROFILE_PATTERN = re.compile(r"Mozilla Firefox(?:\s*-\s*(.+))?$")

# --- Set to True during setup/debugging, False in production ---
COLOR_DETECTION_DEBUG = True

# Browser profiles are now loaded from JSON configuration file managed by the dashboard
# No hardcoded profiles here anymore


class FocusMonitorAgent:
    def __init__(self, output_dir_str: Optional[str] = None, api_url_str: Optional[str] = None):
        self.output_dir = Path(output_dir_str) if output_dir_str else SCRIPT_DIR
        self.api_url = api_url_str
        self.active = True  # Internal current state
        self.desired_active_state = True  # State requested by backend (if API used)
        self.last_window_info: Optional[Dict] = None
        self.window_start_time: float = time.time()
        self.today: str = self._get_current_utc_date() # Store date as YYYY-MM-DD
        self.last_summary_time: float = 0 # Initialize to ensure first summary runs
        self.summary_interval: int = 300  # Generate summary every 5 minutes by default

        # Session start timestamp for 5-minute buckets
        self.session_start_time: float = time.time() # UTC timestamp for session start
        self.session_start_tag: str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.time_buckets: Dict[int, Dict[str, Any]] = {} # bucket_index: bucket_data
        
        # NEW: Daily browser profile aggregation
        self.daily_browser_activities: Dict[str, Dict[str, Any]] = {}  # category_id -> aggregated data
        
        # Track bucket finalization state (simplified since no LLM processing)
        self.finalized_buckets: Set[int] = set()  # Track which buckets have been finalized
        self.current_bucket_index: Optional[int] = None  # Track current active bucket

        # Create focus_logs directory within the script or specified directory
        self.focus_logs_dir = self.output_dir / "focus_logs"
        self.focus_logs_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir = self.focus_logs_dir / "screenshots" # For OCR images
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        
        self.ocr_last_times: Dict[int, float] = {} # hwnd: timestamp of last OCR
        self.ocr_results_cache: Dict[int, Dict[str, str]] = {} # hwnd: {"text": ..., "path": ...}

        # Browser profiles - loaded from configuration file managed by dashboard
        self.browser_profiles: List[Dict[str, Any]] = []
        self.last_profiles_load_time: float = 0
        self.profiles_reload_interval: int = 30  # Reload profiles every 30 seconds
        self._load_browser_profiles()

        logger.info(
            f"Initialized FocusMonitorAgent with Browser Profile Daily Aggregation. Output Dir: {self.focus_logs_dir}, API URL: {self.api_url or 'Disabled'}"
        )
        logger.info("LLM processing disabled in agent - handled by dashboard for user control")
        logger.info(f"Loaded {len(self.browser_profiles)} browser profiles for color detection")
        if not OCR_AVAILABLE:
            logger.warning("OCR features disabled as pytesseract/Pillow are not available.")

    def _load_browser_profiles(self):
        """Load browser profiles from JSON configuration file managed by dashboard."""
        profiles_file = self.focus_logs_dir / "browser_profiles.json"
        
        try:
            if profiles_file.exists():
                with open(profiles_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    profiles = data.get("profiles", [])
                    
                    # Filter to only enabled profiles and convert lists back to Tuples where needed
                    self.browser_profiles = []
                    for profile in profiles:
                        if profile.get("enabled", True):  # Default to enabled if not specified
                            # Convert color_rgb and search_rect from lists to Tuples for consistency
                            processed_profile = profile.copy()
                            processed_profile["color_rgb"] = tuple(profile.get("color_rgb", [128, 128, 128]))
                            processed_profile["search_rect"] = tuple(profile.get("search_rect", [10, 5, 25, 25]))
                            self.browser_profiles.append(processed_profile)
                    
                    self.last_profiles_load_time = time.time()
                    logger.debug(f"Loaded {len(self.browser_profiles)} enabled browser profiles from {profiles_file}")
            else:
                # Create default profile file if it doesn't exist
                self._create_default_profiles_file(profiles_file)
                    
        except Exception as e:
            logger.error(f"Error loading browser profiles from {profiles_file}: {e}")
            # Fall back to empty profiles list
            self.browser_profiles = []

    def _create_default_profiles_file(self, profiles_file: Path):
        """Create a default browser profiles file."""
        default_profiles = {
            "profiles": [
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
        }
        
        try:
            with open(profiles_file, "w", encoding="utf-8") as f:
                json.dump(default_profiles, f, indent=2)
            logger.info(f"Created default browser profiles file: {profiles_file}")
            # Load the profiles we just created
            self._load_browser_profiles()
        except Exception as e:
            logger.error(f"Failed to create default browser profiles file: {e}")
            self.browser_profiles = []

    def _reload_browser_profiles_if_needed(self):
        """Reload browser profiles if enough time has passed since last load."""
        if time.time() - self.last_profiles_load_time >= self.profiles_reload_interval:
            old_count = len(self.browser_profiles)
            self._load_browser_profiles()
            new_count = len(self.browser_profiles)
            
            if new_count != old_count:
                logger.info(f"Browser profiles reloaded: {old_count} -> {new_count} profiles")

    def _get_monitor_info(self) -> List[Dict[str, int]]:
        """Get monitor information using Windows API."""
        monitors = []
        try:
            def enum_display_monitors_callback(hmonitor, hdc, rect, data):
                monitors.append({
                    'left': rect[0],
                    'top': rect[1], 
                    'right': rect[2],
                    'bottom': rect[3],
                    'width': rect[2] - rect[0],
                    'height': rect[3] - rect[1]
                })
                return True
            
            # Fix the EnumDisplayMonitors call - it takes 4 arguments, not 2
            win32api.EnumDisplayMonitors(None, None, enum_display_monitors_callback, None)
            return monitors
        except Exception:
            return []

    def analyze_color_region(self, hwnd, target_color_rgb, search_rect, tolerance=20):
        """
        Robust color detection using multiple capture methods with Method 3a prioritized.
        
        Args:
            hwnd: Window handle
            target_color_rgb: Tuple of (R, G, B) values to search for
            search_rect: Tuple of (x, y, width, height) defining search region relative to window
            tolerance: How much RGB values can differ and still match (0-255)
            
        Returns:
            bool: True if the color was found in the region, False otherwise
        """
        if not OCR_AVAILABLE:
            return False
            
        try:
            # Get window position and calculate coordinates
            window_rect = win32gui.GetWindowRect(hwnd)
            
            # Calculate absolute coordinates of search region
            x1 = window_rect[0] + search_rect[0]
            y1 = window_rect[1] + search_rect[1]
            x2 = x1 + search_rect[2]
            y2 = y1 + search_rect[3]
            
            capture_bbox_screen = (x1, y1, x2, y2)
            
            # Verify the coordinates are valid
            if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0:
                return False
                
            # Get monitor information
            monitors = self._get_monitor_info()
            region_img = None
            
            # Method 3a: Try all_screens=True first as it's most robust for multi-monitor
            if not region_img:
                grab_all_successful = False
                try:
                    if monitors:
                        min_x_all = min(m['left'] for m in monitors)
                        min_y_all = min(m['top'] for m in monitors)
                    else:
                        min_x_all = 0 
                        min_y_all = 0

                    full_virtual_desktop_img = ImageGrab.grab(bbox=None, include_layered_windows=False, all_screens=True)
                    
                    crop_x1_virtual = capture_bbox_screen[0] - min_x_all
                    crop_y1_virtual = capture_bbox_screen[1] - min_y_all
                    crop_x2_virtual = capture_bbox_screen[2] - min_x_all
                    crop_y2_virtual = capture_bbox_screen[3] - min_y_all
                    
                    final_crop_box_virtual = (crop_x1_virtual, crop_y1_virtual, crop_x2_virtual, crop_y2_virtual)

                    if (final_crop_box_virtual[0] >= 0 and final_crop_box_virtual[1] >= 0 and
                        final_crop_box_virtual[2] <= full_virtual_desktop_img.width and
                        final_crop_box_virtual[3] <= full_virtual_desktop_img.height and
                        final_crop_box_virtual[2] > final_crop_box_virtual[0] and
                        final_crop_box_virtual[3] > final_crop_box_virtual[1]):
                        region_img = full_virtual_desktop_img.crop(final_crop_box_virtual)
                        grab_all_successful = True

                except (TypeError, AttributeError):
                    pass
                except Exception:
                    pass
                
                if grab_all_successful and region_img:
                    img_array_check = np.array(region_img)
                    if img_array_check.size == 0 or (img_array_check.ndim == 3 and img_array_check[:,:,:3].sum() == 0):
                        region_img = None

            # Method 1: Standard PIL ImageGrab with specific bbox (fallback)
            if not region_img:
                try:
                    region_img = ImageGrab.grab(bbox=capture_bbox_screen)
                    
                    if region_img and region_img.size[0] > 0 and region_img.size[1] > 0:
                        img_array_check = np.array(region_img)
                        if img_array_check.size == 0 or (img_array_check.ndim == 3 and img_array_check[:,:,:3].sum() == 0):
                            region_img = None
                    else:
                        region_img = None
                        
                except Exception:
                    region_img = None
            
            # Method 2: Try grabbing entire primary screen first, then crop (fallback)
            if not region_img:
                try:
                    full_primary_img = ImageGrab.grab()
                    
                    primary_crop_x1 = max(0, capture_bbox_screen[0])
                    primary_crop_y1 = max(0, capture_bbox_screen[1])
                    primary_crop_x2 = min(full_primary_img.width, capture_bbox_screen[2])
                    primary_crop_y2 = min(full_primary_img.height, capture_bbox_screen[3])
                    
                    primary_crop_box = (primary_crop_x1, primary_crop_y1, primary_crop_x2, primary_crop_y2)
                    
                    if (primary_crop_x2 > primary_crop_x1 and primary_crop_y2 > primary_crop_y1 and
                        primary_crop_x1 >= 0 and primary_crop_y1 >= 0):
                        region_img = full_primary_img.crop(primary_crop_box)
                        
                        img_array_check = np.array(region_img)
                        if img_array_check.size == 0 or (img_array_check.ndim == 3 and img_array_check[:,:,:3].sum() == 0):
                            region_img = None
                        
                except Exception:
                    region_img = None

            # Method 3b: Try all monitors capture (if 3a failed and EnumDisplayMonitors works)
            if not region_img and monitors:
                try:
                    min_x_all = min(m['left'] for m in monitors)
                    min_y_all = min(m['top'] for m in monitors)
                    max_x_all = max(m['right'] for m in monitors)
                    max_y_all = max(m['bottom'] for m in monitors)
                    
                    all_monitors_rect = (min_x_all, min_y_all, max_x_all, max_y_all)
                    full_all_monitors_img = ImageGrab.grab(bbox=all_monitors_rect)
                    
                    crop_x1_all = capture_bbox_screen[0] - min_x_all
                    crop_y1_all = capture_bbox_screen[1] - min_y_all
                    crop_x2_all = capture_bbox_screen[2] - min_x_all
                    crop_y2_all = capture_bbox_screen[3] - min_y_all
                    
                    final_crop_box_all = (crop_x1_all, crop_y1_all, crop_x2_all, crop_y2_all)
                    
                    if (final_crop_box_all[0] >= 0 and final_crop_box_all[1] >= 0 and
                        final_crop_box_all[2] <= full_all_monitors_img.width and
                        final_crop_box_all[3] <= full_all_monitors_img.height and
                        final_crop_box_all[2] > final_crop_box_all[0] and
                        final_crop_box_all[3] > final_crop_box_all[1]):
                        region_img = full_all_monitors_img.crop(final_crop_box_all)
                        
                        img_array_check = np.array(region_img)
                        if img_array_check.size == 0 or (img_array_check.ndim == 3 and img_array_check[:,:,:3].sum() == 0):
                            region_img = None
                        
                except Exception:
                    region_img = None
            
            if not region_img:
                return False
                
            # Save debug image if debug is enabled
            if COLOR_DETECTION_DEBUG:
                debug_filename = f"color_debug_{int(time.time())}_{hwnd}.png"
                debug_filepath = self.screenshot_dir / debug_filename
                region_img.save(debug_filepath)
            
            # Convert to numpy array for color analysis
            img_array = np.array(region_img)
            
            # Add minimum percentage threshold to avoid false positives from random pixels
            MIN_MATCH_PERCENTAGE = 1.0  # At least 1% of pixels must match
            
            r_match = np.abs(img_array[:,:,0] - target_color_rgb[0]) <= tolerance
            g_match = np.abs(img_array[:,:,1] - target_color_rgb[1]) <= tolerance
            b_match = np.abs(img_array[:,:,2] - target_color_rgb[2]) <= tolerance
            
            # Combine the matches - we need all three channels to match
            color_match = np.logical_and(r_match, np.logical_and(g_match, b_match))
            
            # Calculate percentage of matching pixels
            total_pixels = color_match.size
            matching_pixels = np.sum(color_match)
            match_percentage = (matching_pixels / total_pixels) * 100
            
            # Check if we have enough matching pixels to be confident
            has_any_match = np.any(color_match)
            meets_threshold = match_percentage >= MIN_MATCH_PERCENTAGE
            
            # Final result: must have matches AND meet minimum percentage
            result = has_any_match and meets_threshold
            
            return result
                
        except Exception:
            return False

    def _check_color_profiles_for_window(self, hwnd, exe_path, window_title):
        """
        Checks if the window matches any of the color profiles and returns the matching category ID.
        First verifies that we're dealing with a supported browser before attempting color detection.
        When a color match is found, the window is automatically categorized.
        
        Args:
            hwnd: Window handle
            exe_path: Path to the executable
            window_title: Window title
            
        Returns:
            str: Category ID from matching profile, or empty string if no match
        """
        # Reload profiles periodically to pick up changes from dashboard
        self._reload_browser_profiles_if_needed()
        
        if not self.browser_profiles:
            return ""  # No profiles configured
        
        exe_basename = os.path.basename(exe_path).lower()
        
        # Find matching profiles for this executable
        matching_profiles = [
            p for p in self.browser_profiles 
            if p.get("exe_pattern", "").lower() == exe_basename
        ]
        
        if not matching_profiles:
            return ""  # No profiles for this browser
        
        for profile in matching_profiles:
            # Check if the color is present in the defined region using the robust method
            if self.analyze_color_region(
                hwnd, 
                profile["color_rgb"], 
                profile["search_rect"], 
                profile.get("color_tolerance", 20)
            ):
                # Color match found - automatically categorize this window
                category_id = profile.get("category_id", "")
                profile_name = profile.get("name", "Unknown Profile")
                logger.info(f"🎯 COLOR MATCH: Window '{window_title[:50]}' automatically categorized as '{category_id}' based on profile '{profile_name}'")
                return category_id
        
        return ""  # No match found

    def _get_current_utc_date(self) -> str:
        # Use UTC date for consistency across timezones
        return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

    async def check_backend_status(self):
        """Periodically check the desired active state from the backend (if API URL is set)."""
        while True:
            if not self.api_url:
                self.desired_active_state = True  # Default to active if no API
                await asyncio.sleep(60) # Check less frequently if no API
                continue
            try:
                response = requests.get(f"{self.api_url}/focus/status", timeout=5) # Increased timeout
                response.raise_for_status()
                new_desired_state = response.json().get("active", True)
                if self.desired_active_state != new_desired_state:
                    logger.info(
                        f"Backend desired state changed from '{self.desired_active_state}' to '{new_desired_state}'"
                    )
                    self.desired_active_state = new_desired_state
            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"Could not reach backend API ({self.api_url}) to check focus status: {e}"
                )
            except Exception as e: # Catch other potential errors like JSONDecodeError
                logger.error(f"Error checking backend status via API: {e}")
            await asyncio.sleep(15)  # Check every 15 seconds

    def toggle_active(self):
        """Toggle the internal active state if it differs from desired state."""
        if self.active == self.desired_active_state:
            return # No change needed

        new_internal_state = self.desired_active_state
        logger.info(f"Focus Monitor internal active state changing to: {new_internal_state}")

        if not new_internal_state:  # Pausing
            if self.last_window_info: # Log any pending activity
                duration = int(time.time() - self.window_start_time)
                if duration > 0:
                    self._log_window_activity(self.last_window_info, duration)
            self.last_window_info = None # Clear last window info when paused
            # self.window_start_time = time.time() # Reset start time, or keep it to measure pause duration?
                                               # For now, reset as if a new window (idle) started.
        else:  # Resuming
            pass # window_start_time will be set when a new window is focused
        
        self.window_start_time = time.time() # Reset window timer on any state change
        self.active = new_internal_state

    def _extract_browser_profile(self, exe_path: str, window_title: str, process_id: int = 0) -> Optional[str]:
        """Extract browser profile information from window title or process details."""
        exe_basename_lower = os.path.basename(exe_path).lower()
        
        if "msedge.exe" == exe_basename_lower:
            match = EDGE_PROFILE_PATTERN.search(window_title)
            if match and match.group(1): return match.group(1).strip()
        elif "chrome.exe" == exe_basename_lower:
            # WMI based profile detection can be complex and permission-heavy.
            # Title-based is often sufficient and more portable.
            # If you need WMI:
            # if process_id > 0:
            #     profile_from_pid = self._get_chrome_profile_from_pid_wmi(process_id) # Ensure WMI is imported if used
            #     if profile_from_pid: return profile_from_pid
            match = CHROME_PROFILE_PATTERN.search(window_title) # Uses improved regex
            if match and match.group(1):
                return match.group(1).strip()
        elif "firefox.exe" == exe_basename_lower:
            match = FIREFOX_PROFILE_PATTERN.search(window_title)
            if match and match.group(1): return match.group(1).strip()
        return None

    def _get_app_identity(self, exe_path: str, window_title: str, process_id: int = 0) -> Tuple[str, str]:
        """
        Generate a consistent app identity string. For browsers, this may include profile info.
        Returns (app_identity_string, original_exe_path).
        """
        profile_name = self._extract_browser_profile(exe_path, window_title, process_id)
        
        # Default app name is the executable basename without .exe
        app_basename = os.path.basename(exe_path).lower()
        app_name_without_ext = app_basename.replace(".exe", "") if app_basename.endswith(".exe") else app_basename
        
        if app_name_without_ext == "unknown": # Handle cases where exe path couldn't be determined
            return "Unknown Application", exe_path

        # Check for color profile match for browsers - only for supported browsers
        if process_id > 0 and any(p.get("exe_pattern", "").lower() == app_basename for p in self.browser_profiles):
            try:
                hwnd = win32gui.GetForegroundWindow()  # Use the current foreground window
                profile_category_id = self._check_color_profiles_for_window(hwnd, exe_path, window_title)
                if profile_category_id:
                    # Find the profile name from the category
                    matching_profile = next(
                        (p for p in self.browser_profiles if p.get("category_id") == profile_category_id), 
                        None
                    )
                    if matching_profile:
                        detected_profile_name = matching_profile.get("name", "")
                        return f"{app_name_without_ext} - {detected_profile_name}", exe_path
            except Exception as e:
                logger.debug(f"Error in color profile detection for app identity: {e}")

        # For browsers with profile info, append the profile to the app name
        if profile_name:
            # Check if it's a known browser where profiles are significant
            known_browsers = ["chrome", "msedge", "firefox"]
            if any(browser_name in app_name_without_ext for browser_name in known_browsers):
                return f"{app_name_without_ext} - {profile_name}", exe_path

        # If no specific profile, just return the base app name
        return app_name_without_ext, exe_path

    def _get_focused_window_details(self) -> Optional[Dict[str, Any]]:
        """Get details (hwnd, pid, exe, title, timestamp, etc.) for the currently focused window."""
        try:
            hwnd = win32gui.GetForegroundWindow()
            if not hwnd: return None # No window in foreground

            window_title = win32gui.GetWindowText(hwnd)
            # GetWindowThreadProcessId returns (threadId, processId)
            _thread_id, process_id = win32process.GetWindowThreadProcessId(hwnd)

            # Filter out uninteresting windows (e.g., desktop, lock screen)
            if not window_title or process_id == 0 or window_title in [
                "Program Manager", "Windows Default Lock Screen", "Windows Input Experience", "Action center"
            ]:
                return None 

            exe_full_path = "Unknown"
            try:
                #PROCESS_QUERY_LIMITED_INFORMATION is safer if available and often sufficient
                h_process = win32api.OpenProcess(win32con.PROCESS_QUERY_LIMITED_INFORMATION | win32con.PROCESS_VM_READ, False, process_id)
                if h_process:
                    try:
                        exe_full_path = win32process.GetModuleFileNameEx(h_process, 0)
                    finally:
                        win32api.CloseHandle(h_process)
            except Exception as e_proc:
                # logger.debug(f"Could not get exe for pid {process_id} (title: {window_title[:30]}...): {e_proc}")
                pass # Silently pass if exe cannot be retrieved, will log "Unknown"

            # Attempt OCR if applicable and enabled
            ocr_text_content = None
            screenshot_file_rel_path = None # Relative path to save in log

            if OCR_AVAILABLE and os.path.basename(exe_full_path).lower() in OCR_APPS:
                last_ocr_capture_time = self.ocr_last_times.get(hwnd, 0)
                # OCR same window at most every 10 seconds to reduce overhead
                if time.time() - last_ocr_capture_time >= 10:
                    try:
                        # Capture only the active window's bounding box for better OCR accuracy
                        window_rect = win32gui.GetWindowRect(hwnd)
                        # ImageGrab.grab can take a bbox argument. Ensure it's not an empty rect.
                        if window_rect[2] > window_rect[0] and window_rect[3] > window_rect[1]:
                            captured_image = ImageGrab.grab(bbox=window_rect)
                            # Create a unique filename for the screenshot
                            timestamp_filename_part = int(time.time())
                            hwnd_filename_part = hwnd if hwnd else "nohwnd" # Handle if hwnd is 0 (unlikely here)
                            screenshot_filename = f"{timestamp_filename_part}_{hwnd_filename_part}.png"
                            
                            full_screenshot_path = self.screenshot_dir / screenshot_filename
                            captured_image.save(full_screenshot_path)
                            
                            ocr_text_content = pytesseract.image_to_string(captured_image)
                            screenshot_file_rel_path = str(Path("screenshots") / screenshot_filename) # Store relative path

                            self.ocr_last_times[hwnd] = time.time()
                            # Cache result in case window changes before next log write
                            self.ocr_results_cache[hwnd] = {"text": ocr_text_content, "path": screenshot_file_rel_path}
                            logger.debug(f"OCR captured for '{window_title[:30]}...': {ocr_text_content[:50]}...")
                    except Exception as e_ocr_capture:
                        logger.error(f"OCR capture failed for '{window_title[:30]}...': {e_ocr_capture}")
            
            # Get the app identity (potentially with profile)
            app_identity_name, _ = self._get_app_identity(exe_full_path, window_title, process_id)

            return {
                "hwnd": hwnd, "pid": process_id, 
                "exe": exe_full_path, # Store the full original exe path
                "title": window_title,
                "app_name_identity": app_identity_name, # This is the name used for initial logging (app_name field)
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "ocr_text": ocr_text_content, # From current capture attempt, if any
                "screenshot_path": screenshot_file_rel_path, # Relative path, if any
            }
        except Exception as e_main_details:
            # Handle specific pywintypes errors that occur if a window closes during query
            if "pywintypes.error" in str(type(e_main_details)) and hasattr(e_main_details, 'args') and e_main_details.args and e_main_details.args[0] in [0, 5, 1400]: # 5: Access Denied, 1400: Invalid window handle
                logger.debug(f"Window likely closed or inaccessible during info retrieval: {e_main_details}")
            else:
                logger.error(f"Error getting focused window details: {e_main_details}", exc_info=False) # exc_info=False to reduce log noise for common errors
            return None

    def _log_window_activity(self, window_info: Dict[str, Any], duration_seconds: int):
        """Log focused window activity to a JSONL file and handle browser profile aggregation."""
        if duration_seconds <= 0: return
        try:
            # focus_logs_dir is already ensured by __init__
            log_file_path = self.focus_logs_dir / f"focus_log_{self.today}.jsonl"

            # The 'app_name' in the log should be the identity potentially including profile.
            # This is used by dashboard_utils.generate_summary_from_logs for initial aggregation.
            # The dashboard's apply_labels_to_logs will then create 'original_app_name' and a new 'app_name' based on labels.
            app_name_to_log = window_info.get("app_name_identity", os.path.basename(window_info.get("exe","Unknown")).replace(".exe",""))

            log_entry: Dict[str, Any] = {
                "timestamp": window_info["timestamp"],
                "exe": window_info["exe"], # Full original exe path
                "app_name": app_name_to_log, # This is the name used by dashboard for initial grouping
                "title": window_info["title"],
                "duration": duration_seconds,
                "pid": window_info.get("pid", 0),
            }

            # Check for color profile match if it's a browser window
            hwnd = window_info.get("hwnd")
            profile_category_id = ""
            is_browser_profile_activity = False
            
            if hwnd:
                profile_category_id = self._check_color_profiles_for_window(
                    hwnd, window_info["exe"], window_info["title"]
                )
                if profile_category_id:
                    is_browser_profile_activity = True
                    # Store the detected profile category in the log entry for later use
                    log_entry["detected_profile_category"] = profile_category_id
                    # Also update the app_name to reflect the detected category
                    matching_profile = next(
                        (p for p in self.browser_profiles if p.get("category_id") == profile_category_id), 
                        None
                    )
                    if matching_profile:
                        profile_name = matching_profile.get("name", profile_category_id)
                        log_entry["app_name"] = f"{os.path.basename(window_info['exe']).replace('.exe', '')} - {profile_name}"
                        logger.info(f"🎯 BROWSER PROFILE: Window automatically categorized and logged as '{log_entry['app_name']}' with category '{profile_category_id}'")

            # Add OCR info if available from current capture or cache
            hwnd = window_info.get("hwnd")
            cached_ocr = self.ocr_results_cache.pop(hwnd, None) if hwnd else None
            
            current_ocr_text = window_info.get("ocr_text")
            current_screenshot_path = window_info.get("screenshot_path")

            if current_ocr_text: log_entry["ocr_text"] = current_ocr_text
            elif cached_ocr: log_entry["ocr_text"] = cached_ocr.get("text")
            
            if current_screenshot_path: log_entry["screenshot_path"] = current_screenshot_path
            elif cached_ocr: log_entry["screenshot_path"] = cached_ocr.get("path")

            # Clean up OCR timestamp cache if entry processed
            if hwnd in self.ocr_last_times: self.ocr_last_times.pop(hwnd, None)

            # Always log to JSONL file for complete record
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

            # Log a snippet for quick console feedback
            title_snippet = log_entry["title"][:60].replace("\n", " ") + ("..." if len(log_entry["title"]) > 60 else "")
            logger.info(f"Logged: {log_entry['app_name']} - '{title_snippet}' ({duration_seconds}s)")

            # NEW: Handle browser profile daily aggregation vs 5-minute buckets
            if is_browser_profile_activity:
                self._update_daily_browser_aggregation(log_entry, profile_category_id)
            else:
                # Update in-memory time bucket with this activity (only non-browser profile activities)
                self._update_time_bucket(log_entry) # Pass the full log_entry

        except Exception as e_log:
            logger.error(f"Error writing to log file {log_file_path}: {e_log}")

    def _update_daily_browser_aggregation(self, log_entry_data: Dict[str, Any], category_id: str):
        """Aggregate browser profile activities daily by category."""
        try:
            # Initialize category aggregation if not exists
            if category_id not in self.daily_browser_activities:
                self.daily_browser_activities[category_id] = {
                    "category_id": category_id,
                    "total_duration": 0,
                    "titles": set(),
                    "ocr_text": set(),
                    "app_names": set(),
                    "start_time": log_entry_data["timestamp"],
                    "last_activity": log_entry_data["timestamp"],
                    "activity_count": 0
                }
            
            # Update aggregation
            agg = self.daily_browser_activities[category_id]
            agg["total_duration"] += log_entry_data.get("duration", 0)
            agg["activity_count"] += 1
            agg["last_activity"] = log_entry_data["timestamp"]
            
            if log_entry_data.get("title"):
                agg["titles"].add(log_entry_data["title"])
            if log_entry_data.get("ocr_text"):
                agg["ocr_text"].add(log_entry_data["ocr_text"])
            if log_entry_data.get("app_name"):
                agg["app_names"].add(log_entry_data["app_name"])
            
            logger.debug(f"Updated daily browser aggregation for category '{category_id}': {agg['total_duration']}s total, {agg['activity_count']} activities")
            
        except Exception as e:
            logger.error(f"Error updating daily browser aggregation: {e}")

    def _update_time_bucket(self, log_entry_data: Dict[str, Any]):
        """Update the in-memory 5-minute bucket with a new log entry (NO LLM processing, excludes browser profiles)."""
        try:
            # Timestamp from log_entry is already UTC ISO format
            log_ts = datetime.datetime.fromisoformat(log_entry_data["timestamp"]).timestamp() # Convert to UNIX timestamp
        except ValueError:
            log_ts = time.time() # Fallback if ISO format is wrong

        # Calculate bucket index based on session start time
        bucket_index = int((log_ts - self.session_start_time) // 300) # 300 seconds = 5 minutes
        
        # Check if we've moved to a new bucket
        if self.current_bucket_index is not None and bucket_index != self.current_bucket_index:
            # We've moved to a new bucket, so finalize the previous one
            self._finalize_bucket(self.current_bucket_index)
        
        self.current_bucket_index = bucket_index
        
        # Define bucket start and end times based on the index
        current_bucket_start_ts = self.session_start_time + (bucket_index * 300)
        current_bucket_end_ts = current_bucket_start_ts + 300

        # Retrieve or initialize the bucket
        bucket_data = self.time_buckets.setdefault(
            bucket_index,
            {
                "start": datetime.datetime.fromtimestamp(current_bucket_start_ts, tz=datetime.timezone.utc).isoformat(),
                "end": datetime.datetime.fromtimestamp(current_bucket_end_ts, tz=datetime.timezone.utc).isoformat(),
                "apps": set(), "titles": set(), "ocr_text": set(), 
                "summary": "", "category_id": "" # Empty - to be filled by dashboard
            }
        )

        # Add data from current log entry to the bucket's sets
        bucket_data["apps"].add(log_entry_data.get("app_name", "Unknown")) # app_name includes profile if any
        if log_entry_data.get("title"): bucket_data["titles"].add(log_entry_data["title"])
        if log_entry_data.get("ocr_text"): bucket_data["ocr_text"].add(log_entry_data["ocr_text"])

    def _finalize_bucket(self, bucket_index: int):
        """Mark a bucket as finalized (no LLM processing in agent)."""
        if bucket_index in self.finalized_buckets:
            return  # Already finalized
            
        if bucket_index not in self.time_buckets:
            return  # Bucket doesn't exist
            
        bucket_data = self.time_buckets[bucket_index]
        
        # Mark as finalized (no LLM processing needed in agent)
        self.finalized_buckets.add(bucket_index)
        
        logger.info(f"Finalized data collection for bucket {bucket_index} ({bucket_data.get('start', '')}) - LLM processing deferred to dashboard")

    def _write_time_bucket_summary(self):
        """Write the current session's time bucket data to a JSON file (no LLM summaries, excludes browser profiles)."""
        if not self.time_buckets:
            logger.info("No time buckets to write for current session.")
            return

        # Filename includes the session start tag to differentiate sessions
        bucket_file_path = self.focus_logs_dir / f"time_buckets_{self.session_start_tag}.json"
        
        serializable_buckets_list = []
        for _bucket_idx in sorted(self.time_buckets.keys()): # Process in chronological order
            bucket_content = self.time_buckets[_bucket_idx]
            serializable_buckets_list.append({
                "start": bucket_content["start"], 
                "end": bucket_content["end"],
                "apps": sorted(list(bucket_content.get("apps", set()))), # Convert sets to sorted lists
                "titles": sorted(list(bucket_content.get("titles", set()))),
                "ocr_text": sorted(list(bucket_content.get("ocr_text", set()))),
                "summary": bucket_content.get("summary", ""), # Will be empty until dashboard processes
                "category_id": bucket_content.get("category_id", "") # May be set by color detection
            })

        try:
            with open(bucket_file_path, "w", encoding="utf-8") as f:
                json.dump(serializable_buckets_list, f, indent=2)
            logger.info(f"Wrote {len(serializable_buckets_list)} time bucket data entries to {bucket_file_path} (summaries to be generated by dashboard, browser profiles excluded)")
        except Exception as e_write_bucket:
            logger.error(f"Error writing time bucket data file {bucket_file_path}: {e_write_bucket}")

    def _write_daily_browser_aggregation(self):
        """Write the current day's browser profile aggregations to a JSON file."""
        if not self.daily_browser_activities:
            logger.info("No browser profile activities to write for current day.")
            return

        browser_file_path = self.focus_logs_dir / f"daily_browser_activities_{self.today}.json"
        
        serializable_browser_data = []
        for category_id, agg_data in self.daily_browser_activities.items():
            serializable_browser_data.append({
                "category_id": category_id,
                "total_duration": agg_data["total_duration"],
                "titles": sorted(list(agg_data.get("titles", set()))),
                "ocr_text": sorted(list(agg_data.get("ocr_text", set()))),
                "app_names": sorted(list(agg_data.get("app_names", set()))),
                "start_time": agg_data["start_time"],
                "last_activity": agg_data["last_activity"],
                "activity_count": agg_data["activity_count"]
            })

        try:
            with open(browser_file_path, "w", encoding="utf-8") as f:
                json.dump(serializable_browser_data, f, indent=2)
            logger.info(f"Wrote {len(serializable_browser_data)} daily browser profile aggregations to {browser_file_path}")
        except Exception as e_write_browser:
            logger.error(f"Error writing daily browser aggregation file {browser_file_path}: {e_write_browser}")

    def _generate_daily_summary(self, date_to_summarize: Optional[str] = None):
        """
        Generate a daily summary using dashboard utilities if available.
        The agent focuses on basic metrics, letting the dashboard handle LLM features.
        """
        target_date_str = date_to_summarize if date_to_summarize else self.today
        logger.info(f"Generating daily summary for {target_date_str}...")

        # Write browser aggregations before generating summary
        if target_date_str == self.today:
            self._write_daily_browser_aggregation()

        # Attempt to generate (or load if already generated by dashboard) the base summary
        base_summary_dict = None
        if DASHBOARD_UTILS_AVAILABLE:
            logger.info(f"Using dashboard.data_utils to generate base summary for {target_date_str}")
            base_summary_dict = generate_dashboard_summary_file(target_date_str) 
        
        if not base_summary_dict:
            logger.warning(f"Base summary from dashboard_utils failed or unavailable for {target_date_str}.")
            logger.info(f"Agent will attempt to create a minimal summary from raw logs for {target_date_str} if they exist.")
            # Minimal fallback if dashboard_utils didn't produce anything (e.g. no logs)
            log_file_for_check = self.focus_logs_dir / f"focus_log_{target_date_str}.jsonl"
            if not log_file_for_check.exists():
                logger.error(f"No log file found at {log_file_for_check}. Cannot generate any summary for {target_date_str}.")
                return None # Cannot proceed
            # If logs exist but dashboard_utils failed severely, create a placeholder.
            base_summary_dict = {"date": target_date_str, "totalTime": 0, "appBreakdown": []}

        # Work with a copy of the base_summary to enhance it
        summary_to_enhance = base_summary_dict.copy()

        # --- Add/Overwrite Agent-Specific Metrics using RAW log data for agent's perspective ---
        raw_log_file_path = self.focus_logs_dir / f"focus_log_{target_date_str}.jsonl"
        if not raw_log_file_path.exists():
            logger.warning(f"Raw log file {raw_log_file_path.name} missing. Cannot calculate agent-specific metrics. Using base summary as is.")
            return summary_to_enhance 

        agent_perspective_log_entries: List[Dict[str, Any]] = []
        try:
            with open(raw_log_file_path, "r", encoding="utf-8") as f_raw:
                for line in f_raw:
                    try:
                        entry = json.loads(line.strip())
                        # Basic validation for agent's needs
                        if isinstance(entry, dict) and all(k in entry for k in ["exe", "title", "duration", "app_name"]):
                            agent_perspective_log_entries.append(entry)
                    except json.JSONDecodeError: continue # Skip malformed lines
        except Exception as e_read_raw:
            logger.error(f"Error reading raw log file {raw_log_file_path.name} for agent metrics: {e_read_raw}")
            return summary_to_enhance # Return base summary if raw logs can't be read

        if not agent_perspective_log_entries:
            logger.info(f"No valid raw log entries in {raw_log_file_path.name} for agent-specific metrics. Using base summary.")
            return summary_to_enhance

        # Agent's calculation of distraction events (e.g., number of distinct activities or switches)
        summary_to_enhance["distractionEvents"] = len(agent_perspective_log_entries) # Simplistic: count of all entries

        # Agent's calculation of meeting time based on its own rules
        summary_to_enhance["meetingTime"] = round(sum(
            e["duration"] for e in agent_perspective_log_entries if self._is_meeting_app(e["exe"], e["title"])
        ))

        # Agent's classification of productive/distraction apps based on RAW data (its own rules)
        agent_total_productive_time = 0.0
        agent_total_distraction_time = 0.0
        agent_classified_productive_apps = set()
        agent_classified_distraction_apps = set()

        # Aggregate raw exe/titles for agent's classification perspective
        raw_app_aggregates_for_agent: Dict[str, Dict[str, Any]] = {}
        for entry in agent_perspective_log_entries:
            # Use 'app_name' as logged by agent (which might be 'app_identity_name')
            app_name_key = entry["app_name"] 
            raw_app_aggregates_for_agent.setdefault(app_name_key, {"titles": set(), "duration": 0.0, "exe_path": entry["exe"]})
            raw_app_aggregates_for_agent[app_name_key]["titles"].add(entry["title"])
            raw_app_aggregates_for_agent[app_name_key]["duration"] += entry["duration"]

        for app_name_key, details in raw_app_aggregates_for_agent.items():
            titles_list = list(details["titles"])
            exe_path_for_eval = details["exe_path"] 
            duration_for_eval = details["duration"]
            
            if self._is_productive_app(exe_path_for_eval, titles_list):
                agent_total_productive_time += duration_for_eval
                agent_classified_productive_apps.add(app_name_key) 
            elif self._is_distraction_app(exe_path_for_eval, titles_list):
                agent_total_distraction_time += duration_for_eval
                agent_classified_distraction_apps.add(app_name_key)
        
        summary_to_enhance["productiveApps"] = sorted(list(agent_classified_productive_apps))
        summary_to_enhance["distractionApps"] = sorted(list(agent_classified_distraction_apps))
        
        # Use totalTime from the base_summary (which is sum of all durations from logs)
        total_time_for_score_calc = summary_to_enhance.get("totalTime", sum(e["duration"] for e in agent_perspective_log_entries))
        if total_time_for_score_calc == 0 and agent_perspective_log_entries: # Recalculate if base was 0 but we have entries
            total_time_for_score_calc = sum(e["duration"] for e in agent_perspective_log_entries)
            summary_to_enhance["totalTime"] = round(total_time_for_score_calc)

        summary_to_enhance["focusScore"] = self._calculate_focus_score(
            agent_total_productive_time, agent_total_distraction_time, total_time_for_score_calc
        )
        
        # Re-save the summary file, now enhanced with agent-specific metrics
        final_summary_file_path = self.focus_logs_dir / f"daily_summary_{target_date_str}.json"
        try:
            with open(final_summary_file_path, "w", encoding="utf-8") as f_final_sum:
                json.dump(summary_to_enhance, f_final_sum, indent=2)
            logger.info(f"Agent-enhanced daily summary for {target_date_str} saved to {final_summary_file_path.name}")
        except Exception as e_save_final:
            logger.error(f"Error re-saving agent-enhanced daily summary {final_summary_file_path.name}: {e_save_final}")
            return summary_to_enhance 

        # Optionally, generate a simple chart from the agent's perspective
        try:
            self._generate_agent_usage_chart(summary_to_enhance, target_date_str)
        except Exception as chart_err:
            logger.warning(f"Could not generate agent usage chart: {chart_err}")
            
        return summary_to_enhance

    def _calculate_focus_score(self, productive_seconds: float, distraction_seconds: float, total_tracked_seconds: float) -> int:
        """Calculate focus score based on productive, distraction, and total time."""
        if total_tracked_seconds <= 0: return 0
        
        # Time not classified as productive or distraction is neutral
        neutral_seconds = max(0, total_tracked_seconds - productive_seconds - distraction_seconds)
        
        # Weighted score: Productive=1, Neutral=0.5, Distraction=-1
        weighted_score_val = (productive_seconds * 1.0) + (neutral_seconds * 0.5) - (distraction_seconds * 1.0)
        
        # Normalize score to be between -1 and 1 (relative to total time)
        normalized_score_val = weighted_score_val / total_tracked_seconds if total_tracked_seconds > 0 else 0
        
        # Scale to 0-100 range
        final_score_val = max(0, min(100, int((normalized_score_val + 1) / 2 * 100)))
        return final_score_val

    def _is_productive_app(self, exe_full_path: str, window_titles: List[str]) -> bool:
        """Determine if app is productive based on agent's rules."""
        exe_basename_lower = os.path.basename(exe_full_path).lower()
        # Concatenate titles for keyword search, ensuring all are strings and lowercased
        titles_concatenated_lower = " ".join(str(t).lower() for t in window_titles if t).strip()
        
        # Check if exe_basename_lower matches any in PRODUCTIVE_EXES
        is_prod_exe = False
        for prod_exe_pattern in PRODUCTIVE_EXES:
            if prod_exe_pattern.endswith(".exe"):
                if exe_basename_lower == prod_exe_pattern:
                    is_prod_exe = True; break
            else: # Pattern does not end with .exe, match against basename without extension
                if exe_basename_lower.replace(".exe", "") == prod_exe_pattern:
                    is_prod_exe = True; break
        
        if is_prod_exe:
            # If it's a productive exe, ensure no distraction keywords are in titles
            if not any(dist_keyword in titles_concatenated_lower for dist_keyword in DISTRACTION_TITLE_KEYWORDS if dist_keyword in titles_concatenated_lower):
                return True
        return False

    def _is_distraction_app(self, exe_full_path: str, window_titles: List[str]) -> bool:
        """Determine if app is a distraction based on agent's rules."""
        exe_basename_lower = os.path.basename(exe_full_path).lower()
        titles_concatenated_lower = " ".join(str(t).lower() for t in window_titles if t).strip()

        is_dist_exe = False
        for dist_exe_pattern in DISTRACTION_EXES:
            if dist_exe_pattern.endswith(".exe"):
                if exe_basename_lower == dist_exe_pattern:
                    is_dist_exe = True; break
            else:
                if exe_basename_lower.replace(".exe", "") == dist_exe_pattern:
                    is_dist_exe = True; break
        if is_dist_exe: return True
        
        # Check if a typically productive app is used for distraction (e.g., browser for YouTube)
        is_potentially_prod_exe = False
        for prod_exe_pattern in PRODUCTIVE_EXES:
            if prod_exe_pattern.endswith(".exe"):
                if exe_basename_lower == prod_exe_pattern: is_potentially_prod_exe = True; break
            else:
                if exe_basename_lower.replace(".exe", "") == prod_exe_pattern: is_potentially_prod_exe = True; break
        
        if is_potentially_prod_exe and any(dist_keyword in titles_concatenated_lower for dist_keyword in DISTRACTION_TITLE_KEYWORDS if dist_keyword in titles_concatenated_lower):
            return True
        return False

    def _is_meeting_app(self, exe_full_path: str, window_title: str) -> bool:
        """Determine if app is a meeting app based on agent's rules."""
        exe_basename_lower = os.path.basename(exe_full_path).lower()
        title_str_lower = str(window_title).lower() # Ensure title is string

        is_meet_exe = False
        for meet_exe_pattern in MEETING_EXES:
            if meet_exe_pattern.endswith(".exe"):
                if exe_basename_lower == meet_exe_pattern: is_meet_exe = True; break
            else:
                if exe_basename_lower.replace(".exe", "") == meet_exe_pattern: is_meet_exe = True; break
        if is_meet_exe: return True
        
        if any(meet_keyword in title_str_lower for meet_keyword in MEETING_TITLE_KEYWORDS if meet_keyword in title_str_lower):
            return True
        return False
    
    def _generate_agent_usage_chart(self, summary_data_dict: Dict, date_str_for_chart: str):
        """Generate a simple app usage pie chart using Matplotlib."""
        # This method requires matplotlib. If not available, it should gracefully skip.
        try:
            import matplotlib.pyplot as plt
            import pandas as pd # Matplotlib often used with pandas for data prep
            MATPLOTLIB_AVAILABLE = True
        except ImportError:
            MATPLOTLIB_AVAILABLE = False
            logger.info("Matplotlib not installed, skipping agent usage chart generation.")
            return

        if not MATPLOTLIB_AVAILABLE: return # Double check

        app_breakdown_list = summary_data_dict.get("appBreakdown", [])
        if not app_breakdown_list:
            logger.info(f"No app breakdown data to generate agent chart for {date_str_for_chart}.")
            return

        # Use pandas DataFrame for easier manipulation if complex filtering/sorting is needed
        df_apps = pd.DataFrame(app_breakdown_list)
        df_apps = df_apps.sort_values("timeSpent", ascending=False)
        
        # Prepare data for pie chart: Top N apps + "Other"
        top_n_apps_for_chart = 10
        if len(df_apps) > top_n_apps_for_chart:
            df_top_apps = df_apps.head(top_n_apps_for_chart).copy() # Use .copy() to avoid SettingWithCopyWarning
            other_apps_time = df_apps.iloc[top_n_apps_for_chart:]["timeSpent"].sum()
            if other_apps_time > 0:
                # Create a new row for 'Other Apps'
                other_row = pd.DataFrame([{"appName": "Other Apps", "timeSpent": other_apps_time}])
                df_top_apps = pd.concat([df_top_apps, other_row], ignore_index=True)
        else:
            df_top_apps = df_apps

        if df_top_apps.empty or df_top_apps["timeSpent"].sum() == 0:
             logger.info(f"Not enough data or zero total time for agent chart on {date_str_for_chart}")
             return

        plt.figure(figsize=(10, 8)) # Adjust figure size as needed
        plt.pie(
            df_top_apps["timeSpent"],
            labels=df_top_apps["appName"],
            autopct="%1.1f%%", # Format percentages
            startangle=90, # Start first slice at the top
            pctdistance=0.85 # Distance of percentage labels from center
        )
        plt.title(f"App Usage (Agent View) - {date_str_for_chart}")
        plt.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
        
        chart_file_path = self.focus_logs_dir / f"agent_usage_chart_{date_str_for_chart}.png"
        try:
            plt.savefig(chart_file_path)
            logger.info(f"Generated agent usage chart: {chart_file_path}")
        except Exception as e_save_chart:
            logger.error(f"Failed to save agent usage chart {chart_file_path}: {e_save_chart}")
        finally:
            plt.close() # Close the figure to free memory

    async def run_agent_loop(self, check_interval_s: int = 5):
        """The main agent loop: tracks focused window and manages summaries with browser profile daily aggregation."""
        logger.info(f"Starting Focus Monitor agent loop with browser profile daily aggregation. Log check interval: {check_interval_s}s")
        self.window_start_time = time.time()
        # Ensure summary runs on first suitable check if interval suggests it
        self.last_summary_time = time.time() - self.summary_interval 

        while True:
            loop_iteration_start_time = time.time()
            try:
                # Check desired state from backend and toggle internal state if needed
                if self.api_url : self.toggle_active() # Only toggle if API is configured

                if not self.active: # If agent is paused
                    # Still check if it's time to generate a daily summary for past activity
                    if time.time() - self.last_summary_time >= self.summary_interval:
                        # Before generating daily summary, finalize any active bucket
                        if self.current_bucket_index is not None:
                            self._finalize_bucket(self.current_bucket_index)
                        
                        self._generate_daily_summary() # Will use self.today
                        self._write_time_bucket_summary() # Write current session buckets
                        self._write_daily_browser_aggregation() # Write browser aggregations
                        self.last_summary_time = time.time()
                    
                    # Sleep until next check, accounting for processing time
                    await asyncio.sleep(max(0.1, check_interval_s - (time.time() - loop_iteration_start_time)))
                    continue # Skip window processing if not active

                # --- Active Monitoring ---
                # Day Change Check (using UTC date)
                current_utc_date_str = self._get_current_utc_date()
                if current_utc_date_str != self.today:
                    logger.info(
                        f"Date changed from {self.today} to {current_utc_date_str}. Processing previous day's data."
                    )
                    # Finalize any active bucket before day change
                    if self.current_bucket_index is not None:
                        self._finalize_bucket(self.current_bucket_index)
                    
                    self._generate_daily_summary(self.today) # Final summary for the day that just ended
                    self._write_time_bucket_summary() # Write out buckets for the session that just ended with the day
                    self._write_daily_browser_aggregation() # Write out browser aggregations for the day that just ended
                    
                    # Reset for the new day
                    self.today = current_utc_date_str
                    self.session_start_time = time.time() # New session starts now
                    self.session_start_tag = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                    self.time_buckets = {} # Clear buckets for the new session
                    self.daily_browser_activities = {} # Clear daily browser aggregation for new day
                    self.finalized_buckets = set() # Clear finalized buckets tracking
                    self.current_bucket_index = None # Reset current bucket tracking
                    self.last_window_info = None # Treat as fresh start for window tracking
                    self.window_start_time = time.time()
                    logger.info(f"Updated current tracking date to {self.today}. New session tag: {self.session_start_tag}. Browser aggregation reset.")

                # Get current focused window details
                current_focused_win_details = self._get_focused_window_details()

                # Determine if the focused window changed significantly
                is_now_idle = current_focused_win_details is None
                window_has_changed = False
                if is_now_idle:
                    if self.last_window_info is not None: window_has_changed = True # Transition: Active -> Idle
                elif self.last_window_info is None:
                    window_has_changed = True # Transition: Idle -> Active
                # Active -> Different Active: Compare key attributes
                elif (current_focused_win_details["hwnd"] != self.last_window_info.get("hwnd") or \
                      current_focused_win_details["title"] != self.last_window_info.get("title") or \
                      current_focused_win_details["exe"] != self.last_window_info.get("exe") or \
                      current_focused_win_details["app_name_identity"] != self.last_window_info.get("app_name_identity")):
                    window_has_changed = True

                if window_has_changed:
                    # Log duration for the *previous* window/state if it existed
                    if self.last_window_info:
                        duration_active = int(time.time() - self.window_start_time)
                        if duration_active > 0:
                            self._log_window_activity(self.last_window_info, duration_active)

                    # Reset timer and update last window info (becomes None if now idle)
                    self.window_start_time = time.time()
                    self.last_window_info = current_focused_win_details # Update to current, or None if idle

                # Check if it's time to generate a periodic daily summary and write buckets
                if time.time() - self.last_summary_time >= self.summary_interval:
                    # Before generating daily summary, finalize any active bucket
                    if self.current_bucket_index is not None:
                        self._finalize_bucket(self.current_bucket_index)
                    
                    self._generate_daily_summary() # Uses self.today
                    self._write_time_bucket_summary() # Writes current session's buckets
                    self._write_daily_browser_aggregation() # Writes current day's browser aggregations
                    self.last_summary_time = time.time()

                # Sleep until next interval, accounting for processing time in this iteration
                elapsed_this_iteration = time.time() - loop_iteration_start_time
                sleep_duration = max(0.1, check_interval_s - elapsed_this_iteration) # Ensure positive sleep
                await asyncio.sleep(sleep_duration)

            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received in agent loop. Exiting.")
                break # Exit the while loop
            except Exception as e_loop: # Catch-all for unexpected errors in the loop
                logger.error(f"Unhandled error in agent loop: {e_loop}", exc_info=True)
                await asyncio.sleep(check_interval_s * 2) # Wait longer after an error before retrying

        # --- Cleanup on loop exit (e.g., due to KeyboardInterrupt) ---
        logger.info("Agent loop finished. Performing final cleanup...")
        
        # Finalize any remaining active bucket before final cleanup
        if self.current_bucket_index is not None:
            self._finalize_bucket(self.current_bucket_index)
            
        if self.last_window_info:  # Log any final pending activity
            duration_final = int(time.time() - self.window_start_time)
            if duration_final > 0:
                self._log_window_activity(self.last_window_info, duration_final)
        
        self._generate_daily_summary()  # Generate final summary for the last active day
        self._write_time_bucket_summary() # Write out any remaining buckets for the session
        self._write_daily_browser_aggregation() # Write out any remaining browser aggregations
        logger.info("Focus Monitor agent stopped.")


async def main_async_entrypoint():
    """Parses arguments and starts the FocusMonitorAgent with browser profile daily aggregation."""
    parser = argparse.ArgumentParser(description="Focus Monitor Agent with Browser Profile Daily Aggregation (Window Tracking)")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Optional directory to store logs (defaults to script directory).")
    parser.add_argument("--api-url", "-a", type=str, default=None,
                        help="Backend API URL for status checks (optional).")
    parser.add_argument("--interval", "-i", type=int, default=5,
                        help="Sampling interval in seconds for checking focused window (default: 5s).")
    parser.add_argument("--summary-interval", "-s", type=int, default=300,
                        help="Interval for generating periodic summaries in seconds (default: 300s = 5min).")
    parser.add_argument("--no-api-check", action="store_true",
                        help="Disable checking backend API for active status (agent runs continuously).")
    parser.add_argument("--force-summary-on-start", action="store_true",
                        help="Force generate a daily summary immediately on start for the current day.")
    parser.add_argument("--tesseract-path", type=str, default=None,
                        help="Explicitly set the path to tesseract.exe (e.g., for Windows if not in PATH).")
    args = parser.parse_args()

    # Configure Tesseract path if provided (especially for Windows)
    if args.tesseract_path and OCR_AVAILABLE:
        try:
            pytesseract.pytesseract.tesseract_cmd = args.tesseract_path
            logger.info(f"Tesseract command path set to: {args.tesseract_path}")
        except Exception as e_tess_path:
            logger.error(f"Failed to set Tesseract command path to '{args.tesseract_path}': {e_tess_path}")


    output_directory_path = Path(args.output_dir) if args.output_dir else SCRIPT_DIR # Use SCRIPT_DIR if None
    if args.output_dir and not output_directory_path.is_dir(): # Check only if specified
        logger.critical(f"Specified output directory not found: {args.output_dir}")
        try: # Attempt to create it
            output_directory_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {output_directory_path}")
        except Exception as e_mkdir:
            logger.critical(f"Failed to create output directory {output_directory_path}: {e_mkdir}")
            sys.exit(1)
    
    api_url_for_agent = None if args.no_api_check else args.api_url

    # Instantiate and configure the agent
    agent_instance = FocusMonitorAgent(str(output_directory_path), api_url_for_agent)
    agent_instance.summary_interval = args.summary_interval

    # Generate a summary immediately if requested
    if args.force_summary_on_start:
        logger.info("Forcing initial daily summary generation as per --force-summary-on-start...")
        agent_instance._generate_daily_summary() # For current day (self.today)
        agent_instance._write_time_bucket_summary() # Write any buckets from very short startup
        agent_instance._write_daily_browser_aggregation() # Write any browser aggregations from startup
        agent_instance.last_summary_time = time.time() # Reset summary timer

    # Create tasks for the agent loop and backend status check (if API is used)
    agent_tasks = [asyncio.create_task(agent_instance.run_agent_loop(args.interval))]
    if api_url_for_agent: # Only run status check if API URL is provided
        agent_tasks.append(asyncio.create_task(agent_instance.check_backend_status()))

    try:
        # Wait for the main agent loop to complete (or be cancelled)
        await agent_tasks[0] 
    except asyncio.CancelledError:
        logger.info("Main agent loop task was cancelled.")
    finally:
        # Ensure all background tasks (like API checker) are cancelled and awaited
        for task_to_cancel in agent_tasks[1:]: # Cancel any other tasks
            if not task_to_cancel.done():
                task_to_cancel.cancel()
        # Await their completion (or cancellation exception)
        if len(agent_tasks) > 1:
            await asyncio.gather(*agent_tasks[1:], return_exceptions=True)
        logger.info("All agent background tasks have been processed.")


if __name__ == "__main__":
    try:
        asyncio.run(main_async_entrypoint())
    except KeyboardInterrupt:
        logger.info("Focus Monitor Agent stopped by user (Ctrl+C in main).")
    except Exception as main_execution_err:
        logger.critical(f"Focus Monitor Agent exited due to an unhandled error in main: {main_execution_err}", exc_info=True)
    finally:
        logger.info("Focus Monitor Agent application shutting down.")