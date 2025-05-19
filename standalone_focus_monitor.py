#!/usr/bin/env python3
"""
Focus Monitor Agent (Window Title Tracking Version)

This script tracks the title and executable of the currently focused window
to generate focus activity logs and daily summaries. Does NOT take screenshots.
Runs completely standalone and saves all files to the script's directory.
Generates live summaries for the current day's activity.
Detects and groups browser profiles for Microsoft Edge.
"""

import os
import sys
import time
import json
import logging
import argparse
import datetime
import re
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
import asyncio

try:
    import win32gui
    import win32process
    import win32api
    import win32con # For constants if needed later
    import requests  # Optional for API features
    import wmi  # For Chrome profile detection
except ImportError:
    print("Required packages not found. Please install with:")
    print("pip install pywin32 requests wmi")
    sys.exit(1)

# Get the directory where this script is located
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(SCRIPT_DIR / "focus_monitor.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("focus_monitor")

# --- Configuration (Copied for backend calculation consistency) ---
PRODUCTIVE_EXES = {"code.exe", "pycharm", "idea", "webstorm", "goland", "clion",
                   "word", "excel", "powerpnt", "outlook",
                   "chrome.exe", "firefox.exe", "msedge.exe", "safari",
                   "cmd.exe", "powershell.exe", "terminal", "wt.exe",
                   "explorer.exe", "photoshop", "illustrator", "figma", "xd",
                   "blender", "unity", "docker", "virtualbox", "vmware",
                   "gitkraken", "postman", "obsidian"}
DISTRACTION_EXES = {"steam.exe", "epicgameslauncher", "origin.exe", "gog galaxy",
                     "spotify.exe", "discord.exe", "slack.exe",
                     "netflix", "hulu", "disneyplus",
                     "whatsapp", "telegram", "signal"}
DISTRACTION_TITLE_KEYWORDS = {"youtube", "facebook", "twitter", "reddit", "netflix",
                              "hulu", "twitch", "instagram", "9gag", "game", "play",
                              "tiktok", "pinterest"}
MEETING_EXES = {"teams.exe", "zoom.exe", "webex", "skype.exe", "slack.exe"}
MEETING_TITLE_KEYWORDS = {"meet", "meeting", "call", "webinar", "huddle",
                           "zoom meeting", "microsoft teams meeting", "google meet"}

# --- Browser Profile Detection Patterns ---
EDGE_PROFILE_PATTERN = re.compile(r'Microsoft Edge(?:\s*-\s*(.+))?$')
# Chrome typically formats titles as: "Page Title - Google Chrome" or "Page Title - Google Chrome - Profile"
CHROME_PROFILE_PATTERN = re.compile(r'.*Google Chrome(?:\s*-\s*(.+))?$')
FIREFOX_PROFILE_PATTERN = re.compile(r'Mozilla Firefox(?:\s*-\s*(.+))?$')

class FocusMonitorAgent:
    def __init__(self, output_dir: Optional[str] = None, api_url: Optional[str] = None):
        # If no output dir specified, use script directory
        self.output_dir = Path(output_dir) if output_dir else SCRIPT_DIR
        self.api_url = api_url
        self.active = True # Internal current state
        self.desired_active_state = True # State requested by backend
        self.last_window_info: Optional[Dict] = None
        self.window_start_time: float = time.time()
        self.today: str = self._get_current_utc_date()
        self.last_summary_time: float = 0
        self.summary_interval: int = 300  # Generate summary every 5 minutes

        # Session start timestamp for 5-minute buckets
        self.session_start_time: float = time.time()
        self.session_start_tag: str = (
            datetime.datetime.utcfromtimestamp(self.session_start_time)
            .strftime("%Y%m%dT%H%M%SZ")
        )
        self.time_buckets: Dict[int, Dict[str, Set[str]]] = {}

        # Create focus_logs directory within the script or specified directory
        self.focus_logs_dir = self.output_dir / "focus_logs"
        self.focus_logs_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized FocusMonitorAgent (Window Tracking). Output: {self.focus_logs_dir}, API: {self.api_url or 'Disabled'}")

    def _get_current_utc_date(self) -> str:
         # Make sure to use UTC date for consistency
         current_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
         logger.debug(f"Current UTC date: {current_date}")
         return current_date

    async def check_backend_status(self):
        """Periodically check the desired active state from the backend."""
        while True:
            if not self.api_url:
                self.desired_active_state = True # Default to active if no API
                await asyncio.sleep(60)
                continue
            try:
                response = requests.get(f"{self.api_url}/focus/status", timeout=3)
                response.raise_for_status()
                new_desired_state = response.json().get("active", True)
                if self.desired_active_state != new_desired_state:
                     logger.info(f"Backend desired state changed to: {new_desired_state}")
                     self.desired_active_state = new_desired_state
            except requests.exceptions.RequestException as e:
                logger.warning(f"Could not reach backend ({self.api_url}) to check focus status: {e}")
            except Exception as e:
                 logger.error(f"Error checking backend status: {e}")
            await asyncio.sleep(15) # Check every 15 seconds

    def toggle_active(self):
        """Toggle the internal active state if it differs from desired state."""
        if self.active == self.desired_active_state: return

        new_state = self.desired_active_state
        logger.info(f"Focus Monitor internal state changing to: {new_state}")

        if not new_state: # Pausing
             if self.last_window_info:
                 duration = int(time.time() - self.window_start_time)
                 if duration > 0: self._log_window_activity(self.last_window_info, duration)
             self.last_window_info = None
             self.window_start_time = time.time()
        else: # Resuming
             self.window_start_time = time.time() # Reset start time

        self.active = new_state

    def _extract_browser_profile(self, exe: str, title: str, pid: int = 0) -> Optional[str]:
        """Extract browser profile information from window title or process details if available."""
        # First try to detect from window title
        if "msedge.exe" in exe.lower():
            match = EDGE_PROFILE_PATTERN.search(title)
            if match and match.group(1):
                return match.group(1).strip()
        elif "chrome.exe" in exe.lower():
            # First check direct profile path from process command line
            if pid > 0:
                try:
                    # Try to get the profile directory from the command line
                    profile = self._get_chrome_profile_from_pid(pid)
                    if profile:
                        return profile
                except Exception as e:
                    logger.debug(f"Error getting Chrome profile from pid: {e}")
                    
            # Fall back to window title detection
            match = CHROME_PROFILE_PATTERN.search(title)
            if match and match.group(1):
                # Chrome profiles might have extra spaces or text, clean it up
                profile = match.group(1).strip()
                # If the profile contains common Chrome text, it's probably not a profile name
                if profile.lower() in ["new tab", "settings", "history"]:
                    return None
                return profile
        elif "firefox.exe" in exe.lower():
            match = FIREFOX_PROFILE_PATTERN.search(title)
            if match and match.group(1):
                return match.group(1).strip()
        return None
        
    def _get_chrome_profile_from_pid(self, pid: int) -> Optional[str]:
        """Extract Chrome profile name from process command line arguments."""
        try:
            # Open the process with necessary access rights
            handle = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ, False, pid)
            if not handle:
                return None
                
            try:
                # Get the full command line 
                import ctypes
                from ctypes import wintypes
                
                GetProcessCommandLine = ctypes.windll.kernel32.GetProcessCommandLine
                GetProcessCommandLine.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD]
                GetProcessCommandLine.restype = wintypes.BOOL
                
                buffer_size = 8192  # Max command line size
                buffer = ctypes.create_unicode_buffer(buffer_size)
                
                # This direct approach doesn't work because GetProcessCommandLine isn't exposed properly
                # Instead, we'll use WMI to get this information
                
                # Alternative approach using WMI
                import wmi
                c = wmi.WMI()
                for process in c.Win32_Process(ProcessId=pid):
                    cmd_line = process.CommandLine
                    if cmd_line:
                        # Look for --profile-directory="ProfileName" or --profile-directory=ProfileName
                        profile_match = re.search(r'--profile-directory[=\s]\"?([^\"]+)\"?', cmd_line)
                        if profile_match:
                            profile_dir = profile_match.group(1).strip()
                            # Convert profile directory to a friendly name
                            # Default is "Default", others are typically named "Profile 1", "Profile 2", etc.
                            # or custom names like "Work", "Personal", etc.
                            if profile_dir == "Default":
                                return "Default"
                            elif profile_dir.startswith("Profile "):
                                return f"Profile {profile_dir.split(' ')[1]}"
                            else:
                                # Try to map the directory to a user-friendly name
                                return self._get_chrome_profile_name_from_dir(profile_dir)
                return None
            finally:
                win32api.CloseHandle(handle)
        except Exception as e:
            logger.debug(f"Error getting Chrome command line: {e}")
            return None
            
    def _get_chrome_profile_name_from_dir(self, profile_dir: str) -> str:
        """Try to map Chrome profile directory to user-friendly name."""
        # Chrome user data directory is typically at:
        # C:\Users\<username>\AppData\Local\Google\Chrome\User Data\
        
        try:
            # Common locations for Chrome user data
            user_data_dirs = [
                os.path.expanduser(r"~\AppData\Local\Google\Chrome\User Data"),
                os.path.expanduser(r"~\AppData\Local\Chromium\User Data"),
                os.path.expanduser(r"~\AppData\Local\Google\Chrome Beta\User Data"),
                os.path.expanduser(r"~\AppData\Local\Google\Chrome Dev\User Data"),
            ]
            
            # Try to find the Preferences file for this profile
            for user_data_dir in user_data_dirs:
                prefs_file = os.path.join(user_data_dir, profile_dir, "Preferences")
                if os.path.exists(prefs_file):
                    try:
                        with open(prefs_file, 'r', encoding='utf-8') as f:
                            prefs = json.load(f)
                            # Extract profile name from preferences
                            if 'profile' in prefs and 'name' in prefs['profile']:
                                return prefs['profile']['name']
                    except Exception as e:
                        logger.debug(f"Error reading Chrome preferences: {e}")
            
            # If we can't find a friendly name, return the directory name
            return profile_dir
        except Exception as e:
            logger.debug(f"Error mapping Chrome profile directory: {e}")
            return profile_dir

    def _get_app_identity(self, exe: str, title: str) -> Tuple[str, str]:
        """Generate a consistent app identity that includes profile info for browsers."""
        profile = self._extract_browser_profile(exe, title)
        
        # Default app name is the executable basename without .exe
        app_name = os.path.basename(exe).replace('.exe', '') if exe != "Unknown" else "Unknown"
        
        # For browsers with profile info, append the profile to the app name
        if profile:
            if "msedge.exe" in exe.lower():
                return f"{app_name} - {profile}", exe
            elif "chrome.exe" in exe.lower():
                return f"{app_name} - {profile}", exe
            elif "firefox.exe" in exe.lower():
                return f"{app_name} - {profile}", exe
        
        # Special case for Chrome - check if we can infer the profile from page content
        if "chrome.exe" in exe.lower() and not profile:
            # Check for domains that might indicate work vs personal
            title_lower = title.lower()
            
            # Work-related domains
            if any(domain in title_lower for domain in ["jira", "confluence", "slack", "teams", "github", "gitlab", "azure", "office365"]):
                return f"{app_name} - Work", exe
                
            # Entertainment domains
            if any(domain in title_lower for domain in ["youtube", "netflix", "twitch", "spotify", "reddit", "facebook", "instagram"]):
                return f"{app_name} - Personal", exe
                
            # If we can't infer, just return the app name
            return app_name, exe
        
        return app_name, exe

    def _get_focused_window_details(self) -> Optional[Dict]:
        """Get details (hwnd, pid, exe, title, timestamp) for the currently focused window."""
        try:
            hwnd = win32gui.GetForegroundWindow()
            if not hwnd: return None

            title = win32gui.GetWindowText(hwnd)
            tid, pid = win32process.GetWindowThreadProcessId(hwnd)

            if not title or pid == 0 or title in ["Program Manager", "Windows Default Lock Screen", "Windows Input Experience"]:
                 return None # Filter out uninteresting windows

            exe = "Unknown"
            try:
                # PROCESS_QUERY_LIMITED_INFORMATION is safer if available
                handle = win32api.OpenProcess(win32con.PROCESS_QUERY_LIMITED_INFORMATION | win32con.PROCESS_VM_READ, False, pid)
                if handle:
                    try: exe = win32process.GetModuleFileNameEx(handle, 0)
                    finally: win32api.CloseHandle(handle)
            except Exception: pass # Ignore permission errors

            # Extract profile info if available - pass PID for Chrome profile detection
            profile = self._extract_browser_profile(exe, title, pid)
            
            return {
                "hwnd": hwnd, "pid": pid, "exe": exe, "title": title,
                "app_profile": profile,  # Store profile info
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
        except Exception as e:
            if "pywintypes.error" in str(type(e)) and e.args[0] in [0, 1400]: # Handle window closing during check
                 logger.debug(f"Window likely closed during info retrieval: {e}")
                 return None
            logger.error(f"Error getting focused window details: {e}", exc_info=False)
            return None

    def _log_window_activity(self, window_info: Dict, duration: int):
        """Log focused window activity to JSONL file."""
        if duration <= 0: return
        try:
            # Make sure focus_logs directory exists
            self.focus_logs_dir.mkdir(parents=True, exist_ok=True)
            log_file = self.focus_logs_dir / f"focus_log_{self.today}.jsonl"
            
            # Include app_identity and profile in the log
            app_identity, exe_path = self._get_app_identity(window_info["exe"], window_info["title"])
            
            log_entry = {
                "timestamp": window_info["timestamp"], 
                "exe": window_info["exe"],
                "app_name": app_identity,  # Include friendly app name with profile
                "app_profile": window_info.get("app_profile"),  # Include profile if available
                "title": window_info["title"], 
                "duration": duration,
                "pid": window_info.get("pid", 0)  # Include PID for potential label lookup
            }
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

            # Log with profile info if available
            title_snip = log_entry['title'][:60].replace('\n', ' ') + ('...' if len(log_entry['title']) > 60 else '')
            profile_info = f" ({log_entry['app_profile']})" if log_entry.get('app_profile') else ""
            logger.info(f"Logged: {app_identity}{profile_info} - '{title_snip}' ({duration}s)")

            # Update in-memory 5-minute bucket summary
            self._update_time_bucket(log_entry)
        except Exception as e:
            logger.error(f"Error writing to log file {log_file}: {e}")

    def _update_time_bucket(self, log_entry: Dict):
        """Update the in-memory 5-minute bucket summary with a new log entry."""
        try:
            ts = datetime.datetime.fromisoformat(log_entry["timestamp"]).timestamp()
        except Exception:
            ts = time.time()

        bucket_index = int((ts - self.session_start_time) // 300)
        bucket_start = self.session_start_time + bucket_index * 300
        bucket_end = bucket_start + 300

        bucket = self.time_buckets.setdefault(
            bucket_index,
            {
                "start": datetime.datetime.utcfromtimestamp(bucket_start).isoformat(),
                "end": datetime.datetime.utcfromtimestamp(bucket_end).isoformat(),
                "apps": set(),
                "titles": set(),
                "ocr_text": set(),
            },
        )

        bucket["apps"].add(log_entry.get("app_name", "Unknown"))
        if log_entry.get("title"):
            bucket["titles"].add(log_entry["title"])
        if log_entry.get("ocr_text"):
            bucket["ocr_text"].add(log_entry["ocr_text"])

    def _write_time_bucket_summary(self):
        """Write the current time bucket summary to a JSON file."""
        bucket_file = self.focus_logs_dir / f"time_buckets_{self.session_start_tag}.json"
        serializable = []
        for idx in sorted(self.time_buckets.keys()):
            b = self.time_buckets[idx]
            serializable.append(
                {
                    "start": b["start"],
                    "end": b["end"],
                    "apps": sorted(b["apps"]),
                    "titles": sorted(b["titles"]),
                    "ocr_text": sorted(b["ocr_text"]),
                }
            )

        try:
            with open(bucket_file, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
            logger.info(f"Wrote time bucket summary to {bucket_file}")
        except Exception as e:
            logger.error(f"Error writing bucket summary: {e}")

    def _generate_daily_summary(self, date_str=None):
        """Generate a daily summary for the specified date or today."""
        if date_str is None:
            date_str = self.today
        
        logger.info(f"Generating daily summary file for {date_str}...")
        summary_file = self.focus_logs_dir / f"daily_summary_{date_str}.json"
        log_file = self.focus_logs_dir / f"focus_log_{date_str}.jsonl"

        if not log_file.exists():
            logger.warning(f"No focus log file found for {date_str}. Cannot generate summary file.")
            return None  # Return None to indicate no data

        summary = { # Structure without screenshots/keywords
            "date": date_str, "totalTime": 0, "appBreakdown": [],
            "focusScore": 0, "distractionEvents": 0, "meetingTime": 0,
            "productiveApps": [], "distractionApps": []
        }
        try:
            log_entries = []
            total_time = 0
            app_time: Dict[str, float] = {}  # Now keyed by app_name (includes profile)
            app_titles: Dict[str, Set[str]] = {}
            app_exes: Dict[str, str] = {}  # Map app_name to exe for classification
            
            # --- Process Log File ---
            with open(log_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        entry = json.loads(line.strip())
                        if not isinstance(entry, dict) or not all(k in entry for k in ["exe", "title", "duration", "timestamp"]): 
                            continue
                        
                        duration = entry.get("duration", 0)
                        if duration <= 0: continue
                        
                        # If entry doesn't have app_name (from older logs), generate it
                        if "app_name" not in entry:
                            app_identity, exe_path = self._get_app_identity(entry["exe"], entry["title"])
                            entry["app_name"] = app_identity
                        
                        log_entries.append(entry)
                        total_time += duration
                        
                        # Use app_name as key (includes profile info)
                        app_name = entry["app_name"]
                        exe = entry["exe"]
                        title = entry["title"] or ""
                        
                        # Track time by app_name (with profile)
                        app_time[app_name] = app_time.get(app_name, 0) + duration
                        
                        # Track titles for each app_name
                        if app_name not in app_titles: 
                            app_titles[app_name] = set()
                            app_exes[app_name] = exe  # Store exe for classification
                        
                        if len(app_titles[app_name]) < 50: 
                            app_titles[app_name].add(title)
                    except Exception as e: 
                        logger.error(f"Err processing line {line_num}: {e}")

            summary["totalTime"] = round(total_time)
            summary["distractionEvents"] = len(log_entries)

            # App breakdown - now with browser profile support
            app_breakdown_list = []
            for app_name, time_spent in app_time.items():
                 exe = app_exes.get(app_name, "Unknown")
                 percentage = (time_spent / total_time * 100) if total_time > 0 else 0
                 app_breakdown_list.append({
                     "appName": app_name,  # Now includes profile info 
                     "exePath": exe,
                     "timeSpent": round(time_spent),
                     "percentage": round(percentage, 2), 
                     "windowTitles": sorted(list(app_titles.get(app_name, set())))
                 })
            app_breakdown_list.sort(key=lambda x: x["timeSpent"], reverse=True)
            summary["appBreakdown"] = app_breakdown_list

            # Metrics
            title_list_map = {app['appName']: app['windowTitles'] for app in app_breakdown_list}
            exe_map = {app['appName']: app['exePath'] for app in app_breakdown_list}
            
            # Meeting time calculation
            summary["meetingTime"] = round(sum(
                e["duration"] for e in log_entries 
                if self._is_meeting_app(e["exe"], e["title"])
            ))
            
            # App classification with profile support
            productive_apps_set = {
                app["appName"] for app in app_breakdown_list 
                if self._is_productive_app(
                    app["exePath"], 
                    title_list_map.get(app["appName"], [])
                )
            }
            
            distraction_apps_set = {
                app["appName"] for app in app_breakdown_list 
                if self._is_distraction_app(
                    app["exePath"], 
                    title_list_map.get(app["appName"], [])
                )
            }
            
            summary["productiveApps"] = sorted(list(productive_apps_set))
            summary["distractionApps"] = sorted(list(distraction_apps_set))
            summary["focusScore"] = self._calculate_focus_score(
                summary["productiveApps"], 
                summary["distractionApps"], 
                summary["appBreakdown"], 
                summary["totalTime"]
            )

            # Write summary file
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Generated daily summary file for {date_str}.")
            
            # Also create a live usage chart if matplotlib is available
            try:
                self._generate_usage_chart(summary, date_str)
            except Exception as chart_err:
                logger.warning(f"Could not generate usage chart: {chart_err}")
                
            return summary

        except Exception as e:
            logger.error(f"Error generating daily summary file for {date_str}: {e}", exc_info=True)
            return None

    # --- Keep calculation helpers needed by _generate_daily_summary ---
    def _calculate_focus_score(self, productive_apps: List[str], distraction_apps: List[str], app_breakdown: List[Dict], total_time: int) -> int:
        if total_time <= 0: return 0
        productive_time = sum(app["timeSpent"] for app in app_breakdown if app["appName"] in productive_apps)
        distraction_time = sum(app["timeSpent"] for app in app_breakdown if app["appName"] in distraction_apps)
        neutral_time = max(0, total_time - productive_time - distraction_time)
        weighted_score = (productive_time * 1.0) + (neutral_time * 0.5) - (distraction_time * 1.0)
        normalized_score = weighted_score / total_time if total_time > 0 else 0
        final_score = max(0, min(100, int((normalized_score + 1) / 2 * 100)))
        return final_score
    def _is_productive_app(self, exe_path: str, titles: List[str]) -> bool:
        exe_lower = exe_path.lower(); title_concat_lower = " ".join(titles).lower()
        if any(pe in exe_lower for pe in PRODUCTIVE_EXES):
             if not any(dk in title_concat_lower for dk in DISTRACTION_TITLE_KEYWORDS): return True
        return False
    def _is_distraction_app(self, exe_path: str, titles: List[str]) -> bool:
        exe_lower = exe_path.lower(); title_concat_lower = " ".join(titles).lower()
        if any(de in exe_lower for de in DISTRACTION_EXES): return True
        if any(pe in exe_lower for pe in PRODUCTIVE_EXES) and any(dk in title_concat_lower for dk in DISTRACTION_TITLE_KEYWORDS): return True
        return False
    def _is_meeting_app(self, exe_path: str, title: str) -> bool:
         exe_lower = exe_path.lower(); title_lower = title.lower()
         if any(me in exe_lower for me in MEETING_EXES): return True
         if any(mk in title_lower for mk in MEETING_TITLE_KEYWORDS): return True
         return False

    # --- Main Agent Loop (Improved with regular summaries) ---
    async def run_agent_loop(self, interval: int = 5):
        """The main agent loop (async) - tracks focused window."""
        logger.info(f"Starting Focus Monitor agent loop (Window Tracking). Interval: {interval}s")
        self.window_start_time = time.time()
        self.last_summary_time = time.time()

        while True:
            main_loop_start_time = time.time()
            try:
                # Check desired state from backend and toggle internal state if needed
                self.toggle_active()

                # Skip window processing if not active, but still check for summary generation
                if not self.active:
                    # Check if it's time to generate a summary
                    time_since_last_summary = time.time() - self.last_summary_time
                    if time_since_last_summary >= self.summary_interval:
                        self._generate_daily_summary()
                        self._write_time_bucket_summary()
                        self.last_summary_time = time.time()
                        
                    await asyncio.sleep(max(0.1, interval - (time.time() - main_loop_start_time)))
                    continue
                
                # Day Change Check (using UTC)
                current_day_utc = self._get_current_utc_date()
                if current_day_utc != self.today:
                    logger.info(f"Date changed from {self.today} to {current_day_utc}. Generating previous day summary.")
                    self._generate_daily_summary(self.today) # Generate final summary for previous day
                    self._write_time_bucket_summary()
                    self.today = current_day_utc
                    logger.info(f"Updated current tracking date to {self.today}")
                
                # Get current focused window details
                current_window_info = self._get_focused_window_details()

                # Determine if the focused window changed significantly
                is_idle = current_window_info is None
                window_changed = False
                if is_idle:
                     if self.last_window_info is not None: window_changed = True # Active -> Idle
                elif self.last_window_info is None: window_changed = True # Idle -> Active
                elif (current_window_info["hwnd"] != self.last_window_info["hwnd"] or
                      current_window_info["title"] != self.last_window_info["title"] or
                      current_window_info["exe"] != self.last_window_info["exe"]): window_changed = True # Active -> Different Active

                if window_changed:
                    # Log duration for the *previous* window/state
                    if self.last_window_info:
                        duration = int(time.time() - self.window_start_time)
                        if duration > 0: self._log_window_activity(self.last_window_info, duration)

                    # Reset timer and update last window info (becomes None if now idle)
                    self.window_start_time = time.time()
                    self.last_window_info = current_window_info

                # Check if it's time to generate a summary
                time_since_last_summary = time.time() - self.last_summary_time
                if time_since_last_summary >= self.summary_interval:
                    self._generate_daily_summary()
                    self._write_time_bucket_summary()
                    self.last_summary_time = time.time()

                # Sleep until next interval
                elapsed = time.time() - main_loop_start_time
                sleep_duration = max(0.1, interval - elapsed)
                await asyncio.sleep(sleep_duration)

            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received in agent loop.")
                break
            except Exception as e:
                 logger.error(f"Unhandled error in agent loop: {e}", exc_info=True)
                 await asyncio.sleep(interval * 2) # Wait longer after error

        # --- Cleanup on exit ---
        logger.info("Agent loop finished. Performing final cleanup...")
        if self.last_window_info: # Log final activity
            duration = int(time.time() - self.window_start_time)
            if duration > 0: self._log_window_activity(self.last_window_info, duration)
        self._generate_daily_summary() # Generate final summary for the last active day
        self._write_time_bucket_summary()
        logger.info("Focus Monitor agent stopped.")


async def main_async():
    parser = argparse.ArgumentParser(description="Focus Monitor Agent (Window Tracking)")
    parser.add_argument("--output-dir", "-o", help="Optional directory to store logs (defaults to script directory)")
    parser.add_argument("--api-url", "-a", help="Backend API URL for status checks (optional)")
    parser.add_argument("--interval", "-i", type=int, default=5, help="Sampling interval in seconds")
    parser.add_argument("--summary-interval", "-s", type=int, default=300, help="Interval for generating summaries (seconds)")
    parser.add_argument("--no-api-check", action="store_true", help="Disable checking backend API for status.")
    parser.add_argument("--force-summary", action="store_true", help="Force generate a summary immediately on start")
    args = parser.parse_args()

    # If output directory is specified, verify it exists
    if args.output_dir:
        output_dir_path = Path(args.output_dir)
        if not output_dir_path.is_dir():
            logger.critical(f"Output directory not found: {args.output_dir}")
            sys.exit(1)
    else:
        output_dir_path = None  # Will default to script directory

    api_url = None if args.no_api_check else args.api_url

    agent = FocusMonitorAgent(output_dir_path, api_url)
    agent.summary_interval = args.summary_interval
    
    # Generate a summary immediately if requested
    if args.force_summary:
        agent._generate_daily_summary()
        agent._write_time_bucket_summary()
        agent.last_summary_time = time.time()

    tasks = [asyncio.create_task(agent.run_agent_loop(args.interval))]
    if api_url:
        tasks.append(asyncio.create_task(agent.check_backend_status()))

    try:
        await tasks[0] # Wait for main loop
    except asyncio.CancelledError: logger.info("Agent loop task cancelled.")
    finally: # Cleanup background tasks
        for task in tasks[1:]:
            if not task.done(): task.cancel()
        await asyncio.gather(*tasks[1:], return_exceptions=True)
        logger.info("All background tasks finished.")


if __name__ == "__main__":
     try:
         asyncio.run(main_async())
     except KeyboardInterrupt:
         logger.info("Focus Monitor stopped by user (main).")
     except Exception as main_err:
         logger.critical(f"Focus Monitor exited: {main_err}", exc_info=True)
