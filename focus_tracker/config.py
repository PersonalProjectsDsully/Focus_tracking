import os
import re
from pathlib import Path

# Base directory of the project (repository root)
BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPT_DIR = BASE_DIR

# Configure logging location
FOCUS_LOG = SCRIPT_DIR / "focus_monitor.log"

# --- Configuration (Copied for backend calculation consistency) ---
PRODUCTIVE_EXES = {
    "code.exe",
    "pycharm",
    "idea",
    "webstorm",
    "goland",
    "clion",
    "word",
    "excel",
    "powerpnt",
    "outlook",
    "chrome.exe",
    "firefox.exe",
    "msedge.exe",
    "safari",
    "cmd.exe",
    "powershell.exe",
    "terminal",
    "wt.exe",
    "explorer.exe",
    "photoshop",
    "illustrator",
    "figma",
    "xd",
    "blender",
    "unity",
    "docker",
    "virtualbox",
    "vmware",
    "gitkraken",
    "postman",
    "obsidian",
}

DISTRACTION_EXES = {
    "steam.exe",
    "epicgameslauncher",
    "origin.exe",
    "gog galaxy",
    "spotify.exe",
    "discord.exe",
    "slack.exe",
    "netflix",
    "hulu",
    "disneyplus",
    "whatsapp",
    "telegram",
    "signal",
}

DISTRACTION_TITLE_KEYWORDS = {
    "youtube",
    "facebook",
    "twitter",
    "reddit",
    "netflix",
    "hulu",
    "twitch",
    "instagram",
    "9gag",
    "game",
    "play",
    "tiktok",
    "pinterest",
}

MEETING_EXES = {"teams.exe", "zoom.exe", "webex", "skype.exe", "slack.exe"}
MEETING_TITLE_KEYWORDS = {
    "meet",
    "meeting",
    "call",
    "webinar",
    "huddle",
    "zoom meeting",
    "microsoft teams meeting",
    "google meet",
}

OCR_APPS = {"notepad.exe", "word.exe", "acrord32.exe"}

# --- Browser Profile Detection Patterns ---
EDGE_PROFILE_PATTERN = re.compile(r"Microsoft Edge(?:\s*-\s*(.+))?$")
CHROME_PROFILE_PATTERN = re.compile(r".*Google Chrome(?:\s*-\s*(.+))?$")
FIREFOX_PROFILE_PATTERN = re.compile(r"Mozilla Firefox(?:\s*-\s*(.+))?$")
