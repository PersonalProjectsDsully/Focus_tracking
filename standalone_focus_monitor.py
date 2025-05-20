#!/usr/bin/env python3
"""Command-line entrypoint for the Focus Monitor Agent."""

import asyncio
import logging

from focus_tracker.agent import main_async

if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logging.getLogger("focus_monitor").info("Focus Monitor stopped by user (main).")
    except Exception as main_err:
        logging.getLogger("focus_monitor").critical(
            f"Focus Monitor exited: {main_err}", exc_info=True
        )
