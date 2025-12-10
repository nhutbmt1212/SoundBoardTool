
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

# Define log directory and file
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "loop_debug.log"

def setup_debug_logger():
    """Sets up a clean logger for loop debugging that resets on each run."""
    # Create logs directory if it doesn't exist
    if not LOG_DIR.exists():
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        
    # Clear previous log file
    with open(LOG_FILE, 'w') as f:
        f.write(f"=== Loop Debug Session Started at {datetime.now()} ===\n")
        
    # Configure logger
    logger = logging.getLogger('LoopDebug')
    logger.setLevel(logging.DEBUG)
    
    # File handler
    file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('[LOOP_DEBUG] %(message)s'))
    
    # Add handlers
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger

# Global logger instance
debug_logger = setup_debug_logger()

def log_loop_action(action, details=None):
    """Helper to log loop-related actions with consistent formatting."""
    msg = f"{action}"
    if details:
        msg += f" | Details: {details}"
    debug_logger.info(msg)
    # Force flush to ensure logs are written immediately
    for handler in debug_logger.handlers:
        handler.flush()
