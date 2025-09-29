import logging
import sys
from pathlib import Path

def setup_logging(log_level = "INFO"):
    """  
    """
    # create logs directory
    Path("logs").mkdir(exist_ok=True)

    # configure root logger
    logging.basicConfig(
        level = getattr(logging, log_level),
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [
            logging.StreamHandler(sys.stdout), # console
            logging.FileHandler('logs/app.log') # File
        ]
    )

    # Reduce noise from third-party libraries

    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('langchain').setLevel(logging.INFO)
