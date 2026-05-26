#!/usr/bin/env python
"""Quick smoke test of gradio_app startup."""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    logger.info("Importing gradio_app...")
    import gradio_app
    logger.info("SUCCESS: gradio_app imported without errors")
    
    # Test demo interface creation
    logger.info("Checking demo interface...")
    if hasattr(gradio_app, 'demo'):
        logger.info(f"SUCCESS: demo interface created")
        print("READY: Gradio app is ready to run")
    else:
        logger.warning("WARNING: demo interface not found")
        
except Exception as e:
    logger.error(f"FAILED: {e}", exc_info=True)
    sys.exit(1)
