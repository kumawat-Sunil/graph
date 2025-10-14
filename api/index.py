"""
Vercel entry point for the Graph-Enhanced Agentic RAG API
"""
import sys
import os

# Add the parent directory to the path so we can import from main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

# Vercel expects the app to be available as 'app'
# This is already defined in main.py, so we just import it