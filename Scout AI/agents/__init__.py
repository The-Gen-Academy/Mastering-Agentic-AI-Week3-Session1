# agents/__init__.py
import sys
import os

# Add parent directory to path so sibling modules can be imported
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from research_agent import research_agent
from analysis_agent import analysis_agent
from writer_agent import writer_agent

SUBAGENTS = [
    research_agent,
    analysis_agent,
    writer_agent,
]

__all__ = ["research_agent", "analysis_agent", "writer_agent", "SUBAGENTS"]
