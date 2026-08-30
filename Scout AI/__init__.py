# agents/__init__.py

from .research_agent import research_agent
from .analysis_agent import analysis_agent
from .writer_agent import writer_agent


SUBAGENTS = [
    research_agent,
    analysis_agent,
    writer_agent,
]