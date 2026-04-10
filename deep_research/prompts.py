"""
Dynamic Prompt Loader for Deep Research Agent.

This module acts as a router to select specific prompt versions based on the
PROMPT_VERSION setting in src/config.py.
"""

from deep_research.config import PROMPT_VERSION

if PROMPT_VERSION == "FINANCE_V1":
    from deep_research.prompts_finance_v1 import *
elif PROMPT_VERSION == "K1_2":
    from deep_research.prompts_k1_2 import *
elif PROMPT_VERSION == "ORIGINAL":
    from deep_research.prompts_original import *
else:
    # Fallback to original if version unknown
    from deep_research.prompts_original import *
