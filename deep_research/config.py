"""
Model Configuration for Deep Research Agent.

This module provides centralized model configuration to easily switch between
different LLM providers (OpenAI, Anthropic, Google Gemini, Cerebras).

To change the model, simply update the DEFAULT_MODEL variable below.

Examples:
    - OpenAI: "gpt-5", "gpt-5-mini"
    - Anthropic: "claude-4-5-opus", "claude-4-5-sonnet"
    - Google Gemini: "gemini-2.5-pro", "gemini-3-flash-preview", "gemini-3-pro-preview"
    - Z.AI GLM: "glm-5"
    - Cerebras: "qwen-3-32b", "qwen-3-235b-a22b-instruct-2507"
"""

import os
from langchain.chat_models import init_chat_model
from langchain_cerebras import ChatCerebras
from langchain_openai import ChatOpenAI

# ===== CONFIGURATION =====
# Change this single variable to switch models across the entire application
# now means the subagents
DEFAULT_MODEL = "gemini-3-flash-preview" #"gemini-3.1-pro-preview" #"gemini-3-flash-preview"

# Supervisor model can be configured independently from subagents.
# Defaults to OpenAI GPT-5 unless overridden via environment variable.
SUPERVISOR_MODEL = os.environ.get("SUPERVISOR_MODEL", "gpt-5.2")

# Select the model for simple tasks like summarization
LITE_MODEL = "gemini-2.5-flash-lite"

# Select the prompt version to use ("ORIGINAL" or "FINANCE_V1")
PROMPT_VERSION = "K1_2"

# Research time window (in minutes) - controls the expected task duration
RESEARCH_TIME_MIN_MINUTES = 15
RESEARCH_TIME_MAX_MINUTES = 30
RESEARCH_STRICT_TIMEOUT_MINUTES = RESEARCH_TIME_MAX_MINUTES + 1.0  # Hard stop limit

# Maximum number of research iterations (tool calls)
MAX_RESEARCHER_ITERATIONS = 30

# Per-subagent time limit (seconds) - graceful stop; agent finishes current iteration then compresses
SUBAGENT_TIMEOUT_SECONDS = 600  # 10 minutes

# Optional: Set max tokens for writer models (None = no limit)
DEFAULT_MAX_TOKENS = None  # Previously was 32000-40000, now unlimited

# Controls whether reports (final and subtopic) are saved to disk as files
SAVE_REPORT_TO_FILE: bool = True  # Set to False to keep everything in memory only

# Controls whether the subtopic evaluation and generation workflow runs after the final report
ENABLE_SUBTOPIC_GENERATION: bool = False  # Set to False to skip subtopic reports entirely

# Controls whether supervisor-subagent research trace logging/compression is enabled
# Set DISABLE_RESEARCH_TRACE=1/true/yes/on to disable trace generation
ENABLE_RESEARCH_TRACE: bool = False #os.environ.get("DISABLE_RESEARCH_TRACE", "").strip().lower() not in {
#    "1",
#    "true",
#    "yes",
#    "on",
#}

# Model fallback chain for rate limiting resilience
# Tried in order: if primary fails with rate limit, try next in chain
# NOTE: Swap order if a model is exhausted for the day
MODEL_FALLBACK_CHAIN = [
    "gemini-3-flash-preview",  # Fallback 1
    "gemini-3.1-pro-preview",  ## Fallback 2
    "gemini-2.5-pro",   # Fallback 3 - more stable and less likely to be exhausted, but may have lower performanc
]

# Optional supervisor-specific fallback chain.
# If SUPERVISOR_MODEL_FALLBACK_CHAIN is unset, defaults to SUPERVISOR_MODEL.
# Example env value: "claude-4-5-sonnet,gpt-5-mini,gemini-2.5-pro"
_supervisor_chain_env = os.environ.get("SUPERVISOR_MODEL_FALLBACK_CHAIN", "").strip()
if _supervisor_chain_env:
    SUPERVISOR_MODEL_FALLBACK_CHAIN = [
        m.strip() for m in _supervisor_chain_env.split(",") if m.strip()
    ]
else:
    SUPERVISOR_MODEL_FALLBACK_CHAIN = [SUPERVISOR_MODEL]




# Controls the destination of the research outputs
# Options: "file" (standard markdown), "db" (structured database), "both"
OUTPUT_MODE = "file"

# Controls where logs are stored
# Options: "file" (default), "db", "both"
LOG_MODE = "file" #os.environ.get("LOG_MODE", "file")




def get_model(model_name: str = DEFAULT_MODEL, temperature: float = 0, max_tokens: int = None):
    """
    Get a chat model instance based on the model name.

    Automatically detects the provider based on the model name and returns
    the appropriate LangChain chat model instance.

    Args:
        model_name: Name of the model (e.g., "gpt-4o", "claude-3-5-sonnet-20240620")
        temperature: Temperature for generation (default: 0)
        max_tokens: Maximum tokens to generate (default: None, uses model default)

    Returns:
        LangChain chat model instance

    Raises:
        ValueError: If the model name cannot be mapped to a known provider
    """
    model_lower = model_name.lower()

    # Z.AI GLM models (OpenAI-compatible API)
    if "glm" in model_lower:
        kwargs = {
            "model": model_name,
            "api_key": os.getenv("ZHIPUAI_API_KEY"),
            "base_url": "https://api.z.ai/api/paas/v4/",
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return ChatOpenAI(**kwargs)

    # Cerebras models (Qwen)
    # had issues running with cerebras through langchain. cannot be relied upon.
    if "qwen" in model_lower:
        kwargs = {
            "model": model_name,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return ChatCerebras(**kwargs)

    # Google Gemini models
    if "gemini" in model_lower:
        # Use google_genai provider prefix for init_chat_model
        if not model_name.startswith("google_genai:"):
            model_name = f"google_genai:{model_name}"
        kwargs = {
            "temperature": temperature,
            "max_retries": 1,  # Set to 1 to disable SDK retries (0 defaults to 5). This allows LangChain fallback to trigger.
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return init_chat_model(model_name, **kwargs)

    # Anthropic Claude models
    if "claude" in model_lower:
        if not model_name.startswith("anthropic:"):
            model_name = f"anthropic:{model_name}"
        kwargs = {"temperature": temperature}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return init_chat_model(model_name, **kwargs)

    # OpenAI models (default fallback)
    if "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
        if not model_name.startswith("openai:"):
            model_name = f"openai:{model_name}"
        kwargs = {"temperature": temperature}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return init_chat_model(model_name, **kwargs)

    # If we can't determine the provider, raise an error
    raise ValueError(
        f"Cannot determine provider for model: {model_name}. "
        f"Supported providers: OpenAI (gpt-*), Anthropic (claude-*), "
        f"Google (gemini-*), Z.AI (glm-*), Cerebras (qwen-*)"
    )


def get_primary_model():
    """
    Get the primary model for reasoning and decision-making.

    Returns:
        LangChain chat model instance configured with DEFAULT_MODEL
    """
    return get_model(DEFAULT_MODEL, temperature=0)


def get_writer_model(max_tokens: int = DEFAULT_MAX_TOKENS):
    """
    Get the writer model for long-form content generation.

    Args:
        max_tokens: Maximum tokens to generate (default: DEFAULT_MAX_TOKENS)

    Returns:
        LangChain chat model instance configured for writing
    """
    return get_model(DEFAULT_MODEL, temperature=0, max_tokens=max_tokens)


def get_lite_model(max_tokens: int = None):
    """
    Get the lite model for simple tasks like summarization.

    Args:
        max_tokens: Maximum tokens to generate

    Returns:
        LangChain chat model instance configured with LITE_MODEL
    """
    return get_model(LITE_MODEL, temperature=0, max_tokens=max_tokens)


# ===== RESILIENT INVOCATION =====

import logging
from langchain_core.runnables import RunnableWithFallbacks

_fallback_logger = logging.getLogger("model_fallback")


def get_resilient_model(
    tools: list = None,
    max_tokens: int = None,
    temperature: float = 0,
    model_chain: list[str] | None = None,
):
    """
    Get a model that automatically falls back to alternates on failure.

    Uses LangChain's built-in .with_fallbacks() mechanism.
    Models are tried in order from MODEL_FALLBACK_CHAIN.

    Fallback triggers on:
    - ClientError (429 RESOURCE_EXHAUSTED - rate limits)
    - ServerError (503 UNAVAILABLE - model overloaded)
    - Any other Exception as catch-all

    Args:
        tools: Optional list of tools to bind to the models
        max_tokens: Optional max tokens for generation
        temperature: Temperature for generation (default: 0)
        model_chain: Optional explicit model chain (ordered primary -> fallbacks)

    Returns:
        A Runnable capable of falling back to alternate models
    """
    # Import exception types from google.genai SDK when available.
    # Keep a generic Exception fallback for non-Google providers.
    try:
        from google.genai.errors import ClientError, ServerError, APIError
        exceptions_to_handle = (ClientError, ServerError, APIError, Exception)
    except Exception:
        exceptions_to_handle = (Exception,)

    chain = model_chain or MODEL_FALLBACK_CHAIN
    if not chain:
        raise ValueError("Model chain cannot be empty")

    # Create the primary model
    primary_name = chain[0]
    primary_model = get_model(primary_name, temperature=temperature, max_tokens=max_tokens)
    if tools:
        primary_model = primary_model.bind_tools(tools)

    # Create the fallback chain
    fallbacks = []
    for model_name in chain[1:]:
        fb = get_model(model_name, temperature=temperature, max_tokens=max_tokens)
        if tools:
            fb = fb.bind_tools(tools)
        fallbacks.append(fb)

    # Return the combined runnable
    if not fallbacks:
        return primary_model

    # Configure which exceptions trigger fallback:
    # - ClientError: 4xx errors including 429 RESOURCE_EXHAUSTED (rate limits)
    # - ServerError: 5xx errors including 503 UNAVAILABLE (model overloaded)
    # - Exception: Catch-all for any other errors
    _fallback_logger.info(f"Created resilient model chain: {chain}")
    return primary_model.with_fallbacks(
        fallbacks,
        exceptions_to_handle=exceptions_to_handle
    )


def get_supervisor_model(tools: list = None, max_tokens: int = None, temperature: float = 0):
    """Get a resilient model runnable for the supervisor role only."""
    return get_resilient_model(
        tools=tools,
        max_tokens=max_tokens,
        temperature=temperature,
        model_chain=SUPERVISOR_MODEL_FALLBACK_CHAIN,
    )
