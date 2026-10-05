"""LLM error classification for resilient refinement loops.

Categorizes litellm/provider exceptions so callers can decide
whether to continue, record-and-break, or re-raise.
"""
from __future__ import annotations


def classify_llm_error(exc: Exception) -> tuple[str, str]:
    """Classify an LLM exception into a category and human-readable detail.

    Returns:
        (category, detail) where category is one of:
        - "llm_timeout": API call exceeded timeout
        - "llm_auth_error": Invalid API key or auth failure
        - "llm_rate_limit": Rate limited by provider
        - "llm_server_error": Provider returned 5xx
        - "llm_bad_request": Provider returned 4xx (not auth/rate)
        - "llm_connection_error": Network/connection failure
        - "llm_error": Uncategorized LLM/litellm error
        - "internal_error": Non-LLM exception
    """
    exc_type = type(exc).__name__
    exc_str = str(exc)
    type_lower = exc_type.lower()
    str_lower = exc_str.lower()

    if "timeout" in type_lower or "timeout" in str_lower:
        return "llm_timeout", f"{exc_type}: {exc_str}"

    if "auth" in type_lower or "authentication" in str_lower or "api key" in str_lower:
        return "llm_auth_error", f"{exc_type}: {exc_str}"

    if "ratelimit" in type_lower or "rate_limit" in str_lower or "rate limit" in str_lower or "429" in exc_str:
        return "llm_rate_limit", f"{exc_type}: {exc_str}"

    if any(code in exc_str for code in ("500", "502", "503", "504")) or "server error" in str_lower:
        return "llm_server_error", f"{exc_type}: {exc_str}"

    if "badrequest" in type_lower or "400" in exc_str:
        return "llm_bad_request", f"{exc_type}: {exc_str}"

    if any(k in str_lower for k in ("connection", "connect", "network", "dns", "socket")):
        return "llm_connection_error", f"{exc_type}: {exc_str}"

    module = type(exc).__module__ or ""
    if any(m in module for m in ("litellm", "openai", "httpx", "httpcore")):
        return "llm_error", f"{exc_type}: {exc_str}"

    return "internal_error", f"{exc_type}: {exc_str}"
