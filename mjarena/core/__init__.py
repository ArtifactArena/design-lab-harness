"""Bot building module for LLM-generated robots."""
from .bot_builder import build_bot
from .build_config import BuildConfig
from .runner import generate_bots
from mjarena.agents.types import BotArtifact

__all__ = ["build_bot", "BuildConfig", "generate_bots", "BotArtifact"]
