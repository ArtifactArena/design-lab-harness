"""Two-stage Elo pipeline.

Stage 1: rank refinement commits per model by qualification composite.
Stage 2: per-model intra-RR over top-N, pick top K_max by intra-model Elo.
Stage 3: cross-model RR over pooled top-K + baselines.
Stage 4: derive top-1/3/K_max model Elos from saved cross-model matches.
"""
# Bootstrap mjarena.dspy_core fully before any caller imports
# mjarena.tournament.tournament transitively. The chain core -> dspy_core ->
# design_shop -> core.bot_builder is circular when entered through
# tournament; importing core first resolves dspy_core cleanly.
import mjarena.core  # noqa: F401
