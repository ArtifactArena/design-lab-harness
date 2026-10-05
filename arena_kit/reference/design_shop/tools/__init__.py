"""Design shop tools — qualification match runner (match_tools) and physics diagnostics.

Import the submodules directly; this package deliberately imports nothing eagerly so
that envs.sumo -> physics_diagnostics and match_tools -> eval.match_runner -> runner.episode
-> envs.sumo cannot form an import cycle.
"""
