import sys
from pathlib import Path

# Put the SELF-CONTAINED bundle first so tests exercise the vendored mjarena
# (design-lab-harness/mjarena), not a live arena checkout. arena_kit/ holds
# config.py + harness_lib.py + agent.py's siblings; design-lab-harness/ holds agent.py
# and the bundled mjarena/ + configs/. We intentionally do NOT add the outer arena
# repo root — the harness must stand on its own.
_MH = Path(__file__).resolve().parent.parent          # design-lab-harness/ (bundle root)
_KIT = _MH / "arena_kit"
for p in (str(_KIT), str(_MH)):                        # _MH ends up first (highest priority)
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _short_sparring_matches(request, monkeypatch):
    """Sparring matches play the tournament's 300 s per seed. Every test that runs a
    match would take minutes, so outside the enforcement-parity tests (which pin the
    real lengths) the match config's match_time is shortened; qualification keeps
    its real 20 s so the inactivity rule still fires."""
    if request.module.__name__.endswith(("test_enforcement_parity", "test_upstream_environment")):
        return
    import harness_lib
    real = harness_lib.get_match_config

    def short():
        cfg = real()
        cfg.match_time = 0.5
        return cfg

    monkeypatch.setattr(harness_lib, "get_match_config", short)
