"""The launchers must launch exactly the roster of record."""
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("launcher", ["run_big_experiment.sh", "run_parallel.sh"])
def test_launcher_default_models_equal_roster(launcher):
    roster = yaml.safe_load((ROOT / "configs/models/run-roster-2026-09.yaml").read_text())["llms"]
    expected = [re.sub(r"^configs/models/(.+)\.yaml$", r"\1", m) for m in roster]
    script = (ROOT / launcher).read_text()
    body = re.search(r"DEFAULT_MODELS=\(\n(.*?)\n\)", script, re.S).group(1)
    launched = [line.strip() for line in body.splitlines() if line.strip() and not line.strip().startswith("#")]
    assert launched == expected
    for name in launched:
        assert (ROOT / "configs/models" / f"{name}.yaml").is_file(), name
