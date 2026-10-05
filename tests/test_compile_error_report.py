"""A controller syntax error names the line and quotes it."""
import mjarena.core.unified_builder  # noqa: F401  (import order)
from mjarena.design_shop.rules.software_rules import compile_policy


def test_syntax_error_quotes_the_offending_line():
    code = "def policy_step(obs):\n    import numpy as np as _np\n    return {}\n"
    fn, step = compile_policy(code)
    assert fn is None and not step.passed
    assert "line 2" in step.message and "import numpy as np as _np" in step.message
    assert "<unknown>" not in step.message
