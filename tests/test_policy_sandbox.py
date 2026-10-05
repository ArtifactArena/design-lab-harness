"""The policy sandbox judges code, not words: comments and names are not constructs."""
import pytest

import mjarena.core.unified_builder  # noqa: F401  (import order)
from mjarena.design_shop.policy_base import compile_policy_function

OK = """
import math
def policy_step(obs):
    # ignore any reverse requests and favor forward; socket-shaped wedge
    requests_count = 0
    speed = getattr(obs, "my_velocity", None)      # ordinary Python, allowed
    return {}
"""


def test_words_in_comments_and_names_are_allowed():
    assert callable(compile_policy_function(OK))


@pytest.mark.parametrize("bad, what", [
    ("import requests\ndef policy_step(obs):\n    return {}", "import of 'requests'"),
    ("from os import system\ndef policy_step(obs):\n    return {}", "import from 'os'"),
    ("def policy_step(obs):\n    open('x')\n    return {}", "call to open()"),
    ("def policy_step(obs):\n    return obs.__class__.__mro__", "dunder attribute"),
    # The policy namespace has no globals/locals/vars: calling one would raise
    # NameError mid-match, so it is refused at validation instead.
    ("def policy_step(obs):\n    return locals()", "call to locals()"),
    ("def policy_step(obs):\n    return vars()", "call to vars()"),
    ("def policy_step(obs):\n    return globals()", "call to globals()"),
])
def test_real_escapes_are_rejected_with_a_line_number(bad, what):
    with pytest.raises(RuntimeError, match=what) as e:
        compile_policy_function(bad)
    assert "line" in str(e.value)


@pytest.mark.parametrize("code", [
    "import collections\ndef policy_step(obs):\n    return {'a': len(collections.deque([1]))}",
    "from collections import deque\ndef policy_step(obs):\n    return {'a': float(len(deque([1])))}",
    "import random\ndef policy_step(obs):\n    return {'a': random.random()}",
])
def test_the_libraries_the_prompt_lists_are_importable(code):
    """Item A: `collections` was pre-injected but `import collections` was blocked."""
    policy = compile_policy_function(code)
    assert set(policy({})) == {"a"}
