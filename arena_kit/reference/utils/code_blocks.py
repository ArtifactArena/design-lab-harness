"""Unambiguous extraction of the single fenced code block in a model reply.

The build loop must never guess which of several designs the model meant, and it
must never hand a half-stripped fence to a validator. One rule, applied to every
artefact (MJCF and controller source alike):

* text that does not open with a fence is already code and is returned byte for
  byte — no stripping, no re-indentation, no line-ending rewrite;
* text that opens with a fence must contain exactly one block, closed, with
  nothing but blank lines and thematic breaks (``---``, ``***``, ``___``) around
  it — anything else raises ``ValueError`` so the caller can report it;
* the block's content is returned exactly as written, including its own shorter
  fences, its indentation and its trailing blank line.
"""
from __future__ import annotations

import re

__all__ = ["extract_code_block"]

# CommonMark: a fence is 3+ backticks or tildes, indented by at most 3 spaces.
_OPEN_FENCE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
_LINE_END = re.compile(r"\r\n|\r|\n")


def _split_lines(text: str) -> list[str]:
    """Split into lines keeping each line's exact terminator (CR, LF or CRLF)."""
    lines: list[str] = []
    start = 0
    for match in _LINE_END.finditer(text):
        lines.append(text[start:match.end()])
        start = match.end()
    if start < len(text):
        lines.append(text[start:])
    return lines


def _body(line: str) -> str:
    """The line without its terminator."""
    return _LINE_END.sub("", line)


def _is_skippable(line: str) -> bool:
    """Blank lines and thematic breaks are formatting, never content."""
    stripped = _body(line).strip()
    if not stripped:
        return True
    condensed = stripped.replace(" ", "").replace("\t", "")
    # A thematic break: three or more of the same -, * or _ character.
    return (len(condensed) >= 3 and condensed[0] in "-*_"
            and condensed == condensed[0] * len(condensed))


def _opens_fence(line: str) -> re.Match | None:
    match = _OPEN_FENCE.match(_body(line))
    if match is None:
        return None
    # An info string may not contain the fence character (CommonMark); `a```b is
    # inline code, not a block opener.
    if match.group("fence")[0] in match.group("info"):
        return None
    return match


def _closes_fence(line: str, fence: str) -> bool:
    body = _body(line)
    closing = re.match(r"^ {0,3}(?P<fence>%s{%d,})[ \t]*$" % (re.escape(fence[0]), len(fence)), body)
    return closing is not None


def extract_code_block(text: str) -> str:
    """Return the one fenced block in *text*, or *text* itself when unfenced.

    Raises:
        ValueError: the reply opens a fence but does not contain exactly one
            closed block surrounded only by blank lines and thematic breaks.
    """
    lines = _split_lines(text)
    first = 0
    while first < len(lines) and _is_skippable(lines[first]):
        first += 1
    if first >= len(lines):
        return text

    opening = _opens_fence(lines[first])
    if opening is None:
        return text  # raw code: hand it back untouched
    fence = opening.group("fence")

    for index in range(first + 1, len(lines)):
        if not _closes_fence(lines[index], fence):
            continue
        content_start = sum(len(line) for line in lines[:first + 1])
        content_end = sum(len(line) for line in lines[:index])
        for trailing in lines[index + 1:]:
            if _is_skippable(trailing):
                continue
            if _opens_fence(trailing) is not None:
                raise ValueError(
                    "Expected one code block, found more than one: send a single "
                    "block containing the design you chose."
                )
            raise ValueError(
                "Expected one code block, found text after the closing fence: "
                f"{_body(trailing).strip()!r}"
            )
        return text[content_start:content_end]

    raise ValueError(
        f"Expected one code block: the {fence} fence opened on line {first + 1} "
        "is never closed."
    )
