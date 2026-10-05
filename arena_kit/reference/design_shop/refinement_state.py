"""State machine for Engineer refinement with checkpoint + rollback + journal.

Tracks best design across commits, rolls back on regression/failure.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from mjarena.design_shop.types import CommitEntry

logger = logging.getLogger(__name__)

def _qualification_score_label(scores: Optional[Dict[str, Any]] = None) -> str:
    return "Qualification score"


# The ledger shows the newest commits' change summaries in full and cuts older ones
# (on a sentence boundary) so a 50-commit ledger stays readable.
LEDGER_FULL_RECENT = 5
LEDGER_OLDER_CHARS = 300


def _cut_at_sentence(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    head = text[:limit]
    for sep in (". ", ".\n", "; ", "\n"):
        i = head.rfind(sep)
        if i >= limit // 2:
            return head[: i + 1].rstrip() + " …"
    return head.rstrip() + " …"


_SCORE_PARTS = ("engagement", "dominant_contact", "displacement", "destabilization", "self_stability")


def _score_breakdown(scores: Optional[Dict[str, Any]]) -> str:
    """' = wins 1/3 · engagement 0.20 · ...' when the qualification recorded its parts, else ''."""
    scores = scores or {}
    if "wins" not in scores or f"mean_{_SCORE_PARTS[0]}" not in scores:
        return ""
    bits = [f"wins {scores['wins']}/{scores.get('n_seeds', '?')}"]
    bits += [f"{name} {float(scores[f'mean_{name}']):.2f}" for name in _SCORE_PARTS if f"mean_{name}" in scores]
    return " = " + " · ".join(bits)


@dataclass
class RefinementState:
    """Checkpoint + rollback + journal for one refinement lineage.

    Usage::

        state = RefinementState(mode="unified")
        for commit_num in range(n_commits):
            xml, code = engineer(best_commit_robot_xml=state.current_best_for_creator,
                                 last_commit_robot_xml=state.last_commit_xml_for_creator, ...)
            score, feedback = verify(xml, code)
            improved = state.update(commit_num, score, xml, feedback, creator_summary=summary)
    """

    mode: Literal["morphology", "controller", "unified"]
    output_dir: Optional[Path] = None

    # Current best checkpoint
    best_score: float = field(default=float("-inf"))
    best_commit: int = -1
    best_content: str = ""
    best_processed_xml: str = ""
    best_feedback: str = ""
    best_creator_outputs: Dict[str, Any] = field(default_factory=dict)

    # For unified mode
    best_controller_code: str = ""
    best_qualification_passed: bool = False
    best_scores: Dict[str, Any] = field(default_factory=dict)

    # Highest-scoring bot (regardless of qualification) — for LLM reference

    # Latest attempt (regardless of pass/fail) — used as fallback when no best yet
    latest_content: str = ""
    latest_controller_code: str = ""

    # Journal of all commits
    journal: List[CommitEntry] = field(default_factory=list)

    # Raw LM history (prompt/response pairs) for --dump-prompt
    lm_history: List[Any] = field(default_factory=list)

    @property
    def best_round(self) -> int:
        """Backwards compat alias for best_commit."""
        return self.best_commit

    def update(
        self,
        round_num: int,
        score: float,
        content: str,
        verifier_feedback: str,
        creator_summary: str,
        processed_xml: str = "",
        creator_outputs: Optional[Dict[str, Any]] = None,
        controller_code: str = "",
        validation_passed: bool = True,
        qualification_passed: bool = False,
        scores: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Record an attempt. Returns True if this is a new best."""
        score = round(score, 2)
        score_meta = dict(scores or {})
        score_meta.setdefault("qualification_score", score)

        candidate_rank = self._selection_rank(
            score=score,
            validation_passed=validation_passed,
            qualification_passed=qualification_passed,
            scores=score_meta,
        )
        best_rank = self._selection_rank(
            score=self.best_score,
            validation_passed=self.best_commit >= 0,
            qualification_passed=self.best_qualification_passed,
            scores=self.best_scores,
        )
        is_best = validation_passed and candidate_rank > best_rank

        diff_from_best = ""
        if not is_best and self.best_commit >= 0:
            diff_from_best = self._build_diff(creator_summary, verifier_feedback)

        outputs = creator_outputs or {}
        bot_name = outputs.get("name", "")


        entry = CommitEntry(
            commit_num=round_num,
            score=score,
            is_best=is_best,
            best_at_time=round_num if is_best else self.best_commit,
            creator_summary=creator_summary,
            diff_from_best=diff_from_best,
            verifier_feedback=verifier_feedback,
            validation_passed=validation_passed,
            qualification_passed=qualification_passed,
            bot_name=bot_name,
            scores=score_meta,
            reasoning=outputs.get("reasoning", ""),
            improvement_plan=outputs.get("improvement_plan", ""),
            design_strategy=outputs.get("design_strategy", ""),
            hardware_plan=outputs.get("hardware_plan", ""),
            combat_plan=outputs.get("combat_plan", ""),
        )
        self.journal.append(entry)

        # Always track the latest attempt for fallback
        if content:
            self.latest_content = content
        if controller_code:
            self.latest_controller_code = controller_code

        if is_best:
            self.best_score = score
            self.best_commit = round_num
            self.best_content = content
            self.best_processed_xml = processed_xml
            self.best_feedback = verifier_feedback
            self.best_creator_outputs = creator_outputs or {}
            self.best_controller_code = controller_code
            self.best_qualification_passed = qualification_passed
            self.best_scores = score_meta

        if self.output_dir:
            self._save_checkpoint(round_num, content, processed_xml,
                                  controller_code, is_best)

        return is_best

    @property
    def current_best_for_creator(self) -> str:
        """XML the Engineer should refine from (best, or latest if no best yet)."""
        if self.best_content:
            return self.best_content
        # No successful commit yet — return the latest attempt so the engineer
        # can see what it produced and fix it, rather than starting from scratch.
        return self.latest_content

    @property
    def current_best_controller_for_creator(self) -> str:
        """Controller code the Engineer should refine from (best, or latest if no best yet)."""
        if self.best_controller_code:
            return self.best_controller_code
        return self.latest_controller_code

    def _same_as_best_pointer(self) -> str:
        if self.best_commit >= 0:
            return f"same as best_commit (commit {self.best_commit + 1})"
        return "same as best_commit (this is the latest attempt; no commit has passed validation yet)"

    @property
    def _last_commit_is_best(self) -> bool:
        """True when the last commit IS the best commit (both slots would repeat verbatim)."""
        return bool(self.journal) and self.journal[-1].commit_num == self.best_commit

    def _last_xml_same_as_best(self) -> bool:
        return bool(self.latest_content) and self.latest_content == self.current_best_for_creator

    def _last_controller_same_as_best(self) -> bool:
        return bool(self.latest_controller_code) and (
            self.latest_controller_code == self.current_best_controller_for_creator
        )

    @property
    def last_commit_xml_for_creator(self) -> str:
        """XML of the most recent attempt — the design the latest feedback describes.

        A one-line pointer when that attempt is also the current best, so the prompt
        does not carry the same design twice.
        """
        if self._last_commit_is_best:
            return self._same_as_best_pointer()
        if self._last_xml_same_as_best():
            note = "" if self._last_controller_same_as_best() else "; the last commit changed only the controller — see last_commit_controller_code"
            return self._same_as_best_pointer() + note
        return self.latest_content

    @property
    def last_commit_controller_for_creator(self) -> str:
        """Controller of the most recent attempt (same shortcut as the XML)."""
        if self._last_commit_is_best:
            return self._same_as_best_pointer()
        if self._last_controller_same_as_best():
            note = "" if self._last_xml_same_as_best() else "; the last commit changed only the robot_xml — see last_commit_robot_xml"
            return self._same_as_best_pointer() + note
        return self.latest_controller_code

    def format_journal_for_creator(self) -> str:
        """Design ledger: full history with PASS/FAIL, scores, and markers."""
        if not self.journal:
            return ""

        lines = []
        seen_xml: Dict[str, int] = {}
        recent = self.journal[-LEDGER_FULL_RECENT:]

        for entry in self.journal:
            name = entry.bot_name or f"Bot"
            score_label = self._format_entry_score(entry)

            if not entry.validation_passed:
                # Extract failure reason from verifier feedback
                fail_reason = self._extract_fail_reason(entry.verifier_feedback)
                header = f"=== Commit {entry.commit_num + 1}: {name} | FAIL: {fail_reason} ==="
            elif not entry.qualification_passed:
                header = f"=== Commit {entry.commit_num + 1}: {name} | QUALIFIED: NO | {score_label} ==="
            elif entry.is_best:
                marker = "CURRENT BEST" if entry.commit_num == self.best_commit else "NEW BEST"
                header = f"=== Commit {entry.commit_num + 1}: {name} | QUALIFIED | {score_label} [{marker}] ==="
            else:
                header = f"=== Commit {entry.commit_num + 1}: {name} | QUALIFIED | {score_label} [REGRESSION] ==="

            lines.append(header)
            if entry.improvement_plan:
                lines.append(f"Improvement plan: {entry.improvement_plan}")
            lines.append(self._summary_for_ledger(entry.creator_summary, recent=entry in recent))
            xml_key = self._commit_xml_key(entry.commit_num)
            if xml_key and xml_key in seen_xml:
                lines.append(
                    f"(robot_xml physically identical to commit {seen_xml[xml_key] + 1} — "
                    "only comments, descriptions or the model name differ)"
                )
            elif xml_key:
                seen_xml[xml_key] = entry.commit_num
            lines.append("")

        if self.best_commit < 0:
            lines.append("Current best: none yet — no commit has passed validation.")
        else:
            qual_str = "QUALIFIED" if self.best_qualification_passed else "NOT QUALIFIED"
            lines.append(
                f"Current best: Commit {self.best_commit + 1} ({qual_str}, "
                f"{self._format_best_score_summary()})"
            )

        return "\n".join(lines)

    def _selection_rank(
        self,
        *,
        score: float,
        validation_passed: bool,
        qualification_passed: bool,
        scores: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """Return a comparison tuple for selecting the best commit."""
        scores = scores or {}
        qualification_score = float(scores.get("qualification_score", score))

        return (
            1 if validation_passed else 0,
            1 if qualification_passed else 0,
            score,
        )

    def _format_entry_score(self, entry: CommitEntry) -> str:
        """Score total plus its parts, so the model can see which part moved."""
        return f"{_qualification_score_label()}: {entry.score:.2f}/6.0{_score_breakdown(entry.scores)}"

    def _format_best_score_summary(self) -> str:
        """Human-readable score summary for the current best commit."""
        return f"{_qualification_score_label()}: {self.best_score:.2f}/6.0"

    @staticmethod
    def _summary_for_ledger(summary: str, *, recent: bool) -> str:
        return summary.strip() if recent else _cut_at_sentence(summary, LEDGER_OLDER_CHARS)

    def _commit_xml_key(self, commit_num: int) -> str:
        """Normalised robot.xml of a commit on disk, or "" when unavailable."""
        if not self.output_dir:
            return ""
        path = self.output_dir / "refinement" / f"commit_{commit_num}" / "robot.xml"
        if not path.is_file():
            return ""
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"<!--.*?-->", "", text, flags=re.S)             # comments
        text = re.sub(r'\s+description="[^"]*"', "", text)              # design-intent labels
        text = re.sub(r'\s+model="[^"]*"', "", text)                    # cosmetic model name
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _extract_fail_reason(feedback: str) -> str:
        """Extract a short failure reason from verifier feedback."""
        if not feedback:
            return "validation failed"
        lines = feedback.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith("──") or "FAILED" not in stripped:
                continue
            label = re.sub(r"^[─\s]+|[─\s]+FAILED\s*$", "", stripped)
            reason = next((l.strip() for l in lines[i + 1:] if l.strip()), "")
            return f"{label}: {reason}"[:140] if reason else label[:100]
        for line in lines:
            if "FAILED" in line or "FAIL" in line:
                return line.strip()[:100]
        return "validation failed"

    def save_ledger_text(self) -> Optional[Path]:
        """Write the ledger exactly as the next prompt will show it: refinement/design_ledger.txt."""
        if not self.output_dir:
            return None
        path = Path(self.output_dir) / "refinement" / "design_ledger.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.format_journal_for_creator(), encoding="utf-8")
        return path

    def to_json(self) -> str:
        """Serialize journal to JSON for disk persistence."""
        data = {
            "mode": self.mode,
            "best_score": self.best_score,
            "best_commit": self.best_commit,
            "best_qualification_passed": self.best_qualification_passed,
            "best_scores": self.best_scores,
            "commits": [
                {
                    "commit_num": e.commit_num,
                    "score": e.score,
                    "validation_passed": e.validation_passed,
                    "qualification_passed": e.qualification_passed,
                    "bot_name": e.bot_name,
                    "is_best": e.is_best,
                    "best_at_time": e.best_at_time,
                    "creator_summary": e.creator_summary,
                    "diff_from_best": e.diff_from_best,
                    "verifier_feedback": e.verifier_feedback,
                    "scores": e.scores,
                    "reasoning": e.reasoning,
                    "improvement_plan": e.improvement_plan,
                    "design_strategy": e.design_strategy,
                    "hardware_plan": e.hardware_plan,
                    "combat_plan": e.combat_plan,
                }
                for e in self.journal
            ],
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    def _build_diff(self, creator_summary: str, verifier_feedback: str) -> str:
        """Build a diff description."""
        parts = []
        if creator_summary:
            parts.append(creator_summary[:200])
        failed = self._extract_failed_lines(verifier_feedback)
        if failed:
            parts.append("Failed: " + "; ".join(failed[:2]))
        return " | ".join(parts) if parts else ""

    @staticmethod
    def _extract_failed_lines(feedback: str) -> List[str]:
        """Extract FAILED step labels from verifier feedback."""
        if not feedback:
            return []
        failed = []
        for line in feedback.split("\n"):
            line = line.strip()
            if "FAILED" in line:
                m = re.match(r"^[─\-]+\s*(.+?)\s*[─\-]+\s*FAILED", line)
                if m:
                    failed.append(m.group(1).strip())
                else:
                    failed.append(line[:100])
        return failed

    def _save_checkpoint(
        self,
        commit_num: int,
        content: str,
        processed_xml: str,
        controller_code: str,
        is_best: bool,
    ) -> None:
        """Save checkpoint files to disk.

        Each commit gets its own directory with robot.xml and controller.py.
        The journal tracks which commit is best.
        """
        if not self.output_dir:
            return

        refine_dir = self.output_dir / "refinement"
        commit_dir = refine_dir / f"commit_{commit_num}"
        commit_dir.mkdir(parents=True, exist_ok=True)

        # Always save this commit's artifacts
        if content:
            (commit_dir / "robot.xml").write_text(content, encoding="utf-8")
        if processed_xml:
            (commit_dir / "robot_processed.xml").write_text(processed_xml, encoding="utf-8")
        if controller_code:
            (commit_dir / "controller.py").write_text(controller_code, encoding="utf-8")

        # Journal is the index — tracks which commit is best
        journal_file = refine_dir / "journal.json"
        journal_file.write_text(self.to_json(), encoding="utf-8")
