"""Check that a learner can follow the lesson using only what is in it.

Usage:  python tests/check_self_contained.py

Fails if a learner-facing page:
- points to something outside the lesson that may not exist (joining instructions, an email, a link "sent" or "in the chat");
- has leftovers from the live event or earlier versions (breakout rooms, "raise your hand", FIXME, TODO, Miniforge, conda);
- links to a file that is not in the lesson;
- uses DoubleML in an episode without installing it first (a fresh Colab notebook has no DoubleML);
- lists different package versions on the Setup page from requirements.txt.
Standard library only.
"""
import re
import sys
from pathlib import Path

LESSON = Path(__file__).resolve().parent.parent
PAGES = sorted(LESSON.glob("episodes/*.md")) + sorted(LESSON.glob("learners/*.md")) + sorted(LESSON.glob("instructors/*.md")) + \
    sorted(LESSON.glob("profiles/*.md")) + [LESSON / n for n in ("index.md", "CODE_OF_CONDUCT.md", "CONTRIBUTING.md", "LICENSE.md", "config.yaml")]
OUTSIDE = r"joining instructions|link (sent|to follow|in the chat)|sent (to you|with|by email)|in the (Teams )?chat|breakout|raise your hand" \
          r"|FIXME|TODO|miniforge|\bconda\b|environment\.yml|Winter School"
problems = []

for page in PAGES:
    text = page.read_text()
    name = page.relative_to(LESSON)
    for n, line in enumerate(text.splitlines(), 1):
        if m := re.search(OUTSIDE, line, flags=re.I):
            problems.append(f"{name}:{n}: '{m.group(0)}' (outside the lesson or left over): {line.strip()[:90]}")
    for target in re.findall(r"\]\((?!https?:|mailto:|#)([^)\s]+)\)", text):
        if not (page.parent / target.split("#")[0]).exists():
            problems.append(f"{name}: link to a file that is not in the lesson: {target}")

for ep in sorted(LESSON.glob("episodes/*.md")):
    blocks = re.findall(r"^```python[^\n]*\n(.*?)^```", ep.read_text(), flags=re.S | re.M)
    uses = next((i for i, b in enumerate(blocks) if re.search(r"\bdoubleml\b|\bdml\.", b)), None)
    if uses is not None and not any("!pip install doubleml" in b for b in blocks[:uses + 1]):
        problems.append(f"{ep.relative_to(LESSON)}: uses DoubleML before installing it (fails in a fresh Colab notebook)")

pins = {l.strip() for l in (LESSON / "requirements.txt").read_text().splitlines() if "==" in l and not l.startswith("#")}
setup_pins = set(re.findall(r"[\w-]+==[\d.]+", (LESSON / "learners/setup.md").read_text()))
if pins != setup_pins:
    problems.append(f"learners/setup.md and requirements.txt list different packages: {sorted(pins ^ setup_pins)}")

print("\n".join(problems) or "Self-contained: no outside references, broken links or install gaps found.")
sys.exit(1 if problems else 0)
