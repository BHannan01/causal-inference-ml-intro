"""Check that the code on the masterclass slides appears, line for line, in the lesson and the workshop notebook.

Usage:  python tests/check_deck_code.py [path/to/deck.pptx]

With no argument it uses the newest "... vN.pptx" or "... - Version N.M - <date>.pptx" deck in the masterclass folder
(the folder above this lesson). Code on the slides is any text set in JetBrains Mono (a slide whose monospace text has
no "(" or "=" is printed output, such as the results table, and is skipped). Non-breaking spaces, which the Google
Slides export puts in the code, count as spaces, with a warning. Each slide's code must appear as one unbroken run of lines
(blank lines and trailing spaces ignored) inside a single ```python block of the episodes, and inside a single code
cell of each workshop notebook. Standard library only, so it runs in any Python 3.
"""
import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

LESSON = Path(__file__).resolve().parent.parent
ROOT = LESSON.parent
NOTEBOOKS = sorted((ROOT / "05 - Lecture and workshop").glob("Workshop notebook*.ipynb"))
NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"


def newest_deck():
    decks = [(tuple(map(int, m.group(1).split("."))), p) for p in ROOT.glob("*.pptx")
             if (m := re.search(r"(?: v| - Version )(\d+(?:\.\d+)*)(?: - [^/]*)?\.pptx$", p.name))]
    return max(decks)[1] if decks else None


def slide_code(deck):
    """Yield (slide number, [code lines]) for each slide that has code, in presentation order."""
    with zipfile.ZipFile(deck) as z:
        rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(z.read("ppt/_rels/presentation.xml.rels"))
                if r.tag == REL + "Relationship"}
        order = [rels[s.get(f"{{{NS['r']}}}id")] for s in ET.fromstring(z.read("ppt/presentation.xml"))
                 .find("p:sldIdLst", NS)]
        for n, target in enumerate(order, 1):
            lines = []
            for sp in ET.fromstring(z.read("ppt/" + target)).iter(f"{{{NS['p']}}}sp"):
                for para in sp.iter(f"{{{NS['a']}}}p"):
                    runs = para.findall("a:r", NS)
                    if runs and all(r.find("a:rPr/a:latin[@typeface='JetBrains Mono']", NS) is not None for r in runs):
                        off = sp.find(".//a:xfrm/a:off", NS)
                        lines.append((int(off.get("y")) if off is not None else 0,
                                      "".join(r.findtext("a:t", "", NS) for r in runs)))
            code = [t for _, t in sorted(lines, key=lambda x: x[0])]  # one text box per line: sort top to bottom
            if any("(" in t or "=" in t for t in code):  # monospace without either is printed output, not code
                yield n, code


def norm(text_lines):
    return [ln.rstrip() for ln in text_lines if ln.strip()]


def contains(block, want):
    return any(block[i:i + len(want)] == want for i in range(len(block) - len(want) + 1))


def main():
    deck = Path(sys.argv[1]) if len(sys.argv) > 1 else newest_deck()
    if deck is None:
        print("No deck found; nothing to check.")
        return
    lesson_blocks = [norm(b.splitlines()) for ep in sorted((LESSON / "episodes").glob("*.md"))
                     for b in re.findall(r"^```python[^\n]*\n(.*?)^```", ep.read_text(), flags=re.S | re.M)]
    nb_blocks = {nb.name: [norm("".join(c["source"]).splitlines()) for c in json.loads(nb.read_text())["cells"]
                           if c["cell_type"] == "code"] for nb in NOTEBOOKS}
    print(f"Deck: {deck.name}")
    failures = 0
    for n, lines in slide_code(deck):
        if any("\xa0" in ln for ln in lines):
            print(f"  slide {n:2d}  note: non-breaking spaces in the code; pasted into Python they raise a SyntaxError")
        want = norm([ln.replace("\xa0", " ") for ln in lines])
        where = {"lesson": any(contains(b, want) for b in lesson_blocks)}
        where.update({name: any(contains(b, want) for b in blocks) for name, blocks in nb_blocks.items()})
        missing = [k for k, ok in where.items() if not ok]
        failures += bool(missing)
        print(f"  slide {n:2d}  {'ok' if not missing else 'MISSING from ' + ', '.join(missing)}   ({want[0][:50]})")
        if missing:
            print("\n".join("      | " + ln for ln in want))
    print("ALL SLIDE CODE MATCHES" if not failures else f"{failures} SLIDE(S) DO NOT MATCH")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
