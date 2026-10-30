"""Run every ```python block in the lesson episodes and check the locked results.

Usage (from any folder, with the lesson environment active):

    python tests/run_episodes.py            # run all episodes
    python tests/run_episodes.py 4 5        # run only episodes 4 and 5
    SAVE_FIGS=1 python tests/run_episodes.py   # also write figures to episodes/fig/

Rules:
- Episodes run in the order given in config.yaml, each in its own fresh Python
  process, with the working directory set to episodes/ so that "data/..." paths
  resolve as they do for learners.
- Blocks run top to bottom, solutions included, in one shared namespace per
  episode, as they would in a single notebook.
- A block that contains "____" is a fill-in-the-gaps template for learners and
  is skipped; its solution block is run instead.
- Lines starting with "!" (shell commands such as the slides' "!pip install")
  are skipped, as the lesson environment is already installed.
- As in Jupyter, the value of a block's last expression is printed.
- Every DoubleML model named in LOCKED that an episode fits must reproduce the
  locked estimate and confidence interval in numbers.json (a copy of the deck
  build's numbers_v5.json) to within TOLERANCE_USD dollars.

Outputs of every block are written to tests/output/<episode>.txt.
"""
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

LESSON = Path(__file__).resolve().parent.parent
EPISODES = LESSON / "episodes"
OUTPUT = LESSON / "tests" / "output"
NUMBERS = LESSON / "tests" / "numbers.json"  # copy of 05 - Lecture and workshop/build/v2/numbers_v5.json
TOLERANCE_USD = 1.0
# model variable in the episodes -> (model, learner) in numbers.json
LOCKED = {f"dml_{m}_{v}": (m, k) for m in ("plr",)
          for v, k in (("lasso", "lasso"), ("forest", "forest"), ("tree", "tree"), ("boost", "xgboost"))}

PRELUDE = textwrap.dedent('''
    import ast, json, os, sys
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    import pandas as _pd
    _pd.set_option("display.max_columns", 20)  # Jupyter's default; plain Python truncates to the terminal width
    _fig_n = [0]
    def _show(*args, **kwargs):
        if os.environ.get("SAVE_FIGS"):
            for num in _plt.get_fignums():
                _fig_n[0] += 1
                _plt.figure(num).savefig(f"fig/{EPISODE}-{_fig_n[0]}.png", dpi=110, bbox_inches="tight")
        _plt.close("all")
    _plt.show = _show
    _ns = {"__name__": "__main__"}
    # exec/eval are deliberate: the input is the lesson's own code blocks, run as a notebook would.
    def _run(src):
        tree = ast.parse(src)
        last = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        if last is not None:
            tree.body = tree.body[:-1]
        exec(compile(tree, "<block>", "exec"), _ns)
        if last is not None:
            value = eval(compile(ast.Expression(last.value), "<block>", "eval"), _ns)
            if value is not None:
                print(repr(value))
''')

CHECKS = textwrap.dedent('''
    _found = {}
    for _name in LOCKED_NAMES:
        _m = _ns.get(_name)
        if _m is not None and hasattr(_m, "summary"):
            _s = _m.summary
            _found[_name] = [float(_s["coef"].iloc[0]), float(_s["2.5 %"].iloc[0]), float(_s["97.5 %"].iloc[0])]
    print("@@RESULTS@@" + json.dumps(_found))
''')


def episode_order():
    lines = (LESSON / "config.yaml").read_text().splitlines()
    start = lines.index("episodes:") + 1
    order = []
    for line in lines[start:]:
        if not line.startswith("- "):
            break
        order.append(line[2:].strip())
    return order


def python_blocks(md_text):
    return re.findall(r"^```python[^\n]*\n(.*?)^```", md_text, flags=re.S | re.M)


def run_episode(name):
    stem = Path(name).stem
    blocks = python_blocks((EPISODES / name).read_text())
    script = [f"EPISODE = {stem!r}", f"LOCKED_NAMES = {sorted(LOCKED)!r}", PRELUDE]
    n_run = 0
    for i, block in enumerate(blocks, 1):
        if "____" in block:
            continue
        n_run += 1
        block = "\n".join(ln for ln in block.splitlines() if not ln.lstrip().startswith("!"))
        script.append(f"print('@@BLOCK {i}@@', flush=True)")
        script.append(f"_run({block!r})")
    script.append(CHECKS)
    env = dict(os.environ, PYTHONWARNINGS="ignore", OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "1"))
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-c", "\n".join(script)], cwd=EPISODES, env=env,
                          capture_output=True, text=True, timeout=1800)
    secs = time.time() - t0
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / f"{stem}.txt").write_text(proc.stdout + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else ""))
    results = {}
    for line in proc.stdout.splitlines():
        if line.startswith("@@RESULTS@@"):
            results = json.loads(line[len("@@RESULTS@@"):])
    return proc.returncode, proc.stderr, len(blocks), n_run, secs, results


def close(a, b, tol):
    return a is not None and abs(a - b) <= tol


def main():
    wanted = sys.argv[1:]
    locked = json.loads(NUMBERS.read_text())
    failures = []
    checks = []
    for name in episode_order():
        if wanted and not any(name.startswith(w + "-") for w in wanted):
            continue
        code, err, n_blocks, n_run, secs, results = run_episode(name)
        status = "ok" if code == 0 else "FAILED"
        print(f"{name:40s} {status:6s} {n_run}/{n_blocks} blocks run  {secs:6.1f} s")
        if code != 0:
            failures.append(name)
            print(textwrap.indent(err[-3000:], "    "))
        for var, got in results.items():
            model, learner = LOCKED[var]
            want = locked[model][learner]
            checks.append((f"{Path(name).stem[:2]}{var}", got, [want["coef"], want["lo"], want["hi"]]))
    unchecked = set(LOCKED) - {label[2:] for label, _, _ in checks}
    if not wanted and unchecked:  # a renamed model variable would otherwise go unchecked silently
        failures.append(f"never fitted: {sorted(unchecked)}")

    for label, got, want in checks:
        ok = all(close(g, w, TOLERANCE_USD) for g, w in zip(got, want))
        print(f"{'PASS' if ok else 'FAIL'}  {label}: got {[round(g, 2) for g in got]}, expected {want}")
        if not ok:
            failures.append(label)
    print("ALL PASSED" if not failures else f"{len(failures)} FAILURE(S): {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
