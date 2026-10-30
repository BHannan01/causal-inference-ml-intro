# Introduction to Causal Inference with Machine Learning

A Carpentries Workbench lesson that follows on from the QMEG (Quantitative Methods in Education Group) masterclass at the University of Southampton, 13 January 2027 (Microsoft Teams). It works through DoubleML's 401(k) example ([Impact of 401(k) on Financial Wealth](https://docs.doubleml.org/stable/examples/py_double_ml_pension.html)) in Python, continuing Part 4 of the masterclass slides. Code that appears on the slides appears in the lesson unchanged.

Status: pre-alpha, not yet published.

## Contents

- `episodes/`: five episodes (Markdown), figures in `episodes/fig/`, and an offline copy of the data in `episodes/data/`
- `learners/`: setup instructions and the reference page (glossary, references)
- `instructors/`: instructor notes (how the lesson fits the masterclass, timings, common errors)
- `requirements.txt`: the pinned Python packages
- `tests/run_episodes.py`: runs every code block in the episodes and checks every DoubleML estimate against the locked numbers
- `tests/check_deck_code.py`: checks that the code on the slides appears unchanged in the lesson and the workshop notebooks
- `tests/check_self_contained.py`: checks that a learner can follow the lesson from what is in it (no outside references, broken links or install gaps)
- `archive/`: earlier versions, kept locally and excluded from git

## Preview locally

Install R, pandoc and the Workbench packages (`sandpaper`, `varnish`, `pegboard` from https://carpentries.r-universe.dev). Copy this folder outside Google Drive, make it a git repository (`git init && git add -A && git commit -m start`), then in R from that folder:

```r
sandpaper::build_lesson()
```

## Check the code

With Python 3.12 or later:

```bash
python -m venv causal-ml-env
source causal-ml-env/bin/activate        # Windows: causal-ml-env\Scripts\activate
python -m pip install -r requirements.txt
python tests/run_episodes.py             # about 3 minutes; needs internet for fetch_401K
python tests/check_deck_code.py          # after any change to the slides
python tests/check_self_contained.py     # after any change to the text
```

The masterclass deck leads: when it changes, update the lesson and the workshop notebook to match, then run all three. `check_self_contained.py` fails on anything a learner could not do from the lesson alone: a reference to joining instructions, an email or a link "in the chat", leftovers such as FIXME or breakout rooms, a link to a missing file, an episode that uses DoubleML without installing it first, or a Setup page whose package versions differ from `requirements.txt`. It fails on the `FIXME` in `config.yaml` until the lesson has a GitHub repository.

On macOS, XGBoost needs the OpenMP library (`brew install libomp`).

## Licence

Instructional material: CC BY 4.0. Code: MIT. See `LICENSE.md`. Data sources and licences: `episodes/data/README.md`.

## Contact

Brody Hannan, Southampton Education School, University of Southampton (brody.hannan@soton.ac.uk).
