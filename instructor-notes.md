---
title: 'Instructor Notes'
---

## How the lesson fits the masterclass

The masterclass (13 January 2027, Microsoft Teams) is an 80-minute lecture followed by a 40-minute workshop in Google Colab. This lesson is the self-paced follow-up, which learners work through afterwards on their own laptops. It picks up where Part 4 of the slides leaves off and uses the same example, DoubleML's [Impact of 401(k) on Financial Wealth][doubleml-401k].

The code on the masterclass slides (slides 35 to 38) appears unchanged in episodes 1 and 3 and in the workshop's Colab notebook, so learners who attended will recognise it. Learners who did not attend can follow the lesson without the slides; the slide numbers are pointers for those who have them.

## If you teach it live

The episodes add up to 110 minutes. If you have a two-hour follow-up session, you could run it like this:

| Clock | Episode | Slides | Minutes |
|---|---|---|---|
| 0:00 | Welcome; everyone runs the install check | | 5 |
| 0:05 | 1. The question and the data (short: the slides covered it) | 8, 10, 11, 21, 29 | 10 |
| 0:15 | 2. How DML works, by hand | 30 to 33 | 25 |
| 0:40 | 3. DML in five steps | 35 to 39 | 15 |
| 0:55 | Break | | 5 |
| 1:00 | 4. Same question, four learners | 40 | 25 |
| 1:25 | 5. Checks, and your own research | 21, 40 | 25 |
| 1:50 | Questions | | 10 |
| 2:00 | Close | | |

## Episode notes

### Episode 1

This episode is mostly a recap of slides 8 to 11, 21 and 29. Challenge 1 (participation is a mediator) is a good moment to talk about bad controls.

### Episode 2

The cross-fitted predictions take about 20 seconds to run. Keep the attention on the three steps and spend as little time as you can on `cross_val_predict` itself. The slides say "residual" on slides 30 to 32 and "leftover" in the slide 32 chart and on slide 33. The lesson says "residual" throughout, and the callout at the start of the episode explains that a leftover is the same thing. The "why the slope is the effect" section covers the algebra that the slides leave out. The Frisch–Waugh check usually lands well, because learners can see the slope of the residuals equal the regression coefficient to the cent.

### Episode 3

The output must match slide 39 exactly ($8,909.63, 95% CI $6,318.91 to $11,500.36). If it does not, the usual causes are a missing `np.random.seed(123)` immediately before `DoubleMLPLR`, an extra random call between the seed and `fit()`, or different package versions.

### Episode 4

This episode follows the DoubleML example cell by cell. Take some time over the raw-variable lasso ($5,722), which the notes to slide 40 also mention. A learner that can only fit straight lines cannot follow the curve in income, so its estimate lands well away from the others.

### Episode 5

The final discussion asks learners where DML could help in their own research. In a live session it works well in pairs, especially when the partners come from different fields.

## Common learner errors

| Symptom | Likely cause and fix |
|---|---|
| `XGBoostError: ... libomp.dylib could not be loaded` (macOS) | XGBoost needs OpenMP. `brew install libomp` with Homebrew, or skip the boosting cells, or use Colab. |
| `XGBoostError` with `Symbol not found: ___kmpc_dispatch_deinit` and a path inside `anaconda3` (macOS) | The environment was built on Anaconda's Python, and XGBoost found Anaconda's old OpenMP. `brew install libomp` fixes it (XGBoost looks for Homebrew's copy first); with Intel Homebrew on an Apple-chip Mac, `arch -x86_64 /usr/local/bin/brew install libomp`. Then restart the kernel. Learners who already had Python skip setup step 1, so the OpenMP step now sits in step 3. |
| `fetch_401K` fails with a URL, SSL or certificate error | No internet access to GitHub, or (macOS) Python's certificates were not installed: run **Install Certificates.command**. Fallback: `pd.read_csv('401k.csv')` with the file from the lesson's data folder. |
| `ModuleNotFoundError: No module named 'doubleml'` | The notebook is using a different Python. Activate the virtual environment before running `jupyter lab`, or choose the right kernel in **Kernel > Change Kernel**. |
| Different DML estimate from the slides | `np.random.seed(123)` missing or not immediately before creating the model, or different package versions (check with the install check). |
| Summaries print with `...` instead of all columns | A narrow display. `pd.set_option('display.width', 120)` or `print(x.summary.round(2))`. |
| `NameError: name '____' is not defined` | A fill-in exercise run unchanged. Each `____` must be replaced first; the solution under the challenge has the answer. |
| `NameError` for `data`, `data_dml_base` or `randomForest` | Cells run out of order, or the kernel restarted. Each episode's first block sets everything up; rerun from the top. |
| A warning that `use_label_encoder` is not used | From the DoubleML example's XGBoost settings; harmless with recent XGBoost. |
| Boosting or forests take minutes | Thread oversubscription on some setups. Restart the kernel and set `os.environ["OMP_NUM_THREADS"] = "1"` first. |

## Checking the lesson code

The script `tests/run_episodes.py` runs every `python` code block in the episodes, in the order given in `config.yaml`. Each episode runs in a fresh Python process with `episodes/` as the working directory. The script skips lines starting with `!`, and it skips blocks containing `____` because these are fill-in templates (their solutions do run). It also checks every DoubleML model in the lesson (the partially linear model with four learners) against `tests/numbers.json`, a copy of the deck build's `numbers_v5.json`, to within $1. Setting `SAVE_FIGS=1` regenerates the figures in `episodes/fig/`. The run needs internet access for `fetch_401K` and takes about three minutes.
