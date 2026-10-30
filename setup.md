---
title: Setup
---

We will run the code on your own laptop in JupyterLab, using an ordinary installation of Python and pip. Setting up takes about 15 minutes and needs around 1 GB of disk space. Please run the install check at the end of this page before you start the lesson.

If you cannot install software on your laptop, every episode also runs in [Google Colab](https://colab.research.google.com/), which needs only a Google account. Start a new notebook there and paste in the code as you go. Colab already has everything you need except DoubleML. Colab forgets installed packages whenever it restarts, so the first code block of every episode starts with `!pip install doubleml`. Package versions in Colab change over time, so your numbers might differ slightly from the ones in the lesson.

## 1. Install Python

You need Python 3.12 or later. To check which version you have, open a terminal (**Command Prompt** on Windows, **Terminal** on macOS) and type `python --version` (or `python3 --version` on macOS). If you already have 3.12 or later, skip to step 2.

:::::::::::::::: spoiler

### Windows

1. Go to [python.org/downloads/windows](https://www.python.org/downloads/windows/). Under the newest release, choose **Download Windows installer (64-bit)**, or the ARM64 installer if your laptop has an ARM processor. The page also offers the "Python install manager", but these instructions use the installer.
2. Run the installer. On the first screen, tick **Add python.exe to PATH**. If you do not have administrator rights on your laptop, untick **Use admin privileges when installing py.exe**. Then choose **Install Now**.
3. Close and reopen Command Prompt. Use Command Prompt for every step on this page; in PowerShell, Windows may block the command that activates the environment in step 2.

::::::::::::::::::::::::

:::::::::::::::: spoiler

### macOS

1. Download the latest Python 3 installer (the "macOS 64-bit universal2 installer") from [python.org/downloads](https://www.python.org/downloads/) and run it.
2. When the installer finishes, it opens the Python folder. Double-click **Install Certificates.command** in that folder, otherwise downloading the data can fail later with a certificate error.

On macOS, type `python3` wherever these instructions say `python`.

::::::::::::::::::::::::

## 2. Create a folder and a virtual environment

We will install the lesson's packages in a **virtual environment**, which keeps them separate from anything else on your computer. A new terminal starts in your home folder. Run the following to make a folder for the lesson there, move into it and create the environment:

```bash
mkdir causal-ml
cd causal-ml
python -m venv causal-ml-env
```

Next, activate the environment. Each time you open a new terminal for the lesson, you will need to run `cd causal-ml` and then this command:

```bash
# Windows
causal-ml-env\Scripts\activate
# macOS
source causal-ml-env/bin/activate
```

You should see that the prompt now starts with `(causal-ml-env)`.

## 3. Install the packages

With the environment active, install the packages. The version numbers are the ones we used to make the lesson and the masterclass slides, so pinning them means you should get the same numbers as we do:

```bash
python -m pip install doubleml==0.11.4 scikit-learn==1.9.1 xgboost==3.4.1 numpy==2.5.4 pandas==3.1.0 scipy==1.18.1 matplotlib==3.11.2 seaborn==0.13.2 statsmodels==0.15.0 jupyterlab==4.6.4
```

It will take a few minutes.

On macOS there is one more step, even if you already had Python. The XGBoost library, which we use for boosting in episode 4, needs the OpenMP library, and macOS does not include it. If you use [Homebrew](https://brew.sh/), run this in Terminal:

```bash
brew install libomp
```

If Homebrew refuses because it is the Intel version on a Mac with an Apple chip, run `arch -x86_64 /usr/local/bin/brew install libomp` instead. If you do not use Homebrew, you can skip the boosting cells in episode 4 or run that episode in Colab.

## 4. Start JupyterLab

Make sure the environment is active and the terminal is in your lesson folder, then start JupyterLab:

```bash
jupyter lab
```

JupyterLab should open in your web browser. Create a new notebook with **File > New > Notebook** and choose the "Python 3" kernel.

## 5. Run the install check

Paste the following into the first cell and run it with **Shift + Enter**. The first run can take a minute or so with nothing on screen, because the packages are loading for the first time and the data are downloading. While it runs, the brackets beside the cell show `[*]`, and they change to a number when it has finished.

```python
import sys
import doubleml, sklearn, xgboost, pandas, numpy, matplotlib, seaborn
from doubleml.datasets import fetch_401K

print("Python      ", sys.version.split()[0])
for module in (doubleml, sklearn, xgboost, pandas, numpy, matplotlib, seaborn):
    print(f"{module.__name__:12s}", module.__version__)
print("401(k) data:", fetch_401K(return_type='DataFrame').shape)
```

You should see:

```output
Python       3.12.11
doubleml     0.11.4
sklearn      1.9.1
xgboost      3.4.1
pandas       3.1.0
numpy        2.5.4
matplotlib   3.11.2
seaborn      0.13.2
401(k) data: (9915, 14)
```

Your Python version may well be newer than ours. Any version from 3.12 onwards is fine, but the package versions should match.

## If something goes wrong

- If you see `XGBoostError ... libomp.dylib could not be loaded` on macOS, install OpenMP as described at the end of step 3, then restart the kernel. The same error can say `Symbol not found: ___kmpc_dispatch_deinit` and name a folder inside `anaconda3`. That means XGBoost found Anaconda's older copy of OpenMP, and installing OpenMP with Homebrew fixes it.
- If `fetch_401K` fails with a connection or certificate error, remember that the data come from GitHub, so you need an internet connection that allows it. On macOS, also run **Install Certificates.command** (step 1). As a last resort, download `401k.csv` from this lesson's [data folder](../episodes/data/401k.csv) and save it in the same folder as your notebook (in Colab, upload it with the folder icon on the left). Then replace `fetch_401K(return_type='DataFrame')` with `pd.read_csv('401k.csv')`, after `import pandas as pd`. We checked that the random forest and lasso estimates in episodes 3 and 4 come out the same to the cent, although summaries of the data show different storage types.
- University-managed laptops may block installs or route traffic through a proxy. On Windows, the python.org installer works without administrator rights if you untick **Use admin privileges when installing py.exe**. If pip fails with an SSL or proxy error, try again off the university VPN, or use Colab instead.
- If boosting is very slow, XGBoost and scikit-learn may be starting too many threads. This happens on a few setups, for example Intel builds of Python running on an Apple Silicon Mac. Restart the kernel and run `import os; os.environ["OMP_NUM_THREADS"] = "1"` before anything else.
