# Lesson data

The episodes load the 401(k) data with `fetch_401K()`, as DoubleML's example and the masterclass slides do. `401k.csv` is a copy for learners without internet access to GitHub (see the Setup page). It was exported on 24 October 2026.

## 401k.csv

- **Content**: 9,915 US households from the 1991 Survey of Income and Program Participation (SIPP), with net financial assets (`net_tfa`), 401(k) eligibility (`e401`) and participation (`p401`), and covariates (`age`, `inc`, `educ`, `fsize`, `marr`, `twoearn`, `db`, `pira`, `hown`, plus `nifa` and `tw`). 14 columns.
- **How it was obtained**: `doubleml.datasets.fetch_401K(return_type="DataFrame")` with DoubleML 0.11.4, which reads `sipp1991.dta` from https://github.com/VC2015/DMLonGitHub (the replication repository of Chernozhukov et al., 2018). Written with `DataFrame.to_csv(index=False)`. Reading the CSV back gives values identical to the downloaded data (only the storage types differ: the download uses float32 and int8). The random forest and lasso estimates in episodes 3 and 4 are identical to the cent with either source (checked 27 October 2026 with the pinned packages).
- **SHA-256**: `d4d4a8a4a06f7ebf89b9b9b96139ef0c2ca31178be728770d6c747f8dc8f51c4`
- **Licence and origin**: the underlying data are US Census Bureau SIPP public-use microdata, prepared by Poterba, Venti and Wise (1995) and used by Chernozhukov and Hansen (2004) and Chernozhukov et al. (2018). The replication repository carries an MIT-style licence (`LICENSE.Rmd`, copyright 2017 Chernozhukov, Chetverikov, Demirer, Duflo, Newey and Robins), which permits redistribution with the copyright notice. DoubleML itself is BSD-3-Clause licensed.

## References

Chernozhukov, V., & Hansen, C. (2004). The effects of 401(k) participation on the wealth distribution: An instrumental quantile regression analysis. *Review of Economics and Statistics*, *86*(3), 735–751. https://doi.org/10.1162/0034653041811734

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, *21*(1), C1–C68. https://doi.org/10.1111/ectj.12097

Chernozhukov, V., Hansen, C., Kallus, N., Spindler, M., & Syrgkanis, V. (2024). *Applied causal inference powered by ML and AI*. arXiv. https://doi.org/10.48550/arXiv.2403.02467

Poterba, J. M., Venti, S. F., & Wise, D. A. (1995). Do 401(k) contributions crowd out other personal saving? *Journal of Public Economics*, *58*(1), 1–32. https://doi.org/10.1016/0047-2727(94)01462-w
