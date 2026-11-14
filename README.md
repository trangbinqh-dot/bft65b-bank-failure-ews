# BFT65B – A Big-Data Pipeline for Bank Failure Early Warning

End-to-end, leak-free, reproducible data engineering and machine learning pipeline built on four decades of U.S. FDIC Call Reports (1984Q1–2026Q2) to rank insured institutions by 4-quarter ahead failure risk.

## 1. Repository Structure

```text
bft65b-bank-failure-ews/
├── src/
│   ├── common.py                 # Shared schema-harmonised loader & logging utilities
│   ├── 00_schema_audit.py        # Part 1/2: Audit schema drift across 170 quarterly Parquet files
│   ├── 01_download_fdic.py       # Part 1: Resumable, paginated, self-verifying FDIC API downloader
│   ├── 02_storage_bench.py       # Part 1: CSV vs. Parquet (Uncompressed/Snappy/Zstd/Hive) benchmark
│   ├── 03_data_quality.py        # Part 2: Automated data quality & YTD cumulative income audit
│   ├── 03a_inspect.py            # Part 2: Supplementary anomaly & extreme-ratio inspection
│   ├── 03b_exit_codes.py         # Part 2: Bank exit vs. failure reconciliation (Institutions vs. Failures)
│   ├── 04_labels.py              # Part 2: Point-in-time (q, q+4] failure labelling & COST isolation
│   ├── 05_clean_panel.py         # Part 2: YTD de-cumulation (*_Q), post-failure removal & panel export
│   ├── 05a_ytd_check.py          # Part 2: Verification of quarterly flow de-cumulation
│   ├── 05b_ntlnls_check.py       # Part 2: Net charge-offs YTD monotonicity check
│   ├── 06a_polars_pipeline.py    # Part 3: Core Polars Lazy pipeline & query plan extraction
│   ├── 06b_benchmark.py          # Part 3: Subprocess-isolated benchmark (Pandas vs. Polars Eager/Lazy)
│   ├── 06c_scaling.py            # Part 3: 1x / 2x / 3x scaling experiment on 8 GB RAM laptop
│   ├── 06d_pyspark_colab.ipynb   # Part 3: PySpark 4.0.5 equivalence check & Catalyst Physical Plan (Colab)
│   ├── 07_features.py            # Part 4: 29 CAMELS ratios (true QIDX+4 lags) + point-in-time truncation test
│   ├── 08_preprocess.py          # Part 4: Leak-free sklearn Pipeline (QuantileWinsorizer -> Imputer -> Scaler)
│   ├── 09_modelling.py           # Part 5: Expanding-window temporal CV, cost threshold & single Test evaluation
│   └── 09b_mllib_colab.ipynb     # Part 5: Distributed Spark MLlib Pipeline demonstration (Colab)
├── outputs/                      # Saved benchmark CSVs, execution plans, quality audits, and model logs
├── data_raw/                     # Raw FDIC API Parquet files & manifests (git-ignored)
├── data_processed/               # Clean panel & engineered feature Parquet tables (git-ignored)
├── requirements.txt              # Pinned Python package versions
└── README.md