# ml-ats-workflow

A minimal educational workflow for understanding machine-learning-based algorithmic trading systems in finance.

This repository is meant for beginners who want to understand the end-to-end structure of a simple ML-based ATS pipeline. It focuses on workflow clarity, chronological evaluation, and clean separation between forecasting and trading.

## What this repository shows

This repository demonstrates how to:

1. load stock data,
2. split it chronologically into train, validation, and test,
3. build LSTM sequences,
4. train and validate an LSTM forecasting model,
5. save prediction timelines,
6. use those predictions inside a simple ATS,
7. tune a trading threshold on ATS validation only,
8. freeze that threshold before final ATS test evaluation.

## Repository structure

```text
ml-ats-workflow/
├── README.md
├── requirements.txt
├── LICENSE
├── src/
│   ├── ats/
│   │   ├── feeds.py
│   │   ├── run_backtest.py
│   │   └── strategy.py
│   ├── data/
│   │   ├── loaders.py
│   │   ├── sequence_builder.py
│   │   └── splits.py
│   ├── models/
│   │   └── lstm/
│   │       └── model.py
│   └── pipeline/
│       └── prediction_timelines.py
├── notebooks/
│   ├── 01_lstm_baseline_workflow.ipynb
│   └── 02_ats_baseline_workflow.ipynb
├── data/
│   └── prices/
└── outputs/
```
## Selected stock universe

This workflow uses a fixed set of 10 large-cap stocks:

- XOM
- AMZN
- AAPL
- JPM
- TSLA
- GOOGL
- WMT
- PG
- KO
- JNJ

## Notebook overview

### `01_lstm_baseline_workflow.ipynb`

Forecasting stage:
- load or download stock data
- split chronologically into train, validation, and test
- build LSTM sequences
- train and validate the model
- save prediction CSVs

### `02_ats_baseline_workflow.ipynb`

ATS stage:
- load saved prediction timelines
- split ATS validation and ATS test
- tune a simple threshold strategy on ATS validation only
- freeze the threshold
- evaluate the final portfolio on ATS test

## Workflow logic

Raw stock data  
→ Chronological train / validation / test split  
→ LSTM forecasting model  
→ Saved prediction timelines  
→ ATS validation / ATS test split  
→ Threshold tuning on ATS validation  
→ Frozen threshold  
→ Final ATS test evaluation

## Why this matters

This repository emphasizes two simple ideas:

- In time-series forecasting, chronology must be respected.
- Trading rules should not be tuned on final unseen test data.

The goal is not to claim profitability, but to show a clean and correct workflow.

## Installation

Install the required packages with:

```bash
pip install -r requirements.txt
```
## How to run

Run the notebooks in this order:

1. `notebooks/01_lstm_baseline_workflow.ipynb`
2. `notebooks/02_ats_baseline_workflow.ipynb`

Notebook 1 saves the prediction files that Notebook 2 uses for ATS evaluation.

## Notes

This repository is a small teaching baseline. It is not a production trading system, a full research codebase, or a claim of financial profitability :)

## Author

Advije Rizvani
