# Data Folder

This folder contains the fixed CSV snapshot used for the final thesis modelling pipeline.

## Required File

```text
canonical_1m_rth_FULL.csv
```

## Description

The dataset contains 1-minute Regular Trading Hours (RTH) OHLC stock data for four NASDAQ symbols:

- AAPL
- MSFT
- NVDA
- TSLA

The final experiment uses the following fixed snapshot period:

```text
2025-01-02 09:30:00 America/New_York
to
2026-02-13 15:59:00 America/New_York
```

## Required Columns

The modelling scripts require at least the following columns:

- symbol
- datetime
- open
- high
- low
- close

The raw source may contain additional columns such as volume, but volume is not used in the final modelling feature set.

## Filtering Policy

Only complete Regular Trading Hours trading days are retained.

A valid trading day must contain exactly:

```text
390 one-minute bars per symbol
```

Incomplete days and half-days are excluded during script execution.

## Timezone Handling

- Raw timestamps are interpreted as UTC.
- Timestamps are converted to America/New_York inside the model scripts.
- Snapshot boundaries are defined in America/New_York time.

## Derived Features

The following features are computed inside the model scripts:

- change_ratio
- macd
- minute_of_day
- day_of_week
- month_of_year

## Reproduction Requirement

Before running the pipeline, make sure the dataset is located exactly at:

```text
Data/canonical_1m_rth_FULL.csv
```

The model scripts use relative project paths, so the project can be cloned and executed on another machine without changing hard-coded server paths.
