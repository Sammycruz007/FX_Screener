# Eagle Logic Trading: Multi-Asset FX Screener Pipeline

## 1. Project Scope, Goals, and Objectives

### Scope
Eagle Logic Trading is a production-grade, deterministic multi-asset quantitative scanner designed for the Forex market. It processes high-frequency technical adjustments down to hourly chunks across 25 major and minor currency pairs. The system is architected as an independent data ingestion, analytics, and signal evaluation pipeline, serving as the core algorithmic engine for an automated MLOps trading framework.

### Goals
To provide real-time, institutional-grade confluence scanning at critical market inflections (such as daily or session closes) without maintaining persistent database overhead, while guaranteeing statistical and algorithmic precision across asset matrices.

### Core Objectives
* **Zero-Lag Data Processing:** Ingest and process live exchange data while programmatically eliminating active, unclosed candle noise.
* **Statistical Regime Mapping:** Model price distribution using fixed-window rolling linear regression channels to determine macroeconomic trend dynamics and mathematical overextension zones.
* **Confluence Signal Generation:** Couple structural volatility boundaries with discrete price-action reversal states to isolate high-probability execution coordinates.
* **AI-Interoperable Determinism:** Maintain pure, mathematical logic blocks so execution outputs can be ingested or replicated identical-to-the-byte by downstream AI agents, microservices, or automated execution APIs.

---

## 2. System Architecture & Data Flow

The pipeline executes as a synchronous, single-threaded batch process divided into four decoupled stages:

[yfinance API]
│
▼ (Stage 1: Ingestion & Data Quality Gate)
[fetch_data()] ──► Flatten Multi-Index ──► Slice Live Candle (data.iloc[:-1])
│
▼ (Stage 2: Analytics & Feature Engineering)
[calculate_linreg_channels()] ──► Extract 200 H1 Bars ──► Ordinary Least Squares (OLS)
│
▼ (Stage 3: Pattern Identification)
[detect_price_action()] ──► Evaluate t-1 and t-2 Candlestick Vectors
│
▼ (Stage 4: State Machine Confluence)
[evaluate_setup()] ──► Structural State Matrix Mapping ──► STDOUT / Production Matrix


### Architectural Component Specifications
1.  **Ingestion Engine:** Targets the public Yahoo Finance API vector points. Configured to request a bound data horizon (`period="1mo"`) at localized interval steps (`interval="1h"`).
2.  **Transformation Layer:** Dynamically identifies and flattens column multi-indexing generated during batch execution. Converts multi-asset data matrices into index-clean pandas DataFrame structures.
3.  **Analysis Execution Core:** Converts price arrays into native NumPy arrays to fast-track ordinary least squares (OLS) linear equations. 
4.  **Logging Subsystem:** Implements two discrete, isolated loggers (`SystemLogger.Main` and `SystemLogger.DataIngestion`) using standard python logging interfaces to ensure tracking transparency during automated orchestration tasks.

---

## 3. Quantitative & Algorithmic Logic Specifications

To guarantee a 100% accurate replication, an executing agent must apply the mathematical and conditional logic defined below exactly.

### A. Linear Regression Channel Modeling
Given an array of the 200 most recent finalized closing prices where x ranges from 0 to 199, the regression line is modeled as:

y = mx + c

The slope (m) and intercept (c) are calculated via standard ordinary least squares (OLS) regression over the fixed window size N = 200:

m = (N * sum(x*y) - sum(x) * sum(y)) / (N * sum(x^2) - (sum(x))^2)
c = (sum(y) - m * sum(x)) / N

#### Volatility Band Calculation
The residual vector (r) represents the distance of each closing price from the regression line:

r[i] = y[i] - (m * x[i] + c)

The standard deviation (sigma) of the residuals determines the volatility stretch:

sigma = sqrt((1 / N) * sum(r[i]^2))

#### Boundaries at Current Horizon (x = 199)
* **Regression Mean Value:** Mean = m * 199 + c
* **Upper Channel Boundary (+2 StdDev):** Upper = Mean + 2 * sigma
* **Lower Channel Boundary (-2 StdDev):** Lower = Mean - 2 * sigma

### B. Price Action Reversal Vector Logic
Let the two most recent finalized rows in the dataset be t-2 (previous candle) and t-1 (current fully closed candle). Each row contains vectors for Open (O) and Close (C).

#### Bullish Engulfing Condition
A `BULLISH_REVERSAL` is confirmed if and only if:
1. The previous candle was bearish: C[t-2] < O[t-2]
2. The current candle is bullish: C[t-1] > O[t-1]
3. The body of the current candle completely wraps the body of the previous candle:
   O[t-1] <= C[t-2] AND C[t-1] > O[t-2]

#### Bearish Engulfing Condition
A `BEARISH_REVERSAL` is confirmed if and only if:
1. The previous candle was bullish: C[t-2] > O[t-2]
2. The current candle is bearish: C[t-1] < O[t-1]
3. The body of the current candle completely wraps the body of the previous candle:
   O[t-1] >= C[t-2] AND C[t-1] < O[t-2]

---

## 4. Execution Matrix & State Machine

The confluence logic maps three intermediate dimensions (Macro Regime, Structural Zone, and Price Action State) into a single deterministic output string (`STRATEGY SIGNAL`).

### State Variable Mapping Formulas
* **Macro Regime:** If m > 0 -> `📈 UP`. If m <= 0 -> `📉 DOWN`.
* **Structural Zone:** * If Close[t-1] >= Upper -> `PREMIUM`
  * If Close[t-1] <= Lower -> `DISCOUNT`
  * Else -> `EQUILIBRIUM`

### Signal Mapping Logic Table

| Macro Regime (m) | Structural Zone Status | Price Action State | Resulting STRATEGY SIGNAL |
| :--- | :--- | :--- | :--- |
| `📈 UP` | `PREMIUM` | Any | `❌ MISMATCH (UP)` |
| `📈 UP` | `DISCOUNT` | `BULLISH_REVERSAL` | `🔥 BUY TRIGGER` |
| `📈 UP` | `DISCOUNT` | `NONE` or `BEARISH` | `⏳ WATCHING BUY` |
| `📈 UP` | `EQUILIBRIUM` | Any | `💤 IDLE` |
| `📉 DOWN` | `DISCOUNT` | Any | `❌ MISMATCH (DOWN)` |
| `📉 DOWN` | `PREMIUM` | `BEARISH_REVERSAL` | `🔥 SELL TRIGGER` |
| `📉 DOWN` | `PREMIUM` | `NONE` or `BULLISH` | `⏳ WATCHING SELL` |
| `📉 DOWN` | `EQUILIBRIUM` | Any | `💤 IDLE` |

---

## 5. Data Quality Gates & Circuit Breakers

To prevent silent failures and corrupted technical indicators from reaching downstream execution layers, the Ingestion Engine passes all flattened arrays through a data validation gate prior to computing features:

* **Null/NaN Threshold:** If any asset dataset contains > 0.5% missing rows within the 200-bar window, that asset is dropped from the current run and an `ERROR` log is emitted.
* **Schema Enforcement:** Validates that incoming matrices strictly contain the required column structure: `[Open, High, Low, Close, Volume]`.
* **Price Sanity Check:** Checks for negative or zero values. If anomalies are detected, a circuit breaker trips, completely skipping calculations for that specific asset ticker to prevent mathematical distortion.

---

## 6. Testing & Determinism Verification

The pipeline enforces a strict verification paradigm. The codebase includes an automated test suite executed via pytest to guarantee logic and environment stability:

* **Mathematical Fixtures:** Uses a hardcoded, static 200-row synthetic price array with predefined slope and intercept values to verify that the OLS NumPy engine calculates to a tolerance of epsilon = 1e-7.
* **Edge Case Mocking:** Mocks external API responses using `unittest.mock` to simulate network timeouts, empty payloads, and half-day market closes, ensuring logging and retry mechanisms handle failures gracefully without crashing the core process.

---

## 7. Operational Deployment & Orchestration

This pipeline is engineered to run headlessly as a containerized, deterministic microservice.

### Containerization
The execution environment is completely isolated using a multi-stage Dockerfile based on `python:3.10-slim`, minimizing the final production image size (< 200MB) for rapid deployment cycles and dependency lock-in.

### Orchestration Interface
The `run_pipeline.py` entry point accepts dynamic command-line arguments via argparse, allowing workflow orchestrators (such as Apache Airflow, Prefect, or cron daemons) to dynamically override the default parameters or targets:


### 8. Project Structure & Environment

```bash
python run_pipeline.py --interval 1h --config path/to/custom_config.yaml

Directory Architecture
FX screener/
├── config/
│   └── config.yaml        # Externalized pipeline parameters
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py   # Ingestion, formatting, and live candle slicing
│   ├── features.py         # OLS channels and price action logic
│   └── pipeline.py         # Core orchestrator and matrix print engine
├── tests/
│   └── test_analytics.py   # Pytest deterministic mathematical validation
├── .venv/                 # Local Python Virtual Environment
├── Dockerfile             # Production container definition
├── README.md              # System blueprint documentation
└── run_pipeline.py        # Monolithic Pipeline Entry point
