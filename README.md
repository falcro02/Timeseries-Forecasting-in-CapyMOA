# Timeseries-Forecasting-in-CapyMOA

Optional project of the [Streaming Data Analytics](https://emanueledellavalle.org/teaching/streaming-data-analytics-2025-26/) course provided by [Politecnico di Milano](https://www11.ceda.polimi.it/schedaincarico/schedaincarico/controller/scheda_pubblica/SchedaPublic.do?&evn_default=evento&c_classe=837284&__pj0=0&__pj1=36cd41e96fcd065c47b49d18e46e3110).

Student: Alessio Caggiano, Angelo Baturi

## Project Overview

This project focuses on enabling **time series forecasting** in **CapyMOA** using **Streaming Machine Learning (SML)** techniques.

The student will design and implement a pipeline that transforms time series data into streaming regression problems, enabling forecasting through incremental learning models.

The final goal is to provide a **complete tutorial** demonstrating how CapyMOA can be used for time series forecasting.

## Background

Time series forecasting is traditionally addressed using batch learning methods. However, many real-world applications require:

- continuous data processing,
- adaptation to distribution shifts,
- real-time predictions.

Streaming Machine Learning (SML) naturally supports these requirements.

CapyMOA provides tools for streaming learning, but it does not offer a **ready-to-use pipeline for time series forecasting**.

This project aims to fill this gap by:

- transforming time series into supervised streaming problems,
- applying incremental regression models,
- evaluating forecasting performance in a prequential setting.

## Goals and Objectives

### 1. Lag-Based Stream Transformation (Core Task)

Design and implement a module that transforms a regression dataset into a forecasting problem by introducing lag features.

The implementation must:

- Generate lag features of the target:
  - Input: [y(t), y(t-1), y(t-2), ..., y(t-k)]
  - Target: y(t+1)
- Support configurable lag size (k)

(Optional):

- Include lagged input features;
- Add temporal features (day, season, hour, holidays).

### 2. Forecasting Task Definition

The student must implement and compare at least two forecasting settings:

- **One-step-ahead forecasting**:
  - Predict y(t+1) using past observations

- **Aggregated forecasting**:
  - Predict an aggregated value over a future horizon:
    - e.g., mean of y(t+1 ... t+H)

The horizon (H) must be configurable.

### 3. Streaming Simulation (Prequential Evaluation)

Simulate a real-time forecasting scenario using:

- Prequential evaluation

This step must clearly show how forecasting can be performed in a streaming setting.

### 4. Experimental Analysis

Evaluate:

- Effect of lag size (k)
- Effect of forecasting horizon (H)
- Comparison across models

Use at least the following models from CapyMOA:

- Hoeffding Tree Regressor
- Adaptive Random Forest Regressor
- SGD Regressor

### 5. Tutorial Development (Key Deliverable)

Develop a tutorial that explains:

- how to transform a regression dataset into a forecasting task;
- how to use lag features in streaming;
- how to perform forecasting with CapyMOA.

The tutorial must include code examples and experimental results.

## Datasets

The student should use two regression datasets available in CapyMOA:

- Bike
- Fried

Data must be processed in a streaming fashion.

## Repository Structure

- `src/forecasting/`: reusable forecasting library code
  - `lag_transformer.py`: lag-based feature transformation
  - `core.py`: horizon-based sample builder (`H=1` for one-step)
  - `__init__.py`: package exports
- `examples/`: runnable scripts
  - `run_part2_bike.py`: Part 2 example on Bike dataset
- `data/`: local datasets used by examples
- `README.md`, `requirements.txt`, `.gitignore`: project metadata and config

## Run Example (Part 2)

From the project root:

```bash
python examples/run_part2_bike.py
python examples/run_part2_fried.py
python examples/run_bike.py
python examples/run_part3_prequential_bike.py
python examples/run_part3_raw_bike.py
python examples/run_part3_prequential_fried.py
python examples/run_part3_raw_fried.py

# custom parameters
python examples/run_part2_bike.py --lag-size 48 --horizon 12 --include-input-lags --max-samples 5000
python examples/run_part2_fried.py --lag-size 24 --horizon 1 --max-samples -1
python examples/run_bike.py --lag-size 2 --horizon 3 --include-input-lags --show-samples 5
python examples/run_part3_prequential_bike.py --lag-size 24 --horizon 1 --model arf --max-samples 5000 --window-size 500
python examples/run_part3_prequential_bike.py --lag-size 24 --horizon 3 --model sgd --max-samples 5000 --window-size 500
python examples/run_part3_raw_bike.py --model arf --max-samples 5000 --window-size 500
python examples/run_part3_prequential_fried.py --lag-size 24 --horizon 1 --model arf --max-samples 2000 --window-size 500
python examples/run_part3_raw_fried.py --model arf --max-samples 2000 --window-size 500
```

Part 3 scripts now save only the plot image in `outputs/` using an automatic
file name based on input parameters:

- plot image (`.png`)

Use `--show-plot` if you also want to open the interactive plot window.

CLI parameters:

- `--lag-size`: lag window size `k`
- `--horizon`: forecasting horizon `H` for aggregated targets
- `--include-input-lags`: include lagged input features
- `--max-samples`: max samples per mode (`-1` means process all)

Note: examples now generate a single transformed dataset using horizon `H`.
Set `--horizon 1` to get one-step-ahead behavior.

`src/forecasting/core.py` is intentionally simplified and now assumes CapyMOA
regression stream instances (`instance.y_value` and `instance.x`) to make the
logic easier to read while covering both required datasets (Bike and Fried).

### Bike dataset reference

The Bike stream used in this project is the hourly transformed dataset stored in [data/bike.arff](data/bike.arff). It contains 12 input attributes and 1 regression target:

- `season`: season code, where 1 = winter, 2 = spring, 3 = summer, 4 = fall
- `yr`: year flag, where 0 = 2011 and 1 = 2012
- `mth`: month, from 1 to 12
- `hr`: hour of day, from 0 to 23
- `holiday`: holiday indicator, where 1 means holiday and 0 means not a holiday
- `weekday`: day of week, encoded from 0 to 6
- `workingday`: 1 if the day is neither weekend nor holiday, otherwise 0
- `weathersit`: weather situation code
  - 1 = clear, few clouds, partly cloudy
  - 2 = mist/cloudy conditions
  - 3 = light snow, light rain, thunderstorm/scattered clouds
  - 4 = heavy rain, ice pellets, thunderstorm, mist or snow/fog
- `temp`: normalized temperature
- `atemp`: normalized feels-like temperature
- `hum`: normalized humidity
- `windspeed`: normalized wind speed
- `cnt`: total number of rented bikes, used as the forecasting target

This means the forecasting task is to predict future values of `cnt` using lagged `cnt` values and, optionally, the other covariates.

## Deliverable

The student must provide:

1. Code implementation
2. Tutorial in the CapyMOA style: https://capymoa.org/tutorials.html
3. Presentation of the notebook and results obtained

## Note for Students

- Clone the created repository offline;
- Add your name and surname into the Readme file;
- Make any changes to your repository, according to the specific assignment;
- Add a `requirement.txt` file for code reproducibility and instructions on how to replicate the results;
- Commit your changes to your local repository;
- Push your changes to your online repository.
