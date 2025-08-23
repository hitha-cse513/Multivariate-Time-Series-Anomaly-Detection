# Multivariate Time Series Anomaly Detection (MTSAD)


## Project Overview
This project implements a **Python-based machine learning solution to detect anomalies in multivariate time series data**. It can:  

- Detect abnormal patterns (anomalies) in time series with multiple variables.  
- Identify the **top 7 contributing features** for each anomaly.  
- Output a CSV with **abnormality scores (0–100)** and top contributing features.  

This is intended for **predictive maintenance, industrial sensor monitoring, and performance analytics**.

---

## Problem Statement
Organizations monitor multiple sensors or IoT devices to identify potential issues, predict failures, and optimize maintenance schedules. Anomalies are patterns or events in the dataset that deviate from the expected normal behavior.  

**Goal:** Develop a CLI-based Python solution that:  
1. Takes a CSV of time series data.  
2. Trains a model on a known normal period.  
3. Detects anomalies and ranks feature contributions.  
4. Outputs the original CSV with 8 additional columns:  
   - `abnormality_score` (0–100)  
   - `top_feature_1` … `top_feature_7`  

**Training Period:** `1/1/2004 0:00` → `1/5/2004 23:59`  
**Analysis Period:** `1/1/2004 0:00` → `1/19/2004 7:59`  

---

## Features
- Supports **PCA** or **Isolation Forest** models.  
- Handles **multivariate time series** with any number of features.  
- Adds **abnormality scores scaled 0–100**.  
- Calculates **top 7 contributing features** for anomalies.  
- Handles **constant features, missing data, and small zero-variance noise**.  
- Validates **training anomalies** to ensure mean <10 and max <25.  
- CLI-based: no GUI required.  

---
## Key Modules

| Module           | Purpose                                                                                  |
| ---------------- | ---------------------------------------------------------------------------------------- |
| `data.py`        | Handles data preprocessing, splitting, timestamp validation, and missing value handling. |
| `model.py`       | Implements PCA and Isolation Forest anomaly detection models.                            |
| `scoring.py`     | Scales raw anomaly scores to 0–100 percentiles.                                          |
| `attribution.py` | Identifies top contributing features for each anomaly.                                   |
| `io_utils.py`    | Reads/writes CSVs with augmented columns.                                                |
| `main.py`        | CLI entry point orchestrating the full MTSAD pipeline.                                   |
| `config.py`      | Default constants for training and analysis periods, timestamp column, etc.              |

## Requirements
Python 3.9+ with the following packages:  
```text
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
```

## Installation

1. Clone the repository:  
```bash
git clone "https://github.com/hitha-cse513/Multivariate-Time-Series-Anomaly-Detection"
cd Multivariate-Time-Series-Anomaly-Detection
```
2.Create a virtual environment and activate it:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```
3.Install required packages:
```bash
pip install -r requirements.txt
```

## Usage
CLI Usage
Run the anomaly detection pipeline using the CLI:
```bash
python -m mtsad.main --input_csv input.csv --output_csv output.csv
```

## Optional: Validation of Training

To check anomaly scores on the training period:
```bash
python -m mtsad.main --input_csv pollution.csv --output_csv output.csv --validate_training
```
Sample output:
```bash
[Training validation] mean=0.37, max=7.72
```

## Output

The pipeline generates a CSV with 8 additional columns:
```bash

abnormality_score → Float, 0–100

top_feature_1 → Top contributing feature name

top_feature_2 → Top contributing feature name

…

top_feature_7 → Top contributing feature name
```
If fewer than 7 features contribute, remaining columns are filled with empty strings.


