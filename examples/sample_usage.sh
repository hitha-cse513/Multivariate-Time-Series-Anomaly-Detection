#!/usr/bin/env bash
python -m mtsad.main   --input_csv ./your_input.csv   --output_csv ./your_output.csv   --model pca   --train_start "2004-01-01 00:00"   --train_end   "2004-01-05 23:59"   --analysis_start "2004-01-01 00:00"   --analysis_end   "2004-01-19 07:59"   --timestamp_col "Time"
