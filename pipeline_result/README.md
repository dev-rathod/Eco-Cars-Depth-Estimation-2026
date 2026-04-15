# Pipeline Result

This folder contains calibration reports generated from the GitHub `model_inferencing` pipeline inputs for 22 segments.

## What Is Here

- `summary.csv`: one row per segment with the selected calibration model and evaluation metrics
- `summary.json`: aggregate counts and average metrics across the processed segments
- `reports/*.md`: human-readable per-segment calibration reports
- `reports/*.json`: machine-readable per-segment outputs for the segments generated in the later batch reruns
- `generate_calibration_reports.py`: batch utility used to generate these reports from the downloaded GT zip plus Drive prediction/image zips

## Metric Meaning

- `AbsRel`: mean absolute relative error. Lower is better.
- `RMSE`: root mean squared error in meters. Lower is better.
- `delta1`: fraction of valid pixels whose prediction is within a 1.25x ratio of ground truth. Higher is better.

## Calibration Logic

- The first `5` matched frames are used for calibration.
- The script fits `hyperbolic`, `power`, `log`, and `exponential` mappings.
- The report selects the model with the lowest calibration RMSE for each segment.

## Important Caveat

The current outputs show perfect metrics for all 22 segments:

- `avg_AbsRel = 0.0`
- `avg_RMSE = 0.0`
- `avg_delta1 = 1.0`

That can mean the prediction arrays and GT arrays were already numerically identical or already perfectly aligned before calibration. These results should therefore be presented carefully and verified before claiming genuine model performance gains.
