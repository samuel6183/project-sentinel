# Advanced model methodology

The original 80/20 benchmark remains archived and reproducible. The advanced protocol uses approximately 60% training, 10% calibration, 10% validation and 20% final test; entire timestamp groups stay together. Features use only simulated information available before settlement. No settlement outcomes, status or label are model inputs.

Four fixed model families: logistic regression, random forest, XGBoost and isolation forest. Class weights address imbalance in supervised training; isolation forest uses no labels during fitting, but its score is calibrated using labeled calibration data. Each model's score is mapped to probability using a separate logistic calibrator. No test-set threshold tuning. The validation-selected champion minimizes illustrative financial loss, breaking ties by validation average precision. The final test compares the frozen configurations and a no-alert baseline.

Illustrative review cost: KES 25 per flagged transaction; missed synthetic positive cost: KES 1,000; validation alert capacity: 5%. These are assumptions, not observed losses. The validation capacity does not guarantee a test-period alert rate below 5%. A threshold above the largest validation score represents no alerts on validation, not a universal no-alert policy.

Report ROC-AUC, average precision, Brier score, log loss, precision, recall, F1, confusion counts, alert workload and hypothetical cost. Compare average precision with test label prevalence. Baseline probabilities use training prevalence. No test results are used for selection.

This is an exploratory upgrade on the same simulated dataset, whose prior test results have already been seen. A truly fresh external holdout or independent simulation seed, rolling-origin validation, uncertainty intervals, calibration plots and sensitivity analysis are future work. No production fraud claims are justified.
