Provide example time series at TSB-AD-U/M Folder

Link to the dataset:

* TSB-AD-U: https://www.thedatum.org/datasets/TSB-AD-U.zip

* TSB-AD-M: https://www.thedatum.org/datasets/TSB-AD-M.zip

> Disclaimer: The dataset is released for reproducibility purposes. The preprocessing and curation steps are provided under the Apache 2.0 license. If you use any of these datasets in your research, please refer to the original data source. License information for each dataset included in TSB-AD is provided at [[Link]](https://thedatumorg.github.io/TSB-AD/) for your reference.

* File Name Formatting: [index]\_[Dataset Name]\_id\_[id]\_[Domain]\_tr\_[Train Index]\_1st\_[First Anomaly Index].csv
    * Domain ⊆ {Web Service, Sensor, Environment, Traffic, Finance, Facility, Medical, Synthetic}
* Folder Description: `TSB-AD-U/M` contain univariate and multivariate time series respectively. `File-List` contains file lists splitting for evaluation and hyperparameter tunning.

### OTel (multivariate, index 201 to 220)

Twenty multivariate series of OpenTelemetry telemetry from two Kubernetes microservice testbeds (OpenTelemetry Demo and Sock Shop on AWS EKS), one series per (signal, target service): traces (11 features), metrics (65 features) and logs (5 features), aggregated into 60-second windows. Anomalies are Chaos Mesh fault injections (network delay, HTTP 500 errors, CPU stress, memory stress and two-service cascades) from a 400-run campaign with a 24-hour normal baseline before it. A window is labeled anomalous when a run in the campaign manifest targeted that service and the window falls in the fault's active span. The 5-minute cooldown after each fault is labeled normal. Windows during faults on other services are labeled normal. Prevalence is 0.5 to 10.3 percent with 10 to 70 anomaly segments per series. `tr` is the end of the baseline period.

* Source: https://doi.org/10.5281/zenodo.19462083 (CC-BY 4.0). Reference: M. A. Anjum, "Evaluating ML-Based Anomaly Detection on Unified OpenTelemetry Telemetry: An Empirical Study Across Traces, Metrics, and Logs," IEEE Access, vol. 14, 2026, https://doi.org/10.1109/ACCESS.2026.3705430
* `OTel/convert_otel_to_tsbad_m.py` regenerates the CSVs from the Zenodo feature tables and the campaign manifest. `OTel/OTel_series_summary.csv` lists each series with its signal, service, length, prevalence and segment count.
