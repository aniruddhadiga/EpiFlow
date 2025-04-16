# Epiflow

EpiFlow: Epidemic Predictive Intelligence Framework Leveraging Observables from Wastewater Surveillance

The codebase contains scripts for preprocessing and extracting metrics from wastewater viral load datasets, and evaluating their utility in predicting various clinical and syndromic surveillance datasets. 

1. Data Preprocessing Module - This module primarily focuses on formatting time series data and ensuring temporal and spatial alignment across multiple datasets, facilitating accurate comparison. Key steps include data cleansing and harmonization of temporal and spatial dimensions. The necessary functions are provided in `viral_utils.py`. An example of the data preprocessing workflow is provided in `analysis_workflow.ipynb`.
2. Signal Analysis Module - This module focuses on ($i$) determining the appropriate window length based on predictability, ($ii$) denoising, and ($iii$) the causal relationship between the time series.
3. 

For examples related to the analysis of the wastewater data and understanding its relationship to burden indicators (hospitalizations), refer to
```
analysis_workflow.ipynb
```
For examples related to generating forecasts using the vector autoregressive model, refer to
```
forecast_workflow.ipynb
```
